"""Host (CPU) Fourier-domain bin-mean line reduction (#1230).

The NumPy-backend counterpart of ``_fourier_cuda.cuda_fourier_bin_mean_reduce``.
It uses the same exact-cutoff mode partition (``_fourier_cutoffs``), period
(``fourier_period``), broad-line route (``_bin_broad``) and envelope-image
correction (``_bin_fourier``). Large rectangles go through a two-row type-1
NUFFT: the exponential-of-semicircle (ES) kernel of ``_fourier_spread``
(Barnett, Magland and af Klinteberg, SIAM J. Sci. Comput. 41 (2019) C479),
spread by a serial numba loop and transformed with ``scipy.fft``. Smaller
rectangles take a direct numba sum. Both are serial, so a host result does
not depend on thread count.

Without it, CPU bin-mean scoring builds a dense ``(line, bin)`` matrix, and
its cost grows with lines × bins. With it, the cost on a uniform axis grows
mainly with lines, as on the device.

Validation: cxr-line-fourier-reduction
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import scipy.fft as sf
from numba import njit, prange

from ._bin_broad import broad_bin_mean_reduce, broad_width_for_axis
from ._bin_fourier import (
    FOURIER_MAX_LINE_MODES,
    FOURIER_MIN_RESIDUAL_TOL,
    FOURIER_RESIDUAL_TOL,
    _envelope_image_correction,
    fourier_period,
)
from ._bin_mean_jit import host_near_far_bin_mean_reduce
from ._fourier_cutoffs import cutoff_rectangle_array
from ._fourier_spread import _BETA_PER_WIDTH, _deconvolution, _fine_size, es_width

#: Lines queued per target before a host flush (three FP64 arrays each).
HOST_FOURIER_FLUSH_LINES = 4_000_000
#: A rectangle at least this large (sources × modes) takes the NUFFT; the
#: device route's threshold.
_NUFFT_MIN_WORK = 262_144
_NUFFT_MIN_SOURCES = 64
_NUFFT_MIN_MODES = 32


@lru_cache(maxsize=8)
def _cached_deconvolution(modes, n_fine, width):
    """Immutable host factors; callers admit at most 1M entries (64 MiB cache)."""
    factors = _deconvolution(np, modes, n_fine, width)
    factors.flags.writeable = False
    return factors


@njit(cache=True, nogil=True)
def _es_spread(x, sources, n_fine, width, beta, grid):  # pragma: no cover - numba
    two_pi = 2.0 * np.pi
    step = two_pi / n_fine
    half = 0.5 * width
    for k in range(x.shape[0]):
        xk = x[k] % two_pi
        u = xk / step
        l0 = int(np.ceil(u - half))
        for p in range(width):
            index = l0 + p
            z = (index - u) / half
            arg = 1.0 - z * z
            if arg <= 0.0:
                continue
            phi = np.exp(beta * (np.sqrt(arg) - 1.0))
            index %= n_fine
            grid[0, index] += phi * sources[0, k]
            grid[1, index] += phi * sources[1, k]


def host_es_mode_transform(phase, strengths, j0, j1, *, eps=1e-12):
    """Two-row type-1 transform ``sum_k strengths[r, k] exp(-i j phase_k)``, ``j0 <= j < j1``.

    Host counterpart of ``_fourier_spread.es_mode_transform`` with the same
    kernel width, shape parameter, upsampling and deconvolution.

    Validation: cxr-line-fourier-reduction
    """
    modes = j1 - j0
    width = es_width(eps)
    n_fine = _fine_size(modes, width)
    shift = j0 + modes // 2
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    sources = np.ascontiguousarray(strengths * np.exp(-1j * shift * phase), dtype=np.complex128)
    grid = np.zeros((2, n_fine), dtype=np.complex128)
    _es_spread(phase, sources, n_fine, width, _BETA_PER_WIDTH * width, grid)
    spectrum = sf.fft(grid, axis=1)
    centred = np.arange(-(modes // 2), modes - modes // 2) % n_fine
    factors = (
        _cached_deconvolution(modes, n_fine, width)
        if modes <= 1 << 20
        else _deconvolution(np, modes, n_fine, width)
    )
    return spectrum[:, centred] * factors


@njit(parallel=True, cache=True)
def _direct_rectangles(phase, constant, linear, rectangles, dt, out):  # pragma: no cover - numba
    """Accumulate exact triangle moments without one Python callback per rectangle.

    Each (rectangle, mode) fills one scratch slot in parallel, summing its
    lines serially; the slices are then added to ``out`` in
    rectangle order, the serial sum's order, so the result does not depend on
    thread count.
    """
    n = rectangles.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    for r in range(n):
        offsets[r + 1] = offsets[r] + rectangles[r, 1] - rectangles[r, 0]
    scratch = np.empty(offsets[n], dtype=np.complex128)
    # One parallel item per (rectangle, mode): the few high-mode rectangles
    # hold most of the work, so a split over rectangles would serialize them.
    for m in prange(offsets[n]):
        r = np.searchsorted(offsets, m, side="right") - 1
        j0, lo, hi = rectangles[r, 0], rectangles[r, 2], rectangles[r, 3]
        j = j0 + m - offsets[r]
        re0 = im0 = re1 = im1 = 0.0
        for k in range(lo, hi):
            angle = -j * phase[k]
            c, s = np.cos(angle), np.sin(angle)
            re0 += constant[k] * c
            im0 += constant[k] * s
            re1 += linear[k] * c
            im1 += linear[k] * s
        scratch[m] = complex(re0 - dt * j * re1, im0 - dt * j * im1)
    for r in range(n):
        j0, j1 = rectangles[r, 0], rectangles[r, 1]
        base = offsets[r] - j0
        for j in range(j0, j1):
            out[j] += scratch[base + j]


def _positive_modes(E, a, w, c0, period, eps):
    """Host exact-cutoff partition with small rectangles accumulated in one call.

    Same triangle moments as ``partitioned_positive_modes``; only their host
    scheduling differs. Large rectangles retain the ES NUFFT.

    Validation: cxr-line-fourier-reduction
    """
    dt = 2.0 * np.pi / period
    counts = np.floor(2.0 * a / dt).astype(np.int64)
    order = np.argsort(counts, kind="stable")
    counts = counts[order]
    positive = np.zeros(int(counts[-1]) + 1, dtype=np.complex128)
    phase = np.ascontiguousarray((E[order] - c0) * dt)
    constant = np.ascontiguousarray(w[order] * np.pi / a[order])
    linear = np.ascontiguousarray(constant / (2.0 * a[order]))
    rectangles = cutoff_rectangle_array(counts)
    modes = rectangles[:, 1] - rectangles[:, 0]
    sources = rectangles[:, 3] - rectangles[:, 2]
    large = (
        (sources >= _NUFFT_MIN_SOURCES)
        & (modes >= _NUFFT_MIN_MODES)
        & (sources * modes >= _NUFFT_MIN_WORK)
    )
    _direct_rectangles(phase, constant, linear, rectangles[~large], dt, positive)

    def transform(rect):
        j0, j1, lo, hi = rect
        strengths = np.stack((constant[lo:hi], linear[lo:hi]))
        return host_es_mode_transform(phase[lo:hi], strengths, j0, j1, eps=eps)

    # The spreader releases the GIL and scipy.fft threads, so rectangles
    # transform concurrently (#1390); they are added in rectangle order, so the
    # sum does not depend on the worker count.
    jobs = rectangles[large]
    with ThreadPoolExecutor(max_workers=_transform_workers(len(jobs))) as pool:
        for (j0, j1, _, _), moments in zip(jobs, pool.map(transform, jobs), strict=True):
            positive[j0:j1] += moments[0] - dt * np.arange(j0, j1) * moments[1]
    return positive


def _transform_workers(n_jobs):
    """Worker threads for the host ES rectangles: numba's thread budget, at least one."""
    from numba import get_num_threads

    return max(1, min(int(n_jobs), get_num_threads()))


def host_fourier_bin_mean_reduce(
    E_res,
    a_width,
    weight,
    axis,
    *,
    residual_tol=FOURIER_RESIDUAL_TOL,
    broad_width_eV=1000.0,
    eps=1e-12,
    max_line_modes=FOURIER_MAX_LINE_MODES,
):
    """``sum_l w_l mean_k(line l)`` on uniform bins ``axis = (c0, h, n_bins)``, host FP64.

    Same contract and bounds as ``cuda_fourier_bin_mean_reduce``: broad lines
    on the 64-node route, the rest from exact-cutoff coefficients (ES NUFFT
    using requested kernel accuracy ``eps`` on large rectangles, direct sums otherwise), the bin
    multiplier, ``j mod N`` folding, one inverse FFT and the closed-form
    envelope-image correction. Lines with more than ``max_line_modes``
    positive modes (default :data:`FOURIER_MAX_LINE_MODES`, #1320) take the
    numba near/far evaluator (``_bin_mean_jit``) on the same bins, as the
    device route sends them to its near/far kernel; ``None`` keeps every line
    on the Fourier route.

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    c0, h, n_bins = axis
    if not residual_tol >= FOURIER_MIN_RESIDUAL_TOL:
        raise ValueError(f"residual_tol must be >= {FOURIER_MIN_RESIDUAL_TOL:g}")
    E = np.asarray(E_res, dtype=np.float64).ravel()
    a = np.asarray(a_width, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    if not (E.shape == a.shape == w.shape):
        raise ValueError("E_res, a_width and weight must have one shape")
    keep = w != 0.0
    E, a, w = E[keep], a[keep], w[keep]
    out = np.zeros(n_bins, dtype=np.float64)
    if E.size == 0:
        return out
    finite = np.isfinite(E).all() and np.isfinite(w).all() and np.isfinite(a).all()
    if not (finite and (a > 0.0).all()):
        raise ValueError("lines need finite E_res and weight and positive finite a_width")
    if broad_width_eV is not None:
        broad_width_eV = broad_width_for_axis(h, (n_bins - 1) * h, residual_tol, broad_width_eV)
        broad = a <= np.pi / broad_width_eV
        if broad.any():
            out += broad_bin_mean_reduce(
                E[broad], a[broad], w[broad], c0, h, n_bins, tolerance=residual_tol
            )
            E, a, w = E[~broad], a[~broad], w[~broad]
            if E.size == 0:
                return out
    n_fft, P, _ = fourier_period(
        np.array([E.min(), E.max()]), np.array([a.min()]), c0, h, n_bins, residual_tol=residual_tol
    )
    # Same count as ``_positive_modes``; removing lines only shrinks the span,
    # so P keeps its image bounds.
    if max_line_modes is not None:
        over = np.floor(2.0 * a / (2.0 * np.pi / P)) > max_line_modes
        if over.any():
            out += host_near_far_bin_mean_reduce(E[over], a[over], w[over], (c0, h, n_bins))
            E, a, w = E[~over], a[~over], w[~over]
            if E.size == 0:
                return out
    positive = _positive_modes(E, a, w, c0, P, eps)
    t = (2.0 * np.pi / P) * np.arange(positive.size, dtype=np.float64)
    positive *= np.sinc(t * (h / (2.0 * np.pi)))
    positive[0] = 0.0
    if positive.size <= n_fft:
        folded = np.zeros(n_fft, dtype=np.complex128)
        folded[: positive.size] = positive
    else:
        idx = np.arange(positive.size) % n_fft
        folded = np.bincount(idx, positive.real, minlength=n_fft) + 1j * np.bincount(
            idx, positive.imag, minlength=n_fft
        )
    g0 = float(np.sum(w * np.pi / a))
    out += (g0 + 2.0 * (n_fft * sf.ifft(folded)).real[:n_bins]) / P
    out -= _envelope_image_correction(E, a, w, c0, h, n_bins, P)
    return out


#: Shortest uniform run of a non-uniform axis that takes its own Fourier
#: transform (#1390). Each run costs about one transform of every line, the
#: near/far evaluator about ``lines`` exact pairs per bin; on the #1320 flush
#: (4.2M lines) a 64-bin run costs about as much either way.
HOST_SEGMENT_MIN_BINS = 64


def uniform_runs(edges, *, min_bins=HOST_SEGMENT_MIN_BINS):
    """Split ``edges`` into ``(start, stop, axis)`` bin runs; ``axis`` is ``None`` off-Fourier.

    Maximal runs of equal bin widths (to ``1e-9`` relative) with at least
    ``min_bins`` bins carry their :func:`fourier_bin_axis`; every other bin is
    in a run with ``axis=None``. Runs are contiguous and cover every bin.
    """
    from ._bin_quadrature import fourier_bin_axis

    widths = np.diff(np.asarray(edges, dtype=np.float64))
    breaks = np.flatnonzero(np.abs(np.diff(widths)) > 1e-9 * np.abs(widths[1:])) + 1
    runs = []
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, widths.size], strict=True):
        start, stop = int(start), int(stop)
        axis = fourier_bin_axis(edges[start : stop + 1]) if stop - start >= min_bins else None
        if axis is None and runs and runs[-1][2] is None:
            runs[-1] = (runs[-1][0], stop, None)
        else:
            runs.append((start, stop, axis))
    return runs


def host_segmented_bin_mean_reduce(E_res, a_width, weight, edges, inv_width, runs=None):
    """``sum_l w_l mean_k(line l)`` on any ascending ``edges``, host FP64 (#1390).

    A bin mean depends only on its own edges, so each uniform run of
    :func:`uniform_runs` takes :func:`host_fourier_bin_mean_reduce` on its
    sub-axis, and the remaining bins the numba near/far evaluator. A
    piecewise-uniform axis (as built from resonance-local grids) then costs a
    few transforms instead of ``lines * bins`` pairs.

    Validation: cxr-line-fourier-reduction, sinc-bin-far-envelope
    """
    from ._bin_mean_jit import host_near_far_edges_reduce

    edges = np.asarray(edges, dtype=np.float64)
    inv_width = np.asarray(inv_width, dtype=np.float64)
    out = np.zeros(edges.size - 1, dtype=np.float64)
    for start, stop, axis in uniform_runs(edges) if runs is None else runs:
        if axis is None:
            out[start:stop] = host_near_far_edges_reduce(
                E_res, a_width, weight, edges[start : stop + 1], inv_width[start:stop]
            )
        else:
            out[start:stop] = host_fourier_bin_mean_reduce(E_res, a_width, weight, axis)
    return out
