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

import numpy as np
import scipy.fft as sf
from numba import njit

from ._bin_broad import broad_bin_mean_reduce, broad_width_for_axis
from ._bin_fourier import (
    FOURIER_MIN_RESIDUAL_TOL,
    FOURIER_RESIDUAL_TOL,
    _envelope_image_correction,
    fourier_period,
)
from ._fourier_cutoffs import partitioned_positive_modes
from ._fourier_spread import _BETA_PER_WIDTH, _deconvolution, _fine_size, es_width

#: Lines queued per target before a host flush (three FP64 arrays each).
HOST_FOURIER_FLUSH_LINES = 4_000_000
#: A rectangle at least this large (sources × modes) takes the NUFFT; the
#: device route's threshold.
_NUFFT_MIN_WORK = 262_144
_NUFFT_MIN_SOURCES = 64
_NUFFT_MIN_MODES = 32


@njit(cache=True)
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


@njit(cache=True)
def _direct_modes(phase, strengths, j0, j1, out):  # pragma: no cover - numba
    for j in range(j0, j1):
        re0 = im0 = re1 = im1 = 0.0
        for k in range(phase.shape[0]):
            angle = -j * phase[k]
            c = np.cos(angle)
            s = np.sin(angle)
            re0 += strengths[0, k] * c
            im0 += strengths[0, k] * s
            re1 += strengths[1, k] * c
            im1 += strengths[1, k] * s
        out[0, j - j0] = complex(re0, im0)
        out[1, j - j0] = complex(re1, im1)


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
    return spectrum[:, centred] * _deconvolution(np, modes, n_fine, width)


def _hybrid_transform(eps):
    def transform(phase, strengths, j0, j1):
        modes, sources = j1 - j0, phase.size
        large = (
            sources >= _NUFFT_MIN_SOURCES
            and modes >= _NUFFT_MIN_MODES
            and sources * modes >= _NUFFT_MIN_WORK
        )
        if large:
            return host_es_mode_transform(phase, strengths, j0, j1, eps=eps)
        out = np.empty((2, modes), dtype=np.complex128)
        _direct_modes(np.ascontiguousarray(phase), np.ascontiguousarray(strengths), j0, j1, out)
        return out

    return transform


def host_fourier_bin_mean_reduce(
    E_res,
    a_width,
    weight,
    axis,
    *,
    residual_tol=FOURIER_RESIDUAL_TOL,
    broad_width_eV=1000.0,
    eps=1e-12,
):
    """``sum_l w_l mean_k(line l)`` on uniform bins ``axis = (c0, h, n_bins)``, host FP64.

    Same contract and bounds as ``cuda_fourier_bin_mean_reduce``: broad lines
    on the 64-node route, the rest from exact-cutoff coefficients (ES NUFFT
    using requested kernel accuracy ``eps`` on large rectangles, direct sums otherwise), the bin
    multiplier, ``j mod N`` folding, one inverse FFT and the closed-form
    envelope-image correction.

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
    positive = partitioned_positive_modes(E, a, w, c0, P, _hybrid_transform(eps))
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
