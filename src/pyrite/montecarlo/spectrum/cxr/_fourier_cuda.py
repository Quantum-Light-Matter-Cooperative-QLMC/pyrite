"""Device Fourier-domain bin-mean line reduction (#1192); CuPy, ES or cuFINUFFT.

The default device bin-mean reduction on uniform axes (``_bin_quadrature.
resolve_bin_reduction``). Outputs stay on the current stream/device. The
default NUFFT is the portable ES spreader (``_fourier_spread``) on CUDA and
ROCm alike (#1230): on an RTX 5080 it was about 2x faster than cuFINUFFT on
4.4M-11.6M-line production flushes, matched it to <= 1.2e-11, and avoids
cuFINUFFT's ~0.3 s first-plan start-up. cuFINUFFT stays selectable with
``nufft="cufinufft"`` and is imported lazily through :func:`cufinufft_module`.
Validation: cxr-line-fourier-reduction
"""

import ctypes
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from ._bin_broad import broad_bin_mean_reduce, broad_width_for_axis
from ._bin_fourier import (
    _CHEB_NODES,
    FOURIER_MAX_LINE_MODES,
    FOURIER_MIN_RESIDUAL_TOL,
    FOURIER_RESIDUAL_TOL,
    _envelope_image_correction,
    fourier_period,
)
from ._fourier_cutoffs import cutoff_rectangle_array
from ._fourier_spread import SPREADERS, es_gather_moments, es_mode_transform


def cufinufft_mode_transform(phase, strengths, j0, j1, *, eps=1e-12, plans=None):
    """Two-row type-1 transform of device arrays into modes j0..j1-1.

    Same modulation/sign/order convention as the host FINUFFT callback.
    ``plans`` (a dict, keyed by mode count) lets one call site reuse a plan
    across transforms of equal size; only ``setpts`` repeats. The caller
    then owns the plans and must synchronize before dropping them. Without
    it, execution is synchronized before the one-shot plan is destroyed,
    including on a non-default stream. Tolerance is not a final spectrum
    error guarantee.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    modes = j1 - j0
    shift = j0 + modes // 2
    phase = cp.ascontiguousarray(phase, dtype=cp.float64)
    sources = cp.ascontiguousarray(strengths * cp.exp(-1j * shift * phase), dtype=cp.complex128)
    plan = None if plans is None else plans.get(modes)
    if plan is None:
        plan = _plan(cp, modes, eps)
        if plans is not None:
            plans[modes] = plan
    plan.setpts(phase)
    out = plan.execute(sources)
    if plans is None:
        cp.cuda.get_current_stream().synchronize()
    return out


#: CUDA 12 libraries the cuFINUFFT 2.5 wheel links, in dependency order, with
#: the ``nvidia-*-cu12`` wheel package that ships each one.
_CUFINUFFT_RUNTIME = (
    ("nvidia.nvjitlink", "libnvJitLink.so.12"),
    ("nvidia.cufft", "libcufft.so.11"),
    ("nvidia.cuda_runtime", "libcudart.so.12"),
)


@lru_cache(maxsize=1)
def cufinufft_module():
    """Import cuFINUFFT, first preloading its CUDA 12 runtime from wheels.

    The ``nvidia`` extra pairs the CUDA 12 cuFINUFFT wheel with CUDA 13 CuPy.
    Loading ``libcudart.so.12``/``libcufft.so.11`` from their
    ``nvidia-*-cu12`` wheels satisfies the library's dynamic links by soname
    without an ``LD_LIBRARY_PATH`` edit. CuPy's own cuFFT loads first: once
    a ``libcufft`` is resident, CuPy's wheel-library lookup no longer loads
    its CUDA 13 ``libcufft.so.12``. Libraries already on the system path, or
    missing wheels, are left to the dynamic loader.
    """
    import_module("cupy").cuda.cufft  # noqa: B018 - load CuPy's cuFFT first
    for package, soname in _CUFINUFFT_RUNTIME:
        spec = find_spec(package) if find_spec("nvidia") is not None else None
        for root in (spec.submodule_search_locations or []) if spec else []:
            path = Path(root) / "lib" / soname
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
                break
    return import_module("cufinufft")


def _plan(cp, modes, eps):
    """Two-transform FP64 type-1 plan on the current device and stream."""
    cufinufft = cufinufft_module()
    return cufinufft.Plan(
        1,
        (modes,),
        n_trans=2,
        eps=eps,
        isign=-1,
        dtype="complex128",
        modeord=0,
        gpu_device_id=cp.cuda.Device().id,
        gpu_stream=cp.cuda.get_current_stream().ptr,
    )


#: A rectangle needs at least this many (source, mode) pairs for the NUFFT;
#: smaller ones share the one direct launch. By direct precision: the FP32
#: phase factors make direct sums cheaper, so the crossover moves up (RTX
#: 5080, #1230 for FP64, #1296 job 13647 for FP32).
_NUFFT_MIN_WORK = {"fp64": 1 << 22, "fp32": 1 << 24}
#: Flushes with at most this many lines plan cutoff rectangles on the host.
_HOST_PLAN_MAX_LINES = 1 << 20


#: Direct-rectangle phase factors: ``"fp32"`` (default since
#: ``fourier-auto-v6``, #1296) reduces ``j * phase`` modulo 2 pi in FP64 and
#: evaluates ``sincosf``; ``"fp64"`` keeps FP64 ``sincos``. Both accumulate
#: in FP64.
DIRECT_PRECISIONS = ("fp32", "fp64")
#: Bound on the absolute error of each FP32 cosine/sine factor: rounding the
#: reduced phase (<= pi 2^-24) plus ``sincosf`` (<= 2 ulp of 1, CUDA math
#: API), rounded up. Gives ``|dG_j| <= sqrt(2) * this * sum_l |c_l(j)|``.
DIRECT_FP32_PHASE_ERROR = 5e-7

_DIRECT_SOURCE = r"""
extern "C" __global__ void rectangles(
    const double* phase, const double* constant, const double* linear,
    const long long* rect, const double dt, double* out) {
    const long long r = 4LL * blockIdx.x;
    const long long j0 = rect[r], j1 = rect[r+1];
    const long long lo = rect[r+2], hi = rect[r+3];
    for (long long j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        double re = 0.0, im = 0.0;
        const double jd = double(j);
        for (long long l = lo; l < hi; ++l) {
#if DIRECT_FP32
            // FP64 reduction to [-pi, pi]; FP32 rounding of the reduced phase
            // and sincosf are the only single-precision steps (#1296).
            double th = jd * phase[l];
            th = fma(-6.283185307179586, rint(th * 0.15915494309189535), th);
            float sf, cf;
            sincosf(-(float)th, &sf, &cf);
            const double s = sf, c = cf;
#else
            double s, c;
            sincos(-jd * phase[l], &s, &c);
#endif
            const double w = constant[l] - (dt * jd) * linear[l];
            re += w * c;
            im += w * s;
        }
        atomicAdd(out + 2*j, re);
        atomicAdd(out + 2*j+1, im);
    }
}
"""


@lru_cache(maxsize=2)
def _direct_kernel(precision="fp64"):
    """One block per small rectangle; no line-by-mode workspace allocation."""
    cp = import_module("cupy")
    flag = "1" if precision == "fp32" else "0"
    return cp.RawKernel(_DIRECT_SOURCE, "rectangles", options=(f"-DDIRECT_FP32={flag}",))


#: Threads per envelope-image block; a power of two for the tree reduction.
_ENVELOPE_THREADS = 256


@lru_cache(maxsize=1)
def _envelope_kernel():
    """Per-block partial sums of ``q_l g(pi (c_i - E_l) / P)`` at 24 nodes, FP64.

    Same series/direct split and arithmetic as ``_bin_fourier.
    _inv_sin_sq_minus_inv_sq``; one pass over the lines instead of (24, block)
    temporaries. Partials go to a ``(blocks, 24)`` array, so the final sum
    does not depend on atomic ordering.
    """
    cp = import_module("cupy")
    return cp.RawKernel(
        r"""
#define NODES 24
#define THREADS 256
extern "C" __global__ void envelope_nodes(
    const double* E, const double* q, const long long n, const double* nodes,
    const double P, double* partial) {
    __shared__ double red[THREADS];
    double acc[NODES];
    double c[NODES];
    for (int i = 0; i < NODES; ++i) { acc[i] = 0.0; c[i] = nodes[i]; }
    const double pi = 3.141592653589793;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long l = (long long)blockIdx.x * blockDim.x + threadIdx.x; l < n; l += stride) {
        const double e = E[l], ql = q[l];
        for (int i = 0; i < NODES; ++i) {
            const double x = pi * (c[i] - e) / P;
            const double y = x * x;
            double g;
            if (fabs(x) < 0.2) {
                const double tail = 2.0 / 10395 + y * (1382.0 / 58046625 + y * (4.0 / 1403325));
                g = 1.0 / 3 + y * (1.0 / 15 + y * (2.0 / 189 + y * (1.0 / 675 + y * tail)));
            } else {
                const double sx = sin(x);
                g = 1.0 / (sx * sx) - 1.0 / y;
            }
            acc[i] += g * ql;
        }
    }
    for (int i = 0; i < NODES; ++i) {
        red[threadIdx.x] = acc[i];
        __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) partial[(long long)blockIdx.x * NODES + i] = red[0];
        __syncthreads();
    }
}
""",
        "envelope_nodes",
    )


def _device_envelope_node_sums(E_res, q, nodes, P):
    """Device ``_bin_fourier._envelope_node_sums``: one fused pass, host result."""
    cp = import_module("cupy")
    if len(nodes) != _CHEB_NODES or _CHEB_NODES != 24:
        raise ValueError("envelope kernel is compiled for 24 nodes")
    E = cp.ascontiguousarray(E_res, dtype=cp.float64)
    q = cp.ascontiguousarray(q, dtype=cp.float64)
    blocks = int(min(1024, max(1, -(-E.size // _ENVELOPE_THREADS))))
    partial = cp.empty((blocks, _CHEB_NODES), dtype=cp.float64)
    _envelope_kernel()(
        (blocks,),
        (_ENVELOPE_THREADS,),
        (E, q, np.int64(E.size), cp.asarray(nodes, dtype=cp.float64), np.float64(P), partial),
    )
    return cp.asnumpy(partial.sum(axis=0))


#: Direct-kernel tile: at most one block of modes by this many lines, so a
#: few large direct rectangles still fill the device.
_TILE_MODES = 128
_TILE_LINES = 4096


def _direct_tiles(rects):
    """Split ``(j0, j1, l0, l1)`` rectangles into at most 128-mode x 4096-line tiles.

    Tiles partition each rectangle exactly; the kernel's atomic accumulation
    makes the split a summation reordering only.
    """
    rects = np.asarray(rects, dtype=np.int64).reshape(-1, 4)
    n_j = -(-(rects[:, 1] - rects[:, 0]) // _TILE_MODES)
    n_l = -(-(rects[:, 3] - rects[:, 2]) // _TILE_LINES)
    per = n_j * n_l
    row = np.repeat(np.arange(rects.shape[0]), per)
    t = np.arange(int(per.sum())) - np.repeat(np.cumsum(per) - per, per)
    j0 = rects[row, 0] + (t % n_j[row]) * _TILE_MODES
    l0 = rects[row, 2] + (t // n_j[row]) * _TILE_LINES
    j1 = np.minimum(j0 + _TILE_MODES, rects[row, 1])
    l1 = np.minimum(l0 + _TILE_LINES, rects[row, 3])
    return np.stack((j0, j1, l0, l1), axis=1)


def cuda_positive_modes(
    E_res,
    a_width,
    weight,
    c0,
    period,
    *,
    eps=1e-12,
    use_nufft=True,
    nufft="es",
    direct_precision="fp64",
    spread="gather",
):
    """Exact-support hybrid coefficients from validated scalar line arrays.

    Mirrors ``partitioned_positive_modes`` (no zero mode or bin multiplier).
    Large rectangles use a two-row type-1 NUFFT: the portable ES spreader
    (``_fourier_spread``) by default, or cuFINUFFT with
    ``nufft="cufinufft"`` (CUDA only). All others share one direct launch with FP64 atomic
    accumulation over 128-mode x 4096-line tiles; ``direct_precision="fp32"``
    evaluates their phase factors with ``sincosf`` after FP64 range reduction
    (:data:`DIRECT_PRECISIONS`). ``spread`` picks the ES spreader
    (:data:`_fourier_spread.SPREADERS`): the default ``"gather"`` transforms
    all NUFFT rectangles in sorted, batched warp gathers; ``"atomic"`` runs
    one atomic spread per rectangle. Summation order can vary between
    executions. ``use_nufft=False`` is an equal-workload GPU direct baseline.
    ``_NUFFT_MIN_WORK`` was measured on an RTX 5080 (#1230, #1296); the source and
    mode floors are the CPU-audit values.
    Input contract is the host planner's finite arrays/positive widths/period;
    host or CuPy arrays, planned and evaluated on the device.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    if nufft not in ("es", "cufinufft"):
        raise ValueError("nufft must be 'es' or 'cufinufft'")
    if direct_precision not in DIRECT_PRECISIONS:
        raise ValueError(f"direct_precision must be one of {DIRECT_PRECISIONS}")
    if spread not in SPREADERS:
        raise ValueError(f"spread must be one of {SPREADERS}")
    transform = es_mode_transform if nufft == "es" else cufinufft_mode_transform
    dt = 2.0 * np.pi / period
    E_res, a_width, weight = (cp.asarray(x, dtype=cp.float64) for x in (E_res, a_width, weight))
    counts = cp.floor(2.0 * a_width / dt).astype(cp.int64)
    order = cp.argsort(counts)
    counts = counts[order]
    last = int(counts[-1]) if counts.size else 0
    phase = (E_res[order] - c0) * dt
    a = a_width[order]
    constant = weight[order] * np.pi / a
    linear = constant / (2.0 * a)
    out = cp.zeros(last + 1, dtype=cp.complex128)
    # Integer planning: one copy and host passes beat a synchronizing device
    # pass per bit unless the flush is large. Same rectangles either way.
    if counts.size <= _HOST_PLAN_MAX_LINES:
        rects = cutoff_rectangle_array(cp.asnumpy(counts))
    else:
        rects = cp.asnumpy(cutoff_rectangle_array(counts, xp=cp))
    modes, sources = rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2]
    min_work = _NUFFT_MIN_WORK[direct_precision]
    large = (sources >= 64) & (modes >= 32) & (sources * modes >= min_work)
    if not use_nufft:
        large[:] = False
    plans = {}
    if nufft == "es" and spread == "gather":
        moments = es_gather_moments(phase, (constant, linear), rects[large], eps=eps, plans=plans)
    else:
        moments = (
            (
                j0,
                j1,
                transform(
                    phase[lo:hi],
                    cp.stack((constant[lo:hi], linear[lo:hi])),
                    j0,
                    j1,
                    eps=eps,
                    plans=plans,
                ),
            )
            for j0, j1, lo, hi in rects[large].tolist()
        )
    for j0, j1, m in moments:
        t = dt * cp.arange(j0, j1, dtype=cp.float64)
        out[j0:j1] += m[0] - t * m[1]
    if plans and nufft == "cufinufft":
        cp.cuda.get_current_stream().synchronize()
        plans.clear()
    direct = _direct_tiles(rects[~large])
    if direct.size:
        _direct_kernel(direct_precision)(
            (len(direct),),
            (128,),
            (phase, constant, linear, cp.asarray(direct), np.float64(dt), out),
        )
    return out


#: Requested ES/NUFFT accuracy of the production device route (#1296): width
#: 10 instead of 13 at 1e-12; still three orders below ``residual_tol``.
DEVICE_NUFFT_EPS = 1e-9

#: Lines per broad-route block on the device ((block, 64) FP64 temporaries).
_DEVICE_BROAD_BLOCK = 1 << 16


def _near_far_lines(E, a, w, axis):
    """Near/far bin means of the over-cap lines on the reconstructed ``axis`` bins.

    The kernel takes float32 line positions. A bin mean depends only on
    ``E - edge``, so positions and edges are shifted by ``c0`` (FP64) first:
    the float32 position error is then ``2**-24 |E - c0|``, at most
    ``2**-24`` of the band, not ``2**-24 E`` (3e-2 eV at 1 MeV, many times the
    width of the narrow lines that take this route).
    """
    from ._bin_quadrature import run_bin_mean_reduction_kernel

    cp = import_module("cupy")
    c0, h, n_bins = axis
    edges = (cp.arange(n_bins + 1, dtype=cp.float64) - 0.5) * h
    out = cp.zeros(n_bins, dtype=cp.float64)
    return run_bin_mean_reduction_kernel(E - c0, a, w, edges, cp.full(n_bins, 1.0 / h), out=out)


def cuda_fourier_bin_mean_reduce(
    E_res,
    a_width,
    weight,
    axis,
    *,
    residual_tol=FOURIER_RESIDUAL_TOL,
    broad_width_eV=1000.0,
    eps=DEVICE_NUFFT_EPS,
    direct_precision="fp32",
    spread="gather",
    max_line_modes=FOURIER_MAX_LINE_MODES,
):
    """Device counterpart of ``fourier_bin_mean_reduce(..., broad_width_eV=...)``.

    ``axis = (c0, h, n_bins)`` from ``uniform_bin_axis``. Line arrays may be
    host or CuPy arrays of any float dtype; arithmetic is FP64 on the device.
    Returns the ``(n_bins,)`` FP64 device spectrum ``sum_l w_l mean_k(l)``:
    broad lines on the 64-node route, the rest from exact-cutoff hybrid
    coefficients, the bin multiplier, ``j mod N`` folding, one cuFFT inverse
    and the closed-form envelope-image correction (one fused FP64 pass over the
    lines). Same bounds as the host reference, plus the NUFFT tolerance ``eps``
    and, with ``direct_precision="fp32"`` (default since ``fourier-auto-v6``),
    the single-precision phase factors of the direct rectangles. Lines whose
    mode count ``floor(P a_l / pi)`` exceeds ``max_line_modes`` (default
    :data:`FOURIER_MAX_LINE_MODES`, ``fourier-auto-v8``, #1320) take the
    near/far kernel on the same bins, with its own bounds; ``None`` keeps
    every line on the Fourier route (the host reference's behaviour).
    ``eps=1e-12, direct_precision="fp64", spread="atomic"`` is the
    ``fourier-auto-v5`` route up to the envelope kernel's FP64 summation
    order; ``spread="atomic"`` alone is ``fourier-auto-v6`` up to the ES
    spreading's.
    Planning, period choice and the small Chebyshev fits run on host.

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    cp = import_module("cupy")
    c0, h, n_bins = axis
    if not residual_tol >= FOURIER_MIN_RESIDUAL_TOL:
        raise ValueError(f"residual_tol must be >= {FOURIER_MIN_RESIDUAL_TOL:g}")
    E = cp.asarray(E_res, dtype=cp.float64).ravel()
    a = cp.asarray(a_width, dtype=cp.float64).ravel()
    w = cp.asarray(weight, dtype=cp.float64).ravel()
    if not (E.shape == a.shape == w.shape):
        raise ValueError("E_res, a_width and weight must have one shape")
    keep = cp.flatnonzero(w != 0.0)
    E, a, w = E[keep], a[keep], w[keep]
    out = cp.zeros(n_bins, dtype=cp.float64)
    if E.size == 0:
        return out
    finite = cp.isfinite(E).all() & cp.isfinite(w).all() & cp.isfinite(a).all()
    if not bool(finite & (a > 0.0).all()):
        raise ValueError("lines need finite E_res and weight and positive finite a_width")
    # ``None`` keeps every line on the Fourier route, as in the host reference.
    # Otherwise ``broad_width_eV`` is a floor, raised until the broad bound
    # holds on this axis (coarse bins, long spans).
    if broad_width_eV is not None:
        broad_width_eV = broad_width_for_axis(h, (n_bins - 1) * h, residual_tol, broad_width_eV)
    broad = a <= (0.0 if broad_width_eV is None else np.pi / broad_width_eV)
    if bool(broad.any()):
        sel = cp.flatnonzero(broad)
        values = broad_bin_mean_reduce(
            E[sel],
            a[sel],
            w[sel],
            c0,
            h,
            n_bins,
            tolerance=residual_tol,
            xp=cp,
            block=_DEVICE_BROAD_BLOCK,
        )
        out += cp.asarray(values)
        rest = cp.flatnonzero(~broad)
        E, a, w = E[rest], a[rest], w[rest]
        if E.size == 0:
            return out
    # The period depends only on the resonance extrema and the widest line.
    extrema = cp.asnumpy(cp.stack((E.min(), E.max(), a.min())))
    n_fft, P, _ = fourier_period(extrema[:2], extrema[2:], c0, h, n_bins, residual_tol=residual_tol)
    # Removing lines only shrinks the span, so P keeps its image bounds.
    # Same count as ``cuda_positive_modes``.
    over = cp.floor(2.0 * a / (2.0 * np.pi / P)) > (
        np.inf if max_line_modes is None else max_line_modes
    )
    if bool(over.any()):
        sel = cp.flatnonzero(over)
        out += _near_far_lines(E[sel], a[sel], w[sel], (c0, h, n_bins))
        rest = cp.flatnonzero(~over)
        E, a, w = E[rest], a[rest], w[rest]
        if E.size == 0:
            return out
    positive = cuda_positive_modes(
        E, a, w, c0, P, eps=eps, direct_precision=direct_precision, spread=spread
    )
    t = (2.0 * np.pi / P) * cp.arange(positive.size, dtype=cp.float64)
    positive *= cp.sinc(t * (h / (2.0 * np.pi)))
    positive[0] = 0.0
    if positive.size <= n_fft:
        folded = cp.zeros(n_fft, dtype=cp.complex128)
        folded[: positive.size] = positive
    else:
        idx = cp.arange(positive.size) % n_fft
        folded = cp.bincount(idx, positive.real, minlength=n_fft) + 1j * cp.bincount(
            idx, positive.imag, minlength=n_fft
        )
    g0 = float(cp.sum(w * np.pi / a))
    out += (g0 + 2.0 * (n_fft * cp.fft.ifft(folded)).real[:n_bins]) / P
    out -= cp.asarray(
        _envelope_image_correction(
            E, a, w, c0, h, n_bins, P, cp, node_sums=_device_envelope_node_sums
        )
    )
    return out
