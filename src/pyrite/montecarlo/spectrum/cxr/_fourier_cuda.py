"""Experimental CUDA scalar coefficients; optional CuPy/cuFINUFFT imports.

Production reaches it only through the opt-in ``method="fourier"`` bin-mean
reduction. Host planning and uploads are included
in each call; output stays on the current CUDA stream/device in complex128.
Validation: cxr-line-fourier-reduction
"""

from functools import lru_cache
from importlib import import_module

import numpy as np

from ._bin_broad import broad_bin_mean_reduce
from ._bin_fourier import (
    FOURIER_MIN_RESIDUAL_TOL,
    FOURIER_RESIDUAL_TOL,
    _envelope_image_correction,
    fourier_period,
)
from ._fourier_cutoffs import cutoff_rectangle_array


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


def _plan(cp, modes, eps):
    """Two-transform FP64 type-1 plan on the current device and stream."""
    cufinufft = import_module("cufinufft")
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


@lru_cache(maxsize=1)
def _direct_kernel():
    """One block per small rectangle; no line-by-mode workspace allocation."""
    cp = import_module("cupy")
    return cp.RawKernel(
        r"""
extern "C" __global__ void rectangles(
    const double* phase, const double* constant, const double* linear,
    const long long* rect, const double dt, double* out) {
    const long long r = 4LL * blockIdx.x;
    const long long j0 = rect[r], j1 = rect[r+1];
    const long long lo = rect[r+2], hi = rect[r+3];
    for (long long j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        double re = 0.0, im = 0.0;
        for (long long l = lo; l < hi; ++l) {
            double s, c;
            sincos(-double(j) * phase[l], &s, &c);
            const double w = constant[l] - (dt * double(j)) * linear[l];
            re += w * c;
            im += w * s;
        }
        atomicAdd(out + 2*j, re);
        atomicAdd(out + 2*j+1, im);
    }
}
""",
        "rectangles",
    )


def cuda_positive_modes(E_res, a_width, weight, c0, period, *, eps=1e-12, use_nufft=True):
    """Exact-support hybrid coefficients from validated scalar line arrays.

    Mirrors ``partitioned_positive_modes`` (no zero mode or bin multiplier).
    Large rectangles use two-row cuFINUFFT; all others share one direct CUDA
    launch with FP64 atomic accumulation. Summation order can vary between
    executions. ``use_nufft=False`` is an equal-workload GPU direct baseline.
    Thresholds are provisional CPU-audit thresholds, not tuned GPU crossovers.
    Input contract is the host planner's finite arrays/positive widths/period;
    host or CuPy arrays, planned and evaluated on the device.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    if cp.cuda.runtime.is_hip:
        raise ValueError("cuFINUFFT prototype requires CUDA")
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
    rects = cp.asnumpy(cutoff_rectangle_array(counts, xp=cp))
    modes, sources = rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2]
    large = (sources >= 64) & (modes >= 32) & (sources * modes >= 262144)
    if not use_nufft:
        large[:] = False
    plans = {}
    for j0, j1, lo, hi in rects[large].tolist():
        strengths = cp.stack((constant[lo:hi], linear[lo:hi]))
        moments = cufinufft_mode_transform(phase[lo:hi], strengths, j0, j1, eps=eps, plans=plans)
        t = dt * cp.arange(j0, j1, dtype=cp.float64)
        out[j0:j1] += moments[0] - t * moments[1]
    if plans:
        cp.cuda.get_current_stream().synchronize()
        plans.clear()
    direct = rects[~large]
    if direct.size:
        _direct_kernel()(
            (len(direct),),
            (128,),
            (phase, constant, linear, cp.asarray(direct), np.float64(dt), out),
        )
    return out


#: Lines per broad-route block on the device ((block, 64) FP64 temporaries).
_DEVICE_BROAD_BLOCK = 1 << 16


def cuda_fourier_bin_mean_reduce(
    E_res,
    a_width,
    weight,
    axis,
    *,
    residual_tol=FOURIER_RESIDUAL_TOL,
    broad_width_eV=1000.0,
    eps=1e-12,
):
    """Device counterpart of ``fourier_bin_mean_reduce(..., broad_width_eV=...)``.

    ``axis = (c0, h, n_bins)`` from ``uniform_bin_axis``. Line arrays may be
    host or CuPy arrays of any float dtype; arithmetic is FP64 on the device.
    Returns the ``(n_bins,)`` FP64 device spectrum ``sum_l w_l mean_k(l)``:
    broad lines on the 64-node route, the rest from exact-cutoff hybrid
    coefficients, the bin multiplier, ``j mod N`` folding, one cuFFT inverse
    and the closed-form envelope-image correction. Same bounds as the host
    reference, plus the cuFINUFFT tolerance ``eps``. Planning, period choice
    and the small Chebyshev fits run on host.

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
    broad = a <= np.pi / broad_width_eV
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
    positive = cuda_positive_modes(E, a, w, c0, P, eps=eps)
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
    out -= cp.asarray(_envelope_image_correction(E, a, w, c0, h, n_bins, P, cp))
    return out
