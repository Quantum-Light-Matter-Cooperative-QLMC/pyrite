"""Experimental CUDA scalar coefficients; optional CuPy/cuFINUFFT imports.

No production scorer calls this module. Host planning and uploads are included
in each call; output stays on the current CUDA stream/device in complex128.
Validation: cxr-line-fourier-reduction
"""

from functools import lru_cache
from importlib import import_module

import numpy as np

from ._fourier_cutoffs import cutoff_rectangles


def cufinufft_mode_transform(phase, strengths, j0, j1, *, eps=1e-12):
    """Two-row type-1 transform of device arrays into modes j0..j1-1.

    Same modulation/sign/order convention as the host FINUFFT callback.
    Plan execution is synchronized before plan destruction, including on a
    non-default stream. Tolerance is not a final spectrum error guarantee.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    cufinufft = import_module("cufinufft")
    modes = j1 - j0
    shift = j0 + modes // 2
    phase = cp.ascontiguousarray(phase, dtype=cp.float64)
    sources = cp.ascontiguousarray(strengths * cp.exp(-1j * shift * phase), dtype=cp.complex128)
    stream = cp.cuda.get_current_stream()
    plan = cufinufft.Plan(
        1,
        (modes,),
        n_trans=2,
        eps=eps,
        isign=-1,
        dtype="complex128",
        modeord=0,
        gpu_device_id=cp.cuda.Device().id,
        gpu_stream=stream.ptr,
    )
    plan.setpts(phase)
    out = plan.execute(sources)
    stream.synchronize()
    return out


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
    """Exact-support hybrid coefficients from validated host scalar arrays.

    Mirrors ``partitioned_positive_modes`` (no zero mode or bin multiplier).
    Large rectangles use two-row cuFINUFFT; all others share one direct CUDA
    launch with FP64 atomic accumulation. Summation order can vary between
    executions. ``use_nufft=False`` is an equal-workload GPU direct baseline.
    Thresholds are provisional CPU-audit thresholds, not tuned GPU crossovers.
    Input contract is the host planner's finite arrays/positive widths/period.

    Validation: cxr-line-fourier-reduction
    """
    cp = import_module("cupy")
    if cp.cuda.runtime.is_hip:
        raise ValueError("cuFINUFFT prototype requires CUDA")
    dt = 2.0 * np.pi / period
    counts = np.floor(2.0 * a_width / dt).astype(np.int64)
    order = np.argsort(counts, kind="stable")
    counts = counts[order]
    last = int(counts[-1]) if counts.size else 0
    phase = cp.asarray((E_res[order] - c0) * dt, dtype=cp.float64)
    a = a_width[order]
    constant = cp.asarray(weight[order] * np.pi / a, dtype=cp.float64)
    linear = cp.asarray(weight[order] * np.pi / (2.0 * a * a), dtype=cp.float64)
    out = cp.zeros(last + 1, dtype=cp.complex128)
    direct = []
    for j0, j1, lo, hi in cutoff_rectangles(counts):
        modes, sources = j1 - j0, hi - lo
        if use_nufft and sources >= 64 and modes >= 32 and sources * modes >= 262144:
            strengths = cp.stack((constant[lo:hi], linear[lo:hi]))
            moments = cufinufft_mode_transform(phase[lo:hi], strengths, j0, j1, eps=eps)
            t = dt * cp.arange(j0, j1, dtype=cp.float64)
            out[j0:j1] += moments[0] - t * moments[1]
        else:
            direct.append((j0, j1, lo, hi))
    if direct:
        rectangles = cp.asarray(direct, dtype=cp.int64)
        _direct_kernel()(
            (len(direct),), (128,), (phase, constant, linear, rectangles, np.float64(dt), out)
        )
    return out
