"""Single-launch CuPy twins of characteristic scoring's elementwise chains (#1005).

Per segment chunk, characteristic scoring ran one ``cupy.interp`` chain per
ionization subshell (search, interpolate, two range tests, mask) and, per line
group, the ~13 float64 operations of :func:`segment_escape.mean_transmission`
and the yield product. On the GPU, host launch time and repeated passes over
the ``(chunk, line)`` arrays, not arithmetic, bounded both.

Each kernel replays its chain in the original order. Float64 operations that
ran as separate CuPy kernels are explicit round-to-nearest intrinsics, so the
compiler cannot contract a multiply-add across what used to be two launches;
the interpolation body is CuPy 14's ``cupy_interp`` body verbatim. The results
are therefore bit-for-bit those of the chains they replace; only
:func:`segment_sum` reorders a float64 sum. Every entry point
returns ``None`` off the CuPy path or for unexpected dtypes, and the caller
then runs its original chain.
"""

import numpy as np

from ..._backend import xp

_PREAMBLE = r"""
__device__ __forceinline__ double _nan_f64() { return __longlong_as_double(0x7ff8000000000000LL); }

// cupy.minimum float loop: NaN in either operand wins.
__device__ __forceinline__ double _min_nan(double a, double b) {
    return (isnan(a) | isnan(b)) ? _nan_f64() : fmin(a, b);
}
"""

_KERNELS: dict = {}


def _kernel(name, in_params, out_params, body):
    """Compile once per process; ``None`` without CuPy elementwise kernels."""
    if name not in _KERNELS:
        factory = getattr(xp, "ElementwiseKernel", None)
        _KERNELS[name] = (
            None
            if factory is None
            else factory(
                in_params, out_params, body, f"pyrite_characteristic_{name}", preamble=_PREAMBLE
            )
        )
    return _KERNELS[name]


def _cupy_arrays(*arrays):
    return getattr(xp, "__name__", "") == "cupy" and all(isinstance(a, xp.ndarray) for a in arrays)


# ``cupy.searchsorted(fx, x, side="right")`` then CuPy's ``cupy_interp`` body
# (left/right = fy[0]/fy[-1]), followed by the scorer's
# ``where((x >= fx[0]) & (x <= fx[-1]), interp, 0)`` mask. One element per
# (segment, shell); shell s owns ``fx/fy[offset[s]:offset[s + 1]]``.
_INTERP_BODY = r"""
const long long r = i / n_shell;
const long long s = i - r * n_shell;
const long long o = offset[s];
const long long len = offset[s + 1] - o;
const float* fx = &energy[o];
const float* fy = &sigma[o];
const float xv = x[r];
double y;
if (!((xv >= fx[0]) & (xv <= fx[len - 1]))) {
    y = 0.0;
} else {
    long long lo = 0;
    long long hi = len;
    while (lo < hi) {
        const long long m = lo + (hi - lo) / 2;
        if (fx[m] <= xv) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    const long long x_idx = lo - 1;
    if (isnan(xv)) { y = xv; }
    else if (x_idx < 0) { y = fy[0]; }
    else if (xv == fx[len - 1]) { y = fy[len - 1]; }
    else if (x_idx >= len - 1) { y = fy[len - 1]; }
    else if (xv == fx[x_idx]) { y = fy[x_idx]; }
    else {
        const double slope = (double)(fy[x_idx+1] - fy[x_idx]) / \
                             ((double)fx[x_idx+1] - (double)fx[x_idx]);
        double out = slope * ((double)xv - (double)fx[x_idx]) \
                     + (double)fy[x_idx];
        if (isnan(out)) {
            out = slope * ((double)xv - (double)fx[x_idx+1]) \
                  + (double)fy[x_idx+1];
            if (isnan(out) && (fy[x_idx] == fy[x_idx+1])) {
                out = fy[x_idx];
            }
        }
        y = out;
    }
}
sigma_out = y;
"""


def pack_shell_tables(shell_tables):
    """Concatenate device ``(energy, sigma)`` shell tables for :func:`shell_sigma`.

    Returns ``(energy, sigma, offset)`` or ``None`` off the CuPy float32 path.
    """
    if not shell_tables or not _cupy_arrays(*(a for pair in shell_tables for a in pair)):
        return None
    if any(e.dtype != np.float32 or f.dtype != np.float32 for e, f in shell_tables):
        return None
    lengths = [int(e.size) for e, _f in shell_tables]
    offset = xp.asarray(np.concatenate(([0], np.cumsum(lengths))), dtype=np.int64)
    energy = xp.concatenate([e for e, _f in shell_tables])
    sigma = xp.concatenate([f for _e, f in shell_tables])
    return energy, sigma, offset


def shell_sigma(query_eV, packed):
    """``(n_segment, n_shell)`` float64 masked interpolation, or ``None``."""
    if packed is None or not _cupy_arrays(query_eV) or query_eV.dtype != np.float32:
        return None
    kernel = _kernel(
        "shell_interp",
        "raw float32 x, raw float32 energy, raw float32 sigma, raw int64 offset, int64 n_shell",
        "float64 sigma_out",
        _INTERP_BODY,
    )
    if kernel is None:
        return None
    energy, sigma, offset = packed
    n_shell = int(offset.size) - 1
    out = xp.empty((int(query_eV.size), n_shell), dtype=np.float64)
    kernel(xp.ascontiguousarray(query_eV), energy, sigma, offset, n_shell, out)
    return out


# ``mean_transmission(tau_start, tau_end)`` times ``path_cm[:, None] *
# effective_sigma``, in the original operation order:
#   lo = minimum(t0, t1); d = |t1 - t0|; safe = where(d > 1e-4, d, 1)
#   ratio = where(d > 1e-4, -expm1(-safe) / safe, 1 - 0.5 d + d d / 6)
#   out = (path * sigma) * (exp(-lo) * ratio)
_YIELD_BODY = r"""
const double lo = _min_nan(t0, t1);
const double d = fabs(__dsub_rn(t1, t0));
const bool big = d > 1.0e-4;
const double safe = big ? d : 1.0;
const double direct = __ddiv_rn(-expm1(-safe), safe);
const double series = __dadd_rn(
    __dsub_rn(1.0, __dmul_rn(0.5, d)), __ddiv_rn(__dmul_rn(d, d), 6.0));
const double ratio = big ? direct : series;
const double transmission = __dmul_rn(exp(-lo), ratio);
out = __dmul_rn(__dmul_rn((double)path, sigma), transmission);
"""


def weighted_transmission(tau_start, tau_end, path_cm, effective_sigma):
    """Fused per-(segment, line) yield before the segment sum, or ``None``.

    ``tau_*`` and ``effective_sigma`` are ``(n_segment, n_line)`` float64 and
    ``path_cm`` is ``(n_segment,)`` float32/float64, as the scorer forms them.
    """
    arrays = (tau_start, tau_end, effective_sigma)
    if not _cupy_arrays(*arrays, path_cm) or any(a.dtype != np.float64 for a in arrays):
        return None
    if path_cm.dtype not in (np.float32, np.float64):
        return None
    kernel = _kernel(
        "line_yield",
        "float64 t0, float64 t1, P path, float64 sigma",
        "float64 out",
        _YIELD_BODY,
    )
    if kernel is None:
        return None
    return kernel(tau_start, tau_end, path_cm[:, None], effective_sigma)


def segment_sum(weighted):
    """Column sums of a ``(n_segment, n_line)`` CuPy float64 array, or ``None``.

    CuPy's ``sum(axis=0)`` over a few dozen long columns is a poorly parallel
    reduction; a ones-vector matrix product reads the array once through
    cuBLAS. Only the float64 summation order differs (deterministic per
    shape), far below the float32 line yields it accumulates into.
    """
    if not _cupy_arrays(weighted) or weighted.dtype != np.float64 or weighted.ndim != 2:
        return None
    return xp.ones(int(weighted.shape[0]), dtype=np.float64) @ weighted
