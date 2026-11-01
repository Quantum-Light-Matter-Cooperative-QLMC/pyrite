"""Single-launch CuPy twins of the batched line route's elementwise chains.

Each segment block of the batched route used to run ~200 CuPy calls, most of
them one float32 operation over the ``(n_block, N_g)`` pair grid. On a fast GPU
the ~10 us of host time per call, not the device, bounded the route (#906).
Every kernel here replays one such chain in its original order, with each
float32 operation an explicit round-to-nearest intrinsic so the compiler cannot
contract a multiply-add, and with Python-scalar operands rounded to float32 as
CuPy's elementwise calls do. The results are therefore bit-for-bit those of the
chain they replace; no new validation debt. Every entry point returns ``None``
off the CuPy float32 path, and the caller then runs its original chain.

Bracketing follows ``cupy.searchsorted(side="left")`` on an ascending grid and
the ``clip(idx, 1, n - 1)`` of ``_kernels._interp_index``; ``maximum`` and
``minimum`` propagate NaN as CuPy's float loops do.
"""

import numpy as np

from ...._backend import xp

_PREAMBLE = r"""
__device__ __forceinline__ float _nan_f32() { return __int_as_float(0x7fc00000); }

// cupy.maximum / cupy.minimum float loops: NaN in either operand wins.
__device__ __forceinline__ float _max_nan(float a, float b) {
    return (isnan(a) | isnan(b)) ? _nan_f32() : fmaxf(a, b);
}
__device__ __forceinline__ float _min_nan(float a, float b) {
    return (isnan(a) | isnan(b)) ? _nan_f32() : fminf(a, b);
}

// cupy.searchsorted(bins, x, side="left") on ascending bins, then
// clip(., 1, n - 1) as in _kernels._interp_index.
__device__ long long _bracket(const float* bins, long long n, float x) {
    long long y;
    if (isnan(x)) {
        y = n;
        while (y > 0 && isnan(bins[y - 1])) {
            --y;
        }
    } else if (x > bins[n - 1]) {
        y = n;
    } else {
        long long left = 0;
        long long right = n - 1;
        while (left < right) {
            const long long m = left + (right - left) / 2;
            if (bins[m] < x) {
                left = m + 1;
            } else {
                right = m;
            }
        }
        y = right;
    }
    return y < 1 ? 1 : (y > n - 1 ? n - 1 : y);
}

// _interp_index blend fraction and _interp_gather1d of table f.
__device__ __forceinline__ float _gather1d(
    const float* grid, const float* f, long long n, long long idx, float x) {
    const float x0 = grid[idx - 1];
    const float frac = __fdiv_rn(__fsub_rn(x, x0), __fsub_rn(grid[idx], x0));
    const float f0 = f[idx - 1];
    float y = __fadd_rn(f0, __fmul_rn(frac, __fsub_rn(f[idx], f0)));
    y = (x <= grid[0]) ? f[0] : y;
    y = (x >= grid[n - 1]) ? f[n - 1] : y;
    return y;
}
"""

_KERNELS = {}


def _kernel(name, in_params, out_params, body):
    """Compile once per process; ``None`` without CuPy elementwise kernels."""
    if name not in _KERNELS:
        factory = getattr(xp, "ElementwiseKernel", None)
        _KERNELS[name] = (
            None
            if factory is None
            else factory(in_params, out_params, body, f"pyrite_cxr_{name}", preamble=_PREAMBLE)
        )
    return _KERNELS[name]


def _float32_cupy(*arrays):
    return getattr(xp, "__name__", "") == "cupy" and all(
        isinstance(a, xp.ndarray) and a.dtype == np.float32 for a in arrays
    )


# ``_kernels._in_medium_kinematics``: three fixed-point passes of
# E = hbarc (v.g / denom), delta = interp(E), denom = denom_vac - delta v.grad,
# then the convergence and critical-angle guards.
_IN_MEDIUM_BODY = r"""
float denom = denom_vac;
float previous = denom;
float d = 0.0f;
for (int pass = 0; pass < 3; ++pass) {
    const float E = __fmul_rn(hbarc, __fdiv_rn(v_dot_g, denom));
    const long long idx = _bracket(&E_tab[0], n_tab, E);
    d = _gather1d(&E_tab[0], &delta_tab[0], n_tab, idx, E);
    previous = denom;
    denom = __fsub_rn(denom_vac, __fmul_rn(d, v_dot_grad));
}
bool settled = fabsf(__fsub_rn(denom, previous)) <= __fmul_rn(rtol, fabsf(denom));
if (use_grad2) {
    settled = settled & (__fmul_rn(__fmul_rn(2.0f, fabsf(d)), grad2) < 1.0f);
}
denom_out = settled ? denom : _nan_f32();
delta_out = d;
"""


def in_medium_root(denom_vac, v_dot_g, v_dot_grad, grad2, delta_tab, E_tab, *, hbarc, rtol):
    """Fused ``_in_medium_kinematics`` with a supplied ``v_dot_grad``.

    Returns ``(denom, delta)`` broadcast over the operands, or ``None``.
    """
    use_grad2 = grad2 is not None
    if grad2 is None:
        grad2 = denom_vac
    if not _float32_cupy(denom_vac, v_dot_g, v_dot_grad, grad2, delta_tab, E_tab):
        return None
    kernel = _kernel(
        "in_medium_root",
        "float32 denom_vac, float32 v_dot_g, float32 v_dot_grad, float32 grad2, "
        "raw float32 delta_tab, raw float32 E_tab, int64 n_tab, float32 hbarc, "
        "float32 rtol, bool use_grad2",
        "float32 denom_out, float32 delta_out",
        _IN_MEDIUM_BODY,
    )
    if kernel is None:
        return None
    return kernel(
        denom_vac,
        v_dot_g,
        v_dot_grad,
        grad2,
        xp.ascontiguousarray(delta_tab),
        xp.ascontiguousarray(E_tab),
        np.int64(E_tab.size),
        np.float32(hbarc),
        np.float32(rtol),
        bool(use_grad2),
    )


# ``_batched._batched_block`` after ``_cxr_kin_core``: refracted |k|, E_res,
# the spectral keep mask, then ``_interp_index`` and ``_log_interp_fraction``.
_LINE_BRACKET_BODY = r"""
const float one_minus = __fsub_rn(1.0f, __fmul_rn(2.0f, delta));
const float radicand = (grad2 == 0.0f)
    ? 1.0f : __fadd_rn(one_minus, __fmul_rn(__fmul_rn(delta, delta), grad2));
k_mag = __fmul_rn(omega_res, __fsqrt_rn(radicand));
const float E = __fmul_rn(hbarc, omega_res);
E_res = E;
keep = cxr_electron & (E > lo_keep) & (E > 10.0f) & (E < hi_keep);
const long long n = n_tab;
const long long ix = _bracket(&E_tab[0], n, E);
const float x0 = E_tab[ix - 1];
const float x1 = E_tab[ix];
idx = ix;
frac = __fdiv_rn(__fsub_rn(E, x0), __fsub_rn(x1, x0));
below = E <= E_tab[0];
above = E >= E_tab[n - 1];
const float bounded = _min_nan(_max_nan(E, x0), x1);
log_frac = __fdiv_rn(
    log1pf(__fdiv_rn(__fsub_rn(bounded, x0), x0)),
    log1pf(__fdiv_rn(__fsub_rn(x1, x0), x0)));
"""


def line_bracket(omega_res, delta, grad2, cxr_electron, E_tab, *, hbarc, lo_keep, hi_keep):
    """Fused ``k_mag``, ``E_res``, keep mask and shared interpolation bracket.

    Returns ``(k_mag, E_res, keep, idx, frac, log_frac, below, above)`` or ``None``.
    """
    if not _float32_cupy(omega_res, delta, grad2, E_tab) or cxr_electron.dtype != np.bool_:
        return None
    kernel = _kernel(
        "line_bracket",
        "float32 omega_res, float32 delta, float32 grad2, bool cxr_electron, "
        "raw float32 E_tab, int64 n_tab, float32 hbarc, float32 lo_keep, float32 hi_keep",
        "float32 k_mag, float32 E_res, bool keep, int64 idx, float32 frac, "
        "float32 log_frac, bool below, bool above",
        _LINE_BRACKET_BODY,
    )
    if kernel is None:
        return None
    return kernel(
        omega_res,
        delta,
        grad2,
        cxr_electron,
        xp.ascontiguousarray(E_tab),
        np.int64(E_tab.size),
        np.float32(hbarc),
        np.float32(lo_keep),
        np.float32(hi_keep),
    )


# ``_batched._batched_incoherent_block``, one polarization: v.e and the
# refracted k_eff.e = omega delta (grad L . e).
_POLARIZATION_BODY = r"""
v_dot_e = __fadd_rn(__fadd_rn(__fmul_rn(vx, ex), __fmul_rn(vy, ey)), __fmul_rn(vz, ez));
k_dot_e = __fmul_rn(
    __fmul_rn(omega_res, delta),
    __fadd_rn(__fadd_rn(__fmul_rn(g0, ex), __fmul_rn(g1, ey)), __fmul_rn(g2, ez)));
"""


def polarization_terms(vx, vy, vz, ex, ey, ez, g0, g1, g2, omega_res, delta):
    """Fused ``(v.e, k_eff.e)`` for one polarization, or ``None``."""
    if not _float32_cupy(vx, vy, vz, ex, ey, ez, g0, g1, g2, omega_res, delta):
        return None
    kernel = _kernel(
        "polarization_terms",
        "float32 vx, float32 vy, float32 vz, float32 ex, float32 ey, float32 ez, "
        "float32 g0, float32 g1, float32 g2, float32 omega_res, float32 delta",
        "float32 v_dot_e, float32 k_dot_e",
        _POLARIZATION_BODY,
    )
    if kernel is None:
        return None
    return kernel(vx, vy, vz, ex, ey, ez, g0, g1, g2, omega_res, delta)


# ``segment_escape.piece_mean_transmission`` for one layer, times the piece
# fraction: sum_p frac_p mean_transmission(path_start_p mu, path_end_p mu).
_TRANSMISSION_BODY = r"""
const long long row = i / n_g;
float total = 0.0f;
for (long long p = 0; p < n_pieces; ++p) {
    const long long at = row * n_pieces + p;
    const float tau_start = __fmul_rn(path_start[at], mu);
    const float tau_end = __fmul_rn(path_end[at], mu);
    const float lo = _min_nan(tau_start, tau_end);
    const float d = fabsf(__fsub_rn(tau_end, tau_start));
    const bool steep = d > 1.0e-4f;
    const float safe = steep ? d : 1.0f;
    const float ratio = steep
        ? __fdiv_rn(-expm1f(-safe), safe)
        : __fadd_rn(__fsub_rn(1.0f, __fmul_rn(0.5f, d)), __fdiv_rn(__fmul_rn(d, d), 6.0f));
    const float term = __fmul_rn(fraction[at], __fmul_rn(expf(-lo), ratio));
    total = (p == 0) ? term : __fadd_rn(total, term);
}
T_abs = __fmul_rn(total, piece_fraction);
"""


def piece_transmission(fraction, path_start, path_end, mu, piece_fraction):
    """Fused single-layer piece-mean transmission times ``piece_fraction``.

    ``fraction`` is ``(N, P)``, ``path_*`` ``(N, P, 1)``, ``mu`` ``(N, M)`` and
    ``piece_fraction`` ``(N, 1)``. Returns ``(N, M)`` or ``None`` (also for a
    layered absorber or a piece-free row set).
    """
    if path_start.ndim != 3 or path_start.shape[2] != 1 or fraction.shape[1] == 0:
        return None
    if not _float32_cupy(fraction, path_start, path_end, mu, piece_fraction):
        return None
    kernel = _kernel(
        "piece_transmission",
        "raw float32 fraction, raw float32 path_start, raw float32 path_end, float32 mu, "
        "float32 piece_fraction, int64 n_g, int64 n_pieces",
        "float32 T_abs",
        _TRANSMISSION_BODY,
    )
    if kernel is None:
        return None
    return kernel(
        xp.ascontiguousarray(fraction),
        xp.ascontiguousarray(path_start),
        xp.ascontiguousarray(path_end),
        mu,
        piece_fraction,
        np.int64(mu.shape[-1]),
        np.int64(fraction.shape[1]),
    )


# ``_batched._batched_incoherent_block`` after ``_cxr_weight_core``: the sinc
# half-width, the mosaic-weighted line weight, and the surviving-line mask.
_LINE_WEIGHT_BODY = r"""
a_width = __fmul_rn(__fdiv_rn(__fmul_rn(denom, t_L), two_hbarc), 1.0f);
const float w = __fmul_rn(__fmul_rn(pref, A2), WM);
weight = w;
good = keep & isfinite(w) & (w > 0.0f);
"""


def line_weights(denom, t_L, pref, A2, WM, keep, *, two_hbarc):
    """Fused ``(a_width, weight, good)``, or ``None``."""
    if not _float32_cupy(denom, t_L, pref, A2, WM) or keep.dtype != np.bool_:
        return None
    kernel = _kernel(
        "line_weights",
        "float32 denom, float32 t_L, float32 pref, float32 A2, float32 WM, bool keep, "
        "float32 two_hbarc",
        "float32 a_width, float32 weight, bool good",
        _LINE_WEIGHT_BODY,
    )
    if kernel is None:
        return None
    return kernel(denom, t_L, pref, A2, WM, keep, np.float32(two_hbarc))
