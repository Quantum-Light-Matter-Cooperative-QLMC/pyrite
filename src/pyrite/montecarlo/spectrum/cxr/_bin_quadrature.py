"""Closed-form bin integration of the finite-time ``sinc^2`` line profile.

The default line quadrature samples ``sinc^2(a_width (E - E_res) / pi)`` at each
grid node. When a line is narrower than the node spacing that sample aliases
and the integrated yield is wrong. The ``"bin-mean"`` quadrature instead
writes the profile's *mean over each node's bin* into the same density array,
so the photons/eV units do not change and ``sum(density * width)`` is the line
mass inside the grid at any spacing.

With ``x = a_width (E - E_res) / pi`` (``np.sinc`` convention),

    ``F(x) = integral_0^x sinc^2 = Si(2 pi x)/pi - x sinc^2(x)``,
    ``F(+-inf) = +-1/2``,

and a bin ``[E_lo, E_hi]`` holds energy mass ``(pi / a_width) [F(x_hi) - F(x_lo)]``.
``x sinc^2(x)`` is the ``sin^2(pi x)/(pi^2 x)`` term written so ``x = 0`` needs
no series. The antiderivative is evaluated through its tail complement

    ``R(x) = F(x) - s(x)/2``, ``s(x) = +1 for x >= 0, -1 otherwise``,

so a bin mass is ``R(x_hi) - R(x_lo)`` plus one whole line (``1``) when the bin
contains ``E_res`` (``x_lo < 0 <= x_hi``). ``R`` decays like ``1/x``, so the
difference never subtracts two numbers near ``+-1/2``, and ``Si(t) - pi/2`` is
computed directly instead of by cancelling against ``pi/2``. Everything is
evaluated in FP64 and cast to the working precision only as a finished bin mean.

``Si`` has three evaluators with one partition at ``t = 48``:

* NumPy: ``scipy.special.sici`` below 48, the auxiliary-function asymptotic
  series above.
* CUDA/ROCm (both the CuPy fallback and the fused reduction): the C preamble
  below -- power series for ``t <= 4``, rational auxiliary-function fits
  for ``4 < t < 48`` (Cephes; split at 8), the same asymptotic series above. Measured
  against 60-digit ``mpmath`` in the branch-boundary regressions: absolute
  error <= 1e-15 for ``t <= 4`` and <= 2.1e-15/t beyond, relative to the
  ``1/t`` envelope of ``Si(t) - pi/2``.
* Other accelerators (SYCL): the NumPy evaluator on host.

Production routes (``sincsq_bin_lineshape``, ``run_bin_mean_reduction_kernel``)
evaluate this exact mean only for bins within ``BIN_MEAN_EXACT_WIDTHS``
first-zero widths of the resonance. A bin wholly beyond that on one side
(``x_lo >= K`` or ``x_hi <= -K``) takes the exact bin mean of the
``sin^2``-averaged envelope ``1/(2 pi^2 x^2)``,

    ``mean = 1 / (2 pi^2 x_lo x_hi)``,

in float32 arithmetic. Adjacent envelope bins share edges, so their masses
telescope; the only yield change is the omitted oscillatory term
``-cos(2 pi x)/(2 pi^2 x^2)`` over the far region, at most ``3/(4 pi^3 K^2)``
of the line per side (integration by parts), ``1.2e-5`` in total at ``K = 64``.
It removes the FP64 sine-integral evaluation from all but ~``4K`` bins per
line, which dominates bin-mean cost on consumer GPUs.

Validation: sinc-bin-integration
Validation: sinc-bin-far-envelope
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ._sine_integral import SI_RATIONAL_PREAMBLE

#: Line quadratures ``mc_cxr_spectrum`` accepts. ``"node"`` is the default.
CXR_QUADRATURES = ("node", "bin-mean")
NODE_QUADRATURE = "node"
BIN_MEAN_QUADRATURE = "bin-mean"

#: ``Si`` evaluator partition: exact library/rational fits below, the
#: auxiliary-function asymptotic series at and above.
_SI_ASYMPTOTIC_T = 48.0
#: Asymptotic terms. The first omitted term at ``t = 48`` is 26!/48**26 < 1e-17
#: of the leading one, and it only shrinks as ``t`` grows.
_SI_ASYMPTOTIC_TERMS = 13
#: First-zero widths from the resonance within which production bin means are
#: exact; beyond, the averaged envelope (yield error <= 3/(2 pi^3 K^2) = 1.2e-5
#: per line). ``None`` on the routes' ``exact_widths`` keeps every bin exact.
#: Validation: sinc-bin-far-envelope
BIN_MEAN_EXACT_WIDTHS = 64.0
#: Host evaluator block size in (line, bin) elements, bounding its temporaries.
_HOST_BLOCK_ELEMENTS = 1 << 22

#: Midpoint-series window: bins narrower than this many lobe widths and centred
#: at least ``_SERIES_MIN_X`` lobes from the resonance take the series mean
#: (``PYRITE_BIN_MEAN_SERIES=0`` compiles it out for comparison).
#: Validation: sinc-bin-midpoint-series
_SERIES_MAX_DX = 0.1
_SERIES_MIN_X = 2.0

#: C source shared by both CUDA/ROCm routes. ``__device__`` is CuPy's
#: host/device qualifier for kernel preambles.
SINCSQ_BIN_PREAMBLE = (
    f"""
#define PYRITE_SERIES_MAX_DX {_SERIES_MAX_DX!r}
#define PYRITE_SERIES_MIN_X {_SERIES_MIN_X!r}
"""
    + SI_RATIONAL_PREAMBLE
    + r"""
static const double PYRITE_PI = 3.14159265358979323846264338327950288;

/* Si(t) - pi/2 for t >= 0. */
static __device__ double pyrite_si_minus_half_pi(double t) {
    if (t <= 4.0) {
        const double t2 = t * t;
        double term = t;
        double sum = t;
        for (int k = 1; k < 40; ++k) {
            const double m = 2.0 * k;
            term = -term * t2 / (m * (m + 1.0));
            const double add = term / (m + 1.0);
            sum += add;
            if (fabs(add) <= 1.0e-17 * fabs(sum)) {
                break;
            }
        }
        return sum - 0.5 * PYRITE_PI;
    }
    if (t < 48.0) {
        return pyrite_si_rational_complement(t);
    }
    /* Si = pi/2 - f cos t - g sin t, f and g by their asymptotic series. */
    const double y = 1.0 / (t * t);
    double f_term = 1.0, g_term = 1.0, f = 0.0, g = 0.0;
    for (int k = 0; k < 13; ++k) {
        f += f_term;
        g += g_term;
        f_term *= -(2.0 * k + 1.0) * (2.0 * k + 2.0) * y;
        g_term *= -(2.0 * k + 2.0) * (2.0 * k + 3.0) * y;
    }
    return -(f / t * cos(t) + g * y * sin(t));
}

/* R(x) = F(x) - s(x)/2, the tail complement of the sinc^2 antiderivative. */
static __device__ double pyrite_sincsq_tail(double x) {
    const double u = fabs(x);
    const double t = 2.0 * PYRITE_PI * u;
    const double y1 = PYRITE_PI * u;
    double r;
    if (t >= 48.0) {
        /* Far tail, where nearly every (line, edge) pair lands: with
           f = (1 + d)/t, sin t = 2 s c and cos t = 1 - 2 s^2 (s, c of pi u),
           R = -(1 + d)/(2 pi^2 u) + d s^2/(pi^2 u) - 2 g s c/pi exactly, which
           needs two trig calls and has no s^2 cancellation. */
        const double s1 = sin(y1);
        const double c1 = cos(y1);
        const double y = 1.0 / (t * t);
        double d = 0.0, g = 0.0;
        double f_term = -2.0 * y, g_term = 1.0;
        for (int k = 1; k < 13; ++k) {
            d += f_term;
            f_term *= -(2.0 * k + 1.0) * (2.0 * k + 2.0) * y;
            if (fabs(f_term) < 1.0e-18) {
                break;
            }
        }
        for (int k = 0; k < 13; ++k) {
            g += g_term;
            g_term *= -(2.0 * k + 2.0) * (2.0 * k + 3.0) * y;
            if (fabs(g_term) < 1.0e-18) {
                break;
            }
        }
        const double pi2u = PYRITE_PI * PYRITE_PI * u;
        r = -(1.0 + d) / (2.0 * pi2u) + d * s1 * s1 / pi2u - 2.0 * g * y * s1 * c1 / PYRITE_PI;
    } else {
        const double s = (u == 0.0) ? 1.0 : sin(y1) / y1;
        r = pyrite_si_minus_half_pi(t) / PYRITE_PI - u * s * s;
    }
    return (x >= 0.0) ? r : -r;
}

#ifndef PYRITE_BIN_MEAN_SERIES
#define PYRITE_BIN_MEAN_SERIES 1
#endif

/* Mean of f = sinc^2 over a bin much narrower than a lobe, by its midpoint
   series mean = f + f2 dx^2/24 + f4 dx^4/1920 + f6 dx^6/322560 (fk the k-th
   derivative; Validation: sinc-bin-midpoint-series). f = A(x) H(x) / (2 pi^2)
   with A = 1 - cos(2 pi x) and H = x^-2, so every derivative comes from one
   sincospi. Constant divisions are reciprocal products: FP64 division is
   costly on consumer GPUs. */
static __device__ double pyrite_sincsq_midpoint_mean(double xc, double dx) {
    double sp, cp;
    sincospi(xc, &sp, &cp);
    const double om = 2.0 * PYRITE_PI, om2 = om * om;
    const double a0 = 2.0 * sp * sp;          /* 1 - cos(2 pi x) */
    const double s2 = 2.0 * sp * cp;          /* sin(2 pi x) */
    const double c2 = (cp - sp) * (cp + sp);  /* cos(2 pi x) */
    const double a1 = om * s2, a2 = om2 * c2;
    const double a3 = -om2 * a1, a4 = -om2 * a2, a5 = -om2 * a3, a6 = -om2 * a4;
    const double q = 1.0 / xc;
    const double h0 = q * q, h1 = -2.0 * h0 * q, h2 = -3.0 * h1 * q;
    const double h3 = -4.0 * h2 * q, h4 = -5.0 * h3 * q, h5 = -6.0 * h4 * q;
    const double h6 = -7.0 * h5 * q;
    const double f0 = a0 * h0;
    const double f2 = a2 * h0 + 2.0 * a1 * h1 + a0 * h2;
    const double f4 = a4 * h0 + 4.0 * a3 * h1 + 6.0 * a2 * h2 + 4.0 * a1 * h3 + a0 * h4;
    const double f6 = a6 * h0 + 6.0 * a5 * h1 + 15.0 * a4 * h2 + 20.0 * a3 * h3
        + 15.0 * a2 * h4 + 6.0 * a1 * h5 + a0 * h6;
    const double d2 = dx * dx;
    const double series = f0 + d2 * (f2 * (1.0 / 24.0)
        + d2 * (f4 * (1.0 / 1920.0) + d2 * (f6 * (1.0 / 322560.0))));
    return series * (1.0 / (2.0 * PYRITE_PI * PYRITE_PI));
}

/* True, with the midpoint-series mean, when [e_lo, e_hi] is inside the series window. */
static __device__ bool pyrite_sincsq_series_mean(
    double a, double e_r, double e_lo, double e_hi, double *mean
) {
#if PYRITE_BIN_MEAN_SERIES
    const double scale = a * (1.0 / PYRITE_PI);
    const double dx = scale * (e_hi - e_lo);
    const double xc = scale * ((0.5 * (e_lo + e_hi)) - e_r);
    if (dx <= PYRITE_SERIES_MAX_DX && fabs(xc) >= PYRITE_SERIES_MIN_X) {
        *mean = pyrite_sincsq_midpoint_mean(xc, dx);
        return true;
    }
#endif
    return false;
}

/* Mean of sinc^2(a (E - e_r) / pi) over [e_lo, e_hi]; inv_width = 1/(e_hi - e_lo). */
static __device__ double pyrite_sincsq_bin_mean(
    double a, double e_r, double e_lo, double e_hi, double inv_width
) {
    double series;
    if (pyrite_sincsq_series_mean(a, e_r, e_lo, e_hi, &series)) {
        return series;
    }
    const double x_lo = a * (e_lo - e_r) / PYRITE_PI;
    const double x_hi = a * (e_hi - e_r) / PYRITE_PI;
    double mass = pyrite_sincsq_tail(x_hi) - pyrite_sincsq_tail(x_lo);
    if (x_lo < 0.0 && x_hi >= 0.0) {
        mass += 1.0;
    }
    return mass * (PYRITE_PI / a) * inv_width;
}

/* Far-field classification and value, all in float32 (Validation:
   sinc-bin-far-envelope). Each FP64 edge and the resonance are split into a
   float32 head and float32 remainder, so edge - e_r stays near float32 relative
   accuracy (degrading only for E_res/|edge - e_r| >~ 1e6) (an FP64 resonance rounded to float32 first
   would shift x by ulp(E_res)/w, beyond the bound once E_res/w nears 2^23). x = (a / pi)(edge - e_r); a bin is far when both edges lie
   >= exact_widths widths from e_r on one side, and its mean is then the
   sin^2-averaged envelope's, 1/(2 pi^2 x_lo x_hi). Host and device evaluate
   this same float32 sequence, so they classify every bin alike. */
static __device__ bool pyrite_sincsq_far_mean(
    float a, float e_f, float e_c, float lo_f, float lo_c, float hi_f, float hi_c,
    float exact_widths, float *mean
) {
    const float scale = a * 0.318309886f;
    const float x_lo = scale * ((lo_f - e_f) + (lo_c - e_c));
    const float x_hi = scale * ((hi_f - e_f) + (hi_c - e_c));
    if (x_lo >= exact_widths || x_hi <= -exact_widths) {
        *mean = 1.0f / (19.7392088f * (x_lo * x_hi));
        return true;
    }
    return false;
}

/* Production bin mean: the far envelope beyond exact_widths, else exact. */
static __device__ double pyrite_sincsq_bin_mean_hybrid(
    double a, double e_r, double e_lo, double e_hi, double inv_width, double exact_widths
) {
    const float lo_f = (float)e_lo, hi_f = (float)e_hi;
    const float e_f = (float)e_r;
    float mean;
    if (pyrite_sincsq_far_mean(
            (float)a, e_f, (float)(e_r - (double)e_f), lo_f, (float)(e_lo - (double)lo_f), hi_f,
            (float)(e_hi - (double)hi_f), (float)exact_widths, &mean)) {
        return (double)mean;
    }
    return pyrite_sincsq_bin_mean(a, e_r, e_lo, e_hi, inv_width);
}
"""
)


def validate_cxr_quadrature(value: object) -> str:
    """Return ``value`` as a known line quadrature name, or raise."""
    if value not in CXR_QUADRATURES:
        raise ValueError(f"line_quadrature must be one of {list(CXR_QUADRATURES)}, got {value!r}")
    return str(value)


def si_minus_half_pi(t: object) -> np.ndarray:
    """``Si(t) - pi/2`` for ``t >= 0`` on host, without cancelling against ``pi/2``.

    Validation: sinc-bin-integration
    """
    from scipy.special import sici

    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    near = t < _SI_ASYMPTOTIC_T
    out[near] = sici(t[near])[0] - 0.5 * np.pi
    far = t[~near]
    y = 1.0 / (far * far)
    d, g = _asymptotic_series(y)
    out[~near] = -((1.0 + d) / far * np.cos(far) + g * y * np.sin(far))
    return out


def _asymptotic_series(y):
    """``(f t - 1, g t^2)`` as series in ``y = 1/t^2``.

    ``f t - 1`` is summed from its first correction term, so no ``1`` is ever
    subtracted back out.
    """
    d = np.zeros_like(y)
    g = np.zeros_like(y)
    f_term = -2.0 * y
    g_term = np.ones_like(y)
    for k in range(_SI_ASYMPTOTIC_TERMS):
        if k:
            d += f_term
            f_term = f_term * (-(2.0 * k + 1.0) * (2.0 * k + 2.0) * y)
        g += g_term
        g_term = g_term * (-(2.0 * k + 2.0) * (2.0 * k + 3.0) * y)
    return d, g


def sincsq_tail(x: object) -> np.ndarray:
    """``R(x) = F(x) - s(x)/2`` on host (see the module docstring).

    Validation: sinc-bin-integration
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.abs(x)
    t = 2.0 * np.pi * u
    r = np.empty_like(u)
    near = t < _SI_ASYMPTOTIC_T
    un = u[near]
    s = np.sinc(un)
    r[near] = si_minus_half_pi(t[near]) / np.pi - un * s * s
    # Far tail: the preamble's two-trig form (see SINCSQ_BIN_PREAMBLE).
    uf = u[~near]
    y = 1.0 / (t[~near] * t[~near])
    d, g = _asymptotic_series(y)
    s1 = np.sin(np.pi * uf)
    c1 = np.cos(np.pi * uf)
    pi2u = np.pi * np.pi * uf
    r[~near] = -(1.0 + d) / (2.0 * pi2u) + d * s1 * s1 / pi2u - 2.0 * g * y * s1 * c1 / np.pi
    return np.where(x >= 0.0, r, -r)


def sincsq_midpoint_mean(xc: object, dx: object) -> np.ndarray:
    """Midpoint series of the ``sinc^2`` mean over ``[xc - dx/2, xc + dx/2]``.

    ``f + f2 dx^2/24 + f4 dx^4/1920 + f6 dx^6/322560`` (``fk`` the k-th
    derivative) with ``f = (1 - cos 2 pi x)/(2 pi^2 x^2)``: the host mirror of
    the device ``pyrite_sincsq_midpoint_mean``. Valid only in the window
    ``dx <= _SERIES_MAX_DX``, ``|xc| >= _SERIES_MIN_X``, where the omitted
    eighth-derivative term is below ``2.5e-9`` of the envelope ``1/(2 pi^2 xc^2)``.

    Validation: sinc-bin-midpoint-series
    """
    xc = np.asarray(xc, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    om = 2.0 * np.pi
    sp, cp = np.sin(np.pi * xc), np.cos(np.pi * xc)
    a0 = 2.0 * sp * sp
    s2 = 2.0 * sp * cp
    c2 = (cp - sp) * (cp + sp)
    a1, a2 = om * s2, om * om * c2
    a3, a4 = -(om**3) * s2, -(om**4) * c2
    a5, a6 = (om**5) * s2, (om**6) * c2
    q = 1.0 / xc
    h0 = q * q
    h1 = -2.0 * h0 * q
    h2 = -3.0 * h1 * q
    h3 = -4.0 * h2 * q
    h4 = -5.0 * h3 * q
    h5 = -6.0 * h4 * q
    h6 = -7.0 * h5 * q
    f0 = a0 * h0
    f2 = a2 * h0 + 2.0 * a1 * h1 + a0 * h2
    f4 = a4 * h0 + 4.0 * a3 * h1 + 6.0 * a2 * h2 + 4.0 * a1 * h3 + a0 * h4
    f6 = (
        a6 * h0
        + 6.0 * a5 * h1
        + 15.0 * a4 * h2
        + 20.0 * a3 * h3
        + 15.0 * a2 * h4
        + 6.0 * a1 * h5
        + a0 * h6
    )
    d2 = dx * dx
    series = f0 + d2 * (f2 / 24.0 + d2 * (f4 / 1920.0 + d2 * (f6 / 322560.0)))
    return series / (2.0 * np.pi * np.pi)


def sincsq_antiderivative(x: object) -> np.ndarray:
    """``F(x) = Si(2 pi x)/pi - x sinc^2(x)``, the integral of ``sinc^2`` from 0.

    Validation: sinc-bin-integration
    """
    x = np.asarray(x, dtype=np.float64)
    return sincsq_tail(x) + np.where(x >= 0.0, 0.5, -0.5)


def _far_envelope_float32(a_width, E_res, edges, exact_widths):
    """``(far, mean)`` of the far-field envelope, ``[line, bin]``, in the exact
    float32 operation order of ``pyrite_sincsq_far_mean``.

    Validation: sinc-bin-far-envelope
    """
    f32 = np.float32
    edges = np.asarray(edges, dtype=np.float64)
    head = edges.astype(f32)
    carry = (edges - head.astype(np.float64)).astype(f32)
    a = np.asarray(a_width).astype(f32)[:, None]
    e_r64 = np.asarray(E_res, dtype=np.float64)
    e_head = e_r64.astype(f32)
    e_carry = (e_r64 - e_head.astype(np.float64)).astype(f32)
    scale = a * f32(0.318309886)
    x = scale * ((head[None, :] - e_head[:, None]) + (carry[None, :] - e_carry[:, None]))
    x_lo, x_hi = x[:, :-1], x[:, 1:]
    limit = f32(exact_widths)
    far = (x_lo >= limit) | (x_hi <= -limit)
    with np.errstate(divide="ignore", over="ignore"):
        mean = f32(1.0) / (f32(19.7392088) * (x_lo * x_hi))
    return far, mean.astype(np.float64)


def _host_bin_mean(a_width, E_res, edges, inv_width, exact_widths=None):
    """``[line, bin]`` bin means on host, in FP64.

    ``exact_widths=None`` evaluates every bin exactly; a number applies the
    far-field envelope beyond that many first-zero widths, as the device route.

    Validation: sinc-bin-far-envelope
    """
    a = np.asarray(a_width, dtype=np.float64)[:, None]
    x = a * (np.asarray(edges, dtype=np.float64)[None, :] - np.asarray(E_res, np.float64)[:, None])
    x = x / np.pi
    scale = (np.pi / a) * np.asarray(inv_width, dtype=np.float64)[None, :]
    if exact_widths is None:
        tail = sincsq_tail(x)
        mass = tail[:, 1:] - tail[:, :-1]
        mass += (x[:, :-1] < 0.0) & (x[:, 1:] >= 0.0)
        return mass * scale
    # Validation: sinc-bin-far-envelope -- the device's float32 sequence
    far, far_mean = _far_envelope_float32(a[:, 0], E_res, edges, float(exact_widths))
    x_lo, x_hi = x[:, :-1], x[:, 1:]
    needed = np.zeros(x.shape, dtype=bool)
    needed[:, :-1] |= ~far
    needed[:, 1:] |= ~far
    tail = np.zeros_like(x)
    tail[needed] = sincsq_tail(x[needed])
    mass = tail[:, 1:] - tail[:, :-1]
    mass += (x_lo < 0.0) & (x_hi >= 0.0)
    mean = mass * scale
    mean[far] = far_mean[far]
    return mean


def sincsq_bin_masses(a_width: object, E_res: object, edges_eV: object) -> np.ndarray:
    """Energy mass [eV] of each line in each bin, shape ``(n_line, n_bin)``.

    Summed over a line's bins this is ``pi / a_width`` minus
    :func:`sincsq_window_mass`'s truncated part, to rounding.

    Validation: sinc-bin-integration
    """
    edges = np.asarray(edges_eV, dtype=np.float64)
    widths = np.diff(edges)
    a = np.atleast_1d(np.asarray(a_width, dtype=np.float64))
    e_r = np.atleast_1d(np.asarray(E_res, dtype=np.float64))
    return _host_bin_mean(a, e_r, edges, 1.0 / widths) * widths[None, :]


def sincsq_window_mass(
    a_width: object, E_res: object, edges_eV: object
) -> tuple[np.ndarray, np.ndarray]:
    """Per-line ``(captured, truncated)`` energy mass [eV] of the grid window.

    ``captured + truncated == pi / a_width``: the whole-line integral. The
    truncated part is the two tails beyond the outer bin edges; bin-mean
    quadrature does not fold it back into the window.

    Validation: sinc-bin-integration
    """
    edges = np.asarray(edges_eV, dtype=np.float64)
    a = np.atleast_1d(np.asarray(a_width, dtype=np.float64))
    e_r = np.atleast_1d(np.asarray(E_res, dtype=np.float64))
    x_bot = a * (edges[0] - e_r) / np.pi
    x_top = a * (edges[-1] - e_r) / np.pi
    r_bot = sincsq_tail(x_bot)
    r_top = sincsq_tail(x_top)
    above = np.where(x_top >= 0.0, -r_top, 1.0 - r_top)  # 1/2 - F(x_top)
    below = np.where(x_bot >= 0.0, r_bot + 1.0, r_bot)  # F(x_bot) + 1/2
    scale = np.pi / a
    truncated = scale * (above + below)
    return scale - truncated, truncated


_BIN_MEAN_KERNEL = None
_BIN_REDUCE_KERNEL = None


def _cupy_backend() -> bool:
    return getattr(xp, "__name__", "") == "cupy" and hasattr(xp, "ElementwiseKernel")


def _bin_mean_kernel():
    """``(line, bin)`` bin-mean matrix kernel. Built on first use; CuPy only."""
    global _BIN_MEAN_KERNEL
    if _BIN_MEAN_KERNEL is None:
        import cupy

        _BIN_MEAN_KERNEL = cupy.ElementwiseKernel(
            "raw T aw, raw T e_r, raw float64 edges, raw float64 inv_width, int64 n_e, "
            "float64 exact_widths",
            "T out",
            r"""
            const long long row = (long long)(i / n_e);
            const long long col = (long long)(i % n_e);
            out = (T)pyrite_sincsq_bin_mean_hybrid(
                (double)aw[row], (double)e_r[row], edges[col], edges[col + 1], inv_width[col],
                exact_widths
            );
            """,
            "pyrite_sincsq_bin_mean_matrix",
            preamble=SINCSQ_BIN_PREAMBLE,
        )
    return _BIN_MEAN_KERNEL


#: Threads per bin in the fused reduction; lines stride across them.
_REDUCE_THREADS = 256

_BIN_REDUCE_SOURCE = r"""
/* One block per (bin, line chunk); lines of the chunk stride across its
   threads a tile at a time. Block row x evaluates bin bin_index[x]. A single
   chunk (gridDim.y == 1) covers every line; more chunks fill the device when
   bins are few, and each writes its own partial, summed by the caller with a
   fixed device reduction.
   Far pairs add to a compensated float32 sum on the spot. Near pairs, which
   need the FP64 sine integral, are compacted into a shared queue in lane
   order (warp ballot + prefix) and evaluated together once the queue fills,
   so FP64 work runs on full warps instead of one lane of a divergent warp.
   Queue order and the queue-to-thread assignment are fixed, so the result is
   deterministic. */
extern "C" __global__ void pyrite_sincsq_bin_mean_reduce(
    const float *e_r, const float *aw, const float *w, const double *edges,
    const double *inv_width, double *out, const long long n_lines, const float far_widths,
    const long long chunk_lines, const long long *bin_index
) {
    extern __shared__ double shared[];
    const unsigned int tid = threadIdx.x;
    const unsigned int threads = blockDim.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int n_warps = (threads + 31u) >> 5;
    double *partial = shared;
    long long *queue = (long long *)(partial + threads);
    int *warp_offset = (int *)(queue + 2 * threads);
    int *queue_count = warp_offset + 32;

    const long long bin = bin_index[blockIdx.x];
    const long long first = (long long)blockIdx.y * chunk_lines;
    const long long last = min(n_lines, first + chunk_lines);
    const double lo = edges[bin];
    const double hi = edges[bin + 1];
    const double iw = inv_width[bin];
    const float lo_f = (float)lo, hi_f = (float)hi;
    const float lo_c = (float)(lo - (double)lo_f), hi_c = (float)(hi - (double)hi_f);
    if (tid == 0) {
        *queue_count = 0;
    }
    double near = 0.0;
    float far_sum = 0.0f, far_carry = 0.0f;
    for (long long base = first; base < last; base += threads) {
        const long long line = base + tid;
        bool is_near = false;
        if (line < last) {
            const float wt = w[line];
            if (wt != 0.0f) {
                float mean;
                /* e_r is already float32 here: no resonance remainder. */
                if (pyrite_sincsq_far_mean(
                        aw[line], e_r[line], 0.0f, lo_f, lo_c, hi_f, hi_c, far_widths, &mean)) {
                    const float y = wt * mean - far_carry;
                    const float t = far_sum + y;
                    far_carry = (t - far_sum) - y;
                    far_sum = t;
                } else {
                    is_near = true;
                }
            }
        }
#if defined(__HIP_PLATFORM_AMD__)
        /* HIP ballots return a 64-bit wavefront mask; keep this 32-lane group's half. */
        const unsigned int mask = (unsigned int)(__ballot(is_near) >> (__lane_id() & 32u));
#else
        const unsigned int mask = __ballot_sync(0xffffffffu, is_near);
#endif
        if (lane == 0) {
            warp_offset[warp] = __popc(mask);
        }
        __syncthreads();
        if (tid == 0) {
            int running = *queue_count;
            for (unsigned int k = 0; k < n_warps; ++k) {
                const int count = warp_offset[k];
                warp_offset[k] = running;
                running += count;
            }
            *queue_count = running;
        }
        __syncthreads();
        if (is_near) {
            queue[warp_offset[warp] + __popc(mask & ((1u << lane) - 1u))] = line;
        }
        __syncthreads();
        const int pending = *queue_count;
        if (pending >= (int)threads || base + threads >= last) {
            for (int k = tid; k < pending; k += threads) {
                const long long q = queue[k];
                near += (double)w[q] * pyrite_sincsq_bin_mean(
                    (double)aw[q], (double)e_r[q], lo, hi, iw);
            }
            __syncthreads();
            if (tid == 0) {
                *queue_count = 0;
            }
            __syncthreads();
        }
    }
    partial[tid] = near + ((double)far_sum - (double)far_carry);
    __syncthreads();
    for (unsigned int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[(long long)blockIdx.x * gridDim.y + blockIdx.y] = partial[0];
    }
}
"""


#: Resident blocks per SM the all-pairs launch aims for before splitting lines.
_REDUCE_BLOCKS_PER_SM = 8
#: Fewest lines per chunk: four 256-thread tiles keep the queue and the final
#: block reduction a small share of a chunk.
_REDUCE_MIN_CHUNK_LINES = 4 * 256


@cache
def _sm_count(device: int) -> int:
    import cupy

    return int(cupy.cuda.runtime.getDeviceProperties(device)["multiProcessorCount"])


def _line_chunks(n_bins: int, n_lines: int) -> tuple[int, int]:
    """``(chunks, lines per chunk)`` for the all-pairs kernel grid ``(n_bins, chunks)``.

    One block per bin leaves most of the device idle when bins are few (the
    ~50 leftover bins of a ``"segmented"`` flush ran ~50 blocks over every
    line, #1403). Lines are then split into chunks until the grid reaches
    ``_REDUCE_BLOCKS_PER_SM`` blocks per SM. One chunk keeps the single-block
    summation order. More only regroup the sums: each chunk keeps its own FP64
    near sum and compensated float32 far sum, and the chunk totals add in FP64
    with a fixed CuPy reduction, so the result is repeatable for a fixed device
    and CuPy version. The chunk count follows the SM count, so it can differ
    between devices (hence ``CXR_BIN_REDUCTION_MODEL``).
    """
    import cupy

    target = _REDUCE_BLOCKS_PER_SM * _sm_count(cupy.cuda.Device().id)
    chunks = min(-(-target // n_bins), max(1, n_lines // _REDUCE_MIN_CHUNK_LINES))
    chunks = max(1, chunks)
    chunk_lines = -(-n_lines // chunks)
    return -(-n_lines // chunk_lines), chunk_lines


def _bin_reduce_kernel():
    """Fused bin-mean reduction: one block per bin, lines strided across its
    threads (coalesced, like the node kernel), one FP64 sum per bin. Built on
    first use; CuPy only.

    Validation: sinc-bin-integration
    Validation: sinc-bin-far-envelope
    """
    global _BIN_REDUCE_KERNEL
    if _BIN_REDUCE_KERNEL is None:
        import cupy

        _BIN_REDUCE_KERNEL = cupy.RawKernel(
            SINCSQ_BIN_PREAMBLE + _BIN_REDUCE_SOURCE, "pyrite_sincsq_bin_mean_reduce"
        )
    return _BIN_REDUCE_KERNEL


def bin_axis(E_grid_eV: object):
    """``(edges, inv_width)`` for the bin-mean routes, on the routes' device.

    Edges come from the characteristic-line convention
    (``characteristic._energy_bin_edges_and_widths``: midpoints, reflected outer
    half-widths, low edge clamped at 0 eV), computed from the uncast FP64
    coordinates so a float32 grid cast does not move them. CuPy routes get
    device FP64 copies; every other backend keeps them on host.
    """
    # Lazy: ``characteristic`` imports this package at module scope.
    from ..characteristic import _energy_bin_edges_and_widths

    edges, widths = _energy_bin_edges_and_widths(E_grid_eV)
    inv_width = 1.0 / widths
    if _cupy_backend():
        return xp.asarray(edges, dtype=xp.float64), xp.asarray(inv_width, dtype=xp.float64)
    return edges, inv_width


def sincsq_bin_lineshape(
    a_width_j, E_res_j, edges, inv_width, *, exact_widths: float | None = BIN_MEAN_EXACT_WIDTHS
):
    """Bin-mean counterpart of ``_sincsq_lineshape``: ``(n_line, n_bin)`` in ``REAL``.

    ``a_width_j`` and ``E_res_j`` are 1-D device arrays of lines; ``edges`` and
    ``inv_width`` come from :func:`bin_axis`. Bins beyond ``exact_widths``
    first-zero widths take the far-field envelope; ``None`` keeps all exact.

    Validation: sinc-bin-integration
    Validation: sinc-bin-far-envelope
    """
    far_widths = np.inf if exact_widths is None else float(exact_widths)
    n_e = int(inv_width.shape[0])
    if _cupy_backend():
        out = xp.empty((int(a_width_j.size), n_e), dtype=REAL)
        if out.size:
            _bin_mean_kernel()(
                xp.ascontiguousarray(a_width_j, dtype=REAL),
                xp.ascontiguousarray(E_res_j, dtype=REAL),
                xp.ascontiguousarray(edges),
                xp.ascontiguousarray(inv_width),
                np.int64(n_e),
                np.float64(far_widths),
                out,
            )
        return out
    a_host = _to_cpu(a_width_j)
    e_host = _to_cpu(E_res_j)
    edges = _to_cpu(edges)
    inv_width = _to_cpu(inv_width)
    host = np.empty((a_host.size, n_e), dtype=REAL)
    # The host evaluator holds several FP64 (row, edge) temporaries; bound them.
    rows = max(1, _HOST_BLOCK_ELEMENTS // max(n_e, 1))
    for r0 in range(0, a_host.size, rows):
        r1 = min(r0 + rows, a_host.size)
        host[r0:r1] = _host_bin_mean(a_host[r0:r1], e_host[r0:r1], edges, inv_width, exact_widths)
    if getattr(xp, "__name__", "") == "numpy":
        return host
    return xp.asarray(host, dtype=REAL)


#: Numerical generation of device bin-mean reductions (case identity marker).
#: v4 (#1192): CUDA flushes on a uniform ``c_k +- h/2`` axis use the exact
#: Fourier-domain reduction (no far-field envelope); other axes and backends
#: keep the near/far kernel. v5 (#1230): the NumPy backend reduces uniform
#: axes through the host Fourier route (``_fourier_host``) instead of the
#: dense exact evaluator, and CUDA device flushes use the ES NUFFT
#: (``_fourier_spread``) as ROCm does, instead of cuFINUFFT (agreement
#: <= 1.2e-11) and without the cuFINUFFT requirement. v6 (#1296): device
#: flushes evaluate direct-rectangle phase factors with FP32 ``sincosf`` after
#: FP64 range reduction, request ES accuracy 1e-9 (was 1e-12), move the
#: NUFFT crossover to 2^24 pairs and fuse the envelope-image pass; <= 8e-11
#: of the peak from v5 on the mos2 30 keV flushes. Host flushes are unchanged.
#: v7 (#1297): device ES spreading sums each fine point's window by sorted,
#: batched warp gathers instead of per-source FP64 atomics (same terms, FP64
#: summation order only). v8 (#1320): device and host lines with more than
#: ``_bin_fourier.FOURIER_MAX_LINE_MODES`` positive modes take a near/far
#: evaluator (device kernel; host numba ``_bin_mean_jit``) instead of the
#: Fourier route; the kernel transform and gather grid are evaluated in
#: bounded blocks. Flushes without such lines change only in FP64 summation
#: order. v9 (#1390): NumPy-backend bin-mean lines on non-uniform axes are
#: queued; each uniform run of at least 64 bins takes the host Fourier route
#: on its sub-axis (no far envelope there), the other bins one numba near/far
#: call (device-preamble ``Si``, FP64 per-bin sums), instead of the dense
#: ``scipy.special.sici`` matrix product. The device route mirrors it
#: (``"segmented"``: CUDA Fourier per run, one near/far launch for the rest).
#: v10 (#1392): the uniform runs of a ``"segmented"`` axis share one line
#: transform at the largest run period and broad threshold, each run
#: evaluated at its centres by an ES type-2 transform (``_fourier_runs``).
#: v12 (#1407): device Si complement uses Cephes rational auxiliary fits on
#: 4 < t < 48 instead of the modified-Lentz continued fraction; host near/far
#: mirrors it. Classification, series window, and far envelope are unchanged.
CXR_BIN_REDUCTION_MODEL = "fourier-auto-v12"
#: Lines per Fourier-route flush: amortises the per-call FFT and planning.
FOURIER_FLUSH_LINES = 10_000_000

#: ``(method, flush_lines)`` override of the resolved device bin-mean
#: reduction, for benchmarks and cross-checks. Not case identity.
_REDUCTION_OVERRIDE: ContextVar[tuple[str, int | None] | None] = ContextVar(
    "pyrite_bin_reduction_override", default=None
)
_REDUCTION_METHODS = ("auto", "pairs", "tree", "sparse", "fourier", "segmented")


@contextmanager
def bin_reduction_override(method: str, *, flush_lines: int | None = None):
    """Force device bin-mean reductions in this scope to ``method``.

    ``"fourier"`` requires a uniform axis; ``"auto"``/``"pairs"``/``"tree"``/
    ``"sparse"`` select the near/far kernel. ``flush_lines`` replaces the
    batch target. Not case identity: for benchmarks and cross-checks only.
    """
    if method not in _REDUCTION_METHODS:
        raise ValueError(f"method must be one of {_REDUCTION_METHODS}")
    if flush_lines is not None and flush_lines < 1:
        raise ValueError("flush_lines must be positive")
    token = _REDUCTION_OVERRIDE.set((method, flush_lines))
    try:
        yield
    finally:
        _REDUCTION_OVERRIDE.reset(token)


def fourier_bin_axis(edges):
    """``(c0, h, n)`` when ``edges`` are uniform ``c_k +- h/2`` bins, else ``None``."""
    from ._bin_fourier import uniform_bin_axis

    edges = np.asarray(_to_cpu(edges), dtype=np.float64)
    try:
        axis = uniform_bin_axis(0.5 * (edges[:-1] + edges[1:]))
    except ValueError:
        return None
    if abs(edges[0] - (axis[0] - 0.5 * axis[1])) > 1e-9 * axis[1]:
        return None
    return axis


def fourier_backend_available() -> bool:
    """Whether this process can run the device Fourier reduction.

    Needs only the CuPy backend: the default NUFFT is the portable ES
    transform (``_fourier_spread``) on CUDA and ROCm alike (#1230), so
    cuFINUFFT is no longer required.
    """
    return _cupy_backend()


def resolve_bin_reduction(edges):
    """``(method, axis, flush_lines)`` for device bin-mean flushes on ``edges``.

    Default: ``"fourier"`` with its uniform axis when available; on a
    non-uniform axis holding a uniform run of at least ``HOST_SEGMENT_MIN_BINS``
    bins, ``"segmented"`` with the :func:`uniform_runs` list as ``axis``
    (#1390); else ``"auto"`` (near/far). An active
    :func:`bin_reduction_override` wins. ``flush_lines`` is ``None`` for the
    caller's default batch target.

    Validation: cxr-line-fourier-reduction
    """
    override = _REDUCTION_OVERRIDE.get()
    method, flush = override if override is not None else ("auto", None)
    axis = None
    if method == "fourier" or (override is None and fourier_backend_available()):
        axis = fourier_bin_axis(edges)
        if axis is None and method == "fourier":
            raise ValueError("Fourier reduction needs a uniform c_k +- h/2 bin axis")
        method = "auto" if axis is None else "fourier"
        if axis is None:
            from ._fourier_host import uniform_runs

            runs = uniform_runs(np.asarray(_to_cpu(edges), dtype=np.float64))
            if any(run[2] is not None for run in runs):
                method, axis = "segmented", tuple(runs)
    if method in ("fourier", "segmented") and flush is None:
        flush = FOURIER_FLUSH_LINES
    return method, axis, flush


def resolve_host_bin_reduction(edges):
    """``(axis, flush_lines)`` for host (NumPy) bin-mean lines on ``edges``, or ``None``.

    The host Fourier route (``_fourier_host``) applies to every uniform
    ``c_k +- h/2`` axis unless :func:`bin_reduction_override` selects a
    near/far method, which on host means the exact dense evaluator.

    Validation: cxr-line-fourier-reduction
    """
    from ._fourier_host import HOST_FOURIER_FLUSH_LINES

    override = _REDUCTION_OVERRIDE.get()
    method, flush = override if override is not None else ("fourier", None)
    if method != "fourier":
        return None
    axis = fourier_bin_axis(edges)
    if axis is None:
        if override is not None:
            raise ValueError("Fourier reduction needs a uniform c_k +- h/2 bin axis")
        return None
    return axis, HOST_FOURIER_FLUSH_LINES if flush is None else flush


def host_near_far_applies() -> bool:
    """Whether host bin-mean lines on a non-uniform axis take the numba near/far queue.

    True unless a :func:`bin_reduction_override` is active: a non-Fourier
    override keeps the dense host evaluator for comparisons (#1390).

    Validation: sinc-bin-far-envelope
    """
    return _REDUCTION_OVERRIDE.get() is None


def _run_segmented_reduction(E_r, aw, w, edges, inv_width, *, out, exact_widths, runs):
    """Device ``"segmented"`` flush: shared-mode Fourier on the uniform runs, near/far elsewhere.

    #1390 split the axis; #1392 shares one line transform across its runs.

    Validation: cxr-line-fourier-reduction, sinc-bin-far-envelope
    """
    import cupy

    from ._fourier_cuda import cuda_fourier_runs_reduce
    from ._fourier_host import leftover_bins, leftover_inv_width, uniform_runs

    host_edges = np.asarray(_to_cpu(edges), dtype=np.float64)
    if runs is None:
        runs = uniform_runs(host_edges)
    fourier = [(start, stop, axis) for start, stop, axis in runs if axis is not None]
    if fourier:
        # One line transform serves every run (#1392).
        parts = cuda_fourier_runs_reduce(E_r, aw, w, [r[2] for r in fourier])
        for (start, stop, _), part in zip(fourier, parts, strict=True):
            out[start:stop] += part.astype(out.dtype)
    leftover = leftover_bins(host_edges, runs)
    if leftover is not None:
        # One launch over every leftover bin: a launch per single bin ran one
        # block over all lines (~0.6 s each on the #1320 flush, nsys 15404).
        # Only the real bins run: the gaps between Fourier runs are wide bins
        # whose near pairs took most of the FP64 Si work (#1403).
        sub_edges, sub_bins, bins = leftover
        sub_inv = leftover_inv_width(sub_edges, sub_bins, bins, _to_cpu(inv_width))
        sub_out = cupy.zeros(sub_bins.size, dtype=cupy.float64)
        run_bin_mean_reduction_kernel(
            E_r,
            aw,
            w,
            cupy.asarray(sub_edges),
            cupy.asarray(sub_inv),
            out=sub_out,
            exact_widths=exact_widths,
            bins=sub_bins,
        )
        out[cupy.asarray(bins)] += sub_out.astype(out.dtype)
    return out


def _run_pairs_reduction(E_r, aw, w, edges, inv_width, out, exact_widths, bins):
    """All-pairs near/far launch: grid ``(bins, line chunks)``, see :func:`_line_chunks`.

    Validation: sinc-bin-near-far
    """
    import cupy

    n_bins = int(inv_width.shape[0]) if bins is None else int(len(bins))
    n_cxr_rows = int(E_r.size)
    threads = int(_REDUCE_THREADS)
    chunks, chunk_lines = _line_chunks(n_bins, n_cxr_rows)
    per_chunk = cupy.empty((n_bins, chunks), dtype=cupy.float64)
    _bin_reduce_kernel()(
        (n_bins, chunks),
        (threads,),
        (
            cupy.ascontiguousarray(E_r, dtype=cupy.float32),
            cupy.ascontiguousarray(aw, dtype=cupy.float32),
            cupy.ascontiguousarray(w, dtype=cupy.float32),
            cupy.ascontiguousarray(edges, dtype=cupy.float64),
            cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
            per_chunk,
            np.int64(n_cxr_rows),
            np.float32(np.inf if exact_widths is None else float(exact_widths)),
            np.int64(chunk_lines),
            cupy.arange(n_bins, dtype=cupy.int64)
            if bins is None
            else cupy.asarray(bins, dtype=cupy.int64),
        ),
        # partial sums, a two-tile line queue, warp offsets, the queue count
        shared_mem=threads * 8 + 2 * threads * 8 + 33 * 4,
    )
    per_bin = per_chunk[:, 0] if chunks == 1 else per_chunk.sum(axis=1)
    out += per_bin.astype(out.dtype)
    return out


def _bin_reduction_markers(resolved):
    """Numerical generation for bin-mean cases, including policy-only identities."""
    policy = resolved.get("line_grid_policy") or {}
    quadrature = resolved.get("line_quadrature", policy.get("quadrature", "node"))
    return {"line_reduction_model": CXR_BIN_REDUCTION_MODEL} if quadrature == "bin-mean" else {}


def _check_bins(bins, n_axis: int, out, method: str) -> None:
    """Reject a ``bins`` subset the kernel would read out of bounds or misplace."""
    if method in ("segmented", "fourier"):
        raise ValueError("bins applies to the near/far methods only")
    host_bins = np.asarray(_to_cpu(bins))
    if host_bins.ndim != 1 or host_bins.dtype.kind not in "iu":
        raise ValueError("bins must be a 1-D integer index array")
    if host_bins.size and (host_bins.min() < 0 or host_bins.max() >= n_axis):
        raise ValueError("bins must index inv_width")
    if host_bins.size != int(np.prod(out.shape)):
        raise ValueError("out must hold one value per listed bin")


def run_bin_mean_reduction_kernel(
    E_r,
    aw,
    w,
    edges,
    inv_width,
    *,
    out,
    exact_widths: float | None = BIN_MEAN_EXACT_WIDTHS,
    method: str = "auto",
    fourier_axis=None,
    bins=None,
):
    """Fused CUDA bin-mean line reduction: ``out[k] += sum_l w_l * mean_k(line l)``.

    The bin-mean counterpart of ``cxr_jit_kernel.run_reduction_kernel``: no
    ``(line, bin)`` matrix, one launch per flush with one block per bin (and
    per line chunk when bins are few, :func:`_line_chunks`), FP64
    accumulation of near pairs, compensated float32 far pairs, and one cast to
    the spectrum dtype. Bins beyond
    ``exact_widths`` first-zero widths take the far-field envelope.
    ``method="fourier"`` instead evaluates every bin without the envelope
    through the device Fourier route on a uniform axis (``fourier_axis`` from
    :func:`fourier_bin_axis`, computed when omitted); ``exact_widths`` does not
    apply to it. ``method="segmented"`` takes ``fourier_axis`` as the
    :func:`uniform_runs` list of a non-uniform axis: each uniform run takes the
    device Fourier route on its sub-axis and the other bins ``"auto"`` near/far
    with ``exact_widths`` (#1390); a bin mean depends only on its own edges.
    Callers pick the method with :func:`resolve_bin_reduction`. Near/far
    methods take optional ``bins``, indices into ``inv_width``: only those bins
    are evaluated and ``out[i]`` accumulates bin ``bins[i]`` (#1403).

    Validation: sinc-bin-integration
    Validation: sinc-bin-far-envelope
    Validation: sinc-bin-near-far
    Validation: cxr-line-fourier-reduction
    """
    import cupy

    n_bins = int(inv_width.shape[0]) if bins is None else int(len(bins))
    n_cxr_rows = int(E_r.size)
    if n_bins == 0 or n_cxr_rows == 0:
        return out
    if method not in _REDUCTION_METHODS:
        raise ValueError(f"method must be one of {_REDUCTION_METHODS}")
    if bins is not None:
        _check_bins(bins, inv_width.shape[0], out, method)
    if method == "segmented":
        return _run_segmented_reduction(
            E_r, aw, w, edges, inv_width, out=out, exact_widths=exact_widths, runs=fourier_axis
        )
    if method == "fourier":
        from ._fourier_cuda import cuda_fourier_bin_mean_reduce

        axis = fourier_axis if fourier_axis is not None else fourier_bin_axis(edges)
        if axis is None:
            raise ValueError("Fourier reduction needs a uniform c_k +- h/2 bin axis")
        out += cuda_fourier_bin_mean_reduce(E_r, aw, w, axis).astype(out.dtype)
        return out
    # Keep cheap small flushes on all pairs. From 1e9 pairs the fused line-major
    # route beat the tree and all pairs on every measured flush (#1003, job 4059).
    # An infinite exact window has no far nodes, so it stays on all pairs.
    if method == "auto" and exact_widths is not None and n_bins * n_cxr_rows >= 1_000_000_000:
        method = "sparse"
    if method in ("sparse", "tree") and exact_widths is not None:
        if method == "sparse":
            from ._bin_sparse import run_sparse_reduction as run
        else:
            from ._bin_tree import run_tree_reduction as run
        if bins is None:
            return run(E_r, aw, w, edges, inv_width, out=out, exact_widths=exact_widths)
        full = run(
            E_r,
            aw,
            w,
            edges,
            inv_width,
            out=cupy.zeros(inv_width.shape[0], dtype=cupy.float64),
            exact_widths=exact_widths,
        )
        out += full[cupy.asarray(bins)].astype(out.dtype)
        return out
    return _run_pairs_reduction(E_r, aw, w, edges, inv_width, out, exact_widths, bins)
