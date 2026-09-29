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
  below -- power series for ``t <= 4``, the modified-Lentz continued fraction
  of ``E1(i t)`` for ``4 < t < 48``, the same asymptotic series above. Measured
  against 40-digit ``mpmath`` on 1,500 points over ``[0, 1e12]``: absolute
  error <= 1e-15 for ``t <= 4`` and <= 2.1e-15/t beyond, i.e. relative to the
  ``1/t`` envelope of ``Si(t) - pi/2``. The ``t <= 4`` bound is stated with
  margin: it is libm- and compiler-dependent, and an independent host
  ``gcc -O2 -ffp-contract=off`` build measures 7.8e-16.
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

import numpy as np

from ...._backend import REAL, _to_cpu, xp

#: Line quadratures ``mc_spectrum`` accepts. ``"node"`` is the default.
LINE_QUADRATURES = ("node", "bin-mean")
NODE_QUADRATURE = "node"
BIN_MEAN_QUADRATURE = "bin-mean"

#: ``Si`` evaluator partition: exact library/continued fraction below, the
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

#: C source shared by both CUDA/ROCm routes. ``__device__`` is CuPy's
#: host/device qualifier for kernel preambles.
SINCSQ_BIN_PREAMBLE = r"""
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
        /* E1(i t) exp(i t) by modified Lentz; Si - pi/2 is Im(h exp(-i t)). */
        double b_re = 1.0, b_im = t;
        double c_re = 1.0e300, c_im = 0.0;
        double den = b_re * b_re + b_im * b_im;
        double d_re = b_re / den, d_im = -b_im / den;
        double h_re = d_re, h_im = d_im;
        for (int k = 1; k < 200; ++k) {
            const double an = -(double)k * (double)k;
            b_re += 2.0;
            const double x_re = an * d_re + b_re;
            const double x_im = an * d_im + b_im;
            den = x_re * x_re + x_im * x_im;
            d_re = x_re / den;
            d_im = -x_im / den;
            den = c_re * c_re + c_im * c_im;
            c_re = b_re + an * c_re / den;
            c_im = b_im - an * c_im / den;
            const double e_re = c_re * d_re - c_im * d_im;
            const double e_im = c_re * d_im + c_im * d_re;
            const double n_re = h_re * e_re - h_im * e_im;
            h_im = h_re * e_im + h_im * e_re;
            h_re = n_re;
            if (fabs(e_re - 1.0) + fabs(e_im) < 1.0e-16) {
                break;
            }
        }
        return h_im * cos(t) - h_re * sin(t);
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

/* Mean of sinc^2(a (E - e_r) / pi) over [e_lo, e_hi]; inv_width = 1/(e_hi - e_lo). */
static __device__ double pyrite_sincsq_bin_mean(
    double a, double e_r, double e_lo, double e_hi, double inv_width
) {
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


def validate_line_quadrature(value: object) -> str:
    """Return ``value`` as a known line quadrature name, or raise."""
    if value not in LINE_QUADRATURES:
        raise ValueError(f"line_quadrature must be one of {list(LINE_QUADRATURES)}, got {value!r}")
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
/* One block per bin; lines stride across its threads a tile at a time.
   Far pairs add to a compensated float32 sum on the spot. Near pairs, which
   need the FP64 sine integral, are compacted into a shared queue in lane
   order (warp ballot + prefix) and evaluated together once the queue fills,
   so FP64 work runs on full warps instead of one lane of a divergent warp.
   Queue order and the queue-to-thread assignment are fixed, so the result is
   deterministic. */
extern "C" __global__ void pyrite_sincsq_bin_mean_reduce(
    const float *e_r, const float *aw, const float *w, const double *edges,
    const double *inv_width, double *out, const long long n_lines, const float far_widths
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

    const long long bin = blockIdx.x;
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
    for (long long base = 0; base < n_lines; base += threads) {
        const long long line = base + tid;
        bool is_near = false;
        if (line < n_lines) {
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
        const unsigned int mask = __ballot_sync(0xffffffffu, is_near);
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
        if (pending >= (int)threads || base + threads >= n_lines) {
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
        out[bin] = partial[0];
    }
}
"""


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


def run_bin_mean_reduction_kernel(
    E_r, aw, w, edges, inv_width, *, out, exact_widths: float | None = BIN_MEAN_EXACT_WIDTHS,
    method: str = "auto",
):
    """Fused CUDA bin-mean line reduction: ``out[k] += sum_l w_l * mean_k(line l)``.

    The bin-mean counterpart of ``line_jit_kernel.run_reduction_kernel``: no
    ``(line, bin)`` matrix, one launch per flush with one block per bin, FP64
    accumulation of near pairs, compensated float32 far pairs, and one cast to
    the spectrum dtype. Bins beyond
    ``exact_widths`` first-zero widths take the far-field envelope.

    Validation: sinc-bin-integration
    Validation: sinc-bin-far-envelope
    Validation: sinc-bin-near-far
    """
    import cupy

    n_bins = int(inv_width.shape[0])
    n_lines = int(E_r.size)
    if n_bins == 0 or n_lines == 0:
        return out
    if method not in ("auto", "pairs", "tree"):
        raise ValueError("method must be 'auto', 'pairs', or 'tree'")
    # The host tree setup is O(lines) (~0.09 s for 400k synthetic lines on
    # the development CPU). The #192 all-pairs kernel measured ~19B pairs/s
    # on the lab GPU; use a conservative crossover until matched case timing
    # establishes a better one. An infinite exact window has no far nodes.
    use_tree = method == "tree" or (method == "auto" and n_bins * n_lines >= 5_000_000_000)
    if use_tree and exact_widths is not None:
        from ._bin_tree import run_tree_reduction

        return run_tree_reduction(E_r, aw, w, edges, inv_width, out=out, exact_widths=exact_widths)
    per_bin = cupy.empty(n_bins, dtype=cupy.float64)
    threads = int(_REDUCE_THREADS)
    _bin_reduce_kernel()(
        (n_bins,),
        (threads,),
        (
            cupy.ascontiguousarray(E_r, dtype=cupy.float32),
            cupy.ascontiguousarray(aw, dtype=cupy.float32),
            cupy.ascontiguousarray(w, dtype=cupy.float32),
            cupy.ascontiguousarray(edges, dtype=cupy.float64),
            cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
            per_bin,
            np.int64(n_lines),
            np.float32(np.inf if exact_widths is None else float(exact_widths)),
        ),
        # partial sums, a two-tile line queue, warp offsets, the queue count
        shared_mem=threads * 8 + 2 * threads * 8 + 33 * 4,
    )
    out += per_bin.astype(out.dtype)
    return out
