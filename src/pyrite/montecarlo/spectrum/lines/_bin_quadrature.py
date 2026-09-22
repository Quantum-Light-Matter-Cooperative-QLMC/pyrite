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

Validation: sinc-bin-integration
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


def _host_bin_mean(a_width, E_res, edges, inv_width):
    """``[line, bin]`` bin means on host, in FP64."""
    a = np.asarray(a_width, dtype=np.float64)[:, None]
    x = a * (np.asarray(edges, dtype=np.float64)[None, :] - np.asarray(E_res, np.float64)[:, None])
    x = x / np.pi
    tail = sincsq_tail(x)
    mass = tail[:, 1:] - tail[:, :-1]
    mass += (x[:, :-1] < 0.0) & (x[:, 1:] >= 0.0)
    return mass * (np.pi / a) * np.asarray(inv_width, dtype=np.float64)[None, :]


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
            "raw T aw, raw T e_r, raw float64 edges, raw float64 inv_width, int64 n_e",
            "T out",
            r"""
            const long long row = (long long)(i / n_e);
            const long long col = (long long)(i % n_e);
            out = (T)pyrite_sincsq_bin_mean(
                (double)aw[row], (double)e_r[row], edges[col], edges[col + 1], inv_width[col]
            );
            """,
            "pyrite_sincsq_bin_mean_matrix",
            preamble=SINCSQ_BIN_PREAMBLE,
        )
    return _BIN_MEAN_KERNEL


def _bin_reduce_kernel():
    """Fused ``(bin, line block)`` reduction. Built on first use; CuPy only.

    One CUDA thread per (bin, block of lines) pair, so both axes run in
    parallel; the per-block partial sums are then added along the block axis.
    """
    global _BIN_REDUCE_KERNEL
    if _BIN_REDUCE_KERNEL is None:
        import cupy

        _BIN_REDUCE_KERNEL = cupy.ElementwiseKernel(
            "raw float32 e_r, raw float32 aw, raw float32 w, raw float64 edges, "
            "raw float64 inv_width, int64 n_lines, int64 n_blocks, int64 block",
            "float64 acc",
            r"""
            const long long bin = (long long)(i / n_blocks);
            const long long first = (long long)(i % n_blocks) * block;
            const long long stop = (first + block < n_lines) ? first + block : n_lines;
            const double lo = edges[bin];
            const double hi = edges[bin + 1];
            const double iw = inv_width[bin];
            double total = 0.0;
            for (long long line = first; line < stop; ++line) {
                const double wt = (double)w[line];
                if (wt == 0.0) {
                    continue;
                }
                total += wt * pyrite_sincsq_bin_mean(
                    (double)aw[line], (double)e_r[line], lo, hi, iw
                );
            }
            acc = total;
            """,
            "pyrite_sincsq_bin_mean_reduce",
            preamble=SINCSQ_BIN_PREAMBLE,
        )
    return _BIN_REDUCE_KERNEL


#: Fused reduction scratch bound, in (bin, line block) float64 partial sums.
_REDUCE_SCRATCH_ELEMENTS = 1 << 24
#: Smallest line block the fused reduction splits a flush into.
_REDUCE_MIN_BLOCK = 64


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


def sincsq_bin_lineshape(a_width_j, E_res_j, edges, inv_width):
    """Bin-mean counterpart of ``_sincsq_lineshape``: ``(n_line, n_bin)`` in ``REAL``.

    ``a_width_j`` and ``E_res_j`` are 1-D device arrays of lines; ``edges`` and
    ``inv_width`` come from :func:`bin_axis`.

    Validation: sinc-bin-integration
    """
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
        host[r0:r1] = _host_bin_mean(a_host[r0:r1], e_host[r0:r1], edges, inv_width)
    if getattr(xp, "__name__", "") == "numpy":
        return host
    return xp.asarray(host, dtype=REAL)


def run_bin_mean_reduction_kernel(E_r, aw, w, edges, inv_width, *, out):
    """Fused CUDA bin-mean line reduction: ``out[k] += sum_l w_l * mean_k(line l)``.

    The bin-mean counterpart of ``line_jit_kernel.run_reduction_kernel``: no
    ``(line, bin)`` matrix, one launch per flush over (bin, line block) pairs
    with at most ``_REDUCE_SCRATCH_ELEMENTS`` FP64 partial sums, FP64
    accumulation, and one cast to the float32 spectrum.

    Validation: sinc-bin-integration
    """
    import cupy

    n_bins = int(inv_width.shape[0])
    n_lines = int(E_r.size)
    if n_bins == 0 or n_lines == 0:
        return out
    n_blocks = max(1, min(-(-n_lines // _REDUCE_MIN_BLOCK), _REDUCE_SCRATCH_ELEMENTS // n_bins))
    block = -(-n_lines // n_blocks)
    n_blocks = -(-n_lines // block)
    partial = cupy.empty((n_bins, n_blocks), dtype=cupy.float64)
    _bin_reduce_kernel()(
        cupy.ascontiguousarray(E_r, dtype=cupy.float32),
        cupy.ascontiguousarray(aw, dtype=cupy.float32),
        cupy.ascontiguousarray(w, dtype=cupy.float32),
        cupy.ascontiguousarray(edges, dtype=cupy.float64),
        cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
        np.int64(n_lines),
        np.int64(n_blocks),
        np.int64(block),
        partial,
    )
    out += partial.sum(axis=1).astype(out.dtype)
    return out
