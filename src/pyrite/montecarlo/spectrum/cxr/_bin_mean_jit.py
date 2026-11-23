"""Numba host near/far bin-mean line sum (#1320, #1390).

The host Fourier route (``_fourier_host``) sends lines whose positive mode
count exceeds ``FOURIER_MAX_LINE_MODES`` here, and host bin-mean lines on
axes the Fourier route cannot take (non-uniform bins) are reduced here
through ``_bin_queue.HostNearFarLineQueues``. It evaluates the same bin
mean as ``_bin_quadrature._host_bin_mean(..., exact_widths)``: the exact
``sinc^2`` bin mean within ``exact_widths`` first-zero widths of a resonance
and, beyond, the float32 far-field envelope ``1 / (2 pi^2 x_lo x_hi)`` in the
operation order of ``_far_envelope_float32`` (head/carry float32 splits). The
exact part transliterates the device preamble ``SINCSQ_BIN_PREAMBLE``
(``pyrite_si_minus_half_pi``, ``pyrite_sincsq_tail``: power series for
``t <= 4``, modified-Lentz continued fraction of ``E1(i t)`` below ``t = 48``,
auxiliary-function asymptotic series above), not ``scipy.special.sici``, so it
differs from the NumPy evaluator only at FP64 rounding (the preamble's stated
accuracy). No ``(line, bin)`` temporaries are allocated; bins run in parallel
and each bin sums its lines serially, so the result does not depend on
thread count.

Validation: sinc-bin-integration
Validation: sinc-bin-far-envelope
"""

import numpy as np
from numba import njit, prange

_PI = np.pi
_INV_PI_F32 = np.float32(0.318309886)
_TWO_PI_SQ_F32 = np.float32(19.7392088)
_ONE_F32 = np.float32(1.0)


@njit(cache=True)
def _si_minus_half_pi(t):  # pragma: no cover - numba
    """``Si(t) - pi/2`` for ``t >= 0``; ``pyrite_si_minus_half_pi`` of the preamble."""
    if t <= 4.0:
        t2 = t * t
        term = t
        total = t
        for k in range(1, 40):
            m = 2.0 * k
            term = -term * t2 / (m * (m + 1.0))
            add = term / (m + 1.0)
            total += add
            if abs(add) <= 1.0e-17 * abs(total):
                break
        return total - 0.5 * _PI
    if t < 48.0:
        b_re, b_im = 1.0, t
        c_re, c_im = 1.0e300, 0.0
        den = b_re * b_re + b_im * b_im
        d_re, d_im = b_re / den, -b_im / den
        h_re, h_im = d_re, d_im
        for k in range(1, 200):
            an = -float(k) * float(k)
            b_re += 2.0
            x_re = an * d_re + b_re
            x_im = an * d_im + b_im
            den = x_re * x_re + x_im * x_im
            d_re = x_re / den
            d_im = -x_im / den
            den = c_re * c_re + c_im * c_im
            c_re = b_re + an * c_re / den
            c_im = b_im - an * c_im / den
            e_re = c_re * d_re - c_im * d_im
            e_im = c_re * d_im + c_im * d_re
            n_re = h_re * e_re - h_im * e_im
            h_im = h_re * e_im + h_im * e_re
            h_re = n_re
            if abs(e_re - 1.0) + abs(e_im) < 1.0e-16:
                break
        return h_im * np.cos(t) - h_re * np.sin(t)
    y = 1.0 / (t * t)
    f_term, g_term, f, g = 1.0, 1.0, 0.0, 0.0
    for k in range(13):
        f += f_term
        g += g_term
        f_term *= -(2.0 * k + 1.0) * (2.0 * k + 2.0) * y
        g_term *= -(2.0 * k + 2.0) * (2.0 * k + 3.0) * y
    return -(f / t * np.cos(t) + g * y * np.sin(t))


@njit(cache=True)
def _sincsq_tail(x):  # pragma: no cover - numba
    """``R(x) = F(x) - s(x)/2``; ``pyrite_sincsq_tail`` of the preamble."""
    u = abs(x)
    t = 2.0 * _PI * u
    y1 = _PI * u
    if t >= 48.0:
        s1 = np.sin(y1)
        c1 = np.cos(y1)
        y = 1.0 / (t * t)
        d, g = 0.0, 0.0
        f_term, g_term = -2.0 * y, 1.0
        for k in range(1, 13):
            d += f_term
            f_term *= -(2.0 * k + 1.0) * (2.0 * k + 2.0) * y
            if abs(f_term) < 1.0e-18:
                break
        for k in range(13):
            g += g_term
            g_term *= -(2.0 * k + 2.0) * (2.0 * k + 3.0) * y
            if abs(g_term) < 1.0e-18:
                break
        pi2u = _PI * _PI * u
        r = -(1.0 + d) / (2.0 * pi2u) + d * s1 * s1 / pi2u - 2.0 * g * y * s1 * c1 / _PI
    else:
        s = 1.0 if u == 0.0 else np.sin(y1) / y1
        r = _si_minus_half_pi(t) / _PI - u * s * s
    return r if x >= 0.0 else -r


@njit(parallel=True, cache=True)
def _near_far_bins(
    E, a, w, e_head, e_carry, scale32, edges, head, carry, inv_width, limit, out
):  # pragma: no cover - numba
    """``out[k] += sum_j w_j mean_k(line j)``: float32 far envelope, else the exact mean.

    Validation: sinc-bin-integration, sinc-bin-far-envelope
    """
    for k in prange(out.shape[0]):
        lo, hi = edges[k], edges[k + 1]
        acc = 0.0
        for j in range(E.shape[0]):
            x_lo32 = scale32[j] * ((head[k] - e_head[j]) + (carry[k] - e_carry[j]))
            x_hi32 = scale32[j] * ((head[k + 1] - e_head[j]) + (carry[k + 1] - e_carry[j]))
            if x_lo32 >= limit or x_hi32 <= -limit:
                acc += w[j] * float(_ONE_F32 / (_TWO_PI_SQ_F32 * (x_lo32 * x_hi32)))
                continue
            x_lo = a[j] * (lo - E[j]) / _PI
            x_hi = a[j] * (hi - E[j]) / _PI
            mass = _sincsq_tail(x_hi) - _sincsq_tail(x_lo)
            if x_lo < 0.0 and x_hi >= 0.0:
                mass += 1.0
            acc += w[j] * (mass * ((_PI / a[j]) * inv_width[k]))
        out[k] += acc


def host_near_far_bin_mean_reduce(E_res, a_width, weight, axis, *, exact_widths=None):
    """``sum_l w_l mean_k(line l)`` on uniform bins ``axis = (c0, h, n_bins)``, host FP64.

    ``exact_widths`` (default ``BIN_MEAN_EXACT_WIDTHS``) sets the far-envelope
    boundary; ``numpy.inf`` keeps every bin exact. Bin ``k`` spans
    ``c0 + (k -+ 1/2) h``, the edges ``fourier_bin_axis`` accepts.

    Validation: sinc-bin-integration, sinc-bin-far-envelope
    """
    c0, h, n_bins = axis
    edges = c0 + (np.arange(int(n_bins) + 1, dtype=np.float64) - 0.5) * h
    return host_near_far_edges_reduce(E_res, a_width, weight, edges, exact_widths=exact_widths)


def host_near_far_edges_reduce(E_res, a_width, weight, edges, inv_width=None, *, exact_widths=None):
    """``sum_l w_l mean_k(line l)`` on arbitrary ascending ``edges``, host FP64 (#1390).

    The host mirror of the device near/far reduction for any bin axis,
    including the non-uniform axes the Fourier route cannot take.
    ``inv_width`` defaults to ``1 / diff(edges)``; ``exact_widths`` as in
    :func:`host_near_far_bin_mean_reduce`. Zero-weight lines are skipped.

    Validation: sinc-bin-integration, sinc-bin-far-envelope
    """
    from ._bin_quadrature import BIN_MEAN_EXACT_WIDTHS

    widths = BIN_MEAN_EXACT_WIDTHS if exact_widths is None else exact_widths
    E = np.ascontiguousarray(E_res, dtype=np.float64).ravel()
    a = np.ascontiguousarray(a_width, dtype=np.float64).ravel()
    w = np.ascontiguousarray(weight, dtype=np.float64).ravel()
    keep = w != 0.0
    if not keep.all():
        E, a, w = E[keep], a[keep], w[keep]
    edges = np.ascontiguousarray(edges, dtype=np.float64).ravel()
    if inv_width is None:
        inv_width = 1.0 / np.diff(edges)
    inv_width = np.ascontiguousarray(inv_width, dtype=np.float64).ravel()
    out = np.zeros(edges.size - 1, dtype=np.float64)
    if E.size == 0 or out.size == 0:
        return out
    f32 = np.float32
    head = edges.astype(f32)
    carry = (edges - head.astype(np.float64)).astype(f32)
    e_head = E.astype(f32)
    e_carry = (E - e_head.astype(np.float64)).astype(f32)
    scale32 = a.astype(f32) * _INV_PI_F32
    _near_far_bins(
        E, a, w, e_head, e_carry, scale32, edges, head, carry, inv_width, f32(widths), out
    )
    return out
