"""Fused NumPy-backend line reduction: ``sum_j w_j sinc^2`` without temporaries.

The CPU fallback used to build the dense ``(n_line, n_E)`` ``_sincsq_lineshape``
matrix eagerly (five full-size temporaries per chunk) and contract it with a
BLAS matvec per target. Profiling a Si 300 keV CPU run put ~85% of the line
phase in that elementwise pass. This kernel evaluates each ``(line, energy)``
profile value once, accumulates every target from it, and parallelises over
energies, so no two iterations write the same output element.

Per-element profile arithmetic matches ``_sincsq_lineshape`` (same ``x``, same
``x == 0`` guard); only the summation order over lines differs from the BLAS
contraction, a rounding-level reassociation.

Validation: line-hkl-batch
"""

import numpy as np
from numba import njit, prange

_PI = np.pi


@njit(parallel=True, cache=True)
def _sincsq_reduce(E_r, a_width, weights, E_grid, out):  # pragma: no cover - numba
    # Three scalar accumulators (unused weight rows are zero) keep the inner
    # loop free of output stores; ``sin`` dominates, the spare FMAs are free.
    n_l = E_r.shape[0]
    for e in prange(E_grid.shape[0]):
        E = E_grid[e]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for j in range(n_l):
            x = a_width[j] * (E - E_r[j]) / _PI
            if x == 0.0:
                x = 1.0e-20
            y = _PI * x
            s = np.sin(y) / y
            v = s * s
            s0 += weights[0, j] * v
            s1 += weights[1, j] * v
            s2 += weights[2, j] * v
        out[0, e] += s0
        out[1, e] += s1
        out[2, e] += s2


def numpy_backend() -> bool:
    """True when the spectrum array namespace is NumPy (CPU scoring)."""
    from ...._backend import xp

    return getattr(xp, "__name__", "") == "numpy"


def accumulate_sincsq(E_r, a_width, weights, E_grid, targets):
    """Add ``weights[t] @ sinc^2(a_width (E_grid - E_r))`` into ``targets[t]``.

    ``E_r``, ``a_width`` are 1-D line arrays; ``weights`` is a sequence of
    1-D per-line weights, one per 1-D ``targets`` spectrum on ``E_grid``.
    NumPy arrays only; callers keep the array-API path for other backends.

    Validation: line-hkl-batch
    """
    if E_r.size == 0 or E_grid.size == 0:
        return
    if not 1 <= len(targets) <= 3 or len(weights) != len(targets):
        raise ValueError("accumulate_sincsq takes one to three weight/target pairs")
    dtype = np.result_type(E_r, a_width, E_grid, *weights)
    w = np.zeros((3, E_r.size), dtype=dtype)
    for t, row in enumerate(weights):
        w[t] = row
    out = np.zeros((3, E_grid.size), dtype=dtype)
    _sincsq_reduce(
        np.ascontiguousarray(E_r, dtype=dtype),
        np.ascontiguousarray(a_width, dtype=dtype),
        w,
        np.ascontiguousarray(E_grid, dtype=dtype),
        out,
    )
    for t, tgt in enumerate(targets):
        tgt += out[t]


# ---------------------------------------------------------------------------
# Flight-grouped incoherent reduction (substepped transport).
#
# ``sum_f sum_pol |sum_{j in f} c_pol,j F_j(E) exp(i phi_j(E))|^2`` -- the same
# per-row complex field ``_per_hkl`` §7b builds with dense ``(rows, n_E)``
# temporaries and ``np.add.reduceat``, evaluated here one ``(row, energy)``
# element at a time. ``F`` is :func:`._formation.formation_factor` (closed form
# and its ``|w|^2 < FORMATION_SERIES_W2`` series, same expressions), and
# ``phi = d omega - g.r - L_esc delta omega``. Through a K-layer stack the
# escape terms are per layer, ``sum_l L_l delta_l omega`` (and likewise the
# ``half_dL`` slope); ``K = 1`` keeps the single-slab products verbatim.
# Validation: xray-layered-escape-phase. Rows arrive sorted by flight
# with ``starts`` the group offsets (length ``n_flights + 1``); each flight's
# rows are summed in input order, so only the summation of whole-flight
# intensities over flights is reassociated against the reference.
# Validation: coherent-formation-absorption
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _grouped_formation_reduce(  # pragma: no cover - numba
    E_vac, a_vac, half_dL, apb, bma, q, d, g_phase, L_esc, coefs, starts,
    E_grid, omega_grid, delta_omega_grid, series_w2, out,
):  # fmt: skip
    n_pol = coefs.shape[0]
    n_flights = starts.shape[0] - 1
    n_layers = delta_omega_grid.shape[0]
    for e in prange(E_grid.shape[0]):
        E = E_grid[e]
        om = omega_grid[e]
        # K = 1: the row scalars multiply this grid value, as before. K > 1:
        # the per-row sums below already carry delta_l omega, times 1.0.
        dom = delta_omega_grid[0, e] if n_layers == 1 else 1.0
        total = 0.0
        for f in range(n_flights):
            s_re = 0.0
            s_im = 0.0
            p_re = 0.0
            p_im = 0.0
            for j in range(starts[f], starts[f + 1]):
                if n_layers == 1:
                    h_j = half_dL[j, 0]
                    l_j = L_esc[j, 0]
                else:
                    h_j = 0.0
                    l_j = 0.0
                    for k in range(n_layers):
                        h_j += half_dL[j, k] * delta_omega_grid[k, e]
                        l_j += L_esc[j, k] * delta_omega_grid[k, e]
                v = a_vac[j] * (E - E_vac[j]) - h_j * dom
                qj = q[j]
                w2 = qj * qj + v * v
                if w2 < series_w2:
                    h = 0.5 * apb[j]
                    f_re = h * (1.0 - qj * qj / 3.0 - v * v / 6.0)
                    f_im = -h * qj * v / 3.0
                else:
                    sv = np.sin(v)
                    cv = np.cos(v)
                    inv = 1.0 / (2.0 * w2)
                    f_re = (v * apb[j] * sv - qj * bma[j] * cv) * inv
                    f_im = -(v * bma[j] * cv + qj * apb[j] * sv) * inv
                arg = d[j] * om - g_phase[j] - l_j * dom
                ca = np.cos(arg)
                sa = np.sin(arg)
                x_re = f_re * ca - f_im * sa
                x_im = f_re * sa + f_im * ca
                c = coefs[0, j]
                s_re += c.real * x_re - c.imag * x_im
                s_im += c.real * x_im + c.imag * x_re
                if n_pol > 1:
                    c = coefs[1, j]
                    p_re += c.real * x_re - c.imag * x_im
                    p_im += c.real * x_im + c.imag * x_re
            total += s_re * s_re + s_im * s_im + p_re * p_re + p_im * p_im
        out[e] += total


def accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, starts, grids, out, weight):
    """Add ``weight * sum_f sum_pol |field_{f,pol}|^2`` into ``out`` (NumPy only).

    ``lines`` is the ``(E_vac, a_vac, half_dL, apb, bma, q)`` formation tuple,
    ``d``/``g_phase``/``L_esc`` the per-row phase terms and ``coefs`` one or
    two per-row complex polarization coefficients, all already sorted so that
    flight ``f`` occupies rows ``starts[f]:starts[f + 1]``. ``grids`` is
    ``(E_grid, omega_grid, delta_omega_grid)``. A single slab passes ``(n,)``
    ``L_esc``/``half_dL`` and an ``(n_E,)`` delta-omega grid; a ``K``-layer
    stack passes ``(n, K)`` per-layer paths and a ``(K, n_E)`` grid. Memory is
    O(rows + n_E): no ``(rows, n_E)`` temporary is formed.
    Validation: coherent-formation-absorption, xray-layered-escape-phase
    """
    from ._formation import FORMATION_SERIES_W2

    if not 1 <= len(coefs) <= 2:
        raise ValueError("accumulate_grouped_formation takes one or two polarizations")
    E_grid = grids[0]
    if E_grid.size == 0 or starts.size < 2:
        return
    E_vac, a_vac, half_dL, apb, bma, q = lines
    n_rows = int(np.asarray(E_vac).shape[0])

    def per_layer(a):
        return np.ascontiguousarray(np.reshape(a, (n_rows, -1)), dtype=np.float64)

    f64 = [np.ascontiguousarray(a, dtype=np.float64) for a in (E_vac, a_vac)]
    f64.append(per_layer(half_dL))
    f64 += [np.ascontiguousarray(a, dtype=np.float64) for a in (apb, bma, q, d, g_phase)]
    f64.append(per_layer(L_esc))
    grid64 = [np.ascontiguousarray(g, dtype=np.float64) for g in grids]
    grid64[2] = np.ascontiguousarray(np.reshape(grid64[2], (-1, E_grid.size)))
    if f64[2].shape[1] != grid64[2].shape[0] or f64[-1].shape[1] != grid64[2].shape[0]:
        raise ValueError("per-layer escape paths and delta-omega rows must agree")
    acc = np.zeros(E_grid.size, dtype=np.float64)
    _grouped_formation_reduce(
        *f64,
        np.ascontiguousarray(np.stack(coefs), dtype=np.complex128),
        np.ascontiguousarray(starts, dtype=np.int64),
        *grid64,
        float(FORMATION_SERIES_W2),
        acc,
    )
    out += acc * weight
