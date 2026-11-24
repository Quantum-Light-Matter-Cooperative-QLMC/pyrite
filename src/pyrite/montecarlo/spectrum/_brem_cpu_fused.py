"""Fused NumPy-backend BremsLib continuum reduction without ``(rows, NE)`` temporaries.

The portable CPU path evaluates the direction-resolved BremsLib cells as about
ten dense ``(chunk, NE)`` NumPy passes per element (``_along_reduced_energy``
twice, the ``ln T`` blend, the physical mask, the Ter--Mikaelian factor, the
per-layer optical depths and :func:`.segment_escape.mean_transmission`) and
contracts them with a BLAS product. Those passes stream far beyond the
last-level cache and stall on DRAM (#1425). This module is the float64 CPU twin
of the CUDA reducer in :mod:`.brem_jit_kernel`
(``_bremslib_or_fallback_weighted_scalar``, ``_transmission_scalar``) for the
covered case: every composition element has a BremsLib table covering every
row energy, so no EEDL/Bethe--Heitler fallback is reached. All elements are
scored in one pass, so each ``(row, photon)`` transmission is evaluated once.

Per-cell arithmetic follows the portable expressions operation for operation:
the 18-node ``k/T`` interpolation with the held top node (the linear scan is
``searchsorted(inner, k/T, side="right")``), the ``ln T`` blend, the physical
mask ``0 < k <= T``, the cm² scaling ``4 pi * (scaled * 1e-27 * Z * Z / k)``,
the optional Ter--Mikaelian ``(k / hypot(k, gamma Ep))**2``, the layer depths
summed in layer order, and the segment-mean transmission with its
``d <= 1e-4`` series. Only reductions are reordered, in a fixed way:

* spectrum: each photon bin sums its rows sequentially in row order (the
  elements of one row in composition order), so the result does not depend on
  the thread count, on row chunking, or on photon blocking;
* per-row band integrals: each row sums its photon bins sequentially in grid
  order, per element, and adds the elements in composition order, so a row's
  value does not depend on its neighbours (#1253).

Validation: bremslib-angular-model, segment-escape-average, dielectric-suppression
"""

import numpy as np
from numba import njit, prange

#: Rows per tile: their node values stay cache-resident while every thread
#: sweeps its photon bins over them.
_ROW_TILE = 256
#: Rows staged per block; bounds the O(rows * elements * 36) node state.
FUSED_ROW_BLOCK = 1 << 16

_MB_CM2 = 1.0e-27
_ELECTRON_REST_EV = 510998.95
_FOUR_PI = 4.0 * np.pi


@njit(cache=True, inline="always")
def _lower_node(reduced, nominal):  # pragma: no cover - numba
    """``searchsorted(nominal[1:-1], reduced, side="right")`` by bisection."""
    lo = 0
    hi = nominal.shape[0] - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if nominal[mid + 1] <= reduced:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always")
def _scaled_along(reduced, lower, top, values, el, r, nominal):  # pragma: no cover - numba
    """``_along_reduced_energy`` for one element row at one ``k/T`` node bracket."""
    x0 = nominal[lower]
    x1 = top if lower == nominal.shape[0] - 2 else nominal[lower + 1]
    fraction = (reduced - x0) / (x1 - x0)
    if fraction < 0.0:
        fraction = 0.0
    elif fraction > 1.0:
        fraction = 1.0
    v0 = values[el, r, lower]
    return v0 + fraction * (values[el, r, lower + 1] - v0)


@njit(cache=True, inline="always")
def _cross_section(el, r, k, reduced, rows, table):  # pragma: no cover - numba
    """Portable ``4 pi d2sigma/(dk dOmega)`` [cm²/eV/sr] of element ``el``, row ``r``.

    Both incident-energy rows share the ``k/T`` node bracket; only the held
    top node differs between them.

    Validation: bremslib-angular-model, dielectric-suppression
    """
    T_keV, fraction, top_lo, top_hi, lower_values, upper_values, _, _ = rows
    nominal, Z, plasma_eV = table
    grid = nominal[el]
    lower = _lower_node(reduced, grid)
    lo = _scaled_along(reduced, lower, top_lo[el, r], lower_values, el, r, grid)
    hi = _scaled_along(reduced, lower, top_hi[el, r], upper_values, el, r, grid)
    scaled = lo + fraction[el, r] * (hi - lo)
    dsig = _FOUR_PI * (scaled * _MB_CM2 * Z[el] * Z[el] / max(k, 1.0e-30))
    if plasma_eV > 0.0:
        scale = (1.0 + T_keV[r] * 1e3 / _ELECTRON_REST_EV) * plasma_eV
        ratio = k / np.hypot(k, scale)
        dsig = dsig * (ratio * ratio)
    return dsig


@njit(cache=True, inline="always")
def _transmission(r, e, rows, layer_mu):  # pragma: no cover - numba
    """:func:`.segment_escape.mean_transmission` of row ``r`` at bin ``e``.

    Validation: segment-escape-average
    """
    start = rows[6]
    end = rows[7]
    tau0 = start[r, 0] * layer_mu[0, e]
    tau1 = end[r, 0] * layer_mu[0, e]
    for layer in range(1, layer_mu.shape[0]):
        tau0 = tau0 + start[r, layer] * layer_mu[layer, e]
        tau1 = tau1 + end[r, layer] * layer_mu[layer, e]
    lo_tau = min(tau0, tau1)
    d = abs(tau1 - tau0)
    if d > 1.0e-4:
        ratio = -np.expm1(-d) / d
    else:
        ratio = 1.0 - 0.5 * d + d * d / 6.0
    return np.exp(-lo_tau) * ratio


@njit(cache=True, inline="always")
def _reduced(r, k, rows):  # pragma: no cover - numba
    """``k/T`` of row ``r``, or ``-1`` outside the physical ``0 < k <= T``."""
    reduced = k / max(rows[0][r] * 1.0e3, 1.0e-30)
    if k > 0.0 and reduced <= 1.0:
        return reduced
    return -1.0


@njit(parallel=True, cache=True)
def _spectrum_tile(rows, weight, first, stop, E_grid, layer_mu, table, out):  # pragma: no cover
    n_el = weight.shape[0]
    for e in prange(E_grid.shape[0]):
        k = E_grid[e]
        acc = out[e]
        for r in range(first, stop):
            reduced = _reduced(r, k, rows)
            if reduced < 0.0:
                continue
            transmission = _transmission(r, e, rows, layer_mu)
            for el in range(n_el):
                dsig = _cross_section(el, r, k, reduced, rows, table)
                acc += weight[el, r] * (dsig * transmission)
        out[e] = acc


@njit(parallel=True, cache=True)
def _band_rows(rows, weight, band_q, E_grid, layer_mu, table, out):  # pragma: no cover
    n_el = weight.shape[0]
    for r in prange(weight.shape[1]):
        acc = np.zeros(n_el)
        for e in range(E_grid.shape[0]):
            k = E_grid[e]
            reduced = _reduced(r, k, rows)
            if reduced < 0.0:
                continue
            transmission = _transmission(r, e, rows, layer_mu)
            for el in range(n_el):
                dsig = _cross_section(el, r, k, reduced, rows, table)
                acc[el] += dsig * transmission * band_q[e]
        total = out[r]
        for el in range(n_el):
            total += weight[el, r] * acc[el]
        out[r] = total


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _staged_rows(staged_tables, T_keV, cos_theta, path_start, path_end, n_layers):
    from .brem_bremslib import bremslib_segment_state

    states = [bremslib_segment_state(staged, T_keV, cos_theta) for staged in staged_tables]
    return (
        _f64(T_keV),
        _f64([state.energy_fraction for state in states]),
        _f64(
            [s.top_reduced_energy[x.lower_row] for s, x in zip(staged_tables, states, strict=True)]
        ),
        _f64(
            [
                s.top_reduced_energy[x.lower_row + 1]
                for s, x in zip(staged_tables, states, strict=True)
            ]
        ),
        _f64([state.lower_values for state in states]),
        _f64([state.upper_values for state in states]),
        _f64(np.reshape(path_start, (-1, n_layers))),
        _f64(np.reshape(path_end, (-1, n_layers))),
    )


def accumulate_bremslib_fused(
    staged_tables,
    row_weights,
    T_keV,
    cos_theta,
    path_start,
    path_end,
    layer_mu,
    E_grid,
    out,
    *,
    plasma_eV=0.0,
    band_weights=None,
):
    """Add covered BremsLib continuum cells into ``out``.

    ``staged_tables`` and ``row_weights`` hold one staged table and one
    ``(rows,)`` weight ``n * 1e24 * path_cm`` per composition element, in
    composition order. ``path_start``/``path_end`` are the ``(rows, K)``
    escape paths in angstrom and ``layer_mu`` the ``(K, NE)`` attenuation in
    1/angstrom. Without ``band_weights`` the call adds
    ``sum_el row_weight_el @ cells_el`` into the ``(NE,)`` spectrum ``out``;
    with ``(NE,)`` ``band_weights`` it adds each row's
    ``sum_el row_weight_el * sum_E q_E cell_el,E`` into the ``(rows,)`` ``out``.
    Cells are the portable ``4 pi d2sigma/(dk dOmega) * <exp(-tau)>`` in cm²
    per eV per sr. Every row must lie inside every table (the caller checks
    coverage). NumPy float64 arrays only.

    Validation: bremslib-angular-model
    """
    n_rows = int(T_keV.shape[0])
    E = _f64(E_grid)
    if n_rows == 0 or E.size == 0 or not staged_tables:
        return
    mu = _f64(np.reshape(layer_mu, (-1, E.size)))
    table = (
        _f64([staged.nominal_reduced_energy for staged in staged_tables]),
        _f64([staged.table.atomic_number for staged in staged_tables]),
        float(plasma_eV),
    )
    q = None if band_weights is None else _f64(band_weights)
    for first in range(0, n_rows, FUSED_ROW_BLOCK):
        block = slice(first, min(first + FUSED_ROW_BLOCK, n_rows))
        rows = _staged_rows(
            staged_tables, T_keV[block], cos_theta[block], path_start[block], path_end[block],
            mu.shape[0],
        )  # fmt: skip
        weight = _f64([w[block] for w in row_weights])
        if q is not None:
            partial = _f64(out[block])
            _band_rows(rows, weight, q, E, mu, table, partial)
            out[block] = partial
            continue
        for tile in range(0, weight.shape[1], _ROW_TILE):
            stop = min(tile + _ROW_TILE, weight.shape[1])
            _spectrum_tile(rows, weight, tile, stop, E, mu, table, out)


def score_covered_bremslib(
    comp, seg_E, *, enabled, model, isotropic_shares, staged, covers, rows, photons, out
):
    """Score a covered float64 CPU BremsLib call fused; return the rows left.

    ``mc_brem_spectrum`` hands over its row state: ``rows`` is
    ``(seg_L, cos_theta, path_start, path_end)``, ``photons`` is
    ``(mu_by_layer, E_grid, plasma_eV)`` and ``out`` is
    ``(spectrum, band_weights, row_band)``. When the fused reducer applies --
    ``enabled``, the BremsLib model on the float64 NumPy backend, no
    lattice-coherent shares, and a staged table covering every row for every
    composition element -- it accumulates every row into ``out`` and returns
    ``0``; otherwise it does nothing and returns ``seg_E.size``, the rows the
    portable chunked path must score. Unsupported cases are never an error.

    Validation: bremslib-angular-model
    """
    from ..._backend import REAL, xp

    applies = (
        enabled
        and model == "bremslib"
        and not isotropic_shares
        and getattr(xp, "__name__", "") == "numpy"
        and np.dtype(REAL) == np.dtype(np.float64)
        and len(staged) == len(comp)
        and all(covers.get(element, False) for element, _ in comp)
    )
    if not applies:
        return int(seg_E.size)
    seg_L, cos_theta, path_start, path_end = rows
    layer_mu, E_grid, plasma_eV = photons
    spectrum, band_weights, row_band = out
    path_cm = seg_L * 1e-8
    accumulate_bremslib_fused(
        [staged[element] for element, _ in comp],
        [n_i * 1e24 * path_cm for _, n_i in comp],
        seg_E,
        cos_theta,
        path_start,
        path_end,
        layer_mu,
        E_grid,
        spectrum if band_weights is None else row_band,
        plasma_eV=plasma_eV,
        band_weights=band_weights,
    )
    return 0
