"""Prepared moment lookup for frequent CPU transport totals; exact event totals stay separate."""

import numpy as np
from numba import njit

from ._jit_radiative import radiative_layer_moments_scalar


@njit(cache=True)
def transport_layer_moments(
    nel,
    nt,
    z,
    density,
    incident,
    nominal,
    top,
    chi,
    layer,
    energy_eV,
    cutoff_eV,
    element_rates,
    plasma,
):
    """Return soft stopping [eV/Å] and hard rate [Å^-1], with strict bounds.

    ``nel`` and the other quadrature arguments preserve the exact call shape;
    they are not read by the lookup. The row-end exact call still writes
    ``element_rates`` before element selection. All values are float64.
    """
    if plasma.size == nel.size:
        return radiative_layer_moments_scalar(
            nel,
            nt,
            z,
            density,
            incident,
            nominal,
            top,
            chi,
            layer,
            energy_eV,
            cutoff_eV,
            element_rates,
            plasma,
        )
    n_layers = nel.size
    offset = n_layers
    start = np.int64(plasma[offset + layer])
    count = np.int64(plasma[offset + n_layers + layer])
    if count < 2:
        raise ValueError("radiative moment table has fewer than two nodes")
    total = (plasma.size - 3 * n_layers) // 5
    energy_base = offset + 2 * n_layers
    log_base = energy_base + total
    value_base = log_base + total
    lo = start
    hi = start + count - 1
    if not np.isfinite(energy_eV) or not (
        plasma[energy_base + lo] <= energy_eV <= plasma[energy_base + hi]
    ):
        raise ValueError("radiative moment query outside prepared energy coverage")
    if energy_eV == plasma[energy_base + lo]:
        return plasma[value_base + 3 * lo], plasma[value_base + 3 * lo + 1]
    if energy_eV == plasma[energy_base + hi]:
        return plasma[value_base + 3 * hi], plasma[value_base + 3 * hi + 1]
    x = np.log(energy_eV)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if plasma[log_base + mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - plasma[log_base + lo]) / (plasma[log_base + hi] - plasma[log_base + lo])
    f = min(max(f, 0.0), 1.0)
    soft = (1.0 - f) * plasma[value_base + 3 * lo] + f * plasma[value_base + 3 * hi]
    hard = (1.0 - f) * plasma[value_base + 3 * lo + 1] + f * plasma[value_base + 3 * hi + 1]
    return soft, hard


# ---- Exact indexed table search (#841) ---------------------------------------
# The bisection above ends on the unique ``lo`` in ``[start, last - 1]`` with
#   (lo == start or L[lo] <= x)  and  (lo + 1 == last or L[lo + 1] > x),
# where ``L`` is the layer's stored log-energy row and ``last = start+count-1``.
# Its invariants are exactly those two clauses for ``lo`` and ``hi``, and for a
# nondecreasing row they select one index: ``start`` plus the number of interior
# nodes with ``L[i] <= x``. ``_guided_index`` reaches the same index from any
# starting guess: the downward walk establishes the first clause, the upward
# walk preserves it and establishes the second. The guide only shortens the
# walk, so its own rounding cannot change the bracket. The interpolation below
# repeats the bisection path's float64 arithmetic expression by expression.
# Rows that are not nondecreasing, or are degenerate, keep the bisection.
_GUIDE_BINS_PER_NODE = 2


@njit(cache=True)
def build_moment_guide(nel, plasma, enabled=True):
    """Uniform log-energy guide over each layer of a packed moment table.

    Returns ``(guide, guide_start, guide_bins, x0, scale, indexed)``. ``guide``
    holds absolute node indices; ``indexed`` is False for exact-quadrature
    buffers and for any row the guided search cannot reproduce bit for bit.
    The packed buffer and its ``moment_table_sha256`` identity are untouched.
    ``enabled=False`` keeps the bisection (equality-test reference cores).
    """
    n_layers = nel.size
    guide_start = np.zeros(n_layers, dtype=np.int64)
    guide_bins = np.ones(n_layers, dtype=np.int64)
    x0 = np.zeros(n_layers)
    scale = np.zeros(n_layers)
    if not enabled or plasma.size == n_layers:
        return np.zeros(1, dtype=np.int64), guide_start, guide_bins, x0, scale, False
    total = (plasma.size - 3 * n_layers) // 5
    log_base = n_layers + 2 * n_layers + total
    size = 0
    for layer in range(n_layers):
        count = np.int64(plasma[n_layers + n_layers + layer])
        guide_start[layer] = size
        guide_bins[layer] = max(np.int64(1), _GUIDE_BINS_PER_NODE * count)
        size += guide_bins[layer]
    guide = np.zeros(size, dtype=np.int64)
    for layer in range(n_layers):
        start = np.int64(plasma[n_layers + layer])
        count = np.int64(plasma[n_layers + n_layers + layer])
        if count < 2:
            return guide, guide_start, guide_bins, x0, scale, False
        last = start + count - 1
        first_x, last_x = plasma[log_base + start], plasma[log_base + last]
        if not (np.isfinite(first_x) and np.isfinite(last_x) and first_x < last_x):
            return guide, guide_start, guide_bins, x0, scale, False
        for i in range(start, last):
            if not plasma[log_base + i] <= plasma[log_base + i + 1]:
                return guide, guide_start, guide_bins, x0, scale, False
        bins = guide_bins[layer]
        x0[layer] = first_x
        scale[layer] = bins / (last_x - first_x)
        lo = start
        for b in range(bins):
            edge = first_x + b * ((last_x - first_x) / bins)
            lo = _guided_index(plasma, log_base, start, last, lo, edge)
            guide[guide_start[layer] + b] = lo
    return guide, guide_start, guide_bins, x0, scale, True


@njit(cache=True, inline="always")
def _guided_index(plasma, log_base, start, last, lo, x):
    while lo > start and plasma[log_base + lo] > x:
        lo -= 1
    while lo + 1 < last and plasma[log_base + lo + 1] <= x:
        lo += 1
    return lo


@njit(cache=True)
def table_layer_moments_indexed(plasma, n_layers, layer, energy_eV, guide):
    """Table-mode lookup bit-identical to ``transport_layer_moments``.

    ``guide`` is the tuple returned by :func:`build_moment_guide` with
    ``indexed`` True. Errors and endpoint returns match the bisection path.
    """
    guide_index, guide_start, guide_bins, x0, scale, _ = guide
    offset = n_layers
    start = np.int64(plasma[offset + layer])
    count = np.int64(plasma[offset + n_layers + layer])
    if count < 2:
        raise ValueError("radiative moment table has fewer than two nodes")
    total = (plasma.size - 3 * n_layers) // 5
    energy_base = offset + 2 * n_layers
    log_base = energy_base + total
    value_base = log_base + total
    last = start + count - 1
    if not np.isfinite(energy_eV) or not (
        plasma[energy_base + start] <= energy_eV <= plasma[energy_base + last]
    ):
        raise ValueError("radiative moment query outside prepared energy coverage")
    if energy_eV == plasma[energy_base + start]:
        return plasma[value_base + 3 * start], plasma[value_base + 3 * start + 1]
    if energy_eV == plasma[energy_base + last]:
        return plasma[value_base + 3 * last], plasma[value_base + 3 * last + 1]
    x = np.log(energy_eV)
    t = (x - x0[layer]) * scale[layer]
    bins = guide_bins[layer]
    if t >= bins - 1:
        b = bins - 1
    elif t > 0.0:
        b = np.int64(t)
    else:
        b = np.int64(0)
    lo = _guided_index(plasma, log_base, start, last, guide_index[guide_start[layer] + b], x)
    hi = lo + 1
    f = (x - plasma[log_base + lo]) / (plasma[log_base + hi] - plasma[log_base + lo])
    f = min(max(f, 0.0), 1.0)
    soft = (1.0 - f) * plasma[value_base + 3 * lo] + f * plasma[value_base + 3 * hi]
    hard = (1.0 - f) * plasma[value_base + 3 * lo + 1] + f * plasma[value_base + 3 * hi + 1]
    return soft, hard


@njit(cache=True)
def guided_layer_moments(guide, radiative_args, layer, energy_eV, element_rates):
    """Indexed table search when ``guide`` allows it, else the unchanged lookup.

    ``radiative_args`` is the transport tuple ``(keys, cutoff_eV, nel, nt, z,
    density, incident, nominal, top, chi, plasma)``.
    """
    nel, plasma = radiative_args[2], radiative_args[10]
    if guide[5]:
        return table_layer_moments_indexed(plasma, nel.size, layer, energy_eV, guide)
    return transport_layer_moments(
        nel,
        radiative_args[3],
        radiative_args[4],
        radiative_args[5],
        radiative_args[6],
        radiative_args[7],
        radiative_args[8],
        radiative_args[9],
        layer,
        energy_eV,
        radiative_args[1],
        element_rates,
        plasma,
    )
