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
    return table_moments(plasma, nel.size, layer, energy_eV)


@njit(cache=True, inline="always")
def table_moments(plasma, n_layers, layer, energy_eV):
    """Table-mode body of :func:`transport_layer_moments` (bisection bracket).

    Inlined so the CPU core can call it without the out-of-line call, which
    cost more than the search itself (#841). A guide-table search returning
    the same bracket was 0-6% slower than this inline bisection on the #803
    scenes (~250-310 nodes per layer), so the bisection stays.
    """
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


@njit(cache=True)
def exact_moments(radiative_args, layer, energy_eV, element_rates):
    """``transport_layer_moments`` on the transport tuple ``radiative_args``.

    ``radiative_args`` is ``(keys, cutoff_eV, nel, nt, z, density, incident,
    nominal, top, chi, plasma)``. Out of line on purpose: unpacking the tuple
    or keeping quadrature beside the inline table path in one wrapper made the
    CPU core 1.2-2x slower end to end (#841, SLURM 2896/2902/2903).
    """
    return transport_layer_moments(
        *radiative_args[2:10],
        layer,
        energy_eV,
        radiative_args[1],
        element_rates,
        radiative_args[10],
    )
