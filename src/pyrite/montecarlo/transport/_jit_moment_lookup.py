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
