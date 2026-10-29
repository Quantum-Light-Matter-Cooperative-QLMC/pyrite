"""Prepared moment lookup for frequent CUDA totals, with exact-mode dispatch."""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_radiative_device import _rad_layer_moment


@jit.rawkernel(device=True)
def _rad_transport_moment(
    which,
    layer,
    nel,
    nt,
    z,
    density,
    incident,
    nominal,
    top,
    chi,
    max_el,
    max_t,
    n_red,
    energy_eV,
    cutoff_eV,
    packed,
):
    if packed.size == nel.size:
        return _rad_layer_moment(
            which,
            layer,
            nel,
            nt,
            z,
            density,
            incident,
            nominal,
            top,
            chi,
            max_el,
            max_t,
            n_red,
            energy_eV,
            cutoff_eV,
            packed,
        )
    layers = nel.size
    count = np.int32(packed[layers * np.int32(2) + layer])
    start = np.int32(packed[layers + layer])
    total = (packed.size - layers * np.int32(3)) // np.int32(5)
    energy_base = layers * np.int32(3)
    log_base = energy_base + total
    value_base = log_base + total
    lo = start
    hi = start + count - np.int32(1)
    if not (energy_eV >= packed[energy_base + lo] and energy_eV <= packed[energy_base + hi]):
        # The diagnostic trajectory checker rejects any nonfinite row.
        # This refuses uncovered queries rather than silently extrapolating.
        return np.float64(np.nan)
    if energy_eV == packed[energy_base + lo]:
        return packed[value_base + lo * np.int32(3) + which]
    if energy_eV == packed[energy_base + hi]:
        return packed[value_base + hi * np.int32(3) + which]
    x = xp.log(energy_eV)
    while hi - lo > np.int32(1):
        mid = (lo + hi) // np.int32(2)
        if packed[log_base + mid] <= x:
            lo = mid
        else:
            hi = mid
    weight = (x - packed[log_base + lo]) / (packed[log_base + hi] - packed[log_base + lo])
    weight = min(max(weight, np.float64(0)), np.float64(1))
    return (np.float64(1) - weight) * packed[value_base + lo * np.int32(3) + which] + (
        weight * packed[value_base + hi * np.int32(3) + which]
    )
