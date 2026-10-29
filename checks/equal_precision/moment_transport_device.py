"""Experimental lookup adapter for #788 transport qualification only.

The diagnostic appends caller-owned tables after the layer plasma array.
Physical plasma entries retain their original positions for exact samplers.
No production module imports this adapter.
"""

import cupy as cp
import numpy as np

from pyrite.montecarlo._cupy_jit import jit


@jit.rawkernel(device=True)
def lookup_layer_moment(
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
    x = cp.log(energy_eV)
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
