"""Fused host kernel for the measured line-axis ``stop`` search (#1270).

:func:`.cxr_seeds.resonance_population_stop_eV` sums, block by block, every
line's ``mass * sincsq_upper_tail_bound(width, stop - energy)`` over the lines
with a finite positive mass. NumPy forms that product through about ten full
temporaries per block and trial stop; :func:`tail_products` writes the same
products in one pass. Each product uses the operations, operand dtypes and
order of the NumPy expression, so the values are bit-identical, and the caller
keeps NumPy's pairwise ``sum`` over the compacted products, so block sums are
unchanged too.
"""

import numpy as np
from numba import njit

_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))


@njit(cache=True)
def _tail_products_kernel(energy, weight, width, stop, out):
    pi_squared = np.pi**2
    count = 0
    for index in range(energy.size):
        mass = weight[index] * width[index]
        if not (np.isfinite(mass) and mass > 0.0):
            continue
        distance = np.float64(stop - energy[index])
        bound = 1.0
        if distance > 0.0:
            bound = np.float64(width[index]) / (pi_squared * distance)
            if bound > 1.0:
                bound = 1.0
        out[count] = np.float64(mass) * bound
        count += 1
    return count


def _tail_products_numpy(energy, weight, width, stop, out):
    from .cxr_seeds import sincsq_upper_tail_bound

    mass = weight * width
    keep = np.isfinite(mass) & (mass > 0.0)
    products = mass[keep] * sincsq_upper_tail_bound(width[keep], stop - energy[keep])
    out[: products.size] = products
    return int(products.size)


def tail_products(energy, weight, width, stop, out):
    """Compacted ``mass * tail bound`` of one block's kept lines at ``stop``.

    Equals ``mass[keep] * sincsq_upper_tail_bound(width[keep], stop -
    energy[keep])`` with ``mass = weight * width`` and ``keep`` its finite
    positive entries: ``mass`` and ``stop - energy`` in the arrays' own dtypes
    (``stop`` cast as NumPy casts a Python float, NEP 50), the bound and
    product in float64. Fills the float64 ``out[:count]`` in block order and
    returns ``count``. Blocks whose arrays are not float32 or float64 take the
    NumPy expression itself.

    Validation: line-grid-resonance-bandwidth
    """
    arrays = (energy, weight, width)
    if any(array.dtype not in _FLOATS for array in arrays):
        return _tail_products_numpy(energy, weight, width, stop, out)
    return int(_tail_products_kernel(energy, weight, width, energy.dtype.type(stop), out))
