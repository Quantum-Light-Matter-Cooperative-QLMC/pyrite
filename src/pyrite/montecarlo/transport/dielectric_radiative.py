"""Exact moments of a piecewise-linear BremsLib chi with dielectric suppression.

The parent scaled SDCS chi(k) is linear on each cell. Multiplication by
k**2/(k**2 + a**2), a = gamma * hbar*omega_p, is *not* linear in k.
Integrate its zeroth and first physical moments analytically rather than
resampling chi at BremsLib's coarse photon nodes. This is a low-level leaf
shared by host and Numba transport; it is not a mode/medium resolver.

Validation: dielectric-suppression
"""

import numpy as np
from numba import njit

_ELECTRON_REST_EV = 510998.95


@njit(cache=True)
def dielectric_scale_eV(kinetic_eV, plasma_energy_eV):
    """Return gamma*Ep in eV, with kinetic (not total) electron energy.

    Validation: dielectric-suppression
    """
    return (1.0 + kinetic_eV / _ELECTRON_REST_EV) * plasma_energy_eV


@njit(cache=True)
def dielectric_cell_moments(left, right, chi_left, chi_right, scale_eV):
    r"""Integrate (chi/k)*F and chi*F on [left, right] with linear chi.

    Returns (zeroth in mb, first in mb*eV) *before* multiplying Z**2 and
    converting mb to cm2. F=k**2/(k**2+scale_eV**2). In the vacuum limit
    scale_eV=0 these are precisely the pre-existing linear-cell integrals.

    Validation: dielectric-suppression
    """
    if right <= left:
        return 0.0, 0.0
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    if scale_eV == 0.0:
        # The unsuppressed zeroth moment at k=0 is infrared divergent. The
        # caller always consumes only its first moment in that first cell.
        rate = 0.0 if left == 0.0 else intercept * np.log(right / left) + slope * (right - left)
        first = 0.5 * (chi_left + chi_right) * (right - left)
        return rate, first

    u, v = left / scale_eV, right / scale_eV
    if v < 0.1:
        # Direct primitive differences k-a atan(k/a) and k²-a² log(1+k²/a²)
        # lose precision as k/a -> 0. Six Taylor terms bound the relative
        # error by <1e-12 on [0, 0.1]. This also handles left=0 exactly.
        zeroth = 0.0
        first = 0.0
        sign = 1.0
        for n in range(6):
            p = 2 * n + 2
            zeroth += sign * (
                intercept * (v**p - u**p) / p
                + slope * scale_eV * (v ** (p + 1) - u ** (p + 1)) / (p + 1)
            )
            first += (
                sign
                * scale_eV
                * (
                    intercept * (v ** (p + 1) - u ** (p + 1)) / (p + 1)
                    + slope * scale_eV * (v ** (p + 2) - u ** (p + 2)) / (p + 2)
                )
            )
            sign = -sign
        return zeroth, first

    # Dimensionless primitive differences:
    # A = integral x²/(1+x²) dx = x-atan(x)
    # B = integral x³/(1+x²) dx = (x²-log(1+x²))/2
    # C = integral x/(1+x²) dx = log(1+x²)/2.
    delta_c = 0.5 * (np.log1p(v * v) - np.log1p(u * u))
    delta_a = (v - u) - (np.arctan(v) - np.arctan(u))
    delta_b = 0.5 * (v * v - u * u) - delta_c
    rate = intercept * delta_c + slope * scale_eV * delta_a
    first = scale_eV * (intercept * delta_a + slope * scale_eV * delta_b)
    return rate, first
