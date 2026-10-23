"""Tsai radiation length of elements and number-density compositions.

Used only as a thickness scale for the thick-target screen
(``campaign.thick_target``); no transport or scoring kernel reads it.

Validation: radiation-length-tsai
"""

import math
from collections.abc import Iterable

from .atomic import Z_TABLE
from .crystal import ALPHA_FS, R_E_ANG

# Tsai (1974) Table B.2 radiation logarithms for Z = 1-4, where the
# Thomas-Fermi forms below are inaccurate; PDG RPP Table 34.2.
_L_RAD_LIGHT = {1: 5.31, 2: 4.79, 3: 4.74, 4: 4.71}
_L_RAD_PRIME_LIGHT = {1: 6.144, 2: 5.621, 3: 5.805, 4: 5.924}


def _coulomb_correction(z: int) -> float:
    """Davies--Bethe--Maximon Coulomb correction ``f(Z)`` (PDG Eq. 34.27)."""
    a2 = (ALPHA_FS * z) ** 2
    return a2 * (1.0 / (1.0 + a2) + 0.20206 - 0.0369 * a2 + 0.0083 * a2**2 - 0.002 * a2**3)


def radiation_cross_section_ang2(element: str) -> float:
    """Per-atom ``1/(n X0)`` [Angstrom^2] from Tsai's formula (PDG Eq. 34.26).

    ``1/X0 = n 4 alpha r_e^2 {Z^2 [L_rad - f(Z)] + Z L'_rad}`` for a medium of
    ``n`` atoms per volume, with ``L_rad = ln(184.15 Z^-1/3)`` and
    ``L'_rad = ln(1194 Z^-2/3)`` for ``Z > 4`` and Tsai's tabulated values
    below. Complete screening, high-energy limit; the ``Z L'_rad`` term is the
    atomic-electron contribution.

    Validation: radiation-length-tsai
    """
    z = Z_TABLE[element]
    l_rad = _L_RAD_LIGHT.get(z, math.log(184.15 * z ** (-1.0 / 3.0)))
    l_rad_prime = _L_RAD_PRIME_LIGHT.get(z, math.log(1194.0 * z ** (-2.0 / 3.0)))
    return (
        4.0 * ALPHA_FS * R_E_ANG**2 * (z * z * (l_rad - _coulomb_correction(z)) + z * l_rad_prime)
    )


def radiation_length_ang(composition: Iterable[tuple[str, float]]) -> float:
    """Radiation length [Angstrom] of ``(element, atoms per Angstrom^3)`` pairs.

    Inverse lengths add by number density (Bragg additivity of the per-atom
    radiative cross section), the number-density form of PDG Eq. 34.28.

    Validation: radiation-length-tsai
    """
    inverse = sum(float(n) * radiation_cross_section_ang2(el) for el, n in composition)
    if not inverse > 0.0:
        raise ValueError("radiation length needs a composition with positive number density")
    return 1.0 / inverse
