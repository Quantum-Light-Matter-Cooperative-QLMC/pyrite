"""65-digit references for the exact electron/photon line-recoil mass shell."""

from decimal import Decimal, localcontext

import numpy as np

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV


def mass_shell_root(kinetic, g, direction, *, photon=None, incident=(0.0, 0.0, 1.0)):
    """65-digit bisection of Shi et al. (2024) Eq. (6), without squaring.

    ``direction`` is the photon momentum per unit photon energy: the unit
    detector direction in vacuum, or ``n + delta h`` under first-order Snell.
    The lattice takes ``+hbar g`` [1/Angstrom]; ``incident`` is the unit
    electron direction. With ``photon`` [eV] given, returns the signed
    mismatch ``(E_f + photon - E)/hbar c`` [1/Angstrom] instead of the root.
    """
    with localcontext() as ctx:
        ctx.prec = 65

        def dec(value):
            return Decimal(str(value))

        mass, hc, kinetic = dec(M_E_EV), dec(HBARC_EV_ANG), dec(kinetic)
        total = kinetic + mass
        momentum = (kinetic * (kinetic + 2 * mass)).sqrt()
        transfer = [hc * dec(value) for value in g]
        normal = [dec(value) for value in direction]
        along = [dec(value) for value in incident]

        def residual(photon):
            final = [momentum * along[i] - transfer[i] - photon * normal[i] for i in range(3)]
            return (mass * mass + sum(value * value for value in final)).sqrt() + photon - total

        if photon is not None:
            return float(residual(dec(photon)) / hc)
        if residual(Decimal(0)) >= 0:
            return np.nan
        lo, hi = Decimal(0), kinetic
        for _ in range(190):
            midpoint = (lo + hi) / 2
            if residual(midpoint) > 0:
                hi = midpoint
            else:
                lo = midpoint
        return float((lo + hi) / 2)
