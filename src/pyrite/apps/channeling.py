"""Small numerical diagnostics for the planar channeling demo."""

import numpy as np

from pyrite.materials.channeling import PlanarProfile, longitudinal_kinematics
from pyrite.materials.crystal import M_E_EV


def critical_angle_rad(profile: PlanarProfile, kinetic_energy_eV: float) -> float:
    """Return the existing Lindhard well-depth diagnostic in radians.

    Source: the transverse Hamiltonian of Andersen et al., Phys. Scr. 28,
    308 (1983), and Azadegan (2013), Eqs. 2–6. In the small-angle limit,
    equating ``pv theta**2 / 2`` at the well minimum to the planar well
    depth gives ``sqrt(2 (max U - min U) / pv)``. A flat potential gives
    zero. This is a display diagnostic, not an entry selection rule.

    Validation: cr-transverse-eigenproblem
    """
    minimum, maximum = profile.extrema()
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    return float(np.sqrt(2.0 * (maximum - minimum) / (gamma * M_E_EV * beta**2)))
