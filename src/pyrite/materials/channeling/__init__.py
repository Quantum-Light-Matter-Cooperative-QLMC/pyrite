"""Crystal channeling properties for planar and axial electron channeling.

Beam-independent pieces of the channeling zone (ADR-0018, amended by #1018):
temperature-dependent thermal displacement, the planar continuum potential,
and :mod:`.axial`'s experimental 2D string potential. Planar pieces use the
plane-local frame: ``x`` along the plane normal. Bloch states at a beam energy,
dipole elements, entry populations, Doppler line energies and the sample-tilt
entry geometry live in :mod:`pyrite.montecarlo.channeling` (``bloch``, ``entry``).
Physics: docs/research/physics/channeling-radiation-physics.md.
"""

from .potential import PlanarProfile, planar_profile
from .thermal import (
    ChannelingDataError,
    ThermalSite,
    UniaxialThermalSite,
    debye_b_factor,
    debye_integral,
    fit_debye_temperature,
    load_thermal_sites,
    thermal_b_factor,
)

__all__ = [
    "ChannelingDataError",
    "PlanarProfile",
    "ThermalSite",
    "UniaxialThermalSite",
    "debye_b_factor",
    "debye_integral",
    "fit_debye_temperature",
    "load_thermal_sites",
    "planar_profile",
    "thermal_b_factor",
]
