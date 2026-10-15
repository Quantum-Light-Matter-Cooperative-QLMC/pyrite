"""Static quantum-channeling profiles for planar electron channeling.

Deterministic, scattering-free pieces of the channeling zone (ADR-0018):
temperature-dependent thermal displacement, the planar continuum potential,
transverse Bloch states, dipole matrix elements, ideal-entry populations and
Doppler line energies. Everything is in the plane-local frame: ``x`` along
the plane normal, incidence angle to the plane, photon angle to the beam.
Mapping these angles from the laboratory/sample orientation belongs to the
orientation layer (issue #25). Physics: docs/research/physics/channeling-radiation-physics.md.
"""

from .potential import PlanarProfile, planar_profile
from .thermal import (
    ChannelingDataError,
    ThermalSite,
    debye_b_factor,
    debye_integral,
    fit_debye_temperature,
    load_thermal_sites,
    thermal_b_factor,
)
from .transverse import (
    TransverseStates,
    doppler_denominator,
    entry_populations,
    line_energy_eV,
    longitudinal_kinematics,
    transverse_states,
)

__all__ = [
    "ChannelingDataError",
    "PlanarProfile",
    "ThermalSite",
    "TransverseStates",
    "debye_b_factor",
    "debye_integral",
    "doppler_denominator",
    "entry_populations",
    "fit_debye_temperature",
    "line_energy_eV",
    "load_thermal_sites",
    "longitudinal_kinematics",
    "planar_profile",
    "thermal_b_factor",
    "transverse_states",
]
