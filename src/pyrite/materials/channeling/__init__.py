"""Static quantum-channeling profiles for planar electron channeling.

Deterministic, scattering-free pieces of the channeling zone (ADR-0018):
temperature-dependent thermal displacement, the planar continuum potential,
transverse Bloch states, dipole matrix elements, ideal-entry populations and
Doppler line energies; :mod:`.axial` adds the experimental 2D string potential.
Planar pieces use the plane-local frame: ``x`` along
the plane normal, incidence angle to the plane, photon angle to the beam.
:mod:`.entry` maps a sample tilt and a cut crystal's lattice orientation
(``CrystalCut``) onto the plane-local incidence, with mean-inner-potential
refraction at the face. Physics: docs/research/physics/channeling-radiation-physics.md.
"""

from .entry import PlaneIncidence, mean_inner_potential_eV, plane_incidence
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
    "PlaneIncidence",
    "ThermalSite",
    "TransverseStates",
    "UniaxialThermalSite",
    "debye_b_factor",
    "debye_integral",
    "doppler_denominator",
    "entry_populations",
    "fit_debye_temperature",
    "line_energy_eV",
    "load_thermal_sites",
    "longitudinal_kinematics",
    "mean_inner_potential_eV",
    "planar_profile",
    "plane_incidence",
    "thermal_b_factor",
    "transverse_states",
]
