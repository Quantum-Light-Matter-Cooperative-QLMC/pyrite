"""Deterministic scientific calculators with explicit numeric or Pint units.

Lazy same-object aliases of the established implementations. Importing this
namespace loads no numerical engines, plotting backends or material datasets.
Resolving a name loads its implementation; calculations may use bundled material
reference data. Artifact readers and simulation entry points belong to their
own domain facades. See the Scientific calculators reference for units and pairs.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.detectors.quantities import angular_dispersion_q as angular_dispersion_q
    from pyrite.detectors.quantities import bin_to_pixels_q as bin_to_pixels_q
    from pyrite.detectors.quantities import ccd_parameters as ccd_parameters
    from pyrite.detectors.quantities import ccd_q as ccd_q
    from pyrite.detectors.quantities import charge_cloud_sigma_q as charge_cloud_sigma_q
    from pyrite.detectors.quantities import coating_number_density_q as coating_number_density_q
    from pyrite.detectors.quantities import convolve_detector_q as convolve_detector_q
    from pyrite.detectors.quantities import detected_image_q as detected_image_q
    from pyrite.detectors.quantities import detector_efficiency_q as detector_efficiency_q
    from pyrite.detectors.quantities import detector_position_q as detector_position_q
    from pyrite.detectors.quantities import diffraction_angle_q as diffraction_angle_q
    from pyrite.detectors.quantities import disperse_spectrum_q as disperse_spectrum_q
    from pyrite.detectors.quantities import energy_fwhm_q as energy_fwhm_q
    from pyrite.detectors.quantities import grating_parameters as grating_parameters
    from pyrite.detectors.quantities import grating_q as grating_q
    from pyrite.detectors.quantities import grating_reflectivity_q as grating_reflectivity_q
    from pyrite.detectors.quantities import grating_throughput_q as grating_throughput_q
    from pyrite.detectors.quantities import groove_spacing_q as groove_spacing_q
    from pyrite.detectors.quantities import qe_absorption_q as qe_absorption_q
    from pyrite.detectors.quantities import resolving_power_q as resolving_power_q
    from pyrite.detectors.quantities import wavelength_q as wavelength_q
    from pyrite.detectors.response import convolve_detector as convolve_detector
    from pyrite.detectors.response import detector_efficiency as detector_efficiency
    from pyrite.materials.atomic import atomic_form_factor as atomic_form_factor
    from pyrite.materials.atomic import cromer_mann_f0 as cromer_mann_f0
    from pyrite.materials.atomic import henke_dispersion as henke_dispersion
    from pyrite.materials.attenuation import linear_attenuation_inv_mm as linear_attenuation_inv_mm
    from pyrite.materials.crystal import U_g as U_g
    from pyrite.materials.crystal import absorption_length_ang as absorption_length_ang
    from pyrite.materials.crystal import chi_g as chi_g
    from pyrite.materials.crystal import dominant_reflections as dominant_reflections
    from pyrite.materials.crystal import reciprocal_g_vector as reciprocal_g_vector
    from pyrite.materials.crystal import structure_factor as structure_factor
    from pyrite.materials.emission import photon_energies as photon_energies
    from pyrite.materials.quantities import U_g_q as U_g_q
    from pyrite.materials.quantities import absorption_length_q as absorption_length_q
    from pyrite.materials.quantities import atomic_form_factor_q as atomic_form_factor_q
    from pyrite.materials.quantities import chi_g_q as chi_g_q
    from pyrite.materials.quantities import cromer_mann_f0_q as cromer_mann_f0_q
    from pyrite.materials.quantities import crystal_constants as crystal_constants
    from pyrite.materials.quantities import dominant_reflections_q as dominant_reflections_q
    from pyrite.materials.quantities import henke_dispersion_q as henke_dispersion_q
    from pyrite.materials.quantities import linear_attenuation_q as linear_attenuation_q
    from pyrite.materials.quantities import photon_energies_q as photon_energies_q
    from pyrite.materials.quantities import reciprocal_g_vector_q as reciprocal_g_vector_q
    from pyrite.materials.quantities import structure_factor_q as structure_factor_q
    from pyrite.montecarlo.trajectory_summary import depth_loss_profile as depth_loss_profile

_EXPORTS = {
    "U_g": ("pyrite.materials.crystal", "U_g"),
    "U_g_q": ("pyrite.materials.quantities", "U_g_q"),
    "absorption_length_ang": ("pyrite.materials.crystal", "absorption_length_ang"),
    "absorption_length_q": ("pyrite.materials.quantities", "absorption_length_q"),
    "angular_dispersion_q": ("pyrite.detectors.quantities", "angular_dispersion_q"),
    "atomic_form_factor": ("pyrite.materials.atomic", "atomic_form_factor"),
    "atomic_form_factor_q": ("pyrite.materials.quantities", "atomic_form_factor_q"),
    "bin_to_pixels_q": ("pyrite.detectors.quantities", "bin_to_pixels_q"),
    "ccd_parameters": ("pyrite.detectors.quantities", "ccd_parameters"),
    "ccd_q": ("pyrite.detectors.quantities", "ccd_q"),
    "charge_cloud_sigma_q": ("pyrite.detectors.quantities", "charge_cloud_sigma_q"),
    "chi_g": ("pyrite.materials.crystal", "chi_g"),
    "chi_g_q": ("pyrite.materials.quantities", "chi_g_q"),
    "coating_number_density_q": ("pyrite.detectors.quantities", "coating_number_density_q"),
    "convolve_detector": ("pyrite.detectors.response", "convolve_detector"),
    "convolve_detector_q": ("pyrite.detectors.quantities", "convolve_detector_q"),
    "cromer_mann_f0": ("pyrite.materials.atomic", "cromer_mann_f0"),
    "cromer_mann_f0_q": ("pyrite.materials.quantities", "cromer_mann_f0_q"),
    "crystal_constants": ("pyrite.materials.quantities", "crystal_constants"),
    "depth_loss_profile": ("pyrite.montecarlo.trajectory_summary", "depth_loss_profile"),
    "detected_image_q": ("pyrite.detectors.quantities", "detected_image_q"),
    "detector_efficiency": ("pyrite.detectors.response", "detector_efficiency"),
    "detector_efficiency_q": ("pyrite.detectors.quantities", "detector_efficiency_q"),
    "detector_position_q": ("pyrite.detectors.quantities", "detector_position_q"),
    "diffraction_angle_q": ("pyrite.detectors.quantities", "diffraction_angle_q"),
    "disperse_spectrum_q": ("pyrite.detectors.quantities", "disperse_spectrum_q"),
    "dominant_reflections": ("pyrite.materials.crystal", "dominant_reflections"),
    "dominant_reflections_q": ("pyrite.materials.quantities", "dominant_reflections_q"),
    "energy_fwhm_q": ("pyrite.detectors.quantities", "energy_fwhm_q"),
    "grating_parameters": ("pyrite.detectors.quantities", "grating_parameters"),
    "grating_q": ("pyrite.detectors.quantities", "grating_q"),
    "grating_reflectivity_q": ("pyrite.detectors.quantities", "grating_reflectivity_q"),
    "grating_throughput_q": ("pyrite.detectors.quantities", "grating_throughput_q"),
    "groove_spacing_q": ("pyrite.detectors.quantities", "groove_spacing_q"),
    "henke_dispersion": ("pyrite.materials.atomic", "henke_dispersion"),
    "henke_dispersion_q": ("pyrite.materials.quantities", "henke_dispersion_q"),
    "linear_attenuation_inv_mm": ("pyrite.materials.attenuation", "linear_attenuation_inv_mm"),
    "linear_attenuation_q": ("pyrite.materials.quantities", "linear_attenuation_q"),
    "photon_energies": ("pyrite.materials.emission", "photon_energies"),
    "photon_energies_q": ("pyrite.materials.quantities", "photon_energies_q"),
    "qe_absorption_q": ("pyrite.detectors.quantities", "qe_absorption_q"),
    "reciprocal_g_vector": ("pyrite.materials.crystal", "reciprocal_g_vector"),
    "reciprocal_g_vector_q": ("pyrite.materials.quantities", "reciprocal_g_vector_q"),
    "resolving_power_q": ("pyrite.detectors.quantities", "resolving_power_q"),
    "structure_factor": ("pyrite.materials.crystal", "structure_factor"),
    "structure_factor_q": ("pyrite.materials.quantities", "structure_factor_q"),
    "wavelength_q": ("pyrite.detectors.quantities", "wavelength_q"),
}

__all__ = [
    "U_g",
    "U_g_q",
    "absorption_length_ang",
    "absorption_length_q",
    "angular_dispersion_q",
    "atomic_form_factor",
    "atomic_form_factor_q",
    "bin_to_pixels_q",
    "ccd_parameters",
    "ccd_q",
    "charge_cloud_sigma_q",
    "chi_g",
    "chi_g_q",
    "coating_number_density_q",
    "convolve_detector",
    "convolve_detector_q",
    "cromer_mann_f0",
    "cromer_mann_f0_q",
    "crystal_constants",
    "depth_loss_profile",
    "detected_image_q",
    "detector_efficiency",
    "detector_efficiency_q",
    "detector_position_q",
    "diffraction_angle_q",
    "disperse_spectrum_q",
    "dominant_reflections",
    "dominant_reflections_q",
    "energy_fwhm_q",
    "grating_parameters",
    "grating_q",
    "grating_reflectivity_q",
    "grating_throughput_q",
    "groove_spacing_q",
    "henke_dispersion",
    "henke_dispersion_q",
    "linear_attenuation_inv_mm",
    "linear_attenuation_q",
    "photon_energies",
    "photon_energies_q",
    "qe_absorption_q",
    "reciprocal_g_vector",
    "reciprocal_g_vector_q",
    "resolving_power_q",
    "structure_factor",
    "structure_factor_q",
    "wavelength_q",
]


def __getattr__(name: str) -> Any:
    """Resolve a supported name without changing its implementation or metadata."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
