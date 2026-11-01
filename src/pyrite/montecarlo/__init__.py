"""
montecarlo (package)

Coupled electron/photon Monte Carlo transport and radiation scoring in layered
and crystalline targets. Transport produces trajectory segments and events;
spectrum kernels score radiation from them (CXR = PXR + CBS, bremsstrahlung,
characteristic lines, transition/diffracted-transition radiation, coherent
emission); the runner drives cases through both phases on CPU or GPU.

Physics, assumptions and validation live in ``docs/physics/`` (start at
``docs/physics/how-pyrite-works.md``) and the validation ledger, not here.
Elastic-model choices and their defaults are documented on
:class:`pyrite.Numerics` and in ``docs/physics/beam-transport/elastic-scattering.md``;
sample tilt follows ``docs/physics/geometry/tilt-convention.md``.

Conventions: lab frame with the incident beam along +z; the sample occupies
0 <= z <= thickness with its entrance surface facing -z. Units are carried by
name suffix (``_ang`` Angstrom, ``_eV``/``_keV``, ``_rad``/``_deg``).

Legacy flat names remain importable as ``from pyrite.montecarlo import X``;
they resolve lazily so schema imports do not initialize kernels. Submodules:
  case        -- typed, validated input record for one simulation case
  transport   -- electron, positron and photon transport: elastic, inelastic,
                 shell, radiative and pair interactions; CPU and CUDA kernels
  spectrum    -- radiation scoring: CXR, bremsstrahlung, characteristic,
                 transition/optical/DTR, Cherenkov, coherent emission
  channeling  -- planar/axial channeling and open-system quantum transport
  runner      -- per-case driver, scheduling, chunking and streamed scoring
  geometry    -- tilted-sample, detector-direction and orientation rotations
  detector    -- detector efficiency, line widths and convolution
  groove, bunch_front, transverse
              -- entrance-face grooves, bunch-front tilt, transverse beam
                 phase-space policies
  shell_*, eedl_ionization, eadl_relaxation, bote_salvat_ionization
              -- atomic shell data, ionization and relaxation
  trajectories, trajectory_*
              -- trajectory capture, inventory, selection, export and scenes
  gdf         -- GPT time-output/screen adapter (CPU only)

See ``docs/repo_map.md`` for ownership and entry points.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .case import Case as Case

# Resolve legacy bindings on demand; schema imports must not initialize kernels.
_EXPORT_MODULES = {
    "_GPU": ".._backend",
    "REAL": ".._backend",
    "_to_cpu": ".._backend",
    "cp": ".._backend",
    "xp": ".._backend",
    "_layer_dz": "..materials.attenuation",
    "_layer_path_length": "..materials.attenuation",
    "_mu_total_inv_ang": "..materials.attenuation",
    "_normalize_composition": "..materials.attenuation",
    "_stack_tau": "..materials.attenuation",
    "Case": ".case",
    "aperture_fwhm_eV": ".detector",
    "eds_fwhm_eV": ".detector",
    "mosaic_fwhm_eV": ".detector",
    "mosaic_psi_rad": ".detector",
    "X_MAX": ".geometry",
    "X_MIN": ".geometry",
    "Y_MAX": ".geometry",
    "Y_MIN": ".geometry",
    "Z_MAX": ".geometry",
    "Z_MIN": ".geometry",
    "_mosaic_quadrature": ".geometry",
    "_orientation_R": ".geometry",
    "_small_tilt_R": ".geometry",
    "detector_directions": ".geometry",
    "first_prism_exit": ".geometry",
    "tilted_geometry": ".geometry",
    "validate_transverse_dimensions": ".geometry",
    "GrooveSpec": ".groove",
    "blazed_groove_spec": ".groove",
    "_brem_for_case": ".runner",
    "_cxr_for_case": ".runner",
    "_spectrum_case": ".runner",
    "_transport_case": ".runner",
    "_worker_init": ".runner",
    "run_case": ".runner",
    "run_cases": ".runner",
    "_SEG_ARRAYS": ".spectrum",
    "BREM_ENDF_PARSERPY_VERSION": ".spectrum",
    "BREMSSTRAHLUNG_EEDL_FILENAME": ".spectrum",
    "BREMSSTRAHLUNG_EEDL_SHA256": ".spectrum",
    "BREMSSTRAHLUNG_MODEL": ".spectrum",
    "CHARACTERISTIC_EEDL_FILENAME": ".spectrum",
    "CHARACTERISTIC_EEDL_SHA256": ".spectrum",
    "CHARACTERISTIC_MODEL": ".spectrum",
    "CHARACTERISTIC_XRAYDB_VERSION": ".spectrum",
    "R_E_CM2": ".spectrum",
    "BremsstrahlungCrossSectionTable": ".spectrum",
    "CharacteristicCrossSectionTable": ".spectrum",
    "_brem_dsigma_dk": ".spectrum",
    "_bremsstrahlung_dsigma_dk": ".spectrum",
    "_eedl_brem_dsigma_dk": ".spectrum",
    "_escape_length": ".spectrum",
    "_observation_direction": ".spectrum",
    "_polarization_pair": ".spectrum",
    "_segments_in_layer": ".spectrum",
    "load_bremsstrahlung_cross_sections": ".spectrum",
    "load_characteristic_cross_sections": ".spectrum",
    "load_external_brem": ".spectrum",
    "mc_brem_spectrum": ".spectrum",
    "mc_characteristic_spectrum": ".spectrum",
    "mc_cxr_spectrum": ".spectrum",
    "mc_cxr_spectrum_solid_angle": ".spectrum",
    "A0_SQ_CM2": ".transport",
    "TRANSPORT_ELEMENTS": ".transport",
    "MottTableUnavailableError": ".transport",
    "_alpha_from_first_moment": ".transport",
    "_alpha_sr_joy": ".transport",
    "_dEds_bs_compound": ".transport",
    "_dEds_bs_compound_scalar": ".transport",
    "_dEds_bs_keV_per_ang": ".transport",
    "_dEds_bs_packed_scalar": ".transport",
    "_dEds_compound": ".transport",
    "_dEds_keV_per_ang": ".transport",
    "_load_mott_transport": ".transport",
    "_mott_alpha_table": ".transport",
    "_rotate_directions": ".transport",
    "_sample_cos_theta": ".transport",
    "_sigma_browning_cm2": ".transport",
    "beta_from_keV": ".transport",
    "mott_tables_dir": ".transport",
    "simulate_trajectories": ".transport",
}

__all__ = [
    "xp",
    "cp",
    "REAL",
    "_to_cpu",
    "_GPU",
    "Case",
    "_normalize_composition",
    "_mu_total_inv_ang",
    "_layer_dz",
    "_layer_path_length",
    "_stack_tau",
    "TRANSPORT_ELEMENTS",
    "MottTableUnavailableError",
    "mott_tables_dir",
    "A0_SQ_CM2",
    "beta_from_keV",
    "_sigma_browning_cm2",
    "_alpha_sr_joy",
    "_alpha_from_first_moment",
    "_load_mott_transport",
    "_mott_alpha_table",
    "_sample_cos_theta",
    "_dEds_keV_per_ang",
    "_dEds_compound",
    "_dEds_bs_keV_per_ang",
    "_dEds_bs_compound",
    "_dEds_bs_compound_scalar",
    "_dEds_bs_packed_scalar",
    "_rotate_directions",
    "simulate_trajectories",
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    "Z_MIN",
    "Z_MAX",
    "validate_transverse_dimensions",
    "first_prism_exit",
    "tilted_geometry",
    "detector_directions",
    "_orientation_R",
    "_small_tilt_R",
    "_mosaic_quadrature",
    "GrooveSpec",
    "blazed_groove_spec",
    "_SEG_ARRAYS",
    "_segments_in_layer",
    "_polarization_pair",
    "_observation_direction",
    "_escape_length",
    "CHARACTERISTIC_EEDL_FILENAME",
    "CHARACTERISTIC_EEDL_SHA256",
    "CHARACTERISTIC_MODEL",
    "CHARACTERISTIC_XRAYDB_VERSION",
    "BREM_ENDF_PARSERPY_VERSION",
    "BREMSSTRAHLUNG_EEDL_FILENAME",
    "BREMSSTRAHLUNG_EEDL_SHA256",
    "BREMSSTRAHLUNG_MODEL",
    "BremsstrahlungCrossSectionTable",
    "load_bremsstrahlung_cross_sections",
    "CharacteristicCrossSectionTable",
    "load_characteristic_cross_sections",
    "mc_characteristic_spectrum",
    "R_E_CM2",
    "_brem_dsigma_dk",
    "_bremsstrahlung_dsigma_dk",
    "_eedl_brem_dsigma_dk",
    "mc_brem_spectrum",
    "load_external_brem",
    "eds_fwhm_eV",
    "aperture_fwhm_eV",
    "mosaic_fwhm_eV",
    "mosaic_psi_rad",
    "run_case",
    "_transport_case",
    "_spectrum_case",
    "_brem_for_case",
    "_worker_init",
    "run_cases",
    "mc_cxr_spectrum",
    "mc_cxr_spectrum_solid_angle",
    "_cxr_for_case",
]


def __getattr__(name: str) -> Any:
    """Resolve the historical Monte Carlo exports only when requested."""
    if name in _EXPORT_MODULES:
        value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unloaded exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
