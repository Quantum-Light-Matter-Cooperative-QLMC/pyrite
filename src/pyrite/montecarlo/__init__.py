"""
montecarlo (package)

Monte Carlo CXR (PXR + CBS) from scattered electrons in thick crystals,
following Zhai et al., Nat. Commun. 16, 11218 (2025), SI Sections S1-S5:

  1. Electron transport (SI S2): single-scattering Monte Carlo with
     Joy-Luo continuous slowing-down. Elastic scattering (default "mott"):
     free paths from the Browning fit to the Mott total cross sections, and
     scattering angles from a screened-Rutherford form whose screening
     parameter alpha(E) is calibrated, per element, to reproduce the NIST
     SRD 64 relativistic Mott TRANSPORT cross sections
     (mott_transport_cross_sections/) -- so both the collision rate and the
     momentum-transfer rate match Mott data. Same architecture as CASINO.
     A purely analytic screened-Rutherford fallback is kept ("sr"). Runs
     built from the default Numerics instead select "elsepa": full ELSEPA
     differential cross sections from the released tables (issue #89).
  2. Radiation (SI S1): each straight trajectory segment between elastic
     collisions radiates independently (incoherent across segments, coherent
     across reciprocal vectors within a segment) with the finite-interaction-
     time factor |Q|^2 = t_L^2 sinc^2(P t_L), P = [w - v.(k+g)]/2, replacing
     the absorption-limited delta-function limit of Feranchuk Eq. (9).
     Amplitudes are the same Eqs. (13)/(14) as src/pyrite/validation/feranchuk_spence.py, with
     arbitrary segment velocity direction.
  3. Self-absorption (SI S5): Beer-Lambert along the observation direction
     from each segment midpoint (slab geometry).
  4. Detector (SI S3/S4): Gaussian convolution with
     FWHM^2 = FWHM_EDS^2 + FWHM_dtheta^2.

Conventions: lab frame with the incident beam along +z; the sample is a slab
occupying 0 <= z <= thickness (surface normal -z toward vacuum), surface
perpendicular to the beam (no tilt). The crystal construction frame is used
as-is, so for graphite the c-axis (g(00l)) lies along the beam -- the HOPG
fiber-texture geometry of the paper. Only (00l)-type reflections are coherent
in HOPG (random in-plane grain azimuths), so pass (00l) reflections only.

Nonrelativistic amplitudes (no gamma corrections): fine at <~30 keV; the
paper's gamma factors matter toward 100-300 keV.

Lengths in Angstrom, energies in eV (electron energies in keV where noted).

This module was split from a single montecarlo.py into a package; every public
and internal name remains importable as ``from pyrite.montecarlo import X`` for
backward compatibility. The submodules are:
  pyrite._backend -- GPU/CPU array backend (xp, cp, REAL, _to_cpu, _GPU)
  pyrite.materials.attenuation -- composition normalization + X-ray self-absorption
  transport -- electron transport, scattering, stopping power
  geometry  -- tilted-sample / detector / orientation rotations
  spectrum  -- CXR line spectrum, solid-angle integral, bremsstrahlung
  detector  -- detector efficiency, line widths, convolution
  runner    -- per-case driver and the pipelined run_cases sweep
"""

from .._backend import _GPU, REAL, _to_cpu, cp, xp
from ..materials.attenuation import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _normalize_composition,
    _stack_tau,
)
from .case import Case
from .detector import aperture_fwhm_eV, eds_fwhm_eV, mosaic_fwhm_eV, mosaic_psi_rad
from .geometry import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    _mosaic_quadrature,
    _orientation_R,
    _small_tilt_R,
    detector_directions,
    first_prism_exit,
    tilted_geometry,
    validate_transverse_dimensions,
)
from .groove import GrooveSpec, blazed_groove_spec
from .runner import (
    _brem_for_case,
    _lines_for_case,
    _spectrum_case,
    _transport_case,
    _worker_init,
    run_case,
    run_cases,
)
from .spectrum import (
    _SEG_ARRAYS,
    BREM_ENDF_PARSERPY_VERSION,
    BREMSSTRAHLUNG_DATA_DIR,
    BREMSSTRAHLUNG_EEDL_FILENAME,
    BREMSSTRAHLUNG_EEDL_SHA256,
    BREMSSTRAHLUNG_MODEL,
    CHARACTERISTIC_DATA_DIR,
    CHARACTERISTIC_EEDL_FILENAME,
    CHARACTERISTIC_EEDL_SHA256,
    CHARACTERISTIC_MODEL,
    CHARACTERISTIC_XRAYDB_VERSION,
    R_E_CM2,
    BremsstrahlungCrossSectionTable,
    CharacteristicCrossSectionTable,
    _brem_dsigma_dk,
    _bremsstrahlung_dsigma_dk,
    _eedl_brem_dsigma_dk,
    _escape_length,
    _observation_direction,
    _polarization_pair,
    _segments_in_layer,
    load_bremsstrahlung_cross_sections,
    load_characteristic_cross_sections,
    load_external_brem,
    mc_brem_spectrum,
    mc_characteristic_spectrum,
    mc_spectrum,
    mc_spectrum_solid_angle,
)
from .transport import (
    A0_SQ_CM2,
    TRANSPORT_ELEMENTS,
    MottTableUnavailableError,
    _alpha_from_first_moment,
    _alpha_sr_joy,
    _dEds_bs_compound,
    _dEds_bs_compound_scalar,
    _dEds_bs_keV_per_ang,
    _dEds_bs_packed_scalar,
    _dEds_compound,
    _dEds_keV_per_ang,
    _load_mott_transport,
    _mott_alpha_table,
    _rotate_directions,
    _sample_cos_theta,
    _sigma_browning_cm2,
    beta_from_keV,
    mott_tables_dir,
    simulate_trajectories,
)

__all__ = [
    # backend
    "xp",
    "cp",
    "REAL",
    "_to_cpu",
    "_GPU",
    # case
    "Case",
    # materials
    "_normalize_composition",
    "_mu_total_inv_ang",
    "_layer_dz",
    "_layer_path_length",
    "_stack_tau",
    # transport
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
    # geometry
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
    # groove
    "GrooveSpec",
    "blazed_groove_spec",
    # spectrum
    "_SEG_ARRAYS",
    "_segments_in_layer",
    "_polarization_pair",
    "_observation_direction",
    "_escape_length",
    "CHARACTERISTIC_DATA_DIR",
    "CHARACTERISTIC_EEDL_FILENAME",
    "CHARACTERISTIC_EEDL_SHA256",
    "CHARACTERISTIC_MODEL",
    "CHARACTERISTIC_XRAYDB_VERSION",
    "BREM_ENDF_PARSERPY_VERSION",
    "BREMSSTRAHLUNG_DATA_DIR",
    "BREMSSTRAHLUNG_EEDL_FILENAME",
    "BREMSSTRAHLUNG_EEDL_SHA256",
    "BREMSSTRAHLUNG_MODEL",
    "BremsstrahlungCrossSectionTable",
    "load_bremsstrahlung_cross_sections",
    "CharacteristicCrossSectionTable",
    "load_characteristic_cross_sections",
    "mc_characteristic_spectrum",
    "mc_spectrum",
    "mc_spectrum_solid_angle",
    "R_E_CM2",
    "_brem_dsigma_dk",
    "_bremsstrahlung_dsigma_dk",
    "_eedl_brem_dsigma_dk",
    "mc_brem_spectrum",
    "load_external_brem",
    # detector
    "detector_efficiency",
    "eds_fwhm_eV",
    "aperture_fwhm_eV",
    "mosaic_fwhm_eV",
    "mosaic_psi_rad",
    "convolve_detector",
    # runner
    "run_case",
    "_transport_case",
    "_spectrum_case",
    "_brem_for_case",
    "_lines_for_case",
    "_worker_init",
    "run_cases",
]


_DEPRECATED_RESPONSE_EXPORTS = frozenset({"convolve_detector", "detector_efficiency"})


def __getattr__(name: str):
    """Resolve relocated detector response operators through their old API."""
    if name not in _DEPRECATED_RESPONSE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    from .._module_deprecations import warn_public_export_deprecation

    value = getattr(import_module("pyrite.detectors.response"), name)
    warn_public_export_deprecation(__name__, name)
    globals()[name] = value
    return value
