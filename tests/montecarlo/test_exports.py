"""Re-export contract for the ``montecarlo`` package.

montecarlo was split from a single module into a package; the package must keep
re-exporting every public AND internal name that external code (consumers,
tests, checks/) imports as ``from pyrite.montecarlo import X``. This freezes the
set so a dropped name fails here loudly rather than at some consumer's import.
Importing the package runs the GPU/CPU backend probe and is otherwise cheap; it
does NOT call mc_cxr_spectrum (GPU), so it stays in the fast suite.
"""

import warnings

import pyrite.montecarlo as mc

# Every name imported from pyrite.montecarlo anywhere in src/, tests/ or checks/,
# plus the backend/internal helpers re-exported for safety. Adding a name to the
# package is fine; REMOVING one (or failing to re-export it) breaks this test.
FROZEN_EXPORTS = frozenset(
    {
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
        "mc_cxr_spectrum",
        "mc_cxr_spectrum_solid_angle",
        "R_E_CM2",
        "_brem_dsigma_dk",
        "_bremsstrahlung_dsigma_dk",
        "_eedl_brem_dsigma_dk",
        "mc_brem_spectrum",
        "load_external_brem",
        # detector
        "eds_fwhm_eV",
        "aperture_fwhm_eV",
        "mosaic_fwhm_eV",
        "mosaic_psi_rad",
        # runner
        "run_case",
        "_transport_case",
        "_spectrum_case",
        "_brem_for_case",
        "_cxr_for_case",
        "_worker_init",
        "run_cases",
    }
)


FROZEN_EXPORTS = FROZEN_EXPORTS | frozenset(
    {
        "_dEds_bs_compound_scalar",
        "Y_MAX",
        "beta_from_keV",
        "eds_fwhm_eV",
        "_worker_init",
        "BREM_ENDF_PARSERPY_VERSION",
        "_observation_direction",
        "_load_mott_transport",
        "_cxr_for_case",
        "_mott_alpha_table",
        "_dEds_compound",
        "run_cases",
        "mc_cxr_spectrum",
        "BREMSSTRAHLUNG_MODEL",
        "first_prism_exit",
        "_brem_dsigma_dk",
        "tilted_geometry",
        "_polarization_pair",
        "_mu_total_inv_ang",
        "_spectrum_case",
        "_escape_length",
        "BREMSSTRAHLUNG_EEDL_SHA256",
        "CHARACTERISTIC_MODEL",
        "aperture_fwhm_eV",
        "MottTableUnavailableError",
        "_alpha_sr_joy",
        "CHARACTERISTIC_EEDL_SHA256",
        "CHARACTERISTIC_EEDL_FILENAME",
        "REAL",
        "Y_MIN",
        "xp",
        "mosaic_psi_rad",
        "_dEds_bs_compound",
        "Z_MAX",
        "_brem_for_case",
        "_layer_dz",
        "Z_MIN",
        "load_bremsstrahlung_cross_sections",
        "detector_directions",
        "_sample_cos_theta",
        "_eedl_brem_dsigma_dk",
        "_mosaic_quadrature",
        "run_case",
        "_to_cpu",
        "_dEds_bs_packed_scalar",
        "R_E_CM2",
        "load_characteristic_cross_sections",
        "_bremsstrahlung_dsigma_dk",
        "mc_cxr_spectrum_solid_angle",
        "_stack_tau",
        "_dEds_bs_keV_per_ang",
        "simulate_trajectories",
        "_alpha_from_first_moment",
        "_transport_case",
        "CHARACTERISTIC_XRAYDB_VERSION",
        "X_MAX",
        "validate_transverse_dimensions",
        "_layer_path_length",
        "mott_tables_dir",
        "Case",
        "mosaic_fwhm_eV",
        "mc_brem_spectrum",
        "BremsstrahlungCrossSectionTable",
        "CharacteristicCrossSectionTable",
        "cp",
        "_small_tilt_R",
        "_orientation_R",
        "blazed_groove_spec",
        "_segments_in_layer",
        "X_MIN",
        "_rotate_directions",
        "TRANSPORT_ELEMENTS",
        "_GPU",
        "mc_characteristic_spectrum",
        "A0_SQ_CM2",
        "load_external_brem",
        "_dEds_keV_per_ang",
        "_normalize_composition",
        "BREMSSTRAHLUNG_EEDL_FILENAME",
        "GrooveSpec",
        "_SEG_ARRAYS",
        "_sigma_browning_cm2",
    }
)


def test_every_frozen_name_is_importable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        missing = sorted(n for n in FROZEN_EXPORTS if not hasattr(mc, n))
    assert not missing, f"montecarlo package no longer exports: {missing}"


def test_all_matches_frozen_set():
    assert set(mc.__all__) == FROZEN_EXPORTS


def test_public_names_resolve_to_subpackage():
    # the re-exported callables must come from the new submodules, not a leftover
    # top-level montecarlo.py
    assert mc.mc_cxr_spectrum.__module__ == "pyrite.montecarlo.spectrum"
    assert mc.mc_characteristic_spectrum.__module__ == "pyrite.montecarlo.spectrum"
    assert mc.simulate_trajectories.__module__ == "pyrite.montecarlo.transport"
    assert mc.run_cases.__module__ == "pyrite.montecarlo.runner"
