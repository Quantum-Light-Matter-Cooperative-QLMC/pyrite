"""Re-export contract for the ``montecarlo`` package.

montecarlo was split from a single module into a package; the package must keep
re-exporting every public AND internal name that external code (consumers,
tests, checks/) imports as ``from pyrite.montecarlo import X``. This freezes the
set so a dropped name fails here loudly rather than at some consumer's import.
Importing the package runs the GPU/CPU backend probe and is otherwise cheap; it
does NOT call mc_spectrum (GPU), so it stays in the fast suite.
"""

import warnings

import pytest

import pyrite.montecarlo as mc
from pyrite import detectors
from pyrite._module_deprecations import PUBLIC_EXPORT_DEPRECATIONS

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
    assert mc.mc_spectrum.__module__ == "pyrite.montecarlo.spectrum"
    assert mc.mc_characteristic_spectrum.__module__ == "pyrite.montecarlo.spectrum"
    assert mc.simulate_trajectories.__module__ == "pyrite.montecarlo.transport"
    assert mc.run_cases.__module__ == "pyrite.montecarlo.runner"


def test_relocated_detector_response_exports_keep_the_legacy_montecarlo_surface():
    for name in ("convolve_detector", "detector_efficiency"):
        entry = PUBLIC_EXPORT_DEPRECATIONS[("pyrite.montecarlo", name)]
        mc.__dict__.pop(name, None)
        with pytest.warns(DeprecationWarning, match=entry.replacement):
            assert getattr(mc, name) is getattr(detectors, name)
        assert getattr(detectors, name).__module__ == "pyrite.detectors.response"
