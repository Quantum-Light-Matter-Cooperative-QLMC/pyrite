"""Bundled detector demos resolve usable geometry and separate ownership."""

import numpy as np
import pytest

from pyrite.campaign.config import catalog_detector, default_settings, material_sweep
from pyrite.campaign.observation import resolve_profile_observation
from pyrite.campaign.profiles import named_profile_stem, profile_run_settings
from pyrite.campaign.sweep import build_cases
from pyrite.detectors import IdealPhotonCounter
from pyrite.materials import CATALOG


def test_two_detector_demo_resolves_raw_spectra_at_distinct_angles():
    profile = "two_detector_demo"
    assert CATALOG.profile_materials(profile) == ("hopg",)
    assert tuple(CATALOG.profile_detector_set(profile)) == ("at_90", "at_135")
    spectra = np.array([1.0, 3.0, 2.0])
    energy = np.array([100.0, 200.0, 300.0])
    stems = []
    for detector_id, angle in (("at_90", 90.0), ("at_135", 135.0)):
        detector = catalog_detector(profile, detector_id)
        assert detector.observation_angle_deg == angle
        assert detector.polar_acceptance_deg == catalog_detector("quickstart").polar_acceptance_deg
        assert detector.solid_angle_sr == catalog_detector("quickstart").solid_angle_sr
        assert detector.response is None
        np.testing.assert_array_equal(detector.score(energy, spectra), spectra)
        assert resolve_profile_observation(CATALOG, profile, detector_id=detector_id) is None
        sweep = material_sweep("hopg", catalog_profile=profile, detector_id=detector_id)
        settings = profile_run_settings(default_settings(), profile, "full", sweep=sweep)
        assert settings.precision is None
        cases = build_cases(sweep, 4, 4)
        assert len(cases) == 1
        assert cases[0]["theta_obs_rad"] == np.deg2rad(angle)
        stem = named_profile_stem("hopg", catalog_profile=profile, detector_id=detector_id)
        assert f"detector-{detector_id}" in stem
        stems.append(stem)
    assert stems[0] != stems[1]


def test_pixelated_detector_demo_resolves_native_grid_and_ideal_expected_counts():
    profile = "pixelated_detector_demo"
    assert CATALOG.profile_materials(profile) == ("hopg",)
    assert tuple(CATALOG.profile_detector_set(profile)) == ("camera",)
    spec = CATALOG.profile_detector_set(profile)["camera"]
    assert spec["distance_mm"] == 400.0
    assert spec["polar_deg"] == 90.0
    observation = resolve_profile_observation(CATALOG, profile, detector_id="camera")
    assert observation is not None
    assert observation.detector.pixels.shape == (256, 256)
    assert observation.detector.pixels.pitch_mm == (0.055, 0.055)
    assert isinstance(observation.detector.response, IdealPhotonCounter)
    assert observation.scorer.angular_shape == (2, 2)
    assert observation.acquisition.mode == "expected"
    assert observation.acquisition.exposure_s == 1.0
    assert observation.acquisition.seed is None
    assert observation.acquisition.measured_edges_eV[0] == 0.0
    assert observation.acquisition.measured_edges_eV[-1] == 30000.0
    assert observation.filters == ()
    sweep = material_sweep("hopg", catalog_profile=profile, detector_id="camera")
    settings = profile_run_settings(default_settings(), profile, "full", sweep=sweep)
    assert settings.precision is None


@pytest.mark.parametrize("profile", ["pixelated_detector_demo", "two_detector_demo"])
def test_detector_demos_have_one_fixed_count_case(profile):
    sweep = material_sweep("hopg", catalog_profile=profile)
    assert sweep.beam.energy_keV.tolist() == [30.0]
    assert sweep.n_electrons.tolist() == [500.0]
    assert sweep.n_electrons_brem.tolist() == [100.0]
