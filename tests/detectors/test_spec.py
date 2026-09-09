"""Detector validation, binning, scoring, and compatibility contracts."""

from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from pyrite.detectors import (
    Detector,
    DetectorSpec,
    EnergyBins,
    LegacyEDS,
    Timepix3,
    convolve_detector,
    detector_efficiency,
)


def test_detector_is_frozen_and_owns_acceptance_binning_and_response():
    bins = EnergyBins(
        line=np.array([75.0, 78.0]),
        line_by_energy={30.0: np.array([75.0])},
        brem=np.array([0.0, 100.0]),
    )
    detector = Detector(119.0, 16.6, 0.066, bins, Timepix3(thickness_um=500.0))

    payload = asdict(detector)
    assert payload["observation_angle_deg"] == 119.0
    assert payload["energy_bins"]["line_by_energy"][30.0].tolist() == [75.0]
    assert payload["response"]["thickness_um"] == 500.0
    for retired in (
        "response_model",
        "qe_curve",
        "pixel_pitch_um",
        "sensor_thickness_um",
        "distance_mm",
        "threshold_eV",
    ):
        assert not hasattr(detector, retired)
    with pytest.raises(FrozenInstanceError):
        detector.solid_angle_sr = 1.0


def test_none_response_scores_intrinsic_density_with_acceptance_scale():
    intrinsic = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        Detector().score(np.array([10.0, 20.0, 30.0]), intrinsic, scale=0.25),
        intrinsic * 0.25,
    )


def test_legacy_response_is_bit_for_bit_the_historical_read_path():
    energy = np.arange(100.0, 1000.0, 10.0)
    intrinsic = np.linspace(1.0, 3.0, energy.size)
    expected = (
        convolve_detector(
            energy,
            intrinsic * detector_efficiency(energy),
            130.0,
        )
        * 2.5
    )

    scored = Detector(response=LegacyEDS(apply_qe=True, convolve=True)).score(
        energy,
        intrinsic,
        fwhm_eV=130.0,
        scale=2.5,
    )

    np.testing.assert_array_equal(scored, expected)


def test_detector_spec_is_a_deprecated_compatible_detector():
    with pytest.warns(DeprecationWarning, match="DetectorSpec is deprecated"):
        detector = DetectorSpec(119.0, 16.6, 0.066)
    assert isinstance(detector, Detector)
    assert detector.observation_angle_deg == 119.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"observation_angle_deg": float("nan")}, "must be finite"),
        ({"observation_angle_deg": -1.0}, "must be >= 0 and <= 180"),
        ({"observation_angle_deg": 181.0}, "must be >= 0 and <= 180"),
        ({"polar_acceptance_deg": 0.0}, "must be positive"),
        ({"polar_acceptance_deg": 181.0}, "full polar span"),
        ({"solid_angle_sr": 0.0}, "must be positive"),
        ({"solid_angle_sr": 13.0}, r"must be <= 4\*pi sr"),
        ({"energy_bins": object()}, "must be an EnergyBins"),
        ({"response": object()}, r"must implement score\(\)"),
    ],
)
def test_detector_rejects_invalid_state(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        Detector(**kwargs)
