"""DetectorSpec validation and portability contracts."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from cxr_mc.detectors import DetectorSpec


def test_detector_spec_is_frozen_and_serializes_reserved_metadata():
    detector = DetectorSpec(
        observation_angle_deg=119.0,
        polar_acceptance_deg=16.6,
        solid_angle_sr=0.066,
        response_model="registry/zhai",
        qe_curve="package-data/qe/zhai.csv",
        pixel_pitch_um=55.0,
        sensor_thickness_um=500.0,
        distance_mm=400.0,
        threshold_eV=0.0,
    )

    assert asdict(detector) == {
        "observation_angle_deg": 119.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
        "response_model": "registry/zhai",
        "qe_curve": "package-data/qe/zhai.csv",
        "pixel_pitch_um": 55.0,
        "sensor_thickness_um": 500.0,
        "distance_mm": 400.0,
        "threshold_eV": 0.0,
    }
    with pytest.raises(FrozenInstanceError):
        detector.distance_mm = 1.0  # type: ignore[misc]


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
        ({"pixel_pitch_um": -1.0}, "must be >= 0"),
        ({"threshold_eV": -1.0}, "must be >= 0"),
        ({"qe_curve": "/home/user/qe.csv"}, "portable registry/resource identifier"),
        ({"qe_curve": "../qe.csv"}, "portable registry/resource identifier"),
        ({"response_model": r"C:\models\response"}, "portable registry/resource identifier"),
    ],
)
def test_detector_spec_rejects_invalid_geometry_and_machine_paths(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        DetectorSpec(**kwargs)
