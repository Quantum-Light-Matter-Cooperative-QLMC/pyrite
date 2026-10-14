"""Independent quadrature anchors and public acceptance compatibility."""

import math
import pickle
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.api import build_case, case_content_key
from pyrite.campaign.profiles import dataset_identity
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.results import Settings


@pytest.mark.parametrize("theta,polar", [(0.01, 0.01), (1, 1), (90, 30), (179, 1), (179.99, 0.01)])
def test_angular_acceptance_round_trip(theta, polar):
    # Independent integral over latitude, avoiding the implementation's sin identity.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    half = math.radians(polar) / 2
    omega = math.radians(40) * half * np.dot(weights, np.sin(math.radians(theta) + half * nodes))
    descriptions = (
        pr.AngularAcceptance(polar_deg=polar, azimuthal_deg=40),
        pr.AngularAcceptance(polar_deg=polar, solid_angle_sr=omega),
        pr.AngularAcceptance(azimuthal_deg=40, solid_angle_sr=omega),
    )
    for acceptance in descriptions:
        detector = pr.Detector(observation_angle_deg=theta, acceptance=acceptance)
        assert detector.polar_acceptance_deg == pytest.approx(polar, rel=2e-12)
        assert detector.azimuthal_acceptance_deg == pytest.approx(40, rel=2e-12)
        assert detector.solid_angle_sr == pytest.approx(omega, rel=2e-12)


def test_angular_acceptance_analytic_reference_and_small_patch():
    # θ=90°, Δθ=60°, Δφ=90° covers exactly π/2 sr.
    detector = pr.Detector(acceptance=pr.AngularAcceptance(polar_deg=60, azimuthal_deg=90))
    assert detector.solid_angle_sr == pytest.approx(math.pi / 2, rel=1e-15)
    small = pr.Detector(60, acceptance=pr.AngularAcceptance(polar_deg=0.001, azimuthal_deg=0.002))
    assert small.solid_angle_sr == pytest.approx(
        math.radians(0.001) * math.radians(0.002) * math.sin(math.radians(60)), rel=2e-11
    )


@pytest.mark.parametrize(
    "kwargs", [{}, {"polar_deg": 2}, {"polar_deg": 2, "azimuthal_deg": 4, "solid_angle_sr": 0.01}]
)
def test_angular_acceptance_requires_exactly_two(kwargs):
    with pytest.raises(ValueError, match="polar_deg.*azimuthal_deg.*solid_angle_sr"):
        pr.AngularAcceptance(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polar_deg": 0, "azimuthal_deg": 2},
        {"polar_deg": 181, "azimuthal_deg": 2},
        {"polar_deg": 2, "azimuthal_deg": 361},
        {"polar_deg": 2, "solid_angle_sr": 13},
        {"polar_deg": float("nan"), "solid_angle_sr": 0.01},
        {"polar_deg": True, "solid_angle_sr": 0.01},
        {"polar_deg": 2, "solid_angle_sr": float("inf")},
    ],
)
def test_angular_acceptance_rejects_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        pr.AngularAcceptance(**kwargs)


@pytest.mark.parametrize(
    "theta,acceptance",
    [
        (0, pr.AngularAcceptance(polar_deg=1, azimuthal_deg=1)),
        (180, pr.AngularAcceptance(azimuthal_deg=1, solid_angle_sr=0.01)),
        (1, pr.AngularAcceptance(polar_deg=3, azimuthal_deg=1)),
        (179, pr.AngularAcceptance(azimuthal_deg=360, solid_angle_sr=0.1)),
        (90, pr.AngularAcceptance(azimuthal_deg=1, solid_angle_sr=1)),
        (90, pr.AngularAcceptance(polar_deg=1, solid_angle_sr=1)),
    ],
)
def test_angular_acceptance_rejects_unreachable_patch(theta, acceptance):
    with pytest.raises(ValueError):
        pr.Detector(theta, acceptance=acceptance)


def test_angular_acceptance_full_sphere_boundary():
    for acceptance in (
        pr.AngularAcceptance(polar_deg=180, azimuthal_deg=360),
        pr.AngularAcceptance(polar_deg=180, solid_angle_sr=4 * math.pi),
        pr.AngularAcceptance(azimuthal_deg=360, solid_angle_sr=4 * math.pi),
    ):
        detector = pr.Detector(acceptance=acceptance)
        assert detector.solid_angle_sr == 4 * math.pi
        assert detector.polar_acceptance_deg == 180
        assert detector.azimuthal_acceptance_deg == 360


@pytest.mark.parametrize(
    "distance,width,height", [(1, 2, 2), (2, 8, 3), (1, 10, 4), (400, 14.08, 14.08)]
)
def test_sensor_acceptance_exact_rectangle(distance, width, height):
    # Independent tensor Gauss quadrature of projected area, not a closed-form copy.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    x = nodes[:, None] * width / 2
    y = nodes[None, :] * height / 2
    integral = (
        width
        * height
        / 4
        * np.sum(
            weights[:, None] * weights[None, :] * distance / (distance**2 + x * x + y * y) ** 1.5
        )
    )
    detector = pr.Detector(135, acceptance=pr.SensorAcceptance(distance, width, height))
    assert detector.solid_angle_sr == pytest.approx(integral, rel=3e-12)
    if (distance, width, height) == (1, 2, 2):
        # A cube face subtends one sixth of the sphere.
        assert detector.solid_angle_sr == pytest.approx(2 * math.pi / 3, rel=1e-15)
        assert detector.polar_acceptance_deg == 90
        assert detector.azimuthal_acceptance_deg == 90


def test_sensor_acceptance_small_sensor_and_scale_invariance():
    sensor = pr.SensorAcceptance(1000, 0.01, 0.02)
    detector = pr.Detector(acceptance=sensor)
    assert detector.solid_angle_sr == pytest.approx(2e-10, rel=1e-9)
    assert math.radians(detector.polar_acceptance_deg) == pytest.approx(2e-5, rel=1e-9)
    assert math.radians(detector.azimuthal_acceptance_deg) == pytest.approx(1e-5, rel=1e-9)
    for scale in (1e-200, 1e200):
        scaled = pr.Detector(acceptance=pr.SensorAcceptance(scale, 2 * scale, 2 * scale))
        assert scaled.solid_angle_sr == pytest.approx(2 * math.pi / 3, rel=1e-15)


def test_sensor_acceptance_extreme_aspect_ratio_strip_limit():
    detector = pr.Detector(acceptance=pr.SensorAcceptance(1e-200, 1e200, 1e-200))
    # Infinite-width strip is a spherical lune of area 4 atan(h/(2d)).
    assert detector.polar_acceptance_deg == pytest.approx(53.13010235415598, rel=1e-14)
    assert detector.azimuthal_acceptance_deg == 180
    assert detector.solid_angle_sr == pytest.approx(1.8545904360032244, rel=2e-13)


@pytest.mark.parametrize("field", ["distance_mm", "width_mm", "height_mm"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_sensor_acceptance_rejects_invalid_lengths(field, value):
    values = dict(distance_mm=400, width_mm=14, height_mm=14)
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        pr.SensorAcceptance(**values)


def test_acceptance_and_resolved_detector_are_frozen_and_copies_recompute():
    angular = pr.AngularAcceptance(polar_deg=2, azimuthal_deg=4)
    detector = pr.Detector(135, acceptance=angular)
    for obj, name in (
        (angular, "polar_deg"),
        (pr.SensorAcceptance(1, 2, 2), "width_mm"),
        (detector, "polar_acceptance_deg"),
        (detector, "azimuthal_acceptance_deg"),
        (detector, "solid_angle_sr"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, 1)
    copied = replace(detector, observation_angle_deg=90)
    assert copied.acceptance == angular
    assert copied.solid_angle_sr == pytest.approx(
        detector.solid_angle_sr / math.sin(math.radians(135))
    )
    assert pickle.loads(pickle.dumps(detector)) == detector
    assert replace(detector, response=None) == detector


def test_flat_acceptance_keywords_and_positionals_are_removed():
    for keyword in ("polar_acceptance_deg", "solid_angle_sr"):
        with pytest.raises(TypeError, match=keyword):
            pr.Detector(**{keyword: 2})
    with pytest.raises(TypeError):
        pr.Detector(135, 2, 0.001)
    with pytest.raises(TypeError, match="AngularAcceptance or SensorAcceptance"):
        pr.Detector(acceptance=object())
    assert pr.Detector().polar_acceptance_deg is None


def test_equal_acceptance_preserves_lowered_case_and_both_identities():
    angular = pr.AngularAcceptance(polar_deg=2, solid_angle_sr=0.001)
    modern = pr.Detector(135, acceptance=angular)
    legacy = pr.Detector(135, _legacy_acceptance=(2, 0.001))
    beam = pr.Beam(energy_keV=30, transverse=pr.PointSpot(), longitudinal=pr.PointBunch())
    target = pr.Slab("hopg", thickness_ang=1000, footprint=None)
    numerics = pr.Numerics(cxr_trials=1, brem_trials=1)
    cases = [
        build_case(pr.Scene(beam, target, detector), numerics) for detector in (legacy, modern)
    ]
    assert cases[0] == cases[1]
    assert case_content_key(cases[0]) == case_content_key(cases[1])
    # Captured with this exact input on main at 86db77f5, before acceptance objects.
    assert (
        case_content_key(cases[1])
        == "32b309b2aa6564d8806a66018d433801379ec0ba500648f6357d3fb8c6e44661"
    )
    sweeps = [
        LegacySweep(material="hopg", beam=beam.to_spec(), target=target, detector=d)
        for d in (legacy, modern)
    ]
    identities = [dataset_identity("hopg", "full", Settings(), s) for s in sweeps]
    assert identities[0] == identities[1]
    assert (
        identities[1]["parameter_sha256"]
        == "04c7fc6e9e1f7669c8b08db9efc7d5ddc4664b922a5717daa504401d79a0d29b"
    )
    # Pin to the pre-object payload vocabulary; no acceptance metadata enters identity.
    assert "detector" not in identities[0]["resolved_parameters"]["sweep"]


def test_sensor_and_equal_flat_acceptance_lower_identically():
    modern = pr.Detector(135, acceptance=pr.SensorAcceptance(400, 14.08, 14.08))
    legacy = pr.Detector(
        135, _legacy_acceptance=(modern.polar_acceptance_deg, modern.solid_angle_sr)
    )
    beam = pr.Beam(energy_keV=30, transverse=pr.PointSpot(), longitudinal=pr.PointBunch())
    target = pr.Slab("hopg", thickness_ang=1000, footprint=None)
    cases = [build_case(pr.Scene(beam, target, detector)) for detector in (legacy, modern)]
    assert cases[0] == cases[1]
    assert case_content_key(cases[0]) == case_content_key(cases[1])
