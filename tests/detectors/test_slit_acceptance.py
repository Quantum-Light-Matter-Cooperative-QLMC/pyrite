"""SlitAcceptance: SensorAcceptance equivalence, roll symmetry, limits and wiring."""

import math
import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

import pyrite as pr
from pyrite.api import build_case, case_content_key
from pyrite.config import FixedSampling
from pyrite.detectors import Detector, SensorAcceptance, SlitAcceptance
from pyrite.units import Q_, ureg

GEOMETRIES = [(400, 0.05, 14.0), (250, 0.2, 25.0), (100, 30.0, 3.0), (1, 2, 2)]


def _scene(detector):
    beam = pr.Beam(
        energy_keV=30, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab("hopg", thickness_ang=1000, width=None, height=None)
    numerics = pr.Numerics(sampling=FixedSampling(cxr_histories=1, brem_histories=1))
    return build_case(pr.Scene(beam, target, detector), numerics)


@pytest.mark.parametrize(("distance", "width", "length"), GEOMETRIES)
def test_roll_zero_equals_sensor_with_width_polar(distance, width, length):
    slit = SlitAcceptance(distance, width, length)
    sensor = SensorAcceptance(distance, length, width)  # width_mm=azimuthal, height_mm=polar
    assert slit.roll_deg == 0
    assert slit.resolve() == sensor.resolve()
    detectors = [Detector(135, acceptance=a) for a in (slit, sensor)]
    cases = [_scene(d) for d in detectors]
    assert cases[0] == cases[1]
    assert case_content_key(cases[0]) == case_content_key(cases[1])
    assert detectors[0].polar_acceptance_deg == detectors[1].polar_acceptance_deg
    assert detectors[0].azimuthal_acceptance_deg == detectors[1].azimuthal_acceptance_deg
    assert detectors[0].solid_angle_sr == detectors[1].solid_angle_sr


@pytest.mark.parametrize(("distance", "width", "length"), GEOMETRIES)
def test_roll_ninety_swaps_to_sensor_height_length(distance, width, length):
    slit = SlitAcceptance(distance, width, length, roll_deg=90)
    sensor = SensorAcceptance(distance, width, length)  # azimuthal=w, polar=length
    assert slit.resolve() == pytest.approx(sensor.resolve(), rel=1e-15)
    assert slit.resolve()[2] == SlitAcceptance(distance, width, length).resolve()[2]


@pytest.mark.parametrize("roll", [0.0, 1e-7, 17.0, 45.0, 63.3, 89.999, 90.0])
def test_omega_roll_invariant_and_mirror_symmetry(roll):
    base = SlitAcceptance(400, 0.3, 12.0)
    rolled = replace(base, roll_deg=roll)
    mirrored = replace(base, roll_deg=180 - roll)
    assert rolled.resolve()[2] == base.resolve()[2]
    assert mirrored.resolve() == pytest.approx(rolled.resolve(), rel=1e-13)
    assert mirrored.roll_deg == (180 - roll) % 180


def test_projected_extents_use_opening_angles():
    distance, width, length, roll = 300.0, 0.4, 10.0, 30.0
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    polar, azimuthal, omega = SlitAcceptance(distance, width, length, roll).resolve()
    assert polar == pytest.approx(
        math.degrees(2 * math.atan((width * c + length * s) / (2 * distance))), rel=1e-14
    )
    assert azimuthal == pytest.approx(
        math.degrees(2 * math.atan((length * c + width * s) / (2 * distance))), rel=1e-14
    )
    # Independent quadrature of projected area of the rolled rectangle.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    u = nodes[:, None] * length / 2  # along the long axis
    v = nodes[None, :] * width / 2
    x = u * c - v * s
    y = u * s + v * c
    integral = (
        length
        * width
        / 4
        * np.sum(
            weights[:, None] * weights[None, :] * distance / (distance**2 + x * x + y * y) ** 1.5
        )
    )
    assert omega == pytest.approx(integral, rel=1e-12)


@pytest.mark.parametrize("roll", [0, 20, 45, 70, 90, 135])
def test_small_slit_limits(roll):
    distance, width, length = 1e5, 0.01, 0.2
    psi = math.radians(roll)
    polar, azimuthal, omega = SlitAcceptance(distance, width, length, roll).resolve()
    assert omega == pytest.approx(width * length / distance**2, rel=1e-9)
    assert math.radians(polar) == pytest.approx(
        (width * abs(math.cos(psi)) + length * abs(math.sin(psi))) / distance, rel=1e-9
    )
    assert math.radians(azimuthal) == pytest.approx(
        (length * abs(math.cos(psi)) + width * abs(math.sin(psi))) / distance, rel=1e-9
    )


def test_width_may_exceed_length_and_roll_is_from_length_axis():
    wide = SlitAcceptance(100, 8, 2)
    assert wide.resolve() == SensorAcceptance(100, 2, 8).resolve()
    assert SlitAcceptance(100, 8, 2, 90).resolve() == pytest.approx(
        SensorAcceptance(100, 8, 2).resolve(), rel=1e-15
    )


@pytest.mark.parametrize(
    ("roll", "folded"), [(-30, 150), (180, 0), (210, 30), (360, 0), (-1e-20, 0), (45.5, 45.5)]
)
def test_roll_normalized_to_half_turn(roll, folded):
    slit = SlitAcceptance(1, 1, 1, roll)
    assert slit.roll_deg == pytest.approx(folded)
    assert 0 <= slit.roll_deg < 180
    assert SlitAcceptance(1, 1, 1, roll=Q_(roll, "degree")) == slit


def test_quantities_readback_replacement_and_records():
    slit = SlitAcceptance(
        distance=Q_(40, "cm"), width=Q_(50, "um"), length=Q_(1.4, "cm"), roll=Q_(np.pi / 6, "rad")
    )
    numeric = SlitAcceptance(400, 0.05, 14, roll_deg=30)
    assert slit.distance_mm == 400 and slit.width_mm == 0.05 and slit.length_mm == 14
    assert slit.roll_deg == pytest.approx(30)
    assert numeric.roll_deg == 30
    assert {f.name for f in fields(slit)} == {"distance_mm", "width_mm", "length_mm", "roll_deg"}
    assert asdict(numeric) == dict(distance_mm=400, width_mm=0.05, length_mm=14, roll_deg=30)
    assert pickle.loads(pickle.dumps(numeric)) == numeric
    assert numeric.distance == Q_(400, "mm")
    assert numeric.roll == Q_(30, "degree")
    assert numeric.distance._REGISTRY is ureg and numeric.distance is not numeric.distance
    assert numeric.with_distance(Q_(2, "cm")) == replace(numeric, distance_mm=20)
    assert numeric.with_width(Q_(2, "cm")) == replace(numeric, width_mm=20)
    assert numeric.with_length(Q_(2, "cm")) == replace(numeric, length_mm=20)
    assert numeric.with_roll(Q_(210, "degree")) == replace(numeric, roll_deg=30)
    external = pint.UnitRegistry()
    assert (
        SlitAcceptance(distance=external.Quantity(400, "mm"), width_mm=1, length_mm=2).distance_mm
        == 400
    )


@pytest.mark.parametrize("name", ["distance", "width", "length"])
@pytest.mark.parametrize(
    "value", [1, "1 mm", Q_(1, "s"), Q_(0, "mm"), Q_(-1, "mm"), Q_(np.nan, "mm"), Q_([1], "mm")]
)
def test_invalid_semantic_lengths(name, value):
    options = {"distance": Q_(1, "mm"), "width": Q_(1, "mm"), "length": Q_(1, "mm")}
    options[name] = value
    with pytest.raises((TypeError, ValueError), match=f"SlitAcceptance.{name}"):
        SlitAcceptance(**options)


@pytest.mark.parametrize("value", [1, "1", Q_(1, "mm"), Q_(np.inf, "degree"), Q_([1], "degree")])
def test_invalid_roll_quantity(value):
    with pytest.raises((TypeError, ValueError), match="SlitAcceptance.roll"):
        SlitAcceptance(1, 1, 1, roll=value)
    with pytest.raises((TypeError, ValueError), match="SlitAcceptance.roll"):
        SlitAcceptance(1, 1, 1).with_roll(value)


@pytest.mark.parametrize("field", ["distance_mm", "width_mm", "length_mm"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_numeric_lengths(field, value):
    values = dict(distance_mm=400, width_mm=1, length_mm=10)
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        SlitAcceptance(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_numeric_roll(value):
    with pytest.raises((TypeError, ValueError)):
        SlitAcceptance(1, 1, 1, roll_deg=value)


def test_missing_and_conflicting_spellings():
    with pytest.raises(TypeError, match="distance_mm and distance"):
        SlitAcceptance(1, 1, 1, distance=Q_(1, "mm"))
    with pytest.raises(TypeError, match="roll_deg and roll"):
        SlitAcceptance(1, 1, 1, 0, roll=Q_(0, "degree"))
    with pytest.raises(TypeError, match="requires length_mm or length"):
        SlitAcceptance(1, 1)
    with pytest.raises(TypeError, match="width_mm and width"):
        replace(SlitAcceptance(1, 1, 1), width=Q_(1, "mm"))


def test_detector_wiring_and_error_message():
    slit = SlitAcceptance(400, 0.05, 14, roll_deg=30)
    detector = Detector(135, acceptance=slit)
    assert detector.acceptance is slit
    assert (
        detector.polar_acceptance_deg,
        detector.azimuthal_acceptance_deg,
        detector.solid_angle_sr,
    ) == slit.resolve()
    assert pickle.loads(pickle.dumps(detector)) == detector
    assert detector.with_observation_angle(Q_(90, "degree")).acceptance == slit
    with pytest.raises(TypeError, match="AngularAcceptance, SensorAcceptance or SlitAcceptance"):
        Detector(acceptance=object())
    assert pr.detectors.SlitAcceptance is SlitAcceptance
