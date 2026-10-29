"""Semantic planar pose and pixel-grid lengths/angles stay numeric (#755)."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.detectors import PixelGrid, PlanarPose
from pyrite.units import Q_, ureg


def test_pose_center_matches_numeric_fields() -> None:
    semantic = PlanarPose(center=Q_([0, 1, 10], "cm"), normal=(0, 0, 2))
    numeric = PlanarPose((0.0, 10.0, 100.0))
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert {item.name for item in fields(semantic)} == {"center_mm", "normal", "x_axis"}
    np.testing.assert_array_equal(semantic.center.to("mm").magnitude, [0, 10, 100])
    assert semantic.center.units == ureg.mm
    assert isinstance(semantic.center_mm, tuple)


def test_pose_replacement_and_readback_are_detached() -> None:
    pose = PlanarPose((0.0, 0.0, 100.0), x_axis=(0.0, 1.0, 0.0))
    moved = pose.with_center(Q_([1, 2, 3], "m"))
    assert moved == PlanarPose((1000.0, 2000.0, 3000.0), x_axis=(0.0, 1.0, 0.0))
    readback = moved.center
    readback.magnitude[:] = 0
    readback.ito("m")
    assert moved.center_mm == (1000.0, 2000.0, 3000.0)
    source = Q_(np.array([0.0, 0.0, 5.0]), "mm")
    owned = PlanarPose(center=source)
    source.magnitude[:] = 9
    assert owned.center_mm == (0.0, 0.0, 5.0)
    assert pose.center_mm == (0.0, 0.0, 100.0)


def test_pose_legacy_and_conflicts() -> None:
    legacy = PlanarPose(center_mm=(0, 0, 10 * ureg.cm))
    assert legacy.center_mm == (0.0, 0.0, 100.0)
    with pytest.raises(TypeError, match="mutually exclusive"):
        PlanarPose((0, 0, 1), center=Q_([0, 0, 1], "mm"))
    with pytest.raises(TypeError, match="requires center_mm or center"):
        PlanarPose()


@pytest.mark.parametrize(
    "value, error",
    [
        ((0.0, 0.0, 1.0), TypeError),
        (Q_(1, "mm"), TypeError),
        (Q_([0, 1], "mm"), TypeError),
        (Q_([0, 0, 1j], "mm"), TypeError),
        (Q_([0, 0, 1], "eV"), ValueError),
        (Q_([0, 0, np.inf], "mm"), ValueError),
    ],
)
def test_pose_center_refuses_invalid_values(value, error) -> None:
    with pytest.raises(error, match="PlanarPose.center"):
        PlanarPose(center=value)


def test_from_observation_quantities_match_numeric_pose() -> None:
    semantic = PlanarPose.from_observation(
        distance=Q_(0.1, "m"),
        polar=Q_(np.pi / 6, "rad"),
        azimuth=Q_(-20, "degree"),
        roll=Q_(5, "degree"),
        offset=Q_([1, -2], "mm"),
    )
    numeric = PlanarPose.from_observation(100.0, 30.0, -20.0, 5.0, (1.0, -2.0))
    np.testing.assert_allclose(semantic.center_mm, numeric.center_mm, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(semantic.normal, numeric.normal, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(semantic.x_axis, numeric.x_axis, rtol=1e-14, atol=1e-15)
    assert PlanarPose.from_observation(distance=Q_(100, "mm"), polar=Q_(30, "deg")) == (
        PlanarPose.from_observation(100.0, 30.0)
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"distance": 100.0, "polar": Q_(30, "deg")}, TypeError, "distance"),
        ({"distance": Q_(0, "mm"), "polar": Q_(30, "deg")}, ValueError, "distance"),
        ({"distance": Q_(1, "mm"), "polar": Q_(30, "dimensionless")}, ValueError, "polar"),
        ({"distance": Q_(1, "mm"), "polar": Q_(1, "sr")}, ValueError, "polar"),
        ({"distance": Q_(1, "mm"), "polar": Q_(200, "deg")}, ValueError, "polar"),
        (
            {"distance": Q_(1, "mm"), "polar": Q_(30, "deg"), "offset": Q_(1, "mm")},
            TypeError,
            "offset",
        ),
        ({"distance_mm": 1, "distance": Q_(1, "mm"), "polar_deg": 30}, TypeError, "exclusive"),
        (
            {"distance_mm": 1, "polar_deg": 30, "roll_deg": 0, "roll": Q_(0, "deg")},
            TypeError,
            "exclusive",
        ),
        ({"distance": Q_(1, "mm")}, TypeError, "polar angle"),
    ],
)
def test_from_observation_refuses_invalid_quantities(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        PlanarPose.from_observation(**kwargs)


def test_pixel_pitch_matches_numeric_fields() -> None:
    semantic = PixelGrid((4, 8), pitch=Q_([55, 110], "um"))
    numeric = PixelGrid(shape=(4, 8), pitch_mm=(0.055, 0.11))
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert {item.name for item in fields(semantic)} == {"shape", "pitch_mm"}
    np.testing.assert_allclose(semantic.pitch.to("um").magnitude, [55, 110], rtol=1e-15)
    np.testing.assert_allclose(semantic.active_size.to("mm").magnitude, [0.88, 0.22], rtol=1e-15)
    assert PixelGrid.timepix3_chip() == PixelGrid((256, 256), pitch=Q_([55, 55], "um"))


def test_pixel_pitch_replacement_conflicts_and_registry() -> None:
    grid = PixelGrid((2, 2), (1.0, 1.0))
    assert grid.with_pitch(Q_([2, 3], "mm")) == PixelGrid((2, 2), (2.0, 3.0))
    assert grid.pitch_mm == (1.0, 1.0)
    external = pint.UnitRegistry()
    assert PixelGrid((2, 2), pitch=external.Quantity([1, 1], "mm")) == grid
    assert grid.pitch._REGISTRY is ureg
    with pytest.raises(TypeError, match="mutually exclusive"):
        PixelGrid((2, 2), (1.0, 1.0), pitch=Q_([1, 1], "mm"))
    with pytest.raises(TypeError, match="requires pitch_mm or pitch"):
        PixelGrid((2, 2))
    for value, error in (
        ((1.0, 1.0), TypeError),
        (Q_([1, 1, 1], "mm"), TypeError),
        (Q_([1, 0], "mm"), ValueError),
        (Q_([1, -1], "mm"), ValueError),
        (Q_([1, 1], "s"), ValueError),
    ):
        with pytest.raises(error, match="PixelGrid.pitch"):
            grid.with_pitch(value)
