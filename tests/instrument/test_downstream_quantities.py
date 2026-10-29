"""Semantic filter-plate and planar-detector lengths stay numeric (#755)."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from pyrite.detectors import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
from pyrite.detectors.spec import IdealPhotonCounter
from pyrite.units import Q_, ureg

POSE = PlanarPose((0.0, 0.0, 100.0))
PLATE_POSE = PlanarPose((0.0, 0.0, 50.0))


def test_plate_quantities_match_numeric_fields() -> None:
    semantic = FilterPlate(
        "silicon", thickness=Q_(100, "um"), size=Q_([1, 2], "cm"), pose=PLATE_POSE, name="si"
    )
    numeric = FilterPlate("silicon", 0.1, (10.0, 20.0), PLATE_POSE, "si")
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert {f.name for f in fields(semantic)} == {
        "material",
        "thickness_mm",
        "size_mm",
        "pose",
        "name",
    }
    assert semantic.thickness.units == ureg.mm
    assert semantic.thickness.magnitude == pytest.approx(0.1, rel=1e-15)
    np.testing.assert_array_equal(semantic.size.magnitude, [10.0, 20.0])
    assert semantic.size.units == ureg.mm


def test_plate_replacement_and_readback_are_detached() -> None:
    plate = FilterPlate("silicon", 0.1, (10.0, 10.0), PLATE_POSE)
    thicker = plate.with_thickness(Q_(0.5, "mm")).with_size(Q_([3, 4], "mm"))
    assert thicker == FilterPlate("silicon", 0.5, (3.0, 4.0), PLATE_POSE)
    size = thicker.size
    size.magnitude[:] = 0
    size.ito("m")
    assert thicker.size_mm == (3.0, 4.0)
    assert plate.thickness_mm == 0.1


def test_plate_conflicts_and_required_values() -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        FilterPlate("silicon", 0.1, (1, 1), PLATE_POSE, thickness=Q_(1, "mm"))
    with pytest.raises(TypeError, match="mutually exclusive"):
        FilterPlate("silicon", 0.1, (1, 1), PLATE_POSE, size=Q_([1, 1], "mm"))
    with pytest.raises(TypeError, match="requires"):
        FilterPlate("silicon", size=Q_([1, 1], "mm"), pose=PLATE_POSE)
    with pytest.raises(TypeError, match="requires"):
        FilterPlate("silicon", thickness=Q_(1, "mm"), size=Q_([1, 1], "mm"))


@pytest.mark.parametrize(
    "keyword, value, error",
    [
        ("thickness", 0.1, TypeError),
        ("thickness", Q_([1, 2], "mm"), TypeError),
        ("thickness", Q_(1, "eV"), ValueError),
        ("thickness", Q_(0, "mm"), ValueError),
        ("thickness", Q_(-1, "mm"), ValueError),
        ("size", (1.0, 1.0), TypeError),
        ("size", Q_(1, "mm"), TypeError),
        ("size", Q_([1, 0], "mm"), ValueError),
        ("size", Q_([1, np.inf], "mm"), ValueError),
    ],
)
def test_plate_refuses_invalid_quantities(keyword, value, error) -> None:
    values = {"thickness": Q_(1, "mm"), "size": Q_([1, 1], "mm")} | {keyword: value}
    with pytest.raises(error, match=f"FilterPlate.{keyword}"):
        FilterPlate("silicon", pose=PLATE_POSE, **values)


def test_detector_size_matches_numeric_and_keeps_response() -> None:
    response = IdealPhotonCounter()
    semantic = PlanarDetector(POSE, size=Q_([1, 2], "cm"), response=response)
    numeric = PlanarDetector(POSE, size_mm=(10.0, 20.0))
    assert semantic == numeric
    assert {f.name for f in fields(semantic)} == {"pose", "size_mm", "pixels"}
    assert pickle.loads(pickle.dumps(semantic)).response == response
    resized = semantic.with_size(Q_([5, 6], "mm"))
    assert resized.size_mm == (5.0, 6.0)
    assert resized.response is response
    assert replace(semantic, response=None).response is None
    readback = semantic.size
    readback.magnitude[:] = 0
    assert semantic.size_mm == (10.0, 20.0)


def test_detector_size_with_pixels_and_conflicts() -> None:
    pixels = PixelGrid((2, 4), (0.5, 0.25))
    detector = PlanarDetector(POSE, pixels=pixels, size=Q_([1, 1], "mm"))
    np.testing.assert_array_equal(detector.size.to("mm").magnitude, [1.0, 1.0])
    with pytest.raises(ValueError, match="active size"):
        PlanarDetector(POSE, pixels=pixels, size=Q_([2, 1], "mm"))
    with pytest.raises(TypeError, match="mutually exclusive"):
        PlanarDetector(POSE, (1.0, 1.0), size=Q_([1, 1], "mm"))
    with pytest.raises(TypeError, match="PlanarDetector.size"):
        PlanarDetector(POSE, size=(1.0, 1.0))
    with pytest.raises(ValueError, match="requires size_mm or pixels"):
        PlanarDetector(POSE)
