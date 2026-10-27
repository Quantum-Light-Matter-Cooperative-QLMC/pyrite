"""#755: sensor length quantities preserve exact numeric acceptance and cases."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite import Beam, Numerics, Scene
from pyrite.api import build_case, case_content_key
from pyrite.beams import PointBunch, PointSpot
from pyrite.detectors import Detector, SensorAcceptance
from pyrite.geometry import Slab
from pyrite.units import Q_, ureg

NAMES = ("distance", "width", "height")


def test_quantity_construction_readback_and_numeric_records() -> None:
    sensor = SensorAcceptance(distance=Q_(40, "cm"), width=Q_(1.4, "cm"), height=Q_(7000, "um"))
    numeric = SensorAcceptance(400, 14, 7)
    assert sensor == numeric
    assert asdict(sensor) == asdict(numeric)
    assert {field.name for field in fields(sensor)} == {"distance_mm", "width_mm", "height_mm"}
    assert replace(sensor) == numeric
    assert pickle.loads(pickle.dumps(sensor)) == numeric
    assert sensor.resolve() == numeric.resolve()
    for name, value in zip(NAMES, (400, 14, 7), strict=True):
        assert getattr(sensor, name) == Q_(value, "mm")
        assert getattr(sensor, name)._REGISTRY is ureg
        assert getattr(sensor, name) is not getattr(sensor, name)
    assert sensor.distance.to("cm").magnitude == 40


def test_replacement_and_mixed_legacy_arguments() -> None:
    sensor = SensorAcceptance(400, width=Q_(14, "mm"), height_mm=7)
    for name in NAMES:
        changed = getattr(sensor, f"with_{name}")(Q_(2, "cm"))
        assert changed == replace(sensor, **{f"{name}_mm": 20})
    assert sensor == SensorAcceptance(400, 14, 7)
    reading = sensor.width
    reading *= 2
    assert sensor.width_mm == 14


def test_external_registry_and_scalar_array_input_are_detached() -> None:
    external = pint.UnitRegistry()
    source = external.Quantity(np.array(40.0), "cm")
    sensor = SensorAcceptance(distance=source, width=external.Quantity(14, "mm"), height_mm=7)
    source.magnitude[...] = 100
    assert sensor.distance_mm == 400
    assert sensor.distance._REGISTRY is ureg


def test_equivalent_quantities_keep_detector_lowering_and_identity() -> None:
    sensors = (
        SensorAcceptance(400, 14, 7),
        SensorAcceptance(distance=Q_(40, "cm"), width=Q_(14000, "um"), height=Q_(7, "mm")),
    )
    beam = Beam(energy_keV=30, transverse=PointSpot(), longitudinal=PointBunch())
    target = Slab("hopg", thickness_ang=1000, footprint=None)
    numerics = Numerics(cxr_trials=1, brem_trials=1)
    detectors = [Detector(135, acceptance=sensor) for sensor in sensors]
    assert detectors[0].solid_angle_sr == detectors[1].solid_angle_sr
    assert detectors[0].polar_acceptance_deg == detectors[1].polar_acceptance_deg
    assert detectors[0].azimuthal_acceptance_deg == detectors[1].azimuthal_acceptance_deg
    cases = [build_case(Scene(beam, target, detector), numerics) for detector in detectors]
    assert cases[0] == cases[1]
    assert case_content_key(cases[0]) == case_content_key(cases[1])
    changed = replace(detectors[1], acceptance=sensors[1].with_width(Q_(28, "mm")))
    assert changed.solid_angle_sr > detectors[1].solid_angle_sr


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (1, TypeError),
        ("1 mm", TypeError),
        (None, TypeError),
        (Q_(1, "s"), ValueError),
        (Q_(1, "dimensionless"), ValueError),
        (Q_([1], "mm"), TypeError),
        (Q_([], "mm"), TypeError),
        (Q_([[1]], "mm"), TypeError),
        (Q_(np.array(True), "cm"), TypeError),
        (Q_(1j, "mm"), TypeError),
        (Q_(0, "mm"), ValueError),
        (Q_(-1, "mm"), ValueError),
        (Q_(np.inf, "mm"), ValueError),
        (Q_(np.nan, "mm"), ValueError),
    ],
)
def test_invalid_semantic_lengths(name: str, value: object, error: type[Exception]) -> None:
    options = {field: Q_(1, "mm") for field in NAMES}
    options[name] = value
    with pytest.raises(error, match=f"SensorAcceptance.{name}"):
        SensorAcceptance(**options)
    with pytest.raises(error, match=f"SensorAcceptance.{name}"):
        getattr(SensorAcceptance(400, 14, 7), f"with_{name}")(value)


@pytest.mark.parametrize("name", NAMES)
def test_missing_and_conflicting_spellings(name: str) -> None:
    numeric = {f"{field}_mm": 1 for field in NAMES}
    with pytest.raises(TypeError, match=f"{name}_mm and {name}"):
        SensorAcceptance(**numeric, **{name: Q_(1, "mm")})
    with pytest.raises(TypeError, match=f"{name}_mm and {name}"):
        replace(SensorAcceptance(1, 1, 1), **{name: Q_(1, "mm")})
    del numeric[f"{name}_mm"]
    with pytest.raises(TypeError, match=f"requires {name}_mm or {name}"):
        SensorAcceptance(**numeric)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_float_storage_representability(name: str, value: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range")
    options = {field: Q_(1, "mm") for field in NAMES}
    options[name] = Q_(np.longdouble(value), "mm")
    with pytest.raises(ValueError, match="representable"):
        SensorAcceptance(**options)


def test_numeric_import_construction_and_replace_remain_pint_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from dataclasses import replace; "
                "from pyrite.detectors import SensorAcceptance, Detector; "
                "s=SensorAcceptance(400,14,7); assert replace(s)==s; "
                "assert Detector(acceptance=s).solid_angle_sr>0; "
                "assert 'pint' not in sys.modules"
            ),
        ],
        check=True,
    )
