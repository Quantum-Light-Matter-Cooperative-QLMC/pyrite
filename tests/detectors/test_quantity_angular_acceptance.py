"""Quantity-valued scalar-detector angular boundaries (issue #755)."""

import math
import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.detectors import AngularAcceptance, Detector, IdealPhotonCounter
from pyrite.units import Q_, ureg


def test_acceptance_semantic_equivalence_and_record_compatibility():
    numeric = AngularAcceptance(2, 30)
    quantity = AngularAcceptance(polar=Q_(math.pi / 90, "radian"), azimuthal=30 * ureg.degree)
    assert quantity.polar_deg == pytest.approx(numeric.polar_deg)
    assert quantity.azimuthal_deg == numeric.azimuthal_deg
    assert quantity.solid_angle is None
    assert quantity.polar.to("degree").magnitude == pytest.approx(2)
    assert quantity.azimuthal.to("degree").magnitude == 30
    assert asdict(quantity).keys() == asdict(numeric).keys()
    assert {f.name for f in fields(quantity)} == {"polar_deg", "azimuthal_deg", "solid_angle_sr"}
    detector = Detector(90, acceptance=quantity)
    reference = Detector(90, acceptance=numeric)
    assert detector.solid_angle_sr == pytest.approx(reference.solid_angle_sr)
    assert pickle.loads(pickle.dumps(quantity)) == quantity
    assert replace(quantity) == quantity


def test_acceptance_three_two_field_forms_and_replacements():
    direct = AngularAcceptance(polar=Q_(5, "degree"), azimuthal=Q_(20, "degree"))
    omega = Detector(90, acceptance=direct).solid_angle_sr
    assert omega is not None
    p_omega = AngularAcceptance(polar=Q_(5, "degree"), solid_angle=Q_(omega, "sr"))
    a_omega = AngularAcceptance(azimuthal=Q_(20, "degree"), solid_angle=Q_(omega, "sr"))
    assert p_omega.azimuthal is None
    assert a_omega.polar is None
    assert a_omega.solid_angle.to("sr").magnitude == pytest.approx(omega)
    assert Detector(90, acceptance=p_omega).solid_angle_sr == pytest.approx(omega)
    assert Detector(90, acceptance=a_omega).solid_angle_sr == pytest.approx(omega)
    assert direct.with_polar(Q_(0.1, "radian")).polar_deg == pytest.approx(math.degrees(0.1))
    assert direct.with_azimuthal(Q_(0.2, "radian")).azimuthal_deg == pytest.approx(
        math.degrees(0.2)
    )
    assert p_omega.with_solid_angle(Q_(omega / 2, "sr")).solid_angle_sr == omega / 2
    assert direct.polar_deg == 5
    assert replace(direct, polar_deg=3).polar_deg == 3
    with pytest.raises(ValueError, match="not specified"):
        direct.with_solid_angle(Q_(1, "sr"))


def test_detector_semantic_angle_identity_replacement_and_response():
    response = IdealPhotonCounter()
    patch = AngularAcceptance(polar_deg=2, azimuthal_deg=20)
    numeric = Detector(90, acceptance=patch, response=response)
    quantity = Detector(
        observation_angle=Q_(math.pi / 2, "rad"), acceptance=patch, response=response
    )
    assert quantity == numeric
    assert quantity.observation_angle.to("rad").magnitude == pytest.approx(math.pi / 2)
    assert quantity.observation_angle_deg == 90.0
    assert asdict(quantity) == asdict(numeric)
    assert pickle.loads(pickle.dumps(quantity)) == numeric
    shifted = quantity.with_observation_angle(Q_(math.pi / 3, "rad"))
    assert shifted.observation_angle_deg == pytest.approx(60)
    assert shifted.response == response
    assert shifted.acceptance is patch
    assert quantity.observation_angle_deg == 90
    assert replace(quantity, observation_angle_deg=60).solid_angle_sr == pytest.approx(
        shifted.solid_angle_sr
    )
    assert Detector().observation_angle_deg == 90
    assert Detector(observation_angle=Q_(0, "degree")).observation_angle_deg == 0


def test_external_registry_and_detached_scalar_readback():
    external = pint.UnitRegistry()
    patch = AngularAcceptance(
        polar=external.Quantity(0.1, "radian"),
        solid_angle=external.Quantity(0.01, "steradian"),
    )
    assert patch.polar._REGISTRY is ureg
    assert patch.solid_angle._REGISTRY is ureg
    detector = Detector(observation_angle=external.Quantity(0.5, "radian"), acceptance=patch)
    assert detector.observation_angle._REGISTRY is ureg
    # A scalar read-back is newly constructed, not the source registry value.
    assert detector.observation_angle is not detector.observation_angle


@pytest.mark.parametrize(
    "kwargs",
    [
        {"polar_deg": 2, "polar": Q_(2, "degree"), "azimuthal_deg": 20},
        {"polar": Q_(2, "degree"), "azimuthal_deg": 20, "azimuthal": Q_(20, "degree")},
        {"polar_deg": None, "polar": Q_(2, "degree"), "azimuthal_deg": 20},
    ],
)
def test_acceptance_rejects_dual_names(kwargs):
    with pytest.raises(TypeError, match="both"):
        AngularAcceptance(**kwargs)


@pytest.mark.parametrize(
    "value, error",
    [
        (1, TypeError),
        ("1 degree", TypeError),
        (ureg.Quantity(1, "dimensionless"), ValueError),
        (Q_(1, "sr"), ValueError),
        (Q_([1], "degree"), TypeError),
        (Q_(np.True_, "degree"), TypeError),
        (Q_(1 + 2j, "degree"), TypeError),
        (Q_(float("nan"), "degree"), ValueError),
        (Q_(181, "degree"), ValueError),
        (Q_(0, "degree"), ValueError),
    ],
)
def test_acceptance_rejects_invalid_semantic_angles(value, error):
    with pytest.raises(error, match="AngularAcceptance.polar"):
        AngularAcceptance(polar=value, azimuthal_deg=20)
    with pytest.raises(error, match="AngularAcceptance.polar"):
        AngularAcceptance(polar_deg=2, azimuthal_deg=20).with_polar(value)


@pytest.mark.parametrize(
    "value,error",
    [
        (1, TypeError),
        (Q_(0.1, "degree"), ValueError),
        (ureg.Quantity(1, "dimensionless"), ValueError),
        (Q_([1], "sr"), TypeError),
        (Q_(0, "sr"), ValueError),
        (Q_(13, "sr"), ValueError),
    ],
)
def test_acceptance_solid_angle_requires_angular_measure(value, error):
    with pytest.raises(error, match="AngularAcceptance.solid_angle"):
        AngularAcceptance(polar_deg=5, solid_angle=value)


@pytest.mark.parametrize(
    "value,error",
    [
        (0, TypeError),
        ("0 rad", TypeError),
        (ureg.Quantity(1, "dimensionless"), ValueError),
        (Q_(1, "sr"), ValueError),
        (Q_([1], "radian"), TypeError),
        (Q_(np.True_, "degree"), TypeError),
        (Q_(1 + 1j, "rad"), TypeError),
        (Q_(float("inf"), "degree"), ValueError),
        (Q_(181, "degree"), ValueError),
    ],
)
def test_detector_rejects_invalid_semantic_angles(value, error):
    with pytest.raises(error, match="Detector.observation_angle"):
        Detector(observation_angle=value)
    with pytest.raises(error, match="Detector.observation_angle"):
        Detector().with_observation_angle(value)


def test_detector_rejects_dual_names_and_preserves_numeric_defaults():
    with pytest.raises(TypeError, match="both observation_angle_deg and observation_angle"):
        Detector(90, observation_angle=Q_(90, "degree"))
    with pytest.raises(TypeError, match="both observation_angle_deg and observation_angle"):
        Detector(observation_angle_deg=90, observation_angle=Q_(90, "degree"))
    with pytest.raises(TypeError, match="both observation_angle_deg and observation_angle"):
        replace(Detector(), observation_angle=Q_(90, "degree"))


def test_numeric_detector_construction_does_not_import_pint():
    command = (
        "import builtins,sys; old=builtins.__import__; "
        "builtins.__import__=lambda name,*a,**kw: "
        "(_ for _ in ()).throw(AssertionError('imported Pint')) "
        "if name.split('.')[0]=='pint' else old(name,*a,**kw); "
        "from pyrite.detectors import Detector,AngularAcceptance; "
        "from dataclasses import replace; "
        "a=AngularAcceptance(polar_deg=2,azimuthal_deg=20); "
        "d=Detector(90,acceptance=a); "
        "assert replace(a)==a and replace(d)==d and 'pint' not in sys.modules"
    )
    done = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("text", ["1e400", "1e-400"])
def test_extreme_precision_angles_reject_nonrepresentable_float(text):
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("extended exponent range unavailable")
    source = np.longdouble(text)
    assert source > 0 and np.isfinite(source)
    with pytest.raises(ValueError, match="not representable"):
        AngularAcceptance(polar=Q_(source, "degree"), azimuthal_deg=30)
    with pytest.raises(ValueError, match="not representable"):
        Detector(observation_angle=Q_(source, "degree"))
