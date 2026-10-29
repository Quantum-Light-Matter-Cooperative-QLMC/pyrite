"""Explicit quantum zone-cap and step lengths retain numeric lowering (#755)."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite import Numerics, Physics
from pyrite.quantum import QuantumDynamics, QuantumNumerics, QuantumRegion, QuantumSetup
from pyrite.units import Q_, ureg

PLANE = {"kind": "planar", "plane_hkl": (1, 1, 0)}


def test_semantic_lengths_match_numeric_fields_records_and_lowering():
    region = QuantumRegion(**PLANE, zone_max_length=2 * ureg.nm)
    numerics = QuantumNumerics(basis_size=12, step_length=Q_(0.01, "um"))
    numeric_region = QuantumRegion("planar", (1, 1, 0), None, 20.0)
    numeric_numerics = QuantumNumerics("bloch", "density-matrix", 12, 16, 100.0)
    for semantic, numeric in ((region, numeric_region), (numerics, numeric_numerics)):
        assert semantic == numeric
        assert asdict(semantic) == asdict(numeric)
        assert pickle.loads(pickle.dumps(semantic)) == numeric
        assert replace(semantic) == numeric
    assert region.zone_max_length == Q_(20, "angstrom")
    assert numerics.step_length == Q_(100, "angstrom")
    assert QuantumRegion().zone_max_length is None
    assert QuantumNumerics().step_length == Q_(250, "angstrom")

    def lowered(region, numerics):
        setup = QuantumSetup(region, QuantumDynamics(kind="channeling"), numerics)
        return setup.lower(Physics(radiative_model="uncoupled"), Numerics())

    assert lowered(region, numerics) == lowered(numeric_region, numeric_numerics)


def test_replacement_preserves_other_controls_and_can_remove_cap():
    region = QuantumRegion(**PLANE, zone_max_ang=20.0)
    assert region.with_zone_max_length(5 * ureg.nm) == replace(region, zone_max_ang=50.0)
    assert region.with_zone_max_length(None).zone_max_ang is None
    numerics = QuantumNumerics(basis_size=12)
    assert numerics.with_step_length(1 * ureg.nm) == replace(numerics, step_ang=10.0)
    assert (region.zone_max_ang, numerics.step_ang) == (20.0, 250.0)


def test_external_input_and_readback_are_detached():
    external = pint.UnitRegistry()
    length = external.Quantity(np.array(2.0), "nm")
    numerics = QuantumNumerics(step_length=length)
    length.magnitude[...] = 9.0
    returned = numerics.step_length
    returned.ito("nm")
    assert numerics.step_length == Q_(20, "angstrom")
    assert numerics.step_length._REGISTRY is ureg


INVALID = [
    (20.0, TypeError),
    ("2 nm", TypeError),
    (Q_([2.0], "nm"), TypeError),
    (Q_(1 + 2j, "nm"), TypeError),
    (Q_(1, "keV"), ValueError),
    (Q_(1, "dimensionless"), ValueError),
    (Q_(0, "nm"), ValueError),
    (Q_(-1, "nm"), ValueError),
    (Q_(np.inf, "nm"), ValueError),
    (Q_(np.nan, "nm"), ValueError),
    (Q_(np.longdouble("1e4000"), "nm"), ValueError),
]


@pytest.mark.parametrize("value, error", INVALID)
def test_semantic_lengths_refuse_invalid_quantities(value, error):
    with pytest.raises(error, match="QuantumRegion.zone_max_length"):
        QuantumRegion(**PLANE, zone_max_length=value)
    with pytest.raises(error, match="QuantumNumerics.step_length"):
        QuantumNumerics().with_step_length(value)


def test_semantic_and_numeric_keywords_conflict_even_at_default():
    with pytest.raises(TypeError, match="zone_max_ang and zone_max_length"):
        QuantumRegion(**PLANE, zone_max_ang=None, zone_max_length=2 * ureg.nm)
    with pytest.raises(TypeError, match="step_ang and step_length"):
        QuantumNumerics(step_ang=250.0, step_length=250 * ureg.angstrom)


def test_existing_region_and_numerics_rules_apply_to_quantities():
    with pytest.raises(ValueError, match="free quantum region cannot set"):
        QuantumRegion(zone_max_length=2 * ureg.nm)


def test_numeric_quantum_controls_keep_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.quantum import QuantumNumerics, QuantumRegion; "
            "assert QuantumNumerics(step_ang=100).step_ang == 100; "
            "assert QuantumRegion('planar', (1, 1, 0), None, 20).zone_max_ang == 20; "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
