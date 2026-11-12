"""Explicit channeling zone-cap and step lengths retain numeric lowering (#755, #1017)."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite.quantum import Channeling, ChannelingNumerics
from pyrite.quantum._channeling import channeling_fields
from pyrite.units import Q_, ureg


def test_semantic_lengths_match_numeric_fields_records_and_lowering():
    numerics = ChannelingNumerics(
        basis_size=12, step_length=Q_(0.01, "um"), zone_max_length=2 * ureg.nm
    )
    numeric = ChannelingNumerics(12, 16, 100.0, 20.0)
    assert numerics == numeric
    assert asdict(numerics) == asdict(numeric)
    assert pickle.loads(pickle.dumps(numerics)) == numeric
    assert replace(numerics) == numeric
    assert numerics.zone_max_length == Q_(20, "angstrom")
    assert numerics.step_length == Q_(100, "angstrom")
    assert ChannelingNumerics().zone_max_length is None
    assert ChannelingNumerics().step_length == Q_(250, "angstrom")

    def lowered(numerics):
        return channeling_fields(Channeling("planar", plane=(1, 1, 0), numerics=numerics))

    assert lowered(numerics) == lowered(numeric)


def test_replacement_preserves_other_controls_and_can_remove_cap():
    numerics = ChannelingNumerics(basis_size=12, zone_max_ang=20.0)
    assert numerics.with_zone_max_length(5 * ureg.nm) == replace(numerics, zone_max_ang=50.0)
    assert numerics.with_zone_max_length(None).zone_max_ang is None
    assert numerics.with_step_length(1 * ureg.nm) == replace(numerics, step_ang=10.0)
    assert (numerics.zone_max_ang, numerics.step_ang) == (20.0, 250.0)


def test_external_input_and_readback_are_detached():
    external = pint.UnitRegistry()
    length = external.Quantity(np.array(2.0), "nm")
    numerics = ChannelingNumerics(step_length=length)
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
    with pytest.raises(error, match="ChannelingNumerics.zone_max_length"):
        ChannelingNumerics(zone_max_length=value)
    with pytest.raises(error, match="ChannelingNumerics.step_length"):
        ChannelingNumerics().with_step_length(value)


def test_semantic_and_numeric_keywords_conflict_even_at_default():
    with pytest.raises(TypeError, match="zone_max_ang and zone_max_length"):
        ChannelingNumerics(zone_max_ang=None, zone_max_length=2 * ureg.nm)
    with pytest.raises(TypeError, match="step_ang and step_length"):
        ChannelingNumerics(step_ang=250.0, step_length=250 * ureg.angstrom)


def test_numeric_channeling_controls_keep_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.quantum import Channeling, ChannelingNumerics; "
            "assert ChannelingNumerics(step_ang=100, zone_max_ang=20).step_ang == 100; "
            "assert Channeling('planar', plane=(1, 1, 0)).plane == (1, 1, 0); "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
