"""Semantic beam repetition rate and bunch charge stay numeric (#755)."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.model import Sweep
from pyrite.units import Q_, ureg

OPTIONS = {"transverse": Spot(), "longitudinal": PointBunch()}


def test_semantic_rate_and_charge_match_numeric_fields_and_lowering() -> None:
    semantic = Beam(60, repetition_rate=5 * ureg.kHz, bunch_charge=Q_(2000, "fC"), **OPTIONS)
    numeric = Beam(60, rep_rate_hz=5000.0, bunch_charge_pc=2.0, **OPTIONS)
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert semantic.to_spec() == numeric.to_spec()
    assert Beam.from_spec(semantic.to_spec()) == numeric
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert {"repetition_rate", "bunch_charge"}.isdisjoint(f.name for f in fields(semantic))
    assert semantic.repetition_rate == Q_(5000, "Hz")
    assert semantic.bunch_charge == Q_(2, "pC")
    assert semantic.repetition_rate.units == ureg.Hz
    assert semantic.bunch_charge.units == ureg.pC


def test_defaults_positional_calls_and_legacy_quantities_are_unchanged() -> None:
    beam = Beam(60, **OPTIONS)
    assert (beam.rep_rate_hz, beam.bunch_charge_pc) == (5000.0, 1.0)
    positional = Beam(60, None, 2000, 2, **OPTIONS)
    assert (positional.rep_rate_hz, positional.bunch_charge_pc) == (2000, 2)
    legacy = Beam(60, rep_rate_hz=2 * ureg.kHz, bunch_charge_pc=3 * ureg.pC, **OPTIONS)
    assert (legacy.rep_rate_hz, legacy.bunch_charge_pc) == (2000.0, 3.0)


def test_replacement_preserves_other_fields_and_readback_is_detached() -> None:
    beam = Beam(60, **OPTIONS)
    changed = beam.with_repetition_rate(1 * ureg.MHz).with_bunch_charge(Q_(0.5, "nC"))
    assert changed.bunch_charge_pc == pytest.approx(500.0, rel=1e-15)
    assert changed == replace(beam, rep_rate_hz=1e6, bunch_charge_pc=changed.bunch_charge_pc)
    assert (beam.rep_rate_hz, beam.bunch_charge_pc) == (5000.0, 1.0)
    readback = changed.repetition_rate
    readback.ito("MHz")
    assert changed.repetition_rate == Q_(1e6, "Hz")
    assert changed.with_bunch_charge(0 * ureg.pC).bunch_charge_pc == 0.0


def test_external_registry_input_converts_to_project_registry() -> None:
    external = pint.UnitRegistry()
    beam = Beam(
        60,
        repetition_rate=external.Quantity(np.array(2.0), "kHz"),
        bunch_charge=external.Quantity(3, "pC"),
        **OPTIONS,
    )
    assert (beam.rep_rate_hz, beam.bunch_charge_pc) == (2000.0, 3.0)
    assert beam.repetition_rate._REGISTRY is ureg


def test_dual_spellings_conflict() -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        Beam(60, rep_rate_hz=5000, repetition_rate=5 * ureg.kHz, **OPTIONS)
    with pytest.raises(TypeError, match="mutually exclusive"):
        Beam(60, bunch_charge_pc=1, bunch_charge=1 * ureg.pC, **OPTIONS)


INVALID = [
    (5000.0, TypeError),
    ("5 kHz", TypeError),
    (Q_([5.0], "kHz"), TypeError),
    (Q_(1 + 2j, "kHz"), TypeError),
    (Q_(1, "keV"), ValueError),
    (Q_(1, "dimensionless"), ValueError),
    (Q_(-1, "kHz"), ValueError),
    (Q_(np.inf, "kHz"), ValueError),
    (Q_(np.nan, "kHz"), ValueError),
]


@pytest.mark.parametrize("value, error", INVALID)
def test_repetition_rate_refuses_invalid_quantities(value, error) -> None:
    with pytest.raises(error, match="Beam.repetition_rate"):
        Beam(60, repetition_rate=value, **OPTIONS)
    with pytest.raises(error, match="Beam.repetition_rate"):
        Beam(60, **OPTIONS).with_repetition_rate(value)


@pytest.mark.parametrize(
    "value, error",
    [
        (1.0, TypeError),
        (Q_(1, "Hz"), ValueError),
        (Q_(-1, "pC"), ValueError),
        (Q_(np.longdouble("1e4000"), "C"), ValueError),
    ],
)
def test_bunch_charge_refuses_invalid_quantities(value, error) -> None:
    with pytest.raises(error, match="Beam.bunch_charge"):
        Beam(60, bunch_charge=value, **OPTIONS)


def test_sweep_paths_delegate_to_semantic_owners() -> None:
    from pyrite import Scene
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    base = Scene(
        beam=Beam(60, **OPTIONS),
        target=Slab("Si", thickness_ang=1e4, footprint=Footprint(5.0, 5.0)),
        detector=Detector(),
    )
    quantity = Sweep(
        base=base,
        axes={"beam.repetition_rate": [2 * ureg.kHz], "beam.bunch_charge": [Q_(3, "pC")]},
    )
    numeric = Sweep(base=base, axes={"beam.rep_rate_hz": [2000.0], "beam.bunch_charge_pc": [3.0]})
    assert [s for _, s in quantity.expand()] == [s for _, s in numeric.expand()]
    with pytest.raises(TypeError, match="requires a Pint quantity"):
        Sweep(base=base, axes={"beam.repetition_rate": [2000.0]})
