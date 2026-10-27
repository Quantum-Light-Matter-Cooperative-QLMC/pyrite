"""Semantic bunch timing stays numeric through beam lowering and persistence."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.beams import Beam, BunchLength, Offsets, Spot
from pyrite.units import Q_, ureg


def test_duration_quantity_preserves_numeric_contract() -> None:
    timing = BunchLength(rms_duration=Q_(250, "fs"))
    legacy = BunchLength(250.0, "gaussian")
    assert timing == legacy
    assert asdict(timing) == {"rms_fs": 250.0, "shape": "gaussian"}
    assert {item.name for item in fields(timing)} == {"rms_fs", "shape"}
    assert timing.rms_duration.units == ureg.fs
    assert timing.rms_duration.to("ps").magnitude == pytest.approx(0.25, rel=1e-15)
    assert timing.with_rms_duration(Q_(500, "fs")) == replace(legacy, rms_fs=500.0)
    assert replace(timing) == timing
    assert timing.rms_fs == 250.0


def test_duration_scalar_input_and_quantity_readback_are_independent() -> None:
    source = Q_(np.array(250.0), "fs")
    timing = BunchLength(rms_duration=source)
    source.magnitude[...] = 999
    readback = timing.rms_duration
    readback.ito("ps")
    assert timing.rms_fs == 250.0
    assert timing.rms_duration.units == ureg.fs
    assert isinstance(timing.rms_fs, float)


def test_arrival_offsets_signed_vector_and_readback_ownership() -> None:
    source = Q_(np.array([0.0, -125.0, 250.0]), "fs")
    timing = Offsets(arrival_offsets=source)
    legacy = Offsets((0.0, -125.0, 250.0))
    assert timing == legacy
    assert asdict(timing) == asdict(legacy)
    assert {item.name for item in fields(timing)} == {"offsets_fs", "shape"}
    assert timing.arrival_offsets.units == ureg.fs
    np.testing.assert_allclose(
        timing.arrival_offsets.to("ps").magnitude, [0, -0.125, 0.25], rtol=1e-15, atol=0
    )
    source.magnitude[:] = 999
    timing.arrival_offsets.magnitude[:] = 999
    assert timing == legacy
    assert timing.with_arrival_offsets(Q_([-1, 2], "fs")) == Offsets((-1, 2))
    assert replace(timing, shape="uniform").shape == "uniform"


@pytest.mark.parametrize(
    "quantity,numeric",
    [
        (BunchLength(rms_duration=Q_(250, "fs")), BunchLength(250)),
        (Offsets(arrival_offsets=Q_([-125, 250], "fs")), Offsets((-125, 250))),
    ],
)
def test_timing_beam_lowering_and_pickle_stay_numeric(quantity, numeric) -> None:
    beam = Beam(60, transverse=Spot(), longitudinal=quantity)
    legacy = replace(beam, longitudinal=numeric)
    assert beam.to_spec() == legacy.to_spec()
    assert Beam.from_spec(beam.to_spec()) == legacy
    assert pickle.loads(pickle.dumps(beam)) == legacy
    assert asdict(beam) == asdict(legacy)


def test_external_registry_normalizes_time_and_readback_uses_shared_registry() -> None:
    external = pint.UnitRegistry()
    external.define("timing_tick = 125 * femtosecond")
    duration = BunchLength(rms_duration=external.Quantity(2, "timing_tick"))
    offsets = Offsets(arrival_offsets=external.Quantity([-1, 0, 2], "timing_tick"))
    assert duration.rms_duration == Q_(250, "fs")
    np.testing.assert_array_equal(offsets.arrival_offsets.magnitude, [-125, 0, 250])
    assert offsets.arrival_offsets.units == ureg.fs


def test_alternate_time_units_keep_conversion_roundoff_and_legacy_lowering() -> None:
    duration = BunchLength(rms_duration=Q_(0.25, "ps"))
    assert duration.rms_fs == pytest.approx(250, rel=1e-15)
    assert duration == BunchLength(Q_(0.25, "ps"))
    assert duration.with_rms_duration(Q_(0.5, "ps")).rms_fs == pytest.approx(500, rel=1e-15)
    offsets = Offsets(arrival_offsets=Q_([-0.125, 0.25], "ps"))
    np.testing.assert_allclose(offsets.offsets_fs, [-125, 250], rtol=1e-15, atol=0)
    assert offsets == Offsets(Q_([-0.125, 0.25], "ps"))
    for timing in (duration, offsets):
        beam = Beam(60, transverse=Spot(), longitudinal=timing)
        assert Beam.from_spec(beam.to_spec()) == beam


@pytest.mark.parametrize(
    "owner,legacy,semantic,value",
    [
        (BunchLength, "rms_fs", "rms_duration", Q_(250, "fs")),
        (Offsets, "offsets_fs", "arrival_offsets", Q_([0, 250], "fs")),
    ],
)
def test_timing_dual_arguments_and_dataclass_replace_refused(
    owner, legacy, semantic, value
) -> None:
    numeric = value.magnitude
    with pytest.raises(TypeError, match=f"{legacy}.*{semantic}"):
        owner(**{legacy: numeric, semantic: value})
    with pytest.raises(TypeError, match=f"{legacy}.*{semantic}"):
        owner(numeric, **{semantic: value})
    with pytest.raises(TypeError, match=f"{legacy}.*{semantic}"):
        replace(owner(numeric), **{semantic: value})
    with pytest.raises(TypeError, match=legacy):
        owner()


@pytest.mark.parametrize(
    "value,error",
    [
        (2, TypeError),
        ("2 fs", TypeError),
        ([Q_(2, "fs")], TypeError),
        (Q_(2, "mm"), ValueError),
        (Q_(2, "dimensionless"), ValueError),
        (Q_(np.array(True), "fs"), TypeError),
        (Q_(2j, "fs"), TypeError),
        (Q_(np.nan, "fs"), ValueError),
        (Q_(np.inf, "fs"), ValueError),
        (Q_(0, "fs"), ValueError),
        (Q_(-1, "fs"), ValueError),
        (Q_([1], "fs"), TypeError),
    ],
)
def test_duration_invalid_inputs_refused(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="rms_duration"):
        BunchLength(rms_duration=value)
    with pytest.raises(error, match="rms_duration"):
        BunchLength(1).with_rms_duration(value)


@pytest.mark.parametrize(
    "value,error",
    [
        ([0, 1], TypeError),
        ("1 fs", TypeError),
        (Q_([0, 1], "mm"), ValueError),
        (Q_([0, 1], "dimensionless"), ValueError),
        (Q_([True], "fs"), TypeError),
        (Q_([1j], "fs"), TypeError),
        (Q_([np.nan], "fs"), ValueError),
        (Q_([np.inf], "fs"), ValueError),
        (Q_([], "fs"), ValueError),
        (Q_(1, "fs"), TypeError),
        (Q_([[0, 1]], "fs"), TypeError),
    ],
)
def test_offsets_invalid_inputs_refused(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="arrival_offsets"):
        Offsets(arrival_offsets=value)
    with pytest.raises(error, match="arrival_offsets"):
        Offsets((0,)).with_arrival_offsets(value)


@pytest.mark.parametrize("magnitude", ["1e400", "1e-400", "-1e-400"])
def test_offsets_must_be_representable_in_numeric_storage(magnitude: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range than Python float")
    quantity = Q_(np.array([magnitude], dtype=np.longdouble), "fs")
    with pytest.raises(ValueError, match="arrival_offsets.*representable"):
        Offsets(arrival_offsets=quantity)


@pytest.mark.parametrize("magnitude", ["1e400", "1e-400"])
def test_duration_must_be_representable_in_numeric_storage(magnitude: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range than Python float")
    quantity = Q_(np.longdouble(magnitude), "fs")
    with pytest.raises(ValueError, match="rms_duration.*representable"):
        BunchLength(rms_duration=quantity)
    with pytest.raises(ValueError, match="rms_duration.*representable"):
        BunchLength(1).with_rms_duration(quantity)


def test_numeric_timing_construction_and_loading_stay_pint_lazy() -> None:
    code = """
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] == 'pint':
        raise AssertionError('Pint imported on numeric path')
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from dataclasses import replace
from pyrite.beams import Beam, BunchLength, Offsets, Spot
for timing in (BunchLength(250), Offsets((-125, 250))):
    beam = Beam(60, transverse=Spot(), longitudinal=replace(timing))
    assert Beam.from_spec(beam.to_spec()) == beam
assert 'pint' not in sys.modules
"""
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
