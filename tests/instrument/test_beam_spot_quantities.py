"""Semantic spot widths retain the numeric beam/scene record contract."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.beams import Beam, PointBunch, Spot, TransverseSpot
from pyrite.units import Q_, ureg


def test_round_spot_uses_effective_y_without_changing_numeric_sentinel() -> None:
    spot = Spot(fwhm_x=Q_(200, "micrometer"))
    legacy = Spot(fwhm_x_mm=0.2)
    assert spot == legacy
    assert spot.fwhm_y_mm is None
    assert spot.plane_y_mm == pytest.approx(0.2)
    assert spot.fwhm_x.units == ureg.mm
    assert spot.fwhm_y.units == ureg.mm
    assert spot.fwhm_x.to("micrometer").magnitude == pytest.approx(200)
    assert spot.fwhm_y.to("micrometer").magnitude == pytest.approx(200)
    assert asdict(spot) == {"fwhm_x_mm": 0.2, "fwhm_y_mm": None}
    assert {f.name for f in fields(spot)} == {"fwhm_x_mm", "fwhm_y_mm"}
    assert Spot is TransverseSpot
    assert Spot() == Spot(1.0, None)
    assert replace(spot) == spot


def test_elliptical_spot_unit_conversion_and_independent_axes() -> None:
    source = Q_(np.array(250.0), "micrometer")
    spot = Spot(fwhm_x=source, fwhm_y=Q_(0.5, "mm"))
    legacy = Spot(0.25, 0.5)
    assert spot == legacy
    source.magnitude[...] = 700
    readback = spot.fwhm_x
    readback.ito("meter")
    assert spot == legacy
    assert spot.fwhm_x.units == ureg.mm
    assert spot.fwhm_x_mm == pytest.approx(0.25)
    assert spot.with_fwhm_x(Q_(500, "micrometer")) == Spot(0.5, 0.5)
    assert spot.with_fwhm_y(Q_(250, "micrometer")) == Spot(0.25, 0.25)
    assert spot.with_fwhm_y(None) == Spot(0.25, None)
    assert Spot(fwhm_x_mm=0.25, fwhm_y=Q_(500, "micrometer")) == legacy
    assert Spot(fwhm_x=Q_(250, "micrometer"), fwhm_y_mm=0.5) == legacy


def test_round_spot_replacement_preserves_implicit_roundness() -> None:
    round_spot = Spot(fwhm_x=Q_(0.25, "mm"))
    larger = round_spot.with_fwhm_x(Q_(0.5, "mm"))
    assert larger.fwhm_y_mm is None
    assert larger.fwhm_y.magnitude == pytest.approx(0.5)
    explicit_y = round_spot.with_fwhm_y(Q_(0.4, "mm"))
    assert explicit_y.with_fwhm_x(Q_(0.5, "mm")).plane_y_mm == pytest.approx(0.4)


def test_spot_beam_lowering_pickle_and_dataclass_storage_unchanged() -> None:
    spot = Spot(fwhm_x=Q_(200, "micrometer"), fwhm_y=Q_(400, "micrometer"))
    beam = Beam(60.0, transverse=spot, longitudinal=PointBunch())
    legacy = replace(beam, transverse=Spot(0.2, 0.4))
    assert beam.to_spec() == legacy.to_spec()
    assert Beam.from_spec(beam.to_spec()) == legacy
    assert pickle.loads(pickle.dumps(beam)) == legacy
    assert asdict(beam) == asdict(legacy)


def test_external_registry_spot_widths() -> None:
    external = pint.UnitRegistry()
    external.define("custom_spot_width = 100 * micrometer")
    spot = Spot(fwhm_x=external.Quantity(2, "custom_spot_width"))
    assert spot.fwhm_x_mm == pytest.approx(0.2)
    assert spot.fwhm_x.units == ureg.mm


@pytest.mark.parametrize(
    "axis,legacy,value",
    [
        ("fwhm_x", "fwhm_x_mm", Q_(200, "micrometer")),
        ("fwhm_y", "fwhm_y_mm", Q_(400, "micrometer")),
    ],
)
def test_spot_dual_spelling_and_dataclass_replace_conflicts(axis, legacy, value) -> None:
    with pytest.raises(TypeError, match=f"{legacy}.*{axis}"):
        Spot(**{legacy: 0.2, axis: value})
    with pytest.raises(TypeError, match=f"{legacy}.*{axis}"):
        replace(Spot(), **{axis: value})
    if axis == "fwhm_y":
        with pytest.raises(TypeError, match=f"{legacy}.*{axis}"):
            Spot(fwhm_y_mm=None, fwhm_y=value)
    else:
        with pytest.raises(TypeError, match=f"{legacy}.*{axis}"):
            Spot(0.2, fwhm_x=value)


@pytest.mark.parametrize(
    "invalid,exc",
    [
        (0.2, TypeError),
        ("0.2 mm", TypeError),
        ([Q_(0.2, "mm")], TypeError),
        (Q_([0.2], "mm"), TypeError),
        (Q_(0.2, "eV"), ValueError),
        (Q_(0.2, "dimensionless"), ValueError),
        (Q_(np.array(True), "mm"), TypeError),
        (Q_(0.2j, "mm"), TypeError),
        (Q_(0, "mm"), ValueError),
        (Q_(-0.2, "mm"), ValueError),
        (Q_(np.nan, "mm"), ValueError),
        (Q_(np.inf, "mm"), ValueError),
    ],
)
def test_invalid_spot_quantity_refused(invalid, exc) -> None:
    with pytest.raises(exc):
        Spot(fwhm_x=invalid)
    with pytest.raises(exc):
        Spot(fwhm_y=invalid)
    with pytest.raises(exc):
        Spot().with_fwhm_x(invalid)
    with pytest.raises(exc):
        Spot().with_fwhm_y(invalid)


def test_spot_quantity_overflow_refused() -> None:
    with pytest.raises(ValueError, match="representable"):
        Spot(fwhm_x=Q_(1e308, "kilometer"))


def test_numeric_spot_import_remains_pint_lazy() -> None:
    script = (
        "import sys; "
        "from pyrite.beams import Spot; "
        "s=Spot(0.2); assert s.plane_y_mm == 0.2; "
        "assert 'pint' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
