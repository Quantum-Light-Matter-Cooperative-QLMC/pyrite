"""#755: Footprint quantities retain the original Cartesian geometry sweep."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.geometry import Footprint, Slab
from pyrite.units import Q_, ureg


def test_scalar_semantic_lengths_keep_legacy_fields_and_replacement() -> None:
    legacy = Footprint(5.0, 6.0)
    footprint = Footprint(width=0.5 * ureg.cm, height=6000 * ureg.um)
    assert footprint == legacy
    assert asdict(footprint) == asdict(legacy)
    assert {field.name for field in fields(footprint)} == {"width_mm", "height_mm"}
    assert pickle.loads(pickle.dumps(footprint)) == legacy
    assert replace(footprint) == legacy
    assert footprint.width == Q_(5.0, "mm")
    assert footprint.height.to("um") == Q_(6000, "um")
    assert footprint.with_width(2 * ureg.cm) == Footprint(20.0, 6.0)
    assert footprint.with_height(1 * ureg.cm) == Footprint(5.0, 10.0)
    assert footprint == legacy


def test_sweeps_preserve_order_and_detach_external_input_and_readback() -> None:
    external = pint.UnitRegistry()
    input_width = external.Quantity(np.array([0.4, 0.5]), "cm")
    footprint = Footprint(width=input_width, height=Q_([0.6, 0.7], "cm"))
    assert np.array_equal(footprint.width_mm, [4.0, 5.0])
    assert np.array_equal(footprint.height_mm, [6.0, 7.0])
    assert footprint._pairs() == [(4.0, 6.0), (4.0, 7.0), (5.0, 6.0), (5.0, 7.0)]
    assert Slab("mose2", footprint=footprint)._footprints() == footprint._pairs()
    assert footprint.width.units == ureg.mm
    assert footprint.width._REGISTRY is ureg
    assert footprint.width.shape == (2,)
    assert footprint.height.shape == (2,)
    input_width.magnitude[0] = 100.0
    footprint.width.magnitude[0] = -100.0
    footprint.height.magnitude[0] = -100.0
    assert footprint._pairs() == [(4.0, 6.0), (4.0, 7.0), (5.0, 6.0), (5.0, 7.0)]
    changed = footprint.with_width(Q_([2.0, 3.0], "mm"))
    assert changed._pairs() == [(2.0, 6.0), (2.0, 7.0), (3.0, 6.0), (3.0, 7.0)]
    assert footprint._pairs()[0] == (4.0, 6.0)


def test_legacy_numeric_and_quantity_inputs_still_lower_identically() -> None:
    numeric = Footprint([4.0, 5.0], 6.0)
    quantity = Footprint(width=Q_([0.4, 0.5], "cm"), height=Q_(6.0, "mm"))
    assert numeric._pairs() == quantity._pairs()
    assert asdict(numeric)["width_mm"] == [4.0, 5.0]
    with pytest.raises(TypeError, match="requires a Pint quantity"):
        numeric.with_width([1.0, 2.0])


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (5.0, TypeError),
        ("5 mm", TypeError),
        (np.array([5.0]), TypeError),
        (Q_(1.0, "dimensionless"), ValueError),
        (Q_(1.0, "second"), ValueError),
        (Q_([], "mm"), ValueError),
        (Q_(0.0, "mm"), ValueError),
        (Q_(-1.0, "mm"), ValueError),
        (Q_([1.0, np.inf], "mm"), ValueError),
        (Q_([[1.0, 2.0]], "mm"), TypeError),
        (Q_(np.array(True), "mm"), TypeError),
        (Q_(1.0 + 2.0j, "mm"), TypeError),
    ],
)
def test_invalid_semantic_width_and_replacement(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="Footprint.width"):
        Footprint(width=value, height=Q_(5.0, "mm"))
    with pytest.raises(error, match="Footprint.width"):
        Footprint(5.0, 6.0).with_width(value)


def test_alias_collisions_and_missing_dimensions_are_explicit() -> None:
    with pytest.raises(TypeError, match="width_mm and width"):
        Footprint(5.0, 6.0, width=Q_(5.0, "mm"))
    with pytest.raises(TypeError, match="height_mm and height"):
        Footprint(5.0, 6.0, height=Q_(6.0, "mm"))
    with pytest.raises(TypeError, match="requires height_mm or height"):
        Footprint(width=Q_(5.0, "mm"))
    with pytest.raises(TypeError, match="requires width_mm or width"):
        Footprint(height=Q_(5.0, "mm"))
    with pytest.raises(TypeError, match="height_mm and height"):
        replace(Footprint(5.0, 6.0), height=Q_(7.0, "mm"))


@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_semantic_width_rejects_float_storage_overflow_or_underflow(value: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range than float")
    source = np.longdouble(value)
    assert np.isfinite(source) and source > 0
    with pytest.raises(ValueError, match="Footprint.width.*representable"):
        Footprint(width=Q_(source, "mm"), height=Q_(5.0, "mm"))


def test_importing_numeric_footprint_does_not_import_pint() -> None:
    script = (
        "import sys; from dataclasses import replace; "
        "from pyrite.geometry import Footprint; "
        "a=Footprint(5,6); assert replace(a)==a; "
        "assert 'pint' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
