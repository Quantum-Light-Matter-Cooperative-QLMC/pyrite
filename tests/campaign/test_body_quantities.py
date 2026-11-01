"""#755: shared body quantities retain scalar geometry and numeric identities."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.detectors import PlanarPose
from pyrite.geometry import Filter, RectangularBody, SubstanceRef, Target
from pyrite.units import Q_, ureg

REF = SubstanceRef("crystal", "silicon")


def test_quantity_body_preserves_records_and_both_scene_roles() -> None:
    numeric = RectangularBody(REF, (5, 6), 0.1)
    body = RectangularBody(REF, size=Q_([0.5, 0.6], "cm"), thickness=100 * ureg.um)
    assert body == numeric
    assert asdict(body) == asdict(numeric)
    assert {f.name for f in fields(body)} == {"substance", "size_mm", "thickness_mm"}
    assert pickle.loads(pickle.dumps(body)) == numeric
    assert replace(body) == numeric
    assert body.thickness == Q_(0.1, "mm")
    np.testing.assert_array_equal(body.size.magnitude, [5, 6])
    assert Target(body).to_slab() == Target(numeric).to_slab()
    pose = PlanarPose((0, 0, 50))
    assert Filter(body, pose).to_plate() == Filter(numeric, pose).to_plate()
    assert body.with_size(Q_([7, 8], "mm")) == replace(numeric, size_mm=(7, 8))
    assert body.with_thickness(Q_(0.2, "mm")) == replace(numeric, thickness_mm=0.2)
    assert body == numeric


def test_external_registry_and_array_ownership() -> None:
    external = pint.UnitRegistry()
    size = external.Quantity(np.array([5.0, 6.0]), "mm")
    body = RectangularBody(REF, size=size, thickness=external.Quantity(0.1, "mm"))
    size.magnitude[:] = -1
    body.size.magnitude[:] = -1
    assert body.size_mm == (5, 6)
    assert body.size._REGISTRY is ureg
    assert body.thickness._REGISTRY is ureg
    np.testing.assert_allclose(body.size.to("cm").magnitude, [0.5, 0.6])
    assert body.thickness.to("um").magnitude == 100


@pytest.mark.parametrize("field", ["size", "thickness"])
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (1, TypeError),
        ("1 mm", TypeError),
        ([1, 2], TypeError),
        (Q_(1, "s"), ValueError),
        (Q_(1, "dimensionless"), ValueError),
        (Q_(np.array(True), "mm"), TypeError),
        (Q_(1j, "mm"), TypeError),
        (Q_([[1, 2]], "mm"), TypeError),
        (Q_([], "mm"), TypeError),
    ],
)
def test_wrong_units_types_and_shapes(field: str, value: object, error: type[Exception]) -> None:
    options = dict(size=Q_([5, 6], "mm"), thickness=Q_(0.1, "mm"))
    options[field] = value
    with pytest.raises(error, match=f"RectangularBody.{field}"):
        RectangularBody(REF, **options)
    body = RectangularBody(REF, (5, 6), 0.1)
    with pytest.raises(error, match=f"RectangularBody.{field}"):
        getattr(body, f"with_{field}")(value)


@pytest.mark.parametrize("field", ["size", "thickness"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_nonpositive_and_nonfinite_lengths(field: str, value: float) -> None:
    options = dict(size=Q_([5, 6], "mm"), thickness=Q_(0.1, "mm"))
    options[field] = Q_([5, value] if field == "size" else value, "mm")
    with pytest.raises(ValueError, match=f"RectangularBody.{field}"):
        RectangularBody(REF, **options)


@pytest.mark.parametrize("size", [Q_(1, "mm"), Q_([1], "mm"), Q_([1, 2, 3], "mm")])
def test_size_requires_exactly_two_components(size: object) -> None:
    with pytest.raises(TypeError, match="shape"):
        RectangularBody(REF, size=size, thickness=Q_(1, "mm"))


def test_thickness_cannot_be_a_sweep() -> None:
    with pytest.raises(TypeError, match="shape"):
        RectangularBody(REF, size=Q_([5, 6], "mm"), thickness=Q_([1], "mm"))


@pytest.mark.parametrize("field", ["size", "thickness"])
def test_missing_and_conflicting_aliases(field: str) -> None:
    quantity = Q_([5, 6] if field == "size" else 0.1, "mm")
    body = RectangularBody(REF, (5, 6), 0.1)
    with pytest.raises(TypeError, match=f"{field}_mm and {field}"):
        RectangularBody(REF, (5, 6), 0.1, **{field: quantity})
    with pytest.raises(TypeError, match=f"{field}_mm and {field}"):
        replace(body, **{field: quantity})
    other = {"thickness_mm": 0.1} if field == "size" else {"size_mm": (5, 6)}
    with pytest.raises(TypeError, match=f"requires {field}_mm or {field}"):
        RectangularBody(REF, **other)


@pytest.mark.parametrize("field", ["size", "thickness"])
@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_storage_representability(field: str, value: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range")
    magnitude = np.longdouble(value)
    options = dict(size=Q_([5, 6], "mm"), thickness=Q_(0.1, "mm"))
    options[field] = Q_(np.array([1, magnitude]) if field == "size" else magnitude, "mm")
    with pytest.raises(ValueError, match="representable"):
        RectangularBody(REF, **options)


def test_numeric_body_paths_remain_pint_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from dataclasses import replace; "
                "from pyrite.geometry import RectangularBody, SubstanceRef; "
                "b=RectangularBody(SubstanceRef('crystal','silicon'),(5,6),0.1); "
                "assert replace(b)==b; assert 'pint' not in sys.modules"
            ),
        ],
        check=True,
    )
