"""#755: target thickness quantities must not alter numeric geometry lowering."""

import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pint
import pytest

from pyrite.geometry import Footprint, Layer, Slab, Stack
from pyrite.units import Q_, ureg


def test_slab_scalar_semantic_thickness_preserves_dataclass_identity() -> None:
    footprint = Footprint(5.0, 6.0)
    numeric = Slab("mose2", thickness_ang=20_000.0, footprint=footprint)
    explicit = Slab("mose2", thickness=2 * ureg.um, footprint=footprint)
    assert explicit == numeric
    assert asdict(explicit) == asdict(numeric)
    assert "thickness" not in {item.name for item in fields(explicit)}
    assert explicit.thickness == Q_(20_000.0, "angstrom")
    assert replace(explicit) == numeric
    assert pickle.loads(pickle.dumps(explicit)) == numeric
    assert Slab("mose2", footprint=footprint) == numeric
    assert explicit.with_thickness(500 * ureg.nm) == replace(numeric, thickness_ang=5000.0)
    assert explicit.thickness_ang == 20_000.0


def test_layer_scalar_thickness_keeps_numeric_spec_and_orientation() -> None:
    numeric = Layer("sio2", 5000.0, azimuth_deg=30.0)
    explicit = Layer("sio2", thickness=500 * ureg.nm, azimuth_deg=30.0)
    assert explicit == numeric
    assert asdict(explicit) == asdict(numeric)
    assert "thickness" not in {item.name for item in fields(explicit)}
    assert pickle.loads(pickle.dumps(explicit)) == numeric
    assert explicit.thickness.to("nm").magnitude == pytest.approx(500.0)
    assert explicit.spec(5000) == numeric.spec(5000)
    assert explicit.with_thickness(1 * ureg.um) == replace(numeric, thickness_ang=10_000.0)


def test_quantity_sweeps_lower_in_existing_order_and_preserve_stack_policy() -> None:
    footprint = Footprint(5.0, 5.0)
    slab = Slab("mose2", thickness=Q_([100, 200], "nm"), footprint=footprint)
    assert np.array_equal(slab.thickness_ang, [1000.0, 2000.0])
    assert slab.thickness.shape == (2,)
    assert np.array_equal(slab._film_thickness(), [1000.0, 2000.0])
    numeric = Slab("mose2", thickness_ang=[1000.0, 2000.0], footprint=footprint)
    from pyrite.campaign.geometry import crystal_params

    cp = crystal_params("mose2")

    def lowered(target):
        return [
            point.case_keys()
            for point in target.lower(cp, name_stem="mose2", beam_uvw=cp["beam_uvw"])
        ]

    assert lowered(slab) == lowered(numeric)

    film = Layer("mose2", thickness=Q_([1000.0, 2000.0], "angstrom"))
    substrate = Layer("sio2", thickness=0.5 * ureg.mm)
    stack = Stack(layers=(film, substrate), footprint=footprint)
    assert np.array_equal(stack.layers[0].thickness_ang, [1000, 2000])
    assert substrate.thickness_ang == 5e6
    assert np.array_equal(film.thickness.magnitude, [1000, 2000])
    with pytest.raises(ValueError, match="only the film may sweep thickness_ang"):
        Stack(
            layers=(Layer("mose2", 1000), Layer("sio2", thickness=Q_([1, 2], "um"))),
            footprint=footprint,
        )


def test_external_registry_input_and_readback_arrays_are_detached() -> None:
    external = pint.UnitRegistry()
    source = external.Quantity(np.array([100.0, 200.0]), "nm")
    slab = Slab("mose2", thickness=source, footprint=Footprint(5.0, 5.0))
    layer = Layer("mose2", thickness=source)
    assert slab.thickness._REGISTRY is ureg
    assert layer.thickness._REGISTRY is ureg
    assert slab.thickness.units == ureg.angstrom
    source.magnitude[:] = 999
    slab.thickness.magnitude[:] = -1
    layer.thickness.magnitude[:] = -1
    np.testing.assert_allclose(slab.thickness_ang, [1000, 2000])
    np.testing.assert_allclose(layer.thickness_ang, [1000, 2000])
    updated = slab.with_thickness(Q_([0.3, 0.4], "um"))
    np.testing.assert_allclose(updated.thickness_ang, [3000, 4000])
    np.testing.assert_allclose(slab.thickness_ang, [1000, 2000])


@pytest.mark.parametrize("owner", ["slab", "layer"])
def test_semantic_thickness_requires_pint_and_refuses_alias_collision(owner: str) -> None:
    def create(**kw):
        if owner == "slab":
            return Slab("mose2", footprint=Footprint(5, 5), **kw)
        return Layer("mose2", **kw)

    with pytest.raises(TypeError, match="thickness_ang and thickness"):
        create(thickness_ang=1000, thickness=100 * ureg.nm)
    with pytest.raises(TypeError, match="thickness_ang and thickness"):
        replace(create(thickness_ang=1000), thickness=100 * ureg.nm)
    with pytest.raises(TypeError, match="requires a Pint quantity"):
        create(thickness=1000.0)
    with pytest.raises(TypeError, match="requires a Pint quantity"):
        create(thickness_ang=1000).with_thickness(1000.0)
    if owner == "layer":
        with pytest.raises(TypeError, match="requires thickness_ang or thickness"):
            create()
    else:
        assert create().thickness_ang == 20_000.0


@pytest.mark.parametrize("owner", ["slab", "layer"])
@pytest.mark.parametrize(
    ("quantity", "error"),
    [
        (Q_(1.0, "dimensionless"), ValueError),
        (Q_(2.0, "s"), ValueError),
        (Q_([], "angstrom"), ValueError),
        (Q_(0, "angstrom"), ValueError),
        (Q_(-1, "angstrom"), ValueError),
        (Q_([100.0, np.inf], "angstrom"), ValueError),
        (Q_([[1.0, 2.0]], "angstrom"), TypeError),
        (Q_(np.array(True), "angstrom"), TypeError),
        (Q_(1.0 + 2.0j, "angstrom"), TypeError),
    ],
)
def test_bad_thickness_is_rejected(owner: str, quantity: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="thickness"):
        if owner == "slab":
            Slab("mose2", footprint=Footprint(5, 5), thickness=quantity)
        else:
            Layer("mose2", thickness=quantity)


@pytest.mark.parametrize("owner", ["slab", "layer"])
@pytest.mark.parametrize("source", ["1e400", "1e-400"])
def test_thickness_rejects_stored_float_overflow_underflow(owner: str, source: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble exponent range is not extended")
    magnitude = np.longdouble(source)
    assert np.isfinite(magnitude) and magnitude > 0
    with pytest.raises(ValueError, match="representable"):
        if owner == "slab":
            Slab("mose2", thickness=Q_(magnitude, "angstrom"), footprint=Footprint(5, 5))
        else:
            Layer("mose2", thickness=Q_(magnitude, "angstrom"))


def test_numeric_geometry_constructor_stays_pint_lazy() -> None:
    script = (
        "import sys; from dataclasses import replace; "
        "from pyrite.geometry import Slab, Layer, Footprint; "
        "s=Slab('mose2', footprint=Footprint(5,5)); "
        "l=Layer('sio2',5000); "
        "assert replace(s)==s and replace(l)==l; "
        "assert 'pint' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
