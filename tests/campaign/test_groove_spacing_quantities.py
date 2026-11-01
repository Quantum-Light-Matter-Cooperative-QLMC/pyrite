"""#755: explicit groove periods preserve the numeric entrance-face contract."""

import pickle
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pint
import pytest

from pyrite.geometry import BlazedGrooves, Footprint, Slab
from pyrite.units import Q_, ureg


def test_spacing_preserves_numeric_records_and_lowering() -> None:
    numeric = BlazedGrooves(20_000.0)
    grooves = BlazedGrooves(spacing=2 * ureg.um)
    assert grooves == numeric
    assert asdict(grooves) == {"spacing_ang": 20_000.0}
    assert pickle.loads(pickle.dumps(grooves)) == numeric
    assert replace(grooves) == numeric
    assert grooves.spacing == Q_(20_000, "angstrom")
    assert grooves.with_spacing(1 * ureg.um) == BlazedGrooves(10_000.0)
    assert grooves.spacing_ang == 20_000
    from pyrite.campaign.geometry import crystal_params

    cp = crystal_params("mose2")

    def lower(face):
        slab = Slab(
            "mose2", tilt_deg=30, tilt_azim_deg=180, footprint=Footprint(5, 5), entrance_face=face
        )
        return [
            point.case_keys()
            for point in slab.lower(cp, name_stem="mose2", beam_uvw=cp["beam_uvw"])
        ]

    assert lower(grooves) == lower(numeric)


def test_external_registry_and_detached_scalar_readback() -> None:
    external = pint.UnitRegistry()
    source = external.Quantity(np.array(2.0), "um")
    grooves = BlazedGrooves(spacing=source)
    source.magnitude[...] = 99
    assert grooves.spacing._REGISTRY is ureg
    assert grooves.spacing == Q_(20_000, "angstrom")
    readback = grooves.spacing
    readback *= 2
    assert grooves.spacing_ang == 20_000


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (1, TypeError),
        ("1 um", TypeError),
        (Q_(1, "s"), ValueError),
        (Q_(1, "dimensionless"), ValueError),
        (Q_([1], "um"), TypeError),
        (Q_([], "um"), TypeError),
        (Q_([[1]], "um"), TypeError),
        (Q_(np.array(True), "um"), TypeError),
        (Q_(1j, "um"), TypeError),
        (Q_(0, "um"), ValueError),
        (Q_(-1, "um"), ValueError),
        (Q_(np.inf, "um"), ValueError),
        (Q_(np.nan, "um"), ValueError),
    ],
)
def test_invalid_periods(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="BlazedGrooves.spacing"):
        BlazedGrooves(spacing=value)
    with pytest.raises(error, match="BlazedGrooves.spacing"):
        BlazedGrooves(1).with_spacing(value)


def test_missing_and_conflicting_aliases() -> None:
    with pytest.raises(TypeError, match="requires spacing_ang or spacing"):
        BlazedGrooves()
    with pytest.raises(TypeError, match="spacing_ang and spacing"):
        BlazedGrooves(10_000, spacing=1 * ureg.um)
    with pytest.raises(TypeError, match="spacing_ang and spacing"):
        replace(BlazedGrooves(10_000), spacing=1 * ureg.um)


@pytest.mark.parametrize("value", ["1e400", "1e-400"])
def test_storage_representability(value: str) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range")
    with pytest.raises(ValueError, match="representable"):
        BlazedGrooves(spacing=Q_(np.longdouble(value), "angstrom"))


def test_numeric_construction_remains_pint_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from dataclasses import replace; "
                "from pyrite.geometry import BlazedGrooves; "
                "g=BlazedGrooves(10000); assert replace(g)==g; "
                "assert 'pint' not in sys.modules"
            ),
        ],
        check=True,
    )
