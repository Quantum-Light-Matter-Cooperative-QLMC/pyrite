"""Explicit material-authoring quantities keep numeric definitions and identities (#755)."""

import subprocess
import sys
import tomllib

import numpy as np
import pint
import pytest

from pyrite import _catalog_layout as layout
from pyrite.materials import build_crystal, build_medium
from pyrite.units import Q_, ureg

SILICON_CIF = layout.bundled_catalog().parent / "cifs" / "silicon.cif"


def _cif(**kwargs):
    return build_crystal(
        "my-si", cif=SILICON_CIF, formula="Si", phase="diamond", surface_hkl=(1, 1, 1), **kwargs
    )


def test_medium_density_quantity_matches_numeric_definition_and_identity():
    numeric = build_medium("silica", formula="SiO2", density_g_cm3=2.2)
    semantic = build_medium("silica", formula="SiO2", density=2.2 * ureg.g / ureg.cm**3)
    assert semantic == numeric
    assert semantic.identity == numeric.identity
    assert semantic.density == Q_(numeric.density_g_cm3, "g/cm**3")
    readback = semantic.density
    readback.ito("kg/m**3")
    assert semantic.density.units == ureg.Unit("g/cm**3")
    # Converted units agree within float conversion; no rounding is added to
    # force identity equality.
    converted = build_medium("silica", formula="SiO2", density=2200 * ureg.kg / ureg.m**3)
    assert converted.density_g_cm3 == pytest.approx(numeric.density_g_cm3, rel=1e-14)
    for (element, value), (expected_element, expected) in zip(
        converted.composition, numeric.composition, strict=True
    ):
        assert element == expected_element
        assert value == pytest.approx(expected, rel=1e-14)


def test_crystal_quantities_match_numeric_record_identity_and_checks():
    numeric = _cif(B_ang2=0.46, mosaic_fwhm_deg=0.4)
    semantic = _cif(
        b_factor=Q_(0.46, "angstrom**2"),
        mosaic_fwhm=Q_(0.4, "degree"),
        density=numeric.density,
    )
    assert semantic == numeric
    assert _cif(b_factor=Q_(0, "angstrom**2")).identity == _cif(B_ang2=0).identity
    converted = _cif(b_factor=Q_(0.0046, "nm**2"), mosaic_fwhm=Q_(0.4, "degree").to("mrad"))
    assert converted.composition == numeric.composition
    record = tomllib.loads(converted.record_toml)
    assert record["B_ang2"] == pytest.approx(0.46, rel=1e-14)
    assert record["mosaic_fwhm_deg"] == pytest.approx(0.4, rel=1e-14)
    with pytest.raises(ValueError, match="density disagrees"):
        build_crystal("bad", crystal="silicon", density=10 * ureg.g / ureg.cm**3)
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_crystal("bad", crystal="silicon", b_factor=Q_(0.5, "angstrom**2"))


def test_external_registry_quantities_are_converted():
    external = pint.UnitRegistry()
    built = build_medium("si", formula="Si", density=external.Quantity(np.array(2.33), "g/cm**3"))
    assert built.density_g_cm3 == pytest.approx(2.33)
    assert built.density._REGISTRY is ureg


INVALID = [
    (2.2, TypeError),
    ("2.2 g/cm**3", TypeError),
    (Q_([2.2], "g/cm**3"), TypeError),
    (Q_(1 + 2j, "g/cm**3"), TypeError),
    (Q_(1, "g"), ValueError),
    (Q_(1, "dimensionless"), ValueError),
    (Q_(-1, "g/cm**3"), ValueError),
    (Q_(0, "g/cm**3"), ValueError),
    (Q_(np.nan, "g/cm**3"), ValueError),
    (Q_(np.longdouble("1e4000"), "g/cm**3"), ValueError),
]


@pytest.mark.parametrize("value, error", INVALID)
def test_medium_density_refuses_invalid_quantities(value, error):
    with pytest.raises(error, match="build_medium.density"):
        build_medium("si", formula="Si", density=value)


@pytest.mark.parametrize(
    "keyword, value, error",
    [
        ("b_factor", Q_(-0.1, "angstrom**2"), ValueError),
        ("b_factor", Q_(0.46, "angstrom"), ValueError),
        ("b_factor", 0.46, TypeError),
        ("mosaic_fwhm", Q_(0, "degree"), ValueError),
        ("mosaic_fwhm", Q_(0.4, "dimensionless"), ValueError),
        ("mosaic_fwhm", Q_(0.4, "steradian"), ValueError),
        ("density", Q_(2.33, "g/cm**2"), ValueError),
    ],
)
def test_crystal_quantities_refuse_invalid_values(keyword, value, error):
    with pytest.raises(error, match=f"build_crystal.{keyword}"):
        _cif(**({"b_factor": Q_(0.46, "angstrom**2")} | {keyword: value}))


def test_numeric_spellings_refuse_quantities_and_both_spellings_conflict():
    with pytest.raises(TypeError, match="density_g_cm3 is numeric-only"):
        build_medium("si", formula="Si", density_g_cm3=Q_(2.33, "dimensionless"))
    with pytest.raises(TypeError, match="B_ang2 is numeric-only"):
        _cif(B_ang2=Q_(0.46, "angstrom**2"))
    with pytest.raises(TypeError, match="density_g_cm3 and density are mutually exclusive"):
        build_medium("si", formula="Si", density_g_cm3=2.33, density=2.33 * ureg.g / ureg.cm**3)
    with pytest.raises(TypeError, match="B_ang2 and b_factor are mutually exclusive"):
        _cif(B_ang2=0.46, b_factor=Q_(0.46, "angstrom**2"))
    with pytest.raises(TypeError, match="requires density_g_cm3 or density"):
        build_medium("si", formula="Si")


def test_numeric_authoring_keeps_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.materials import build_medium; "
            "assert build_medium('si', formula='Si', density_g_cm3=2.33).density_g_cm3 > 0; "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
