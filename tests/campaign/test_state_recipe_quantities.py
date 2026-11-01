"""Semantic transverse-state recipe parameters keep the numeric JSON schema (#755)."""

import numpy as np
import pytest

from pyrite.quantum import TransverseStateSpec
from pyrite.units import Q_, ureg


def test_semantic_recipe_matches_numeric_identity() -> None:
    semantic = TransverseStateSpec.recipe(
        "laguerre_gauss", ell=1, p=0, waist=Q_(8, "nm"), center=Q_([1, -2], "angstrom")
    )
    numeric = TransverseStateSpec.recipe(
        "laguerre_gauss", ell=1, p=0, waist_ang=80, center_ang=(1, -2)
    )
    assert semantic == numeric
    assert semantic.digest == numeric.digest
    assert semantic.to_dict() == numeric.to_dict()
    assert TransverseStateSpec.from_dict(semantic.to_dict()) == numeric


def test_parameter_quantities_readback() -> None:
    probe = TransverseStateSpec.recipe(
        "aperture_probe",
        kinetic_energy=Q_(60, "keV"),
        semiangle=Q_(20, "mrad"),
        aberrations={"C10": 5.0},
    )
    quantities = probe.parameter_quantities
    assert set(quantities) == {"kinetic_energy", "semiangle", "inner_semiangle", "center"}
    assert quantities["kinetic_energy"].units == ureg.eV
    assert quantities["kinetic_energy"].magnitude == pytest.approx(60_000.0, rel=1e-15)
    assert quantities["semiangle"].units == ureg.radian
    assert quantities["semiangle"].magnitude == pytest.approx(0.02, rel=1e-15)
    np.testing.assert_array_equal(quantities["center"].magnitude, [0.0, 0.0])
    quantities["center"].magnitude[:] = 9
    assert dict(probe.parameters)["center_ang"] == (0.0, 0.0)
    bessel = TransverseStateSpec.recipe(
        "bessel_gauss", ell=0, k_radial=Q_(0.1, "1/nm"), waist_ang=50
    )
    assert bessel.parameter_quantities["k_radial"].magnitude == pytest.approx(0.01, rel=1e-14)
    assert bessel.parameter_quantities["k_radial"].units == 1 / ureg.angstrom


def test_with_parameters_revalidates() -> None:
    gaussian = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    wider = gaussian.with_parameters(waist=Q_(0.5, "nm"))
    assert wider == TransverseStateSpec.recipe("gaussian", waist_ang=5.0, dimension=1)
    assert gaussian.with_parameters(waist_ang=5.0) == wider
    probe = TransverseStateSpec.recipe(
        "aperture_probe", kinetic_energy_eV=6e4, semiangle_rad=0.02, aberrations={"C10": 5.0}
    )
    assert probe.with_parameters(semiangle=Q_(10, "mrad")).parameter_quantities[
        "semiangle"
    ].magnitude == pytest.approx(0.01)
    with pytest.raises(ValueError, match="positive"):
        gaussian.with_parameters(waist=Q_(-1, "nm"))
    with pytest.raises(TypeError, match="mutually exclusive"):
        gaussian.with_parameters(waist=Q_(1, "nm"), waist_ang=2.0)


def test_translated_and_sampled_quantities(tmp_path) -> None:
    base = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    assert base.translated(offset=Q_([0.2], "nm")) == base.translated((2.0,))
    assert base.translated(offset=Q_([0.2], "nm")).parameter_quantities["offset"].units == (
        ureg.angstrom
    )
    with pytest.raises(TypeError, match="exactly one"):
        base.translated()
    with pytest.raises(TypeError, match="exactly one"):
        base.translated((1.0,), offset=Q_([1], "nm"))
    path = tmp_path / "samples.npy"
    np.save(path, np.ones((4,)))
    semantic = TransverseStateSpec.from_npy(
        path, spacing=Q_([0.1], "nm"), kinetic_energy=Q_(60, "keV")
    )
    numeric = TransverseStateSpec.from_npy(path, spacing_ang=(1.0,), kinetic_energy_eV=6e4)
    assert semantic == numeric
    assert TransverseStateSpec.from_npy(path, spacing_ang=(1.0,)).parameter_quantities.keys() == {
        "spacing",
        "origin",
    }
    with pytest.raises(TypeError, match="spacing"):
        TransverseStateSpec.from_npy(path)


@pytest.mark.parametrize(
    "keyword, value, error",
    [
        ("waist", 80.0, TypeError),
        ("waist", Q_(80, "eV"), ValueError),
        ("semiangle", Q_(0.02, ""), ValueError),
        ("kinetic_energy", Q_(1, "mm"), ValueError),
    ],
)
def test_semantic_recipe_refusals(keyword, value, error) -> None:
    kind = "aperture_probe" if keyword in {"semiangle", "kinetic_energy"} else "gaussian"
    base = (
        {"kinetic_energy_eV": 6e4, "semiangle_rad": 0.02}
        if kind == "aperture_probe"
        else {"waist_ang": 1.0}
    )
    numeric = {"waist": "waist_ang", "semiangle": "semiangle_rad"}.get(keyword)
    base.pop(numeric or "kinetic_energy_eV", None)
    with pytest.raises(error, match=f"TransverseStateSpec.{keyword}"):
        TransverseStateSpec.recipe(kind, **base, **{keyword: value})
    with pytest.raises(TypeError, match="mutually exclusive"):
        TransverseStateSpec.recipe("gaussian", waist_ang=1.0, waist=Q_(1, "angstrom"))
