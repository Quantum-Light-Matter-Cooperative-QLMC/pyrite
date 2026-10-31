"""Quantity read-back on numeric CIF/catalog material data (#816)."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials._schema import CrystalInfo, CrystalSpec, MediumSpec
from pyrite.units import Q_, ureg


def _info():
    return CrystalInfo(
        lattice={"system": "triclinic", "a": 5.0, "b": 4.0, "alpha": 90.0},
        basis=(("Si", np.array((0.0, 0.5, 0.5))),),
        V_cell=20.0,
        composition=(("Si", 0.1), ("O", 0.2)),
        mosaic_fwhm_deg=0.25,
    )


def _crystal(info):
    return CrystalSpec(
        key="test",
        cif=Path("test.cif"),
        validation_id="test",
        formula="SiO2",
        full_name=None,
        phase=None,
        cod_id=None,
        mp_id=None,
        B_ang2=0.46,
        beam_uvw=None,
        hkl_families=(),
        hkl_reason=None,
        layers_per_cell=None,
        info=info,
        surface_hkl=None,
    )


def test_material_readback_has_correct_dimensions_order_and_detachment():
    info = _info()
    spec = _crystal(info)
    lattice = spec.lattice_parameters
    assert lattice["system"] == "triclinic"
    assert set(lattice) == {"system", "a", "b", "alpha"}
    assert lattice["a"].to("nm").magnitude == pytest.approx(0.5)
    assert lattice["alpha"].to("radian").magnitude == pytest.approx(np.pi / 2)
    assert spec.cell_volume.to("nm**3").magnitude == pytest.approx(0.02)
    assert spec.b_factor.to("nm**2").magnitude == pytest.approx(0.0046)
    assert spec.mosaic_fwhm.to("mrad").magnitude == pytest.approx(np.deg2rad(0.25) * 1e3)
    assert [element for element, _ in spec.number_densities] == ["Si", "O"]
    assert spec.number_densities[0][1].to("1/nm**3").magnitude == pytest.approx(100.0)
    assert info.lattice_parameters["b"].magnitude == 4.0
    assert MediumSpec("test", info.composition).number_densities[1][1] == Q_(0.2, "1/angstrom**3")

    lattice["a"].ito("nm")
    lattice["b"] = 99 * ureg.nm
    spec.number_densities[0][1].ito("1/nm**3")
    assert info.lattice["b"] == 4.0
    assert info.lattice["a"] == 5.0
    assert info.composition == (("Si", 0.1), ("O", 0.2))
    assert spec.B_ang2 == 0.46
    assert spec.V_cell == 20.0


def test_optional_mosaic_and_numeric_schema_refusal():
    info = replace(_info(), mosaic_fwhm_deg=None)
    assert info.mosaic_fwhm is None
    assert _crystal(info).mosaic_fwhm is None
    with pytest.raises(TypeError, match="CrystalInfo.V_cell is numeric-only"):
        replace(info, V_cell=Q_(20, "angstrom**3"))
    with pytest.raises(TypeError, match=r"CrystalInfo.lattice\[a\] is numeric-only"):
        replace(info, lattice={"a": Q_(5, "angstrom")})
    with pytest.raises(TypeError, match=r"CrystalInfo.composition\[Si\] is numeric-only"):
        replace(info, composition=(("Si", Q_(0.1, "1/angstrom**3")),))
    with pytest.raises(TypeError, match="CrystalSpec.B_ang2 is numeric-only"):
        replace(_crystal(info), B_ang2=Q_(0.46, "angstrom**2"))
    with pytest.raises(TypeError, match=r"MediumSpec.composition\[Si\] is numeric-only"):
        MediumSpec("test", (("Si", Q_(0.1, "1/angstrom**3")),))


def test_numeric_material_schema_keeps_pint_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pyrite.materials._schema import CrystalInfo, MediumSpec; "
            "CrystalInfo({'a': 5.0}, (), 10.0, (), None); MediumSpec('m', (('Si', 0.1),)); "
            "assert 'pint' not in sys.modules",
        ],
        check=True,
    )
