"""Optional validation-oracle backend tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cxr_mc.materials.crystal import CRYSTALS, reciprocal_g_vector, structure_factor
from cxr_mc.validation import validation_oracles as vo


class FakeCell:
    def __init__(self, crystal: str):
        lattice = CRYSTALS[crystal]["lattice"]
        self._lp = vo._cxr_lattice_tuple(crystal)
        self._volume = CRYSTALS[crystal]["V_cell"]
        self._lattice = lattice

    def lp(self):
        return self._lp

    def volume(self):
        return self._volume

    def Qmag(self, hkls):
        return np.array([reciprocal_g_vector(hkl, self._lattice)[1] for hkl in hkls])


class FakeScatter:
    def __init__(self):
        self.setup_kwargs = None

    def setup_scatter(self, **kwargs):
        self.setup_kwargs = kwargs

    def structure_factor(self, hkl=None, **kwargs):
        return np.array([structure_factor("silicon", one_hkl, 8000.0)[0] for one_hkl in hkl])


class FakeCrystal:
    def __init__(self, crystal: str = "silicon"):
        self.Cell = FakeCell(crystal)
        self.Scatter = FakeScatter()
        self.cell_args = None
        self.atom_kwargs = None

    def new_cell(self, lattice):
        self.cell_args = tuple(lattice)

    def new_atoms(self, **kwargs):
        self.atom_kwargs = kwargs


class OffsetGeometryCrystal(FakeCrystal):
    def __init__(self):
        super().__init__("silicon")
        original_qmag = self.Cell.Qmag
        self.Cell.Qmag = lambda hkls: 1.01 * original_qmag(hkls)


def test_compare_lattice_matches_internal_cell():
    comparison = vo.compare_lattice("silicon", FakeCrystal())

    assert comparison.lattice_delta == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert comparison.volume_delta_ang3 == pytest.approx(0.0)


def test_compare_reflection_geometry_uses_oracle_qmag():
    comparisons = vo.compare_reflection_geometry("silicon", FakeCrystal(), [(1, 1, 1), (2, 2, 0)])

    assert [comparison.hkl for comparison in comparisons] == [(1, 1, 1), (2, 2, 0)]
    assert [comparison.absolute_delta_inv_ang for comparison in comparisons] == pytest.approx(
        [0.0, 0.0]
    )


def test_compare_structure_factor_magnitudes_uses_intensity_safe_quantity():
    oracle = FakeCrystal()

    comparisons = vo.compare_structure_factor_magnitudes(
        "silicon", oracle, [(1, 1, 1), (2, 2, 0)], 8000.0
    )

    assert [comparison.relative_delta for comparison in comparisons] == pytest.approx([0.0, 0.0])
    assert comparisons[0].cxr_abs_f_sq > 0.0
    assert oracle.Scatter.setup_kwargs == {
        "scattering_type": "xray",
        "energy_kev": 8.0,
        "int_hkl": True,
        "use_waaskirf": True,
        "output": False,
    }


def test_build_dans_crystal_from_cxr_transfers_full_basis(monkeypatch):
    created = []

    class ModuleCrystal(FakeCrystal):
        def __init__(self, filename=None):
            super().__init__("silicon")
            self.filename = filename
            created.append(self)

    monkeypatch.setattr(
        vo.importlib,
        "import_module",
        lambda name: SimpleNamespace(Crystal=ModuleCrystal),
    )

    crystal = vo.build_dans_crystal_from_cxr("silicon")

    assert crystal is created[0]
    assert crystal.filename is None
    assert crystal.cell_args == pytest.approx(vo._cxr_lattice_tuple("silicon"))
    assert crystal.atom_kwargs["type"] == [element for element, _ in CRYSTALS["silicon"]["basis"]]
    assert crystal.atom_kwargs["occupancy"] == [1.0] * len(CRYSTALS["silicon"]["basis"])
    assert crystal.atom_kwargs["uiso"] == [0.0] * len(CRYSTALS["silicon"]["basis"])


def test_missing_dans_diffraction_raises_clear_optional_dependency_error(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("Dans_Diffraction")

    monkeypatch.setattr(vo.importlib, "import_module", missing)

    with pytest.raises(vo.DansDiffractionUnavailableError, match="optional 'Dans-Diffraction'"):
        vo.load_dans_crystal_from_cif("missing.cif")


def test_validate_dans_crystal_fails_closed_on_out_of_tolerance_geometry(monkeypatch):
    monkeypatch.setattr(vo, "build_dans_crystal_from_cxr", lambda _crystal: OffsetGeometryCrystal())

    report = vo.validate_dans_crystal(
        "silicon",
        [(1, 1, 1)],
        [8000.0],
        use_henke=False,
    )

    assert not report.passed
    assert report.failures == (
        "silicon (1, 1, 1) reciprocal geometry: 9.900990e-03 > 1.000000e-12",
    )


@pytest.mark.oracle
def test_real_dans_diffraction_backend_stays_within_documented_tolerances():
    pytest.importorskip("Dans_Diffraction")

    cases = {
        "silicon": [(1, 1, 1), (2, 2, 0), (4, 0, 0)],
        "sapphire": [(0, 0, 6), (1, 1, 0), (1, 0, 4)],
        "mote2_product": [(0, 0, 2), (1, 0, 0), (1, 0, 3)],
    }
    reports = [
        vo.validate_dans_crystal(
            crystal,
            hkls,
            [1000.0, 2000.0, 3000.0, 8000.0],
            use_henke=True,
        )
        for crystal, hkls in cases.items()
    ]
    reports.append(
        vo.validate_dans_crystal(
            "sapphire",
            cases["sapphire"],
            [8000.0],
            use_henke=False,
        )
    )

    failures = [failure for report in reports for failure in report.failures]
    assert not failures, "\n".join(failures)
