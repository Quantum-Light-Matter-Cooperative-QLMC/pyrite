"""EPDL2025 subshell photoionization and MF=27 scattering functions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xraydb

from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.materials.photon_differential import (
    X_MAX_INV_ANG,
    _table,
    coherent_form_factor,
    coherent_form_factor_table,
    incoherent_scattering_function,
    incoherent_scattering_function_table,
    photoionization_subshells,
    subshell_photoionization_cross_sections_ang2,
)

REPO = Path(__file__).resolve().parents[2]
_ELEMENTS = tuple(xraydb.atomic_symbol(z) for z in range(1, 101))
_BOHR_ANG = 0.529177210544


@pytest.mark.parametrize(
    "accessor,evaluate",
    [
        (coherent_form_factor_table, coherent_form_factor),
        (incoherent_scattering_function_table, incoherent_scattering_function),
    ],
)
def test_scattering_knot_accessors_preserve_readonly_lin_lin_data(accessor, evaluate):
    for element in _ELEMENTS:
        x, y = accessor(element)
        assert x.ndim == y.ndim == 1 and x.shape == y.shape
        assert x[0] == 0 and x[-1] == X_MAX_INV_ANG
        assert not x.flags.writeable and not y.flags.writeable
        assert np.all(np.diff(x) >= 0)
        positive = np.diff(x) > 0
        midpoints = (x[:-1] + x[1:])[positive] / 2
        np.testing.assert_allclose(
            evaluate(element, midpoints), ((y[:-1] + y[1:]) / 2)[positive], rtol=2e-14
        )


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "release_epdl_table", REPO / "scripts" / "release_epdl_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table_is_pinned_and_covers_every_element() -> None:
    from pyrite.datasets import EPDL_DIFFERENTIAL, require_dataset
    from pyrite.materials.photon_differential import EPDL_DIFFERENTIAL_SHA256

    assert EPDL_DIFFERENTIAL.sha256 == EPDL_DIFFERENTIAL_SHA256
    assert sorted(_table()) == list(range(1, 101))
    with np.load(require_dataset("epdl-differential")) as archive:
        assert archive["tolerance"].item() == 5.0e-4
    for element in _ELEMENTS:
        shells = photoionization_subshells(element)
        designators = [shell.designator for shell in shells]
        assert designators == sorted(set(designators)) and designators[0] == 1
        for shell in shells:
            assert shell.energy_eV[0] == shell.binding_energy_eV
            assert not shell.energy_eV.flags.writeable


def test_subshells_match_eadl_designators_and_binding_energies() -> None:
    """Photoionized vacancies index EADL relaxation without remapping."""
    from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation

    for element in _ELEMENTS:
        epdl = {s.designator: s.binding_energy_eV for s in photoionization_subshells(element)}
        eadl = {
            s.shell_designator: s.binding_energy_eV for s in load_eadl_relaxation(element).subshells
        }
        assert epdl == eadl, element


def test_subshell_sum_reproduces_the_photoelectric_total() -> None:
    """Away from edges both tables carry 5e-4 thinning, so the sum agrees to 1e-3."""
    rng = np.random.default_rng(341)
    energy = np.sort(10.0 ** rng.uniform(2.0, 11.0, 400))
    for element in _ELEMENTS:
        edges = np.array([s.binding_energy_eV for s in photoionization_subshells(element)])
        away = np.min(np.abs(np.log(energy[:, None] / edges[None, :])), axis=1) > 5.0e-3
        _, per_shell = subshell_photoionization_cross_sections_ang2(element, energy[away])
        total = photon_cross_sections_ang2(element, energy[away])["photoelectric"]
        np.testing.assert_allclose(per_shell.sum(axis=0), total, rtol=1.0e-3, err_msg=element)


def test_subshell_thresholds_and_band() -> None:
    designators, sigma = subshell_photoionization_cross_sections_ang2(
        "Fe", [7000.0, 7117.0, 7200.0, 0.5, 2.0e11]
    )
    k = designators.index(1)
    assert sigma[k, 0] == 0.0  # below the K edge
    assert sigma[k, 1] > 0.0 and sigma[k, 2] > 0.0  # right-continuous at the edge
    assert np.all(np.isnan(sigma[:, 3:]))
    assert sigma.shape == (len(designators), 5)


@pytest.mark.parametrize(("element", "edge"), [("Fe", "K"), ("Cu", "K"), ("W", "L3"), ("Pb", "K")])
def test_binding_energies_agree_with_xraydb_edges(element, edge) -> None:
    designator = {"K": 1, "L3": 4}[edge]
    shell = next(s for s in photoionization_subshells(element) if s.designator == designator)
    reference = xraydb.xray_edge(element, edge).energy
    assert shell.binding_energy_eV == pytest.approx(reference, rel=5.0e-3)


def test_hydrogen_matches_the_analytic_1s_form_factor() -> None:
    """``F = (1 + (a0 q / 2)^2)^-2`` and ``S = 1 - F^2``, ``q = 4 pi x``; pins x in 1/Angstrom."""
    x = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0])
    form = (1.0 + (2.0 * np.pi * _BOHR_ANG * x) ** 2) ** -2
    np.testing.assert_allclose(coherent_form_factor("H", x), form, rtol=1.0e-3)
    np.testing.assert_allclose(incoherent_scattering_function("H", x), 1.0 - form**2, rtol=1.0e-3)


@pytest.mark.parametrize("element", ["C", "Si", "Fe", "W", "Pb"])
def test_scattering_function_limits(element) -> None:
    z = _ELEMENTS.index(element) + 1
    form = coherent_form_factor(element, [0.0, 1.0e3])
    incoherent = incoherent_scattering_function(element, [0.0, 1.0e3])
    assert form[0] == pytest.approx(z)
    assert 0.0 <= form[1] < 1.0e-3 * z
    assert incoherent[0] == 0.0
    assert incoherent[1] == pytest.approx(z, rel=1.0e-3)
    assert np.all(np.isnan(coherent_form_factor(element, [-1.0, 2.0 * X_MAX_INV_ANG])))


def test_unknown_element_is_refused() -> None:
    with pytest.raises(KeyError):
        coherent_form_factor("Xx", 0.1)


class _FakeMaterial:
    def __init__(self, z, sections):
        self.za = 1000.0 * z
        self._sections = sections

    def __contains__(self, key):
        return key in self._sections

    def __getitem__(self, key):
        return self._sections[key]


def _tab(x_field, y_field, x, y, **head):
    return {x_field: x, y_field: y, "INT": [2], "NBT": [len(x)], **head}


def _fake_tape(monkeypatch, materials):
    import endf_parserpy

    class FakeTape:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __len__(self):
            return len(materials)

        def __getitem__(self, position):
            return materials[position]

    monkeypatch.setattr(endf_parserpy, "EndfFile", FakeTape)


def _material(z, *, l1_start=10.0):
    e = np.linspace(100.0, 1000.0, 50)
    x = np.concatenate([[0.0], np.geomspace(1.0e-3, 1.0e9, 40)])
    return _FakeMaterial(
        z,
        {
            (23, 535): _tab("Eint", "sigma", [l1_start, l1_start, 50.0], [0.0, 2.0, 1.0], EPE=10.0),
            (23, 534): _tab("Eint", "sigma", np.r_[100.0, e], np.r_[0.0, e**-2.0], EPE=100.0),
            (27, 502): _tab("xint", "H", x, z / (1.0 + x) ** 2),
            (27, 504): _tab("xint", "H", x, z * x / (1.0 + x)),
        },
    )


def test_generator_differential_rows_and_units(monkeypatch) -> None:
    generator = _load_generator()
    monkeypatch.setattr(generator, "Z_MAX", 2)
    _fake_tape(monkeypatch, [_material(1), _material(2)])

    arrays = generator.build_differential_table(Path("fake"), 5.0e-4)

    assert arrays["z"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert arrays["mf"].tolist() == [23, 23, 27, 27] * 2
    assert arrays["mt"].tolist() == [534, 535, 502, 504] * 2
    assert arrays["binding_eV"][:2].tolist() == [100.0, 10.0]
    assert np.all(np.isnan(arrays["binding_eV"][2:4]))
    assert arrays["y"].dtype == np.float32 and arrays["x"].dtype == np.float64
    assert arrays["offsets"][-1] == arrays["x"].size


def test_generator_refuses_a_binding_energy_off_the_first_knot(monkeypatch) -> None:
    generator = _load_generator()
    monkeypatch.setattr(generator, "Z_MAX", 1)
    _fake_tape(monkeypatch, [_material(1, l1_start=11.0)])

    with pytest.raises(ValueError, match="binding energy"):
        generator.build_differential_table(Path("fake"), 5.0e-4)
