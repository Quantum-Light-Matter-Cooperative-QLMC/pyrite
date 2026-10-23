"""Opt-in Bote--Salvat shell-ionization backend for characteristic scoring.

Validation: bote-salvat-ionization-backend
"""

import numpy as np
import pytest

from pyrite import datasets
from pyrite.montecarlo import bote_salvat_ionization as bs
from pyrite.montecarlo.spectrum import characteristic
from pyrite.montecarlo.spectrum.ionization_systematic import ionization_model_disagreement


def _carbon_k_only() -> dict[int, bs.BoteSalvatElement]:
    """A synthetic one-shell Z = 6 fit: ``f(U) = 1`` below the branch switch."""
    return {
        6: bs.BoteSalvatElement(
            6,
            b_e=np.array([1.0]),
            a_nlj=np.array([0.5]),
            g=np.zeros((1, 4)),
            edge_eV=np.array([283.8]),
            a=np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        )
    }


@pytest.fixture
def synthetic(monkeypatch):
    """Route the backend to a synthetic fit; the table cache must not leak it."""
    monkeypatch.setattr(characteristic, "load_bote_salvat_parameters", _carbon_k_only)
    characteristic._load_packaged_characteristic_cross_sections.cache_clear()
    yield _carbon_k_only()[6]
    characteristic._load_packaged_characteristic_cross_sections.cache_clear()


@pytest.fixture(scope="module")
def pinned():
    try:
        return bs.load_bote_salvat_parameters()
    except datasets.DatasetNotFoundError as exc:
        pytest.skip(str(exc))


def test_default_marker_is_unchanged_and_models_never_share_identity():
    default = characteristic.characteristic_model_marker()
    bote = characteristic.characteristic_model_marker(ionization_model="bote-salvat")

    assert characteristic.CHARACTERISTIC_MODEL == default
    assert characteristic.characteristic_model_marker(ionization_model="eedl") == default
    assert "bote-salvat" not in default
    assert bote.startswith(default)
    assert bote.endswith(f"/bote-salvat-ionization-{bs.BOTE_SALVAT_SHA256[:12]}")
    assert characteristic.characteristic_model_marker("elam", "bote-salvat") != bote
    with pytest.raises(ValueError, match="ionization_model"):
        characteristic.characteristic_model_marker(ionization_model="penelope")
    with pytest.raises(ValueError, match="ionization_model"):
        characteristic.load_characteristic_cross_sections("C", ionization_model="penelope")


def test_bote_salvat_dataset_is_opt_in():
    from pyrite.xsgen.install import BARE_FETCH_STEPS, FETCH_CODES

    assert "bote-salvat" not in datasets.DATASETS
    assert datasets.get("bote-salvat") is datasets.BOTE_SALVAT
    assert "bote-salvat" in FETCH_CODES
    assert all(code != "bote-salvat" for code, _projectile in BARE_FETCH_STEPS)
    assert "ships it once" in datasets.fetch_hint("bote-salvat")


def test_default_table_stays_eedl_and_cache_separates_models(synthetic):
    eedl = characteristic.load_characteristic_cross_sections("C")
    bote = characteristic.load_characteristic_cross_sections("C", ionization_model="bote-salvat")

    assert eedl.ionization_model == "eedl"
    assert eedl.ionization_shell_model == ("eedl",)
    assert bote.ionization_model == "bote-salvat"
    assert bote.ionization_shell_model == ("bote-salvat",)
    assert characteristic.load_characteristic_cross_sections("C") is eedl
    assert not np.array_equal(
        eedl.ionization_cross_sections_cm2_by_shell[0],
        bote.ionization_cross_sections_cm2_by_shell[0],
    )
    # Only the primary-vacancy rate changes.
    assert bote.line_labels == eedl.line_labels
    assert bote.line_yield_per_vacancy is eedl.line_yield_per_vacancy
    np.testing.assert_array_equal(bote.shell_binding_energy_eV, eedl.shell_binding_energy_eV)


def test_tabulated_rows_cover_edge_to_eedl_end_and_reproduce_the_fit(synthetic):
    eedl = characteristic.load_characteristic_cross_sections("C")
    bote = characteristic.load_characteristic_cross_sections("C", ionization_model="bote-salvat")
    grid = bote.projectile_energy_eV_by_shell[0]
    values = bote.ionization_cross_sections_cm2_by_shell[0]

    assert grid[0] == 283.8 and values[0] == 0.0
    assert grid[-1] == eedl.projectile_energy_eV_by_shell[0][-1]
    assert not grid.flags.writeable and not values.flags.writeable
    # Low branch with f = 1: sigma = 4 pi a0^2 (U - 1)/U^2, maximal at U = 2.
    four_pi_a0 = 4.0 * np.pi * 5.291772108e-9**2
    assert np.isclose(np.interp(2.0 * 283.8, grid, values), four_pi_a0 / 4.0, rtol=1e-12, atol=0.0)
    below = characteristic._interpolate_shell_cross_sections(bote, np.array([0.28, 0.2838]))
    assert np.all(np.asarray(below) == 0.0)


def test_spectrum_scales_with_the_selected_cross_section(monkeypatch, synthetic):
    monkeypatch.setattr(
        characteristic,
        "_mu_total_inv_ang",
        lambda _composition, energy: np.zeros_like(np.asarray(energy, dtype=float)),
    )
    segments = {
        "r_mid": np.array([[0.0, 0.0, 50.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([100.0]),
        "E_keV": np.array([30.0]),
        "elec_id": np.array([0]),
        "layer": np.array([0]),
        "Ne": 1,
        "thickness_ang": 100.0,
    }
    energy = np.arange(250.0, 291.0, 1.0)
    spectra = {
        model: np.asarray(
            characteristic.mc_characteristic_spectrum(
                segments, energy, composition=[("C", 0.1)], ionization_model=model
            )
        )
        for model in ("eedl", "bote-salvat")
    }
    tables = {
        model: characteristic.load_characteristic_cross_sections("C", ionization_model=model)
        for model in spectra
    }
    sigma = {
        model: np.interp(
            30_000.0,
            table.projectile_energy_eV_by_shell[0],
            table.ionization_cross_sections_cm2_by_shell[0],
        )
        for model, table in tables.items()
    }

    np.testing.assert_allclose(
        spectra["bote-salvat"], spectra["eedl"] * sigma["bote-salvat"] / sigma["eedl"], rtol=1e-6
    )


def test_disagreement_diagnostic_matches_the_cross_section_ratio(synthetic):
    result = ionization_model_disagreement("C", [200.0, 30_000.0])
    eedl = characteristic.load_characteristic_cross_sections("C")
    bote = characteristic.load_characteristic_cross_sections("C", ionization_model="bote-salvat")
    sigma = [
        np.interp(
            30_000.0,
            t.projectile_energy_eV_by_shell[0],
            t.ionization_cross_sections_cm2_by_shell[0],
        )
        for t in (eedl, bote)
    ]

    assert result.line_labels == eedl.line_labels
    assert np.all(np.isnan(result.relative_difference[0]))  # below the K edge
    produced = result.eedl_cm2[1] > 0.0
    np.testing.assert_allclose(
        result.relative_difference[1, produced], sigma[1] / sigma[0] - 1.0, rtol=1e-12
    )
    with pytest.raises(ValueError, match="positive"):
        ionization_model_disagreement("C", [0.0])


def test_vectorized_fit_matches_scalar_and_has_physical_limits(pinned):
    si = pinned[14]
    edge = float(si.edge_eV[0])
    energies = edge * np.array([0.5, 1.0, 1.001, 2.0, 16.0, 100.0, 1.0e6])
    vector = si.cross_sections_cm2("K", energies)

    assert vector[0] == 0.0 and vector[1] == 0.0
    assert np.all(vector[2:] > 0.0)
    assert [si.cross_section_cm2("K", e) for e in energies] == list(vector)
    # Independent scalar transcription of the pinned Si K coefficients
    # (docs/validation/radiation-physics/eedl-shell-ionization-comparison.md).
    np.testing.assert_allclose(
        si.cross_sections_cm2("K", edge * np.array([2.0, 16.0, 100.0])),
        [8.393129394945435e-21, 5.356385964758975e-21, 1.818776286011186e-21],
        rtol=1e-12,
    )


@pytest.mark.parametrize("z", [6, 14, 29, 42, 74, 92])
def test_tabulation_interpolates_the_fit_within_half_a_permille(pinned, z):
    element = pinned[z]
    for shell in element.shells:
        edge = float(element.edge_eV[element.shells.index(shell)])
        grid = bs.bote_salvat_energy_grid(edge, 1.0e11)
        assert grid[0] == edge and grid[-1] == 1.0e11 and np.all(np.diff(grid) > 0.0)
        wide = grid[1:] / grid[:-1] > 1.0 + 1.0e-9  # skip the U = 16 branch pair
        middle = np.sqrt(grid[1:] * grid[:-1])[wide]
        middle = middle[middle > 1.001 * edge]
        exact = element.cross_sections_cm2(shell, middle)
        interpolated = np.interp(middle, grid, element.cross_sections_cm2(shell, grid))

        np.testing.assert_allclose(interpolated, exact, rtol=6e-4, err_msg=f"Z={z} {shell}")


def test_pinned_tungsten_l3_production_runs_above_eedl_near_threshold(pinned):
    # EEDL/Bote-Salvat W L3-family production is 0.47 at 30 keV
    # (eedl-shell-ionization-comparison); the diagnostic carries the same sign.
    characteristic._load_packaged_characteristic_cross_sections.cache_clear()
    result = ionization_model_disagreement("W", [30_000.0])
    line = result.line_labels.index("La1")

    assert 0.5 < result.relative_difference[0, line] < 2.0
