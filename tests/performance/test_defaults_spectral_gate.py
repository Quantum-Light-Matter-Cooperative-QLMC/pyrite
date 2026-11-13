"""The pilot compares a fixed represented band across sampling conventions."""

import runpy
from pathlib import Path

import numpy as np
import pytest


def test_constant_density_preserves_band_mass_and_centroid():
    script = Path(__file__).resolve().parents[2] / "checks/defaults_spectral_gate_642.py"
    ns = runpy.run_path(str(script))
    for quadrature in ("node", "bin-mean"):
        for spacing in (0.375, 3.0, 12.0):
            grid = ns["axis"](500, 10000, spacing, quadrature)
            result = ns["metrics"](grid, np.ones(grid.size), quadrature)
            assert result["yield_per_primary"] == pytest.approx(9500, rel=1e-12)
            assert result["centroid_eV"] == pytest.approx(5250, rel=1e-12)
            np.testing.assert_allclose(result["band_masses_100eV"], 100, rtol=1e-12)
            assert result["dominant_fwhm_eV"] is None


def _gate():
    script = Path(__file__).resolve().parents[2] / "checks/defaults_spectral_gate_642.py"
    return runpy.run_path(str(script))


def test_layered_scene_is_a_shell_admitted_two_crystal_stack():
    case = _gate()["layered_case"](8, 642501)
    assert case["crystal"] == "mos2"
    assert case["inelastic_model"] == "shell-soft-hard"
    assert [layer[:2] for layer in case["abs_layers"]] == [(0.0, 2000.0), (2000.0, 202000.0)]
    assert [r["crystal"] for r in case["layer_radiators"]][:1] == ["mos2"]
    assert case["E0_keV"] == 30.0 and case["seed"] == 642501


def test_mosaic_mc_case_moves_catalog_width_to_the_exact_route():
    case = _gate()["mosaic_mc_case"]({"crystal": "hopg", "mosaic_fwhm_rad": 0.01}, 9)
    assert case["mosaic_fwhm_rad"] is None
    assert case["mosaic_mc_fwhm_rad"] == pytest.approx(np.deg2rad(0.8))
    assert case["mosaic_mc_nodes"] == 9
    with pytest.raises(ValueError, match="no catalog mosaic"):
        _gate()["mosaic_mc_case"]({"crystal": "silicon"}, 5)


def test_versus_judges_each_observable_on_its_own_margin():
    gate = _gate()
    reference = dict(
        yield_per_primary=1.0,
        centroid_eV=1000.0,
        peak_density_per_eV=2.0,
        dominant_fwhm_eV=None,
        band_masses_100eV=[1.0, 1.0],
    )
    # Same mass, same centroid, higher peak: shape fails although mass passes.
    candidate = dict(reference, peak_density_per_eV=2.1, band_masses_100eV=[2.0, 2.0])
    out = gate["versus"](candidate, reference)
    assert out["yield_per_primary"] == 0.0 and out["mass_l1"] == 0.0
    assert out["peak_density_per_eV"] == pytest.approx(0.05)
    assert out["dominant_fwhm_eV"] is None
    margin = out["within_margin"]
    assert margin["yield_per_primary"] and margin["mass_l1"]
    assert margin["peak_density_per_eV"] is False and margin["dominant_fwhm_eV"] is None


def test_matched_grid_keeps_policy_spacing_on_the_fixed_band():
    gate = _gate()
    uniform = gate["matched_grid"](
        np.linspace(10, 20000, 101), {"actual_spacing_eV": 0.5}, "node", 3.0
    )
    assert uniform[0] == 500 and uniform[-1] == 10000
    np.testing.assert_allclose(np.diff(uniform), 0.5)
    local = np.r_[np.arange(10, 900, 3.0), np.arange(900, 12000, 0.25)]
    local_record = {"resolution_policy": "resonance-local"}
    clipped = gate["matched_grid"](local, local_record, "bin-mean", 3.0)
    assert clipped[0] == 500 and clipped[-1] == 10000
    np.testing.assert_array_equal(clipped[1:-1], local[(local > 500) & (local < 10000)])
    # A measured edge below the band top continues at the backbone spacing.
    short = local[local <= 3700]
    filled = gate["matched_grid"](short, local_record, "bin-mean", 3.0)
    assert filled[-1] == 10000 and np.diff(filled).max() <= 3.0 + 1e-9
    np.testing.assert_array_equal(filled[1 : 1 + (short > 500).sum()], short[short > 500])


@pytest.mark.parametrize("name", ["sinc", "sinc-bin", "measured", "measured-bin", "measured-local"])
def test_policy_case_attaches_the_production_payload(name):
    gate = _gate()
    sweep = gate["scene_sweep"]("hopg", 30.0, 20000.0)
    frozen = {"Ne": 8, "seed": 642501, "crystal": "hopg"}
    case = gate["policy_case"](frozen, sweep, gate["POLICY_ARMS"][name])
    payload = case["line_grid_policy"]
    selectors = gate["POLICY_ARMS"][name]
    assert payload["bandwidth"]["policy"] == selectors.get("bandwidth", "kinematic-ceiling")
    assert payload["resolution"]["policy"] == selectors.get("resolution", "sinc-nyquist")
    assert case["line_quadrature"] == selectors.get("quadrature", "node")
    assert case["seed"] == 642501 and case["Ne"] == 8


def test_window_mass_integrates_the_node_interpolant_exactly():
    gate = _gate()
    grid = np.array([1.0, 2.0, 4.0])
    density = np.array([0.0, 2.0, 2.0])
    # Linear ramp 0->2 on [1, 2], flat 2 on [2, 4]; zero past the support.
    assert gate["window_mass"](grid, density, 1.5, 3.0) == pytest.approx(0.75 + 2.0)
    assert gate["window_mass"](grid, density, 3.0, 9.0) == pytest.approx(2.0)


def test_continuum_features_flag_unrefined_substrate_edges():
    gate = _gate()
    layered = gate["continuum_features"](gate["layered_case"](8, 642501))
    assert layered["medium"] == "mos2" and layered["tip_eV"] == 30000.0
    refined = {k for k, v in layered["edges"].items() if v["refined"]}
    substrate = {k for k, v in layered["edges"].items() if not v["refined"]}
    assert any(k.startswith("Mo K") for k in refined)
    assert any(k.startswith("Si K (EPDL)") for k in substrate)
    slab = gate["continuum_features"]({"crystal": "hopg", "E0_keV": 30.0})
    assert all(v["refined"] for v in slab["edges"].values())


def test_continuum_versus_keeps_local_failures_visible_under_conserved_yield():
    gate = _gate()
    features = gate["continuum_features"]({"crystal": "hopg", "E0_keV": 30.0})
    grid = gate["continuum_grid"]("hopg", 30000.0, 0.0025, True)
    smooth = 1.0 / grid
    reference = gate["continuum_metrics"](grid, smooth, features)
    same = gate["continuum_versus"](reference, reference)
    assert all(same["within_margin"].values())
    # Move mass across the Si K edge window without changing the total.
    edge = next(v["edge_eV"] for k, v in features["edges"].items() if k.startswith("Si K"))
    bump = np.where((grid > edge * 0.995) & (grid < edge), 1.2, 1.0)
    bump = np.where((grid > edge) & (grid < edge * 1.005), 0.8, bump)
    out = gate["continuum_versus"](
        gate["continuum_metrics"](grid, smooth * bump, features), reference
    )
    assert out["within_margin"]["yield_per_primary"]
    assert out["within_margin"]["edge_jump"] is False


def test_unrefined_continuum_twin_matches_the_production_step():
    gate = _gate()
    refined = gate["continuum_grid"]("silicon", 300000.0, 0.005, True)
    twin = gate["continuum_grid"]("silicon", 300000.0, 0.005, False)
    assert twin[0] == refined[0] and twin[-1] > 300000.0
    np.testing.assert_allclose(np.diff(np.log(twin)), np.diff(np.log(twin))[0], rtol=1e-9)
    assert abs(np.diff(np.log(twin))[0] - 0.005) < 1e-5
    assert refined.size > twin.size  # edge and tip anchors only add nodes


def test_families_case_varies_only_the_reflection_set():
    gate = _gate()
    frozen = {"crystal": "silicon", "hkl_list": [(1, 1, 1)], "Ne": 8, "seed": 1, "E0_keV": 300.0}
    sweep = gate["scene_sweep"]("silicon", 300.0, 1e5)
    small, (start, stop1), _ = gate["families_case"](frozen, sweep, 1)
    large, (_, stop8), _ = gate["families_case"](frozen, sweep, 8)
    assert {k: v for k, v in small.items() if k != "hkl_list"} == {
        k: v for k, v in frozen.items() if k != "hkl_list"
    }
    assert len(small["hkl_list"]) < len(large["hkl_list"])
    assert set(map(tuple, small["hkl_list"])) <= set(map(tuple, large["hkl_list"]))
    assert start < stop1 <= stop8
    hopg, (_, hopg_stop), _ = gate["families_case"](
        {"crystal": "hopg", "E0_keV": 30.0}, gate["scene_sweep"]("hopg", 30.0, 2e4), 3
    )
    assert hopg["hkl_list"] == [(0, 0, 2), (0, 0, -2), (0, 0, 4), (0, 0, -4), (0, 0, 6), (0, 0, -6)]
    assert hopg_stop > 3700.0  # (006) resolves past the catalog (004) stop
    layered = gate["layered_case"](8, 642501)
    one, (_, film_stop), (_, need) = gate["families_case"](
        layered, gate["scene_sweep"]("mos2-on-si", 30.0, None), 1
    )
    assert [len(r["hkl_list"]) for r in one["layer_radiators"]] == [2, 8]
    assert one["abs_layers"] == layered["abs_layers"]
    assert need > film_stop  # Si substrate lines pass the film-only production stop


def test_families_case_reciprocal_cutoff_keeps_every_reflection_inside_it():
    gate = _gate()
    frozen = {"crystal": "silicon", "hkl_list": [(1, 1, 1)], "E0_keV": 300.0, "B_ang2": 0.46}
    sweep = gate["scene_sweep"]("silicon", 300.0, 1e5)
    full8, *_ = gate["families_case"](frozen, sweep, "g8")
    n12, *_ = gate["families_case"](frozen, sweep, 12)
    assert sorted(full8["hkl_list"]) == sorted(n12["hkl_list"])  # n=12 exhausts g<=8
    wider, *_ = gate["families_case"](frozen, sweep, "g10")
    assert set(full8["hkl_list"]) < set(wider["hkl_list"])
    layered = gate["layered_case"](8, 642501)
    lay, *_ = gate["families_case"](layered, gate["scene_sweep"]("mos2-on-si", 30.0, None), "g6")
    assert lay["hkl_list"] == lay["layer_radiators"][0]["hkl_list"]
    assert layered["layer_radiators"][1]["hkl_list"] != lay["layer_radiators"][1]["hkl_list"]
    with pytest.raises(ValueError, match="HOPG"):
        gate["families_case"]({"crystal": "hopg"}, gate["scene_sweep"]("hopg", 30.0, 2e4), "g8")
