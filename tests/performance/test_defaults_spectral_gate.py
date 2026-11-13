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
