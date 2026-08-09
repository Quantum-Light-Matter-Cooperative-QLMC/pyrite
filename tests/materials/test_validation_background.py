"""Regression tests for external-background validation and subtraction."""

from pathlib import Path

import numpy as np
import pytest

from cxr_mc.montecarlo import load_external_brem
from cxr_mc.validation.validation_background import (
    compare_external_background,
    fit_external_background,
    subtract_external_background,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cxr_mc"
    / "apps"
    / "reference_data"
    / "external_brem"
    / "v1"
)
_BREM = _FIXTURES / "zhai_fig3b_25kev_1mm_brem.csv"
_EXPERIMENT = _FIXTURES / "zhai_fig3b_25kev_1mm_experiment.csv"


def test_scale_fit_and_subtraction_recover_synthetic_signal(tmp_path):
    external = tmp_path / "background.csv"
    external.write_text("energy_eV,intensity\n100,1\n200,2\n300,3\n400,4\n")
    energy = np.array([100.0, 200.0, 300.0, 400.0])
    signal = np.array([0.0, 0.0, 5.0, 0.0])
    observed = 2.5 * np.array([1.0, 2.0, 3.0, 4.0]) + signal
    sigma = np.array([1.0, 2.0, 1.0, 2.0])
    sidebands = np.array([True, True, False, True])

    residual, fitted, fit = subtract_external_background(
        energy,
        observed,
        external,
        sigma=sigma,
        fit_mask=sidebands,
    )

    assert fit.scale == pytest.approx(2.5)
    assert fit.scale_std == pytest.approx(1.0 / np.sqrt(6.0))
    assert fit.reduced_chi2 == pytest.approx(0.0)
    assert fit.n_points == 3
    assert np.allclose(fitted, 2.5 * np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(residual, signal)


def test_fit_rejects_underdetermined_or_negative_scale(tmp_path):
    external = tmp_path / "background.csv"
    external.write_text("100,1\n200,2\n")
    with pytest.raises(ValueError, match="at least two"):
        fit_external_background([100, 200], [1, 2], external, fit_mask=[True, False])
    with pytest.raises(ValueError, match="negative"):
        fit_external_background([100, 200], [-1, -2], external)


def test_versioned_zhai_fixture_uses_existing_external_loader():
    energy = np.array([754.3068515227599, 900.0, 1146.861703954482])
    loaded = load_external_brem(_BREM, energy)

    assert loaded[0] == pytest.approx(0.7845421448001629)
    assert loaded[-1] == pytest.approx(1.8636400647192963)
    assert loaded[0] < loaded[1] < loaded[-1]


def test_zhai_sideband_fit_leaves_positive_coherent_peak():
    data = np.genfromtxt(_EXPERIMENT, delimiter=",", names=True, comments="#", skip_header=5)
    energy = data["energy_eV"]
    intensity = data["intensity_Phs_per_eV_s_nA"]
    sigma = data["sigma_Phs_per_eV_s_nA"]
    sidebands = (energy < 900.0) | (energy > 1040.0)

    residual, fitted, fit = subtract_external_background(
        energy,
        intensity,
        _BREM,
        sigma=sigma,
        fit_mask=sidebands,
    )

    assert fit.n_points == int(np.count_nonzero(sidebands))
    assert 0.95 < fit.scale < 1.10
    assert energy[np.argmax(residual)] == pytest.approx(957.7937706145138)
    assert residual.max() > 0.8
    assert np.all(fitted > 0.0)


def test_background_comparison_preserves_absolute_normalization():
    data = np.genfromtxt(_BREM, delimiter=",", names=True, comments="#", skip_header=5)
    energy = data["energy_eV"]
    intensity = data["intensity_Phs_per_eV_s_nA"]

    same = compare_external_background(energy, intensity, _BREM)
    doubled = compare_external_background(energy, 2.0 * intensity, _BREM)

    assert same.integrated_ratio == pytest.approx(1.0)
    assert same.normalized_rmse == pytest.approx(0.0)
    assert same.correlation == pytest.approx(1.0)
    assert doubled.integrated_ratio == pytest.approx(2.0)
