from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_oracle():
    path = Path(__file__).parents[2] / "checks" / "pixel_reconstruction_oracle.py"
    spec = importlib.util.spec_from_file_location("pixel_reconstruction_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle()


def test_reviewed_geometry_presets_are_explicit_and_distinct() -> None:
    assert ORACLE.GEOMETRIES == {
        "baseline": ORACLE.OracleGeometry(60.0, 30.0, 0.0),
        "broken-symmetry": ORACLE.OracleGeometry(60.0, 30.0, 45.0),
        "detector-on-g": ORACLE.OracleGeometry(20.0, 20.0, 0.0),
        "near-pole-not-g-aligned": ORACLE.OracleGeometry(20.0, 30.0, 0.0),
    }


def test_oracle_cli_parses_reproducible_run_settings(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    args = ORACLE._parse_args(
        [
            "--geometry",
            "broken-symmetry",
            "--n-electrons",
            "4000",
            "--n-electrons-brem",
            "2000",
            "--timepix-n-mc",
            "200000",
            "--seeds",
            "1",
            "2",
            "3",
            "--output",
            str(output),
        ]
    )

    assert args.geometry == "broken-symmetry"
    assert args.n_electrons == 4000
    assert args.n_electrons_brem == 2000
    assert args.timepix_n_mc == 200_000
    assert args.seeds == [1, 2, 3]
    assert args.output == output


def test_spectrum_shape_metrics_report_peak_width_and_multiple_features():
    energy = np.linspace(0.0, 20.0, 2001)
    sigma = 0.5
    spectrum = np.exp(-0.5 * ((energy - 6.0) / sigma) ** 2)
    spectrum += 0.5 * np.exp(-0.5 * ((energy - 14.0) / sigma) ** 2)

    metrics = ORACLE._spectrum_shape_metrics(energy, spectrum)

    assert metrics["dominant_peak_eV"] == pytest.approx(6.0, abs=0.01)
    assert metrics["dominant_fwhm_eV"] == pytest.approx(2.35482 * sigma, rel=2e-3)
    assert metrics["significant_peak_count"] == 2
    assert metrics["significant_peak_energies_eV"] == pytest.approx([6.0, 14.0], abs=0.01)


def test_spectrum_shape_metrics_do_not_call_boundary_ramp_a_peak():
    energy = np.linspace(0.0, 20.0, 101)

    metrics = ORACLE._spectrum_shape_metrics(energy, energy)

    assert metrics["dominant_peak_eV"] == 20.0
    assert metrics["significant_peak_count"] == 0
    assert metrics["significant_peak_energies_eV"] == []


def test_lineshape_total_variation_separates_shape_from_flux():
    energy = np.linspace(0.0, 20.0, 2001)
    direct = np.exp(-0.5 * ((energy - 5.0) / 0.5) ** 2)
    same_shape = 7.0 * direct
    disjoint = np.exp(-0.5 * ((energy - 15.0) / 0.5) ** 2)

    assert ORACLE._lineshape_total_variation(energy, direct, same_shape) == pytest.approx(
        0.0, abs=1e-14
    )
    assert ORACLE._lineshape_total_variation(energy, direct, disjoint) == pytest.approx(
        1.0, abs=1e-12
    )


def test_timepix_pair_is_filtered_before_response():
    class RecordingResponse:
        def __init__(self):
            self.inputs = []

        def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
            self.inputs.append(np.asarray(intrinsic_density))
            return intrinsic_density

    response = RecordingResponse()
    energy = np.asarray([1.0, 2.0, 3.0])
    direct = np.asarray([2.0, 4.0, 6.0])
    reconstructed = np.asarray([3.0, 6.0, 9.0])
    transmission = np.asarray([0.5, 0.25, 0.125])

    measured = ORACLE._score_filtered_pair(response, energy, direct, reconstructed, transmission)

    np.testing.assert_allclose(response.inputs[0], direct * transmission)
    np.testing.assert_allclose(response.inputs[1], reconstructed * transmission)
    np.testing.assert_allclose(measured[0], direct * transmission)
    np.testing.assert_allclose(measured[1], reconstructed * transmission)
