"""Bulk-source map coordinates, normalization, threshold and export."""

import io
import json
from dataclasses import replace

import numpy as np
import pytest

from pyrite.apps.angular_cherenkov import (
    CherenkovConfig,
    calculate_cherenkov,
    cherenkov_npz,
)


def test_cone_bins_match_source_and_integrate_to_path_yield():
    config = CherenkovConfig(energy_points=7, angle_bins=36)
    panel, edges, source = calculate_cherenkov(config)
    np.testing.assert_array_equal(
        panel.values, source.angular_histogram(edges).T * config.path_um * 1e-6
    )
    np.testing.assert_allclose(panel.values.sum(axis=0), source.photons_per_eV())
    index = np.flatnonzero(panel.values[:, 0])[0]
    assert edges[index] <= source.theta_rad[0] < edges[index + 1]
    assert panel.energy_eV[0] == config.start_eV
    assert panel.energy_eV[-1] == config.stop_eV
    assert calculate_cherenkov(config) is calculate_cherenkov(config)
    doubled = calculate_cherenkov(replace(config, path_um=2 * config.path_um))[0]
    np.testing.assert_allclose(doubled.values, panel.values * 2)


def test_below_threshold_and_invalid_inputs():
    config = CherenkovConfig(beam_keV=30)
    panel, _, source = calculate_cherenkov(config)
    assert not source.above_threshold.any()
    np.testing.assert_array_equal(panel.values, 0)
    for updates in (
        {"refractive_index": -1},
        {"refractive_index": np.nan},
        {"path_um": 0},
        {"stop_eV": 1},
        {"stop_eV": 300000},
        {"angle_bins": 1},
        {"energy_points": 1},
        {"energy_points": 100000, "angle_bins": 100000},
    ):
        with pytest.raises(ValueError):
            replace(CherenkovConfig(), **updates)


def test_export_preserves_bins_dielectric_and_internal_observable():
    config = CherenkovConfig(energy_points=5, angle_bins=18)
    archive = np.load(io.BytesIO(cherenkov_npz(config)))
    panel, edges, source = calculate_cherenkov(config)
    np.testing.assert_array_equal(archive["angle_edges_rad"], edges)
    np.testing.assert_array_equal(archive["yield_per_electron_per_eV_per_bin"], panel.values)
    np.testing.assert_array_equal(archive["above_threshold"], source.above_threshold)
    metadata = json.loads(str(archive["metadata"]))
    assert metadata["config"]["path_um"] == config.path_um
    assert metadata["dielectric"]["epsilon"] == config.refractive_index**2
    assert "internal" in metadata["observable"]
    assert "not per sr" in metadata["angular_normalization"]


def test_notebook_cherenkov_view_never_fetches_tables_or_transports(monkeypatch):
    from types import SimpleNamespace

    import pyrite
    from pyrite.apps.angular_heatmap_app import app
    from pyrite.montecarlo import runner

    def forbidden(*args, **kwargs):
        raise AssertionError("bulk diagnostic must not fetch X-ray tables or transport")

    monkeypatch.setattr(pyrite, "fetch_tables", forbidden)
    monkeypatch.setattr(runner, "_transport_case", forbidden)
    _, definitions = app.run(
        defs={
            "source_view": SimpleNamespace(value="Cherenkov (internal bulk)"),
        }
    )
    assert definitions["cone_panel"].values.shape == (180, 81)
    assert "maps" not in definitions
