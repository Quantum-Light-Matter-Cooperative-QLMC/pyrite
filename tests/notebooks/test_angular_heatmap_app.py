"""Cached source rendering, bounded refinement and standalone app contract."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.apps.angular_heatmap import MapConfig, calculate_maps, heatmap_figure


def test_current_color_and_tilt_changes_do_not_score(monkeypatch):
    from pyrite.apps import angular_heatmap as owner
    from pyrite.results.directional import DirectionalSpectrum

    calls = []

    def source(*args, **kwargs):
        calls.append(1)
        return DirectionalSpectrum(
            np.array([-180.0, 0.0, 90.0]),
            np.array([100.0, 200.0]),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.ones((3, 2)),
            np.ones(3, dtype=bool),
            {"tilt_deg": kwargs.get("tilt_deg", 30), "rep_rate_hz": 5000.0, "bunch_charge_pc": 1.0},
            {},
        )

    monkeypatch.setattr(owner, "directional_source", source)
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30.0,))
    maps = calculate_maps(config)
    assert calculate_maps(config) is maps
    for current, log in [(0, True), (2, False), (4, True)]:
        figure = heatmap_figure(maps, current_na=current, logarithmic=log)
        figure.canvas.draw()
        assert figure.axes[0].get_xlim() == (-180.0, 180.0)
        if current == 0 and log:
            assert np.ma.getmaskarray(figure.axes[0].collections[0].get_array()).all()
    assert len(calls) == 1
    calculate_maps.cache_clear()


def test_display_cells_do_not_feed_calculation():
    path = Path(__file__).parents[2] / "src/pyrite/apps/angular_heatmap_app.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(n, ast.Name) and n.id == "calculate_maps" and isinstance(n.ctx, ast.Load)
            for n in ast.walk(node)
        ):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "calculate_maps"
                for n in ast.walk(node)
            ):
                assert not {"current", "color", "selection"} & {arg.arg for arg in node.args.args}


@pytest.mark.parametrize(
    "updates",
    [
        {"electrons": 257},
        {"thickness_ang": 0},
        {"energy_points": 2050},
        {"angle_points": 1000},
        {"start_eV": 1},
        {"tilts_deg": ()},
        {"tilts_deg": (30.0, 30.1)},
        {"electrons": 256, "angle_points": 720},
    ],
)
def test_bounded_configuration(updates):
    with pytest.raises(ValueError):
        replace(MapConfig(), **updates)


@pytest.mark.slow
def test_default_grid_refinement_and_history_normalization():
    """Thin-film example convergence check, not a general grid certificate."""
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30.0,))
    coarse = calculate_maps(config)[0]
    fine = calculate_maps(replace(config, angle_points=144, energy_points=257))[0]
    shared = coarse.supported & fine.supported[::2]
    np.testing.assert_array_equal(coarse.density[shared], fine.density[::2, ::2][shared])
    a = np.trapezoid(coarse.density[shared], coarse.energy_eV, axis=1)
    b = np.trapezoid(fine.density[::2][shared], fine.energy_eV, axis=1)
    assert np.sum(abs(a - b)) / np.sum(b) < 0.01
    assert abs(np.nanmax(coarse.density) / np.nanmax(fine.density) - 1) < 0.03
    # Diagnostic plane-slice quadrature; unsupported grazing points omitted.
    q = []
    for result in (coarse, fine):
        q.append(
            np.trapezoid(np.nan_to_num(result.density), result.energy_eV, axis=1).sum()
            * 360
            / len(result.angle_deg)
        )
    assert abs(q[0] / q[1] - 1) < 0.02
    more = calculate_maps(replace(config, electrons=16))[0]
    assert np.nansum(abs(more.density - coarse.density)) / np.nansum(coarse.density) < 0.1


@pytest.mark.slow
def test_initial_notebook_renders_real_bounded_maps():
    from pyrite.apps.angular_heatmap_app import app

    _, definitions = app.run()
    assert definitions["scene_form"].value is None
    assert definitions["config"] == MapConfig()
    assert len(definitions["maps"]) == 2
    assert definitions["maps"][0].density.shape == (72, 129)
