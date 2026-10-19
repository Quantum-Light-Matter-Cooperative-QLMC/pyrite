"""Cached source rendering, bounded refinement and standalone app contract."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.apps.angular_heatmap import MapConfig, calculate_maps, heatmap_figure


def _fake_map(tilt, density):
    from pyrite.results.directional import DirectionalSpectrum

    angles = np.linspace(-180.0, 180.0, len(density), endpoint=False)
    return DirectionalSpectrum(
        angles,
        np.array([100.0, 200.0]),
        np.zeros((len(angles), 3)),
        np.zeros((len(angles), 3)),
        density,
        np.isfinite(density).all(axis=1),
        {"tilt_deg": tilt, "rep_rate_hz": 5000.0, "bunch_charge_pc": 1.0},
        {"line_window_ceiling_eV": 200.0},
    )


def test_current_color_and_tilt_changes_do_not_score(monkeypatch):
    from pyrite.apps import angular_heatmap as owner

    calls = []

    def source(*args, **kwargs):
        calls.append(1)
        return _fake_map(30.0, np.ones((3, 2)))

    monkeypatch.setattr(owner, "directional_source", source)
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30.0,), stop_eV=5000.0)
    maps = calculate_maps(config)
    assert calculate_maps(config) is maps
    for current, log in [(0, True), (2, False), (4, True)]:
        figure = heatmap_figure(maps, current_na=current, logarithmic=log)
        figure.canvas.draw()
        assert figure.axes[0].get_xlim() == (-180.0, 180.0)
    assert len(calls) == 1
    calculate_maps.cache_clear()


def test_gray_marks_only_unsupported_directions():
    density = np.array([[0.0, 1.0], [np.nan, np.nan], [1e-300, 2.0]])
    figure = heatmap_figure((_fake_map(30.0, density),), logarithmic=True)
    mask = np.ma.getmaskarray(figure.axes[0].collections[0].get_array()).reshape(2, 4)
    # Columns are angles (plus the closing duplicate); only the NaN row is gray.
    np.testing.assert_array_equal(mask.any(axis=0), [False, True, False, False])


def test_side_by_side_labels_only_on_outer_axes():
    maps = (_fake_map(30.0, np.ones((4, 2))), _fake_map(45.0, 2 * np.ones((4, 2))))
    figure = heatmap_figure(maps)
    figure.canvas.draw()
    left, right, colorbar = figure.axes
    assert left.get_ylabel() and not right.get_ylabel()
    assert not any(t.get_visible() and t.get_text() for t in right.get_yticklabels())
    assert colorbar.get_ylabel() and len(figure.axes) == 3
    single = heatmap_figure(maps[:1])
    assert single.axes[0].get_ylabel() and single.axes[1].get_ylabel()
    assert single.get_supxlabel()


def test_heavy_configuration_is_not_capped():
    MapConfig(electrons=100_000, angle_points=3600, energy_points=4097, thickness_ang=1e6)
    MapConfig(tilts_deg=tuple(range(5, 85, 5)))


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
        {"electrons": 0},
        {"thickness_ang": 0},
        {"width_mm": None},
        {"width_mm": 0.0},
        {"energy_points": 1},
        {"start_eV": 1},
        {"stop_eV": 40_000.0},
        {"tilts_deg": ()},
        {"tilts_deg": (30.0, 30.1)},
        {"energy_points": 1_000_000, "angle_points": 3600},
    ],
)
def test_physical_and_memory_validation(updates):
    with pytest.raises(ValueError):
        replace(MapConfig(), **updates)


@pytest.mark.slow
def test_default_grid_refinement_and_history_normalization():
    """Default-example grid convergence check, not a general grid certificate."""
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30.0,))
    coarse = calculate_maps(config)[0]
    fine = calculate_maps(replace(config, angle_points=144, energy_points=2049))[0]
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
    # Per-incident-electron normalization: a 4x history change moves the
    # plane-integrated yield only by statistics (seed spread ~8% at 128).
    fewer = calculate_maps(replace(config, electrons=32))[0]
    assert abs(np.nansum(fewer.density) / np.nansum(coarse.density) - 1) < 0.25


@pytest.mark.slow
def test_initial_notebook_renders_real_bounded_maps():
    from pyrite.apps.angular_heatmap_app import app

    _, definitions = app.run()
    assert definitions["scene_form"].value is None
    assert definitions["config"] == MapConfig()
    assert len(definitions["maps"]) == 2
    assert definitions["maps"][0].density.shape == (72, 1025)
    assert definitions["maps"][0].supported.all()  # finite default footprint
    assert (
        definitions["maps"][0].energy_eV[-1]
        == definitions["maps"][0].provenance["line_window_ceiling_eV"]
    )
