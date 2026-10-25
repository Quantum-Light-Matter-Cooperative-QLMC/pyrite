"""Cached source rendering, bounded refinement and standalone app contract."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.apps.angular_heatmap import MapConfig, calculate_maps, parse_tilts, tilt_panels
from pyrite.apps.angular_maps import heatmap_figure, profile_figure


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
        {"tilt_deg": tilt, "tilt_azim_deg": 0.0, "rep_rate_hz": 5000.0, "bunch_charge_pc": 1.0},
        {"line_window_ceiling_eV": 200.0},
    )


def _maps(*tilts_and_density):
    return tuple({"pxr-cbs": _fake_map(t, d)} for t, d in tilts_and_density)


def test_current_color_and_tilt_changes_do_not_score(monkeypatch):
    from pyrite.apps import angular_heatmap as owner

    calls = []

    def source(*args, **kwargs):
        calls.append(kwargs["components"])
        return {"pxr-cbs": _fake_map(30.0, np.ones((3, 2)))}

    monkeypatch.setattr(owner, "directional_sources", source)
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30.0,), stop_eV=5000.0)
    maps, cached = owner.calculate_maps_reporting_cache(config)
    assert not cached
    assert owner.calculate_maps_reporting_cache(config) == (maps, True)
    for current, log in [(0, True), (2, False), (4, True)]:
        panels = tilt_panels(maps, "pxr-cbs", current_na=current)
        figure = heatmap_figure(panels, value_label="rate", logarithmic=log, periodic=True)
        assert tuple(figure.layout.xaxis.range) == (-180.0, 180.0)
        profile_figure(panels, energy_eV=150, angle_deg=0, value_label="rate")
    assert calls == [("pxr-cbs",)]
    calculate_maps.cache_clear()


def test_unsupported_directions_stay_nan_and_hover_reports_true_values():
    density = np.array([[0.0, 1.0], [np.nan, np.nan], [1e-300, 2.0]])
    panels = tilt_panels(_maps((30.0, density)), "pxr-cbs", current_na=1.0)
    trace = heatmap_figure(panels, value_label="rate", periodic=True).data[0]
    z = np.asarray(trace.z, dtype=float)
    # Columns are angles (plus the closing duplicate); only the NaN angle is gray.
    np.testing.assert_array_equal(np.isnan(z).any(axis=0), [False, True, False, False])
    np.testing.assert_allclose(np.asarray(trace.customdata, dtype=float)[:, 0], panels[0].values[0])


def test_side_by_side_labels_only_on_outer_axes():
    maps = _maps((30.0, np.ones((4, 2))), (45.0, 2 * np.ones((4, 2))))
    figure = heatmap_figure(tilt_panels(maps, "pxr-cbs", current_na=1.0), value_label="rate")
    assert figure.layout.yaxis.title.text and not figure.layout.yaxis2.title.text
    assert all(trace.coloraxis == "coloraxis" for trace in figure.data)
    assert figure.layout.coloraxis.colorbar.title.text == "rate"


def test_tilt_lists_and_ranges():
    assert parse_tilts("30, 45") == (30.0, 45.0)
    assert parse_tilts("0:20:5, 45") == (0.0, 5.0, 10.0, 15.0, 20.0, 45.0)
    for bad in ("a", "0:10", "10:0:5", "0:10:0"):
        with pytest.raises(ValueError):
            parse_tilts(bad)


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
        {"tilts_deg": (30.0, 30.0)},
        {"energy_points": 1_000_000, "angle_points": 3600},
        {"components": ()},
        {"components": ("coherent",)},
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
    coarse = calculate_maps(config)[0]["pxr-cbs"]
    fine = calculate_maps(replace(config, angle_points=144, energy_points=2049))[0]["pxr-cbs"]
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
    fewer = calculate_maps(replace(config, electrons=32))[0]["pxr-cbs"]
    assert abs(np.nansum(fewer.density) / np.nansum(coarse.density) - 1) < 0.25


@pytest.mark.slow
def test_initial_notebook_renders_real_bounded_maps():
    from pyrite.apps.angular_heatmap_app import app

    _, definitions = app.run()
    assert definitions["scene_form"].value is None
    assert definitions["config"] == MapConfig()
    assert len(definitions["maps"]) == 2
    first = definitions["maps"][0]["pxr-cbs"]
    assert first.density.shape == (72, 1025)
    assert first.supported.all()  # finite default footprint
    assert first.energy_eV[-1] == first.provenance["line_window_ceiling_eV"]


def test_download_archive_round_trip():
    import io
    import json

    from pyrite.apps.angular_heatmap import maps_npz

    maps = _maps((30.0, np.ones((4, 2))), (45.0, 2 * np.ones((4, 2))))
    archive = np.load(io.BytesIO(maps_npz(MapConfig(), maps, current_na=2.0)))
    np.testing.assert_array_equal(archive["tilt_deg"], [30.0, 45.0])
    np.testing.assert_array_equal(archive["t1_pxr_cbs_density"], 2 * np.ones((4, 2)))
    np.testing.assert_allclose(archive["t0_pxr_cbs_rate"], maps[0]["pxr-cbs"].rate(2.0))
    metadata = json.loads(str(archive["metadata"]))
    assert metadata["current_na"] == 2.0
    assert metadata["config"]["thickness_ang"] == MapConfig().thickness_ang
    assert metadata["units"]["rate"] == "ph/s/eV/sr"
    assert metadata["components"] == {"pxr-cbs": "incoherent-pxr-cbs"}


def test_close_target_tilts_remain_distinct():
    config = replace(MapConfig(), tilts_deg=(30.123456789, 30.2))
    assert config.tilts_deg == (30.123456789, 30.2)

def test_azimuth_grid_cache_and_export(monkeypatch):
    import io

    from pyrite.apps import angular_heatmap as owner

    calls = []

    def source(beam, target, **kwargs):
        calls.append((target.tilt_deg, target.tilt_azim_deg))
        result = _fake_map(target.tilt_deg, np.ones((4, 2)))
        return {
            "pxr-cbs": replace(
                result,
                case={
                    **result.case,
                    "tilt_azim_deg": target.tilt_azim_deg,
                },
            )
        }

    monkeypatch.setattr(owner, "directional_sources", source)
    calculate_maps.cache_clear()
    config = MapConfig(tilts_deg=(30, 45), tilt_azimuths_deg=(0, 60), stop_eV=200)
    maps = calculate_maps(config)
    assert calls == [(30, 0), (30, 60), (45, 0), (45, 60)]
    assert config.result_elements == 4 * config.angle_points * config.energy_points
    assert calculate_maps(config) is maps
    assert len(calls) == 4
    calculate_maps(replace(config, tilt_azimuths_deg=(45,)))
    assert calls[-2:] == [(30, 45), (45, 45)]
    archive = np.load(io.BytesIO(owner.maps_npz(config, maps, current_na=1)))
    np.testing.assert_array_equal(archive["tilt_deg"], [30, 30, 45, 45])
    np.testing.assert_array_equal(archive["tilt_azim_deg"], [0, 60, 0, 60])
    calculate_maps.cache_clear()


def test_transition_requires_explicit_planar_geometry_and_uses_beam_stop():
    with pytest.raises(ValueError, match="infinite"):
        MapConfig(components=("transition",))
    config = MapConfig(components=("transition",), width_mm=None, height_mm=None, energy_points=8)
    assert config.energy_grid()[-1] == config.beam_keV * 1000
    for azimuths in ((), (0, 360), (0, 0.1), (np.nan,)):
        with pytest.raises(ValueError, match="azimuth"):
            MapConfig(tilt_azimuths_deg=azimuths)


def test_notebook_cached_sources_include_single_azimuth_slider(monkeypatch):
    import pyrite
    from pyrite.apps.angular_heatmap_app import app

    monkeypatch.setattr(pyrite, "fetch_tables", lambda: [])
    maps = _maps((30.0, np.ones((4, 2))), (45.0, np.ones((4, 2))))
    _, definitions = app.run(
        defs={
            "cached": True,
            "config": MapConfig(),
            "elapsed": 0.0,
            "maps": maps,
        }
    )
    assert definitions["azimuth_step"].value == 0
    assert definitions["step"].value == 30
    assert "cone_panel" not in definitions
