"""Regression checks for the marimo trace app's static UI wiring."""

import ast
from pathlib import Path

APP = Path(__file__).parents[2] / "src" / "pyrite" / "apps" / "trace_app.py"


def test_penetration_controls_read_the_active_material_scan() -> None:
    source = APP.read_text()

    assert "if MATERIAL is None:" in source
    assert "def _(CATALOG, MATERIAL, fmt_thickness, mo):" in source
    assert "_scan = CATALOG.material(MATERIAL).scan" in source


def test_crystal_view_hides_reciprocal_vectors_until_switched_on() -> None:
    source = APP.read_text()

    assert '1, 8, value=1, step=1, label="reciprocal vectors"' in source
    assert "crystal_reciprocal_show_ui = mo.ui.switch(value=False" in source
    # Count still drives the overlay, but only while the switch is on; off
    # collapses to zero vectors, which drops the cone trace entirely.
    assert "crystal_reciprocal_ui.value if crystal_reciprocal_show_ui.value else 0" in source


def test_crystal_view_toggles_cell_gridlines_and_axes_independently() -> None:
    source = APP.read_text()

    assert "crystal_grid_ui = mo.ui.switch(value=True" in source
    assert "crystal_axes_ui = mo.ui.switch(value=True" in source
    # One switch owns the WHOLE axis frame (walls, wall rules, ticks, labels,
    # titles), which is what axis-level ``visible`` takes down.
    assert "_fig.update_scenes(" in source
    for axis in ("xaxis", "yaxis", "zaxis"):
        assert f'{axis}={{"visible": crystal_axes_ui.value}}' in source
    # The gridlines switch owns the per-cell edge trace running through the
    # structure -- the depth cue -- not anything on the axis walls.
    assert '_fig.update_traces(visible=crystal_grid_ui.value, selector={"name": "cell"})' in source


def test_crystal_view_defaults_to_an_orthographic_camera() -> None:
    source = APP.read_text()

    assert "crystal_ortho_ui = mo.ui.switch(value=True" in source
    # Projection rides on the one resolved camera (plots.plotly.camera owns the
    # orthographic/perspective mapping), not a separate scene update.
    assert "orthographic=crystal_ortho_ui.value" in source
    assert "_fig.update_scenes(camera=_camera)" in source


def test_both_3d_tabs_drive_an_explicit_camera() -> None:
    source = APP.read_text()

    assert "from pyrite.plots.plotly.camera import (" in source
    for name in ("CAMERA_PRESETS,", "DEFAULT_ANGLES,", "resolve_scene_camera,"):
        assert name in source
    # A dragged camera never reaches Python (marimo drops scene.camera relayout
    # events), so each tab resolves its own explicit one instead.
    assert "camera_controls(DEFAULT_ANGLES)" in source
    assert "camera_controls(VOLUME_CAMERA_ANGLES)" in source
    assert "_volume.update_scenes(camera=penetration_camera)" in source
    # Azimuth orbits ABOUT one axis, so both tabs expose which axis that is --
    # a z-only orbit can never bring the camera over the z pole.
    assert 'mo.ui.dropdown(["z", "x", "y"], value="z", label="orbit axis")' in source
    assert "orbit_axis=crystal_orbit_ui.value" in source
    assert "orbit_axis=penetration_camera_orbit_ui.value" in source


def test_orthographic_crystal_view_zooms_through_the_aspect_ratio() -> None:
    source = APP.read_text()

    # Camera distance cannot zoom an orthographic scene (fixed projection box),
    # so the ortho branch scales the aspect ratio the way plotly's own scroll
    # handler does -- and only "manual" keeps the ratio it is handed.
    assert "if crystal_ortho_ui.value:" in source
    assert 'aspectmode="manual"' in source
    assert "aspectratio=data_aspect_ratio(_fig, crystal_zoom_ui.value)" in source


def test_render_path_uses_the_same_camera_the_trace_tab_shows() -> None:
    source = APP.read_text()

    # The camera is resolved ONCE, upstream of both the lazy tab body (which
    # looks up the cache entry) and the eager cell (which writes it); a
    # mismatch would leave the tab hunting for a file the render never wrote.
    assert source.count("penetration_camera = resolve_scene_camera(") == 1
    assert source.count("        penetration_camera,") == 2  # both cache keys
    assert "camera=penetration_camera," in source
    assert "camera=None," not in source


def test_penetration_view_uses_static_volume_figure_as_primary_track_plot() -> None:
    source = APP.read_text()

    assert "from pyrite.plots.plotly.trajectories import (" in source
    assert "trajectory_volume_data," in source
    assert "trajectory_volume_figure_from_data," in source
    assert "trajectory_volume_animation" not in source
    # Playback controls drive the transport data (Ne/seed/realistic/beam_fwhm);
    # the volume figure itself is a static full reveal.
    assert "_data = trajectory_volume_data(" in source
    for arg in ("Ne=_Ne", "seed=_seed", "realistic=_realistic", "beam_fwhm_mm=_beam_fwhm"):
        assert arg in source
    assert "_volume = trajectory_volume_figure_from_data(" in source
    for arg in ("_nc,", "_data,", "realistic=_realistic,", "reveal_until_fs=None"):
        assert arg in source
    # The 2D cross-section renders eagerly beside the survival chart (no lazy
    # accordion wrapper -- see the rail-free declutter).
    assert "trajectory_chart(_nc, data=_view_data, width=420)" in source
    assert "visible_trajectory_data(" in source
    assert "lateral extent is fitted to the tracks" in source


def test_penetration_view_offers_prerendered_render_button() -> None:
    source = APP.read_text()

    assert "from pyrite.plots.plotly.render import (" in source
    for name in (
        "cached_render_path,",
        "prune_render_cache,",
        "render_cache_key,",
        "render_reveal_animation,",
    ):
        assert name in source
    assert "penetration_render_frames_ui = mo.ui.slider(" in source
    assert "start=24, stop=120, value=60, step=12" in source
    assert 'penetration_render_button_ui = mo.ui.run_button(label="Render animation")' in source
    assert "penetration_speed_ui" not in source
    assert "render_reveal_animation(" in source
    assert "mo.status.progress_bar(" in source
    # Video is served as bytes (file handle, not a local path string) and its
    # display is gated on the cached file existing, not on the transient
    # run_button value -- see the render block's comments in the app.
    assert 'src=_render_path.open("rb")' in source
    # The render itself runs in an EAGER cell outside the lazy tab body:
    # marimo resets run_button.value to False when the click-triggered update
    # completes, and lazy tab content evaluates afterwards, so an in-tab gate
    # on the button value never fires. The eager cell gates on the button and
    # hands the result to the tab via the render-status state.
    assert "mo.stop(not penetration_render_button_ui.value)" in source
    assert "get_penetration_render_status, set_penetration_render_status = mo.state(None)" in source
    assert "set_penetration_render_status((_render_key, None))" in source
    assert "set_penetration_render_status((_render_key, str(_exc)))" in source
    # The tab body stores the case record alongside data so the eager cell
    # can render without redoing transport.
    assert "set_penetration_data((_data_key, _data, _nc))" in source


def test_penetration_tab_wires_groove_control_to_trajectory_sweep() -> None:
    source = APP.read_text()

    # The groove-spacing control exists (default off) ...
    assert "penetration_groove_ui = mo.ui.number(" in source
    assert 'label="Groove spacing (Å, 0 = off)"' in source
    # ... is read in the penetration tab ...
    assert "penetration_groove_ui.value" in source
    # ... and threads into the trajectory sweep only when azim == 180 deg.
    assert "groove_spacing_ang=spacing" in source
    assert "penetration_azim_deg == 180.0" in source
    # the transported dataset cache invalidates on the groove knob
    assert '_nc.get("groove_spacing_ang")' in source


def test_penetration_controls_offer_material_presets_and_uncapped_energy() -> None:
    source = APP.read_text()

    for grid in ("scan.energy_keV", "scan.thickness_ang", "scan.tilt_deg"):
        assert grid in source
    for bound in ("start=1.0", "start=0.001", "stop=10000.0", "stop=89.9"):
        assert bound in source
    assert 'start=1.0, step=1.0, value=_energy_values[0], label="(keV)"' in source
    # The compact control-row layout supplies the "Crystal thickness" label once
    # via the shared row label; the manual widget's own label is just the unit.
    assert 'label="(µm)"' in source
    assert "penetration_thickness_manual_ui.value * 1e4" in source
    for name in (
        "penetration_energy_keV",
        "penetration_thickness_ang",
        "penetration_tilt_deg",
    ):
        assert name in source


def test_thickness_controls_and_context_use_shared_human_units() -> None:
    source = APP.read_text()

    assert "from pyrite._formatting import fmt_thickness" in source
    assert "build_configured_cases(_sweep, settings)" in source
    # The preset-thickness dropdown labels its options in human units.
    assert "fmt_thickness(value)" in source
    assert source.count("fmt_thickness(") >= 2


def test_penetration_ne_defaults_to_50() -> None:
    source = APP.read_text()

    assert "start=25, stop=300, value=50, step=5" in source


def test_penetration_view_surfaces_sampled_beam_diagnostics() -> None:
    source = APP.read_text()

    assert "from pyrite.campaign.beam_metrics import initial_state_metrics" in source
    for key in ("initial_r_ang", "initial_v_hat", "initial_E_keV", "initial_t0_ang"):
        assert key in source
    assert "**Sampled beam**" in source


def test_penetration_control_values_are_read_in_a_downstream_cell() -> None:
    tree = ast.parse(APP.read_text())
    controls_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and {arg.arg for arg in cell.args.args} >= {"CATALOG", "MATERIAL", "fmt_thickness"}
    )

    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "value" for node in ast.walk(controls_cell)
    )
    assert any(
        isinstance(cell, ast.FunctionDef)
        and {arg.arg for arg in cell.args.args}
        >= {
            "penetration_energy_source_ui",
            "penetration_thickness_source_ui",
            "penetration_tilt_source_ui",
        }
        and any(isinstance(node, ast.Attribute) and node.attr == "value" for node in ast.walk(cell))
        for cell in tree.body
    )
