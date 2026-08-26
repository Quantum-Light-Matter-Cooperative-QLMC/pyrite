"""Regression checks for the marimo trace app's static UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).parents[2] / "src" / "pyrite" / "apps" / "trace_app.py"


def test_penetration_controls_read_the_active_material_scan() -> None:
    source = APP.read_text()

    assert "if MATERIAL is None:" in source
    assert "def _(CATALOG, MATERIAL, fmt_thickness, mo):" in source
    assert "_scan = CATALOG.material(MATERIAL).scan" in source


def test_crystal_view_defaults_to_one_ranked_reciprocal_vector() -> None:
    source = APP.read_text()

    assert '1, 8, value=1, step=1, label="reciprocal vectors"' in source
    assert "n_reciprocal_vectors=crystal_reciprocal_ui.value" in source


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
    assert (
        "_cross_section_chart = apply_altair_theme("
        "trajectory_chart(_nc, Ne=40, width=420), _theme)" in source
    )
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


def test_penetration_controls_offer_material_presets_and_bounded_manual_values() -> None:
    source = APP.read_text()

    for grid in ("scan.energy_keV", "scan.thickness_ang", "scan.tilt_deg"):
        assert grid in source
    for bound in ("start=1.0", "stop=300.0", "start=0.001", "stop=10000.0", "stop=89.9"):
        assert bound in source
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

    assert "from pyrite.campaign.sweep import fmt_thickness" in source
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
