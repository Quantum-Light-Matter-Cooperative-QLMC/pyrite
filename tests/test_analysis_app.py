"""Regression checks for the marimo analysis app's static UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).parents[1] / "notebooks" / "analysis_app.py"


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def test_penetration_controls_read_the_active_material_scan() -> None:
    source = APP.read_text()

    assert "def _(CATALOG, MATERIAL, fmt_thickness, mo):" in source
    assert "_scan = CATALOG.material(MATERIAL).scan" in source


def test_penetration_view_uses_interactive_3d_volume_as_primary_track_plot() -> None:
    source = APP.read_text()

    assert "from cxr_mc.plots.plotly_trajectories import (" in source
    assert "trajectory_volume_animation," in source
    assert "trajectory_volume_data," in source
    # Playback controls drive the transport data (Ne/seed/realistic/beam_fwhm);
    # Play/Pause/scrub is native Plotly animation in the browser from there.
    assert "_data = trajectory_volume_data(" in source
    for arg in ("Ne=_Ne", "seed=_seed", "realistic=_realistic", "beam_fwhm_mm=_beam_fwhm"):
        assert arg in source
    assert "_volume = trajectory_volume_animation(" in source
    for arg in ("_nc,", "_data,", "realistic=_realistic,", "beam_fwhm_mm=_beam_fwhm,", "speed="):
        assert arg in source
    # The 2D cross-section renders eagerly beside the survival chart (no lazy
    # accordion wrapper -- see the rail-free declutter).
    assert "_cross_section_chart = trajectory_chart(_nc, Ne=40, width=420)" in source
    assert "lateral extent is fitted to the tracks" in source


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

    assert "from cxr_mc.sweep import build_cases, fmt_thickness" in source
    assert source.count("fmt_thickness(t)") == 5
    assert "fmt_thickness(value)" in source
    # Context-rail summaries (each with their own fmt_thickness call) were
    # dropped in the rail-free declutter, so the floor is lower than it used
    # to be; this still guards against silently losing shared-unit calls.
    assert source.count("fmt_thickness(") >= 13


def test_penetration_control_values_are_read_in_a_downstream_cell() -> None:
    tree = ast.parse(APP.read_text())
    controls_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef) and any(arg.arg == "CATALOG" for arg in cell.args.args)
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


def test_all_ui_values_are_read_downstream_of_creation() -> None:
    tree = ast.parse(APP.read_text())

    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        created = {
            target.id
            for assignment in ast.walk(cell)
            if isinstance(assignment, (ast.Assign, ast.AnnAssign))
            for target in (
                assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            )
            if isinstance(target, ast.Name)
            and isinstance(assignment.value, ast.Call)
            and _attribute_path(assignment.value.func)[:2] == ("mo", "ui")
        }
        read = {
            node.value.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Attribute)
            and node.attr == "value"
            and isinstance(node.value, ast.Name)
        }
        assert not created & read, (
            f"cell at line {cell.lineno} creates and reads UI values: {sorted(created & read)}"
        )


def test_cross_cell_exports_do_not_use_private_names() -> None:
    tree = ast.parse(APP.read_text())

    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        for statement in cell.body:
            if not isinstance(statement, ast.Return) or statement.value is None:
                continue
            exported = {node.id for node in ast.walk(statement.value) if isinstance(node, ast.Name)}
            assert not {name for name in exported if name.startswith("_")}, (
                f"cell at line {cell.lineno} exports private names: {sorted(exported)}"
            )


def test_material_menu_uses_checkpoint_helper_and_custom_select() -> None:
    source = APP.read_text()

    assert "material_menu" in source
    assert "select_initial_material" in source
    assert "mo.ui.anywidget" in source
    assert "MaterialSelect" in source


def test_material_menu_cell_owns_checkpoint_directory_dependency() -> None:
    tree = ast.parse(APP.read_text())

    menu_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(arg.arg == "MaterialSelect" for arg in cell.args.args)
    )

    assert not any(arg.arg == "_DEFAULT_CHECKPOINT_DIR" for arg in menu_cell.args.args)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "cxr_mc.run"
        and any(alias.name == "_DEFAULT_CHECKPOINT_DIR" for alias in node.names)
        for node in menu_cell.body
    )


def test_analysis_app_discovers_materials_directly_from_catalog() -> None:
    source = APP.read_text()

    assert "from cxr_mc.materials import CATALOG" in source
    assert "MATERIAL_LABELS" not in source
    assert "from cxr_mc.config import MATERIALS" not in source


def test_analysis_app_uses_five_top_level_tabs_and_action_names() -> None:
    source = APP.read_text()

    # "Instruments", "Trace", and "Compare" hold a single view each, so
    # they're bare top-level tabs rather than nested action-accordion groups.
    for group in ('"Explore"', '"Optimize"', '"Instruments"', '"Trace"', '"Compare"'):
        assert group in source
    for action in (
        "Compare beam energies",
        "Compare polar angles",
        "Compare azimuths",
        "Rank geometries",
        "Inspect scan maps",
    ):
        assert action in source


def test_no_lazy_view_uses_a_context_rail() -> None:
    # Every context_rail block (per-tab Material/E0/theta/phi/Thickness/Records
    # summary) was dropped in the rail-free declutter -- the controls above
    # each chart already carry that information. Guard against it creeping
    # back in (e.g. a live marimo-pair session resaving from stale kernel state).
    source = APP.read_text()
    assert "context_rail" not in source


def test_control_rows_wrap_at_narrow_widths() -> None:
    tree = ast.parse(APP.read_text())
    hstacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == ("mo", "hstack")
    ]

    assert hstacks
    for call in hstacks:
        wrap = next((keyword.value for keyword in call.keywords if keyword.arg == "wrap"), None)
        assert isinstance(wrap, ast.Constant) and wrap.value is True


def test_analysis_app_has_no_mojibake() -> None:
    # The checkpoint-contents/provenance accordion (path, swept-dimensions
    # table, empty-checkpoint warning) was dropped in the rail-free declutter;
    # only the encoding/stale-name guards remain relevant.
    source = APP.read_text()

    assert "â€”" not in source
    assert "INTRINSIC" not in source


def test_auto_domain_controls_replace_zero_sentinel_copy() -> None:
    source = APP.read_text()

    assert "0 = auto" not in source
    for name in (
        "narrow_auto_ui",
        "broad_auto_ui",
        "polar_auto_ui",
        "polar_broad_auto_ui",
        "azim_auto_ui",
        "azim_broad_auto_ui",
        "detector_auto_ui",
        "detector_broad_auto_ui",
    ):
        assert name in source
    assert "if auto:" in source
    assert "return None" in source
