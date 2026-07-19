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


def test_penetration_controls_offer_material_presets_and_bounded_manual_values() -> None:
    source = APP.read_text()

    for grid in ("scan.energy_keV", "scan.thickness_ang", "scan.tilt_deg"):
        assert grid in source
    for bound in ("start=1.0", "stop=300.0", "start=0.001", "stop=10000.0", "stop=89.9"):
        assert bound in source
    assert 'label="crystal thickness (µm)"' in source
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
    assert source.count("fmt_thickness(") >= 17


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


def test_analysis_app_uses_four_task_groups_and_action_names() -> None:
    source = APP.read_text()

    for group in ('"Explore"', '"Optimize"', '"Instrument"', '"Compare"'):
        assert group in source
    for action in (
        "Compare beam energies",
        "Compare polar angles",
        "Compare azimuths",
        "Rank geometries",
        "Inspect scan maps",
        "Model detectors",
        "Inspect penetration",
        "Compare materials",
    ):
        assert action in source


def test_each_lazy_view_builds_its_own_context_rail() -> None:
    tree = ast.parse(APP.read_text())
    builders = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "spectra_tab",
            "polar_compare_tab",
            "azim_compare_tab",
            "rankings_tab",
            "scans_tab",
            "detectors_tab",
            "penetration_tab",
            "cross_material_tab",
        }
    }

    assert len(builders) == 8
    for name, builder in builders.items():
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "context_rail"
            for node in ast.walk(builder)
        ), f"{name} lacks active-view context"


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


def test_analysis_app_has_checkpoint_summary_and_no_mojibake() -> None:
    source = APP.read_text()

    assert "Checkpoint contents and provenance" in source
    assert "Swept dimensions" in source
    assert "cxr scan {MATERIAL or '<material>'}" in source
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
