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


def test_no_checkpoint_state_displays_without_analysis_tabs() -> None:
    source = APP.read_text()
    tree = ast.parse(source)
    display_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(
            isinstance(node, ast.Constant)
            and node.value
            == "**No checkpoint data available.** Run `cxr scan <material>` to create one."
            for node in ast.walk(cell)
        )
    )

    assert any(arg.arg == "MATERIAL" for arg in display_cell.args.args)
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "MATERIAL"
        and any(isinstance(op, ast.Is) for op in node.ops)
        and any(
            isinstance(value, ast.Constant) and value.value is None for value in node.comparators
        )
        for node in ast.walk(display_cell)
    )


def test_in_progress_checkpoint_uses_analysis_safe_reads() -> None:
    source = APP.read_text()

    assert "from cxr_mc.analyze import load_analysis_checkpoint" in source
    assert "_loaded = load_analysis_checkpoint(_stem)" in source
    assert "if _loaded is None:" in source
    assert "from cxr_mc.analyze import analysis_checkpoint_manifest" in source
    assert "from cxr_mc.analyze import cached_analysis" in source


def test_thickness_controls_and_context_use_shared_human_units() -> None:
    source = APP.read_text()

    assert "fmt_thickness" in source
    assert source.count("fmt_thickness(t)") == 5
    # Context-rail summaries (each with their own fmt_thickness call) were
    # dropped in the rail-free declutter, so the floor is lower than it used
    # to be; this still guards against silently losing shared-unit calls.
    assert source.count("fmt_thickness(") >= 10


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


def test_material_and_face_select_labels_render_bold() -> None:
    source = APP.read_text()
    # MaterialSelect (and its <strong> label ESM) lives in the shared widget
    # module now; both apps import it from there.
    widget_source = (APP.parent / "_widgets.py").read_text()

    assert "from _widgets import MaterialSelect" in source
    assert 'const labelText = document.createElement("strong");' in widget_source
    assert 'labelText.textContent = model.get("label");' in widget_source
    assert 'label="Material"' in source
    assert 'label="Face"' in source
    assert 'label="**Material**"' not in source
    assert 'label="**Face**"' not in source


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


def test_analysis_app_uses_four_top_level_tabs_and_action_names() -> None:
    source = APP.read_text()

    # "Instruments" and "Compare" hold a single view each, so they're bare
    # top-level tabs rather than nested action-accordion groups.
    for group in (
        '"Explore"',
        '"Optimize"',
        '"Instruments"',
        '"Compare"',
    ):
        assert group in source
    for action in (
        "Compare beam energies",
        "Compare polar angles",
        "Compare azimuths",
        "Rank geometries",
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
