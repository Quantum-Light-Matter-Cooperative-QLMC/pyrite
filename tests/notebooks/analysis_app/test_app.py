"""Regression checks for the marimo analysis app's static UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import altair as alt

from pyrite.apps.analysis_ui.models import AnalysisContext

APP = Path(__file__).parents[3] / "src" / "pyrite" / "apps" / "analysis_app.py"


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def test_blazed_face_title_preserves_altair_typography() -> None:
    context = AnalysisContext(
        selected_material="hopg",
        selected_face="blazed",
        selected_profile=None,
        checkpoint_stem=None,
        settings=None,
        checkpoint_results=None,
        results=None,
        cases=None,
    )
    chart = alt.Chart(alt.Data(values=[])).mark_line().properties(
        title=alt.TitleParams(text="Spectrum", fontSize=18)
    )

    spec = context.title_for_face(chart).to_dict()

    assert spec["title"]["text"] == "Spectrum (blazed)"
    assert spec["title"]["fontSize"] == 18


def test_no_checkpoint_state_displays_without_analysis_tabs() -> None:
    # Checkpoint loading moved behind analysis_ui.load_context -> AnalysisContext;
    # the app cell now branches on context.has_data instead of MATERIAL is None.
    source = APP.read_text()
    tree = ast.parse(source)
    display_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(
            isinstance(node, ast.Constant) and node.value == "**No checkpoint data available.** "
            for node in ast.walk(cell)
        )
    )

    assert any(arg.arg == "context" for arg in display_cell.args.args)
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "has_data"
        for node in ast.walk(display_cell)
    )


def test_in_progress_checkpoint_uses_analysis_safe_reads() -> None:
    # Safe-read logic (load_analysis_checkpoint + manifest lookup) moved out of
    # app.py into analysis_ui.data.load_context; the app now delegates via
    # load_context instead of reading checkpoints itself.
    source = APP.read_text()
    data_source = (APP.parent / "analysis_ui" / "data.py").read_text()

    assert "load_context" in source
    assert "load_analysis_checkpoint" in data_source
    assert "loaded = load_analysis_checkpoint(stem)" in data_source
    assert "load_error" in data_source


def test_thickness_controls_and_context_use_shared_human_units() -> None:
    # fmt_thickness calls were consolidated out of app.py entirely into the
    # shared analysis_ui control/view helpers; guard both that app.py no
    # longer duplicates the logic and that the shared modules still call it.
    source = APP.read_text()
    controls_source = (APP.parent / "analysis_ui" / "controls.py").read_text()
    common_view_source = (APP.parent / "analysis_ui" / "views" / "common.py").read_text()

    assert "fmt_thickness" not in source
    assert controls_source.count("fmt_thickness(") + common_view_source.count("fmt_thickness(") >= 4


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

    assert "from pyrite.apps._widgets import MaterialSelect" in source
    assert 'const labelText = document.createElement("strong");' in widget_source
    assert 'labelText.textContent = model.get("label");' in widget_source
    assert 'label="Material"' in source
    assert 'label="Face"' in source
    assert 'label="**Material**"' not in source
    assert 'label="**Face**"' not in source


def test_material_menu_cell_owns_checkpoint_directory_dependency() -> None:
    # DEFAULT_CHECKPOINT_DIR is now a single shared app.setup import (rather
    # than each per-cell dependency re-importing/re-injecting it); menu cells
    # reference the shared name directly instead of taking it as a cell arg.
    source = APP.read_text()
    tree = ast.parse(source)

    setup_block = next(node for node in tree.body if isinstance(node, ast.With))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pyrite.runs.run"
        and any(alias.name == "DEFAULT_CHECKPOINT_DIR" for alias in node.names)
        for node in ast.walk(setup_block)
    )

    menu_cells = [
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call) and _attribute_path(node.func)[-1] == "MaterialSelect"
            for node in ast.walk(cell)
        )
    ]
    assert menu_cells
    for menu_cell in menu_cells:
        assert not any(
            arg.arg in ("DEFAULT_CHECKPOINT_DIR", "_DEFAULT_CHECKPOINT_DIR")
            for arg in menu_cell.args.args
        )


def test_analysis_app_discovers_materials_directly_from_catalog() -> None:
    source = APP.read_text()

    assert "from pyrite.materials import CATALOG" in source
    assert "MATERIAL_LABELS" not in source
    assert "from pyrite.campaign.config import MATERIALS" not in source


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
    # The per-axis *_auto_ui widgets were consolidated into a single
    # make_axis_controls("Auto {prefix} domain") switch factory shared across
    # narrow/broad/polar/azimuth/detector axes.
    source = APP.read_text()
    controls_source = (APP.parent / "analysis_ui" / "controls.py").read_text()
    axes_source = (APP.parent / "analysis_ui" / "axes.py").read_text()

    assert "0 = auto" not in source
    assert "0 = auto" not in controls_source
    assert 'mo.ui.switch(value=auto, label=f"Auto {prefix} domain")' in controls_source
    assert "if auto:" in axes_source
    assert "return None" in axes_source
