"""Regression checks for the marimo analysis app's static UI wiring."""

import ast
from pathlib import Path

import altair as alt

from pyrite.apps.analysis_ui.axes import resolve_axis_spec
from pyrite.apps.analysis_ui.controls import _grid_limits, make_spectrum_axes
from pyrite.apps.analysis_ui.models import AnalysisContext

APP = Path(__file__).parents[3] / "src" / "pyrite" / "apps" / "analysis_app.py"
CONTROLS = APP.parent / "analysis_ui" / "controls.py"
VIEWS = APP.parent / "analysis_ui" / "views"


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
        checkpoint_stem=None,
        settings=None,
        checkpoint_results=None,
        results=None,
        cases=None,
    )
    chart = (
        alt.Chart(alt.Data(values=[]))
        .mark_line()
        .properties(title=alt.TitleParams(text="Spectrum", fontSize=18))
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
    assert "loaded = load_analysis_checkpoint(stem, root)" in data_source
    assert "load_error" in data_source


def test_characteristic_checkbox_matches_spectrum_component_controls() -> None:
    source = APP.read_text()
    controls_source = CONTROLS.read_text()
    view_sources = [(VIEWS / name).read_text() for name in ("spectra.py", "cases.py")]

    # One Spectra component toggle set plus the compare app's case axes.
    assert controls_source.count('"characteristic": mo.ui.checkbox(') == 2
    assert controls_source.count('label="show characteristic radiation"') == 2
    assert "characteristic_ui" not in source
    assert 'components["characteristic"]' in view_sources[0]
    assert 'include_characteristic=component_values["characteristic"]' in view_sources[0]
    assert 'controls["characteristic"]' in view_sources[1]


def test_thickness_controls_and_context_use_shared_human_units() -> None:
    # fmt_thickness calls were consolidated out of app.py entirely into the
    # shared analysis_ui control/view helpers; guard both that app.py no
    # longer duplicates the logic and that the shared modules still call it.
    source = APP.read_text()
    controls_source = (APP.parent / "analysis_ui" / "controls.py").read_text()
    common_view_source = (APP.parent / "analysis_ui" / "views" / "common.py").read_text()

    assert "fmt_thickness" not in source
    assert controls_source.count("fmt_thickness(") + common_view_source.count("fmt_thickness(") >= 3


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
    assert 'label="**Material**"' not in source
    assert 'label="Face"' not in source
    assert 'label="Profile"' not in source
    assert "make_checkpoint_picker(" in source


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


def test_analysis_app_uses_no_legacy_material_registries() -> None:
    source = APP.read_text()

    assert "MATERIAL_LABELS" not in source
    assert "from pyrite.campaign.config import MATERIALS" not in source


def test_analysis_app_has_three_views_and_no_accordions() -> None:
    # Issue #236: Spectra / Map / Detectors, no accordion anywhere except the
    # single collapsible axes panel (replaced by chart zoom in #237).
    navigation = (APP.parent / "analysis_ui" / "navigation.py").read_text()
    for view in ('"Spectra"', '"Map"', '"Detectors"'):
        assert view in navigation
    assert "VIEW_NAMES = (SPECTRA, MAP, DETECTORS)" in navigation
    assert "mo.accordion" not in APP.read_text()
    for view_module in ("spectra.py", "map.py", "detectors.py", "sidebar.py"):
        assert "mo.accordion" not in (VIEWS / view_module).read_text()
    assert CONTROLS.read_text().count("mo.accordion") == 1  # axes_panel only


def test_matplotlib_panes_are_gone_from_the_analysis_views() -> None:
    for view_module in ("spectra.py", "map.py", "detectors.py"):
        source = (VIEWS / view_module).read_text()
        for dropped in (
            "plot_best_spectra",
            "plot_eaglexo_efficiency",
            "plot_eaglexo_charge_map",
            "plot_timepix_efficiency",
        ):
            assert dropped not in source


def test_face_selector_note_is_gone() -> None:
    for app in ("analysis_app.py", "compare_app.py", "pixel_app.py"):
        assert "the face selector does not" not in (APP.parent / app).read_text()


def test_slice_controls_live_in_the_sidebar_outside_every_view() -> None:
    tree = ast.parse(APP.read_text())
    cells = [cell for cell in tree.body if isinstance(cell, ast.FunctionDef)]
    slice_cell = next(cell for cell in cells if "make_slice_controls(" in ast.unparse(cell))
    sidebar_cell = next(cell for cell in cells if "render_sidebar(" in ast.unparse(cell))

    # Not gated on the view and not rebuilt on emission changes.
    assert _stop_guards(slice_cell) == set()
    assert {arg.arg for arg in slice_cell.args.args} == {"base_context"}
    assert "view_nav" not in {arg.arg for arg in sidebar_cell.args.args}
    sidebar_source = (VIEWS / "sidebar.py").read_text()
    assert "mo.sidebar(" in sidebar_source
    # Each view reads the shared slice dropdowns rather than building its own.
    for name in ("energy_ui", "tilt_ui", "azimuth_ui", "thickness_ui"):
        readers = [cell for cell in cells if f"{name}.value" in ast.unparse(cell)]
        assert readers, name


def test_view_local_choices_are_dependency_free() -> None:
    tree = ast.parse(APP.read_text())
    cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef) and "make_vary_control(mo)" in ast.unparse(cell)
    )
    assert not cell.args.args
    for builder in (
        "make_component_controls(mo)",
        "make_map_quantity_control(mo)",
        "make_detector_choice(mo)",
    ):
        assert builder in ast.unparse(cell)


def test_pixel_case_and_cross_material_views_live_in_their_own_apps() -> None:
    # Issue #235: the pixel detector moved to pixel_app.py; the case basket and
    # cross-material comparison moved to compare_app.py.
    source = APP.read_text()

    for moved in (
        "analysis_ui.pixels",
        "render_pixel_detector",
        "render_case_comparison",
        "render_cross_material",
        "CASE_BASKET_CAP",
        "make_case_axes",
        "case_basket",
        '"Pixel detector"',
        '"Compare any cases"',
    ):
        assert moved not in source
    views_package = (VIEWS / "__init__.py").read_text()
    for module in (".pixels", ".cases", ".materials"):
        assert f"from {module} import" not in views_package


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


def test_grid_limits_use_full_line_and_brem_arrays() -> None:
    source = [
        {"E_grid": [40.0, 5000.0], "E_grid_brem": [20.0, 150000.0]},
        {"E_grid": [30.0, 6000.0], "E_grid_brem": [10.0, 200000.0]},
    ]
    assert _grid_limits(source, "E_grid") == (30.0, 6000.0)
    assert _grid_limits(source, "E_grid_brem") == (10.0, 200000.0)


def test_axis_defaults_use_grid_endpoints_and_log_floor() -> None:
    class FakeUI:
        @staticmethod
        def switch(*, value, label):
            return value

        @staticmethod
        def text(*, value, label):
            return value

        @staticmethod
        def dictionary(elements):
            return elements

    class FakeMo:
        ui = FakeUI()

    axes = make_spectrum_axes(
        FakeMo(),
        narrow_auto=False,
        narrow_xmin=30.0,
        narrow_xmax=6000.0,
        broad_auto=False,
        broad_xmin=10.0,
        broad_xmax=200000.0,
        broad_xlog=True,
    )
    assert (axes["narrow"]["xmin"], axes["narrow"]["xmax"]) == ("30", "6000")
    assert (axes["broad"]["xmin"], axes["broad"]["xmax"]) == ("10", "200000")
    assert resolve_axis_spec(axes["broad"]).x_domain == (10.0, 200000.0)
    detector_axis = {**axes["broad"], "ymin": "0", "ymax": "0"}
    assert resolve_axis_spec(detector_axis, include_y_domain=True).warnings == ()


def test_manual_axis_accepts_incomplete_and_invalid_text() -> None:
    base = {"auto": False, "xmin": "20", "xmax": "1,000", "xlog": True}
    assert resolve_axis_spec(base).x_domain == (20.0, 1000.0)
    for xmax in ("", "0", "not a number"):
        assert resolve_axis_spec({**base, "xmax": xmax}).x_domain is None


def _stop_guards(cell: ast.FunctionDef) -> set[str]:
    """View constants named in ``mo.stop(view_nav.value != VIEW)`` calls of a cell."""
    found: set[str] = set()
    for node in ast.walk(cell):
        if isinstance(node, ast.Call) and _attribute_path(node.func) == ("mo", "stop"):
            found.update(
                sub.id
                for sub in ast.walk(node.args[0])
                if isinstance(sub, ast.Name) and sub.id.isupper()
            )
    return found


def test_heavy_view_builders_are_gated_on_the_selected_view() -> None:
    # Only the visible view may build controls/widgets (issue #234).
    tree = ast.parse(APP.read_text())
    guards: dict[str, set[str]] = {}
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        source = ast.unparse(cell)
        for builder in (
            "make_spectra_axes",
            "make_varying_control",
            "render_spectra",
            "make_map_widget",
            "render_map_trends",
            "render_map(",
            "make_detector_axes",
            "render_detectors",
        ):
            if builder in source:
                guards[builder] = _stop_guards(cell)

    for builder in ("make_spectra_axes", "make_varying_control", "render_spectra"):
        assert guards[builder] == {"SPECTRA"}, builder
    for builder in ("make_map_widget", "render_map_trends", "render_map("):
        assert guards[builder] == {"MAP"}, builder
    for builder in ("make_detector_axes", "render_detectors"):
        assert guards[builder] == {"DETECTORS"}, builder


def test_view_nav_is_dependency_free_so_selection_survives_context_changes() -> None:
    tree = ast.parse(APP.read_text())
    nav_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef) and "make_view_nav(mo)" in ast.unparse(cell)
    )
    assert not nav_cell.args.args
