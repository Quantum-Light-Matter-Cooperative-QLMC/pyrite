"""Regression checks for catalog-backed scan app discovery."""

import ast
from pathlib import Path

APP = Path(__file__).parents[2] / "src" / "pyrite" / "apps" / "scan_app.py"


def test_scan_app_discovers_ordered_material_labels_from_catalog() -> None:
    source = APP.read_text()

    assert "from pyrite.materials import CATALOG" in source
    assert "CATALOG.material_keys" in source
    assert ".label" in source
    assert "from pyrite.campaign.config import COLLAPSE_AZIMUTH, MATERIALS" not in source
    assert "data/catalog/" in source
    assert "analysis app" in source
    assert "analysis notebook" not in source


def test_scan_app_gates_cases_by_penetration_before_running() -> None:
    source = APP.read_text()

    assert "gate_cases_by_penetration" in source
    build_at = source.index("build_configured_cases(sweep, settings)")
    gate_at = source.index("gate_cases_by_penetration(cases")
    run_at = source.index("run_sweep(")
    assert build_at < gate_at < run_at


def test_scan_execution_is_downstream_of_explicit_button_value() -> None:
    tree = ast.parse(APP.read_text())
    run_cell = next(
        cell
        for cell in tree.body
        if isinstance(cell, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_sweep"
            for node in ast.walk(cell)
        )
    )

    assert "run_scan_ui" in {arg.arg for arg in run_cell.args.args}
    assert any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "run_scan_ui"
        and node.attr == "value"
        for node in ast.walk(run_cell)
    )


def test_scan_preview_and_analysis_handoff_are_explicit() -> None:
    source = APP.read_text()

    assert "Scan preview ready" in source
    assert "Geometry preview" in source
    assert "Penetration exclusions" in source
    assert "pyrite app analysis launch {MATERIAL}" in source


def test_geometry_preview_is_lazy() -> None:
    source = APP.read_text()

    assert '"Geometry preview": mo.lazy(lambda: geometry_table(cases))' in source


def test_initial_material_does_not_require_hopg_in_selected_catalog() -> None:
    source = APP.read_text()

    assert 'CATALOG.material("hopg")' not in source
    assert '"hopg" if "hopg" in CATALOG.material_keys' in source
