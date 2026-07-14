"""Regression checks for the marimo analysis app's static UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

from cxr_mc.config import PENETRATION_TILT_DEG

APP = Path(__file__).parents[1] / "notebooks" / "analysis_app.py"


def test_penetration_dropdown_default_is_one_of_configured_angles() -> None:
    tree = ast.parse(APP.read_text())

    dropdown = next(
        call
        for cell in ast.walk(tree)
        if isinstance(cell, ast.FunctionDef)
        and any(arg.arg == "PENETRATION_TILT_DEG" for arg in cell.args.args)
        for call in ast.walk(cell)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "dropdown"
    )
    default = next(keyword.value for keyword in dropdown.keywords if keyword.arg == "value")

    configured_options = {f"{tilt:g} deg" for tilt in PENETRATION_TILT_DEG}
    assert ast.literal_eval(default) in configured_options


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
