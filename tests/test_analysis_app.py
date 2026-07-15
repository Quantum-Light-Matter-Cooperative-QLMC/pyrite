"""Regression checks for the marimo analysis app's static UI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).parents[1] / "notebooks" / "analysis_app.py"


def test_penetration_controls_read_the_active_material_scan() -> None:
    source = APP.read_text()

    assert "def _(CATALOG, MATERIAL, mo):" in source
    assert "_scan = CATALOG.material(MATERIAL).scan" in source


def test_penetration_controls_offer_material_presets_and_bounded_manual_values() -> None:
    source = APP.read_text()

    for grid in ("scan.energy_keV", "scan.thickness_ang", "scan.tilt_deg"):
        assert grid in source
    for bound in ("start=1.0", "stop=300.0", "start=0.001", "stop=10.0", "stop=89.9"):
        assert bound in source
    for name in (
        "penetration_energy_keV",
        "penetration_thickness_ang",
        "penetration_tilt_deg",
    ):
        assert name in source


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
