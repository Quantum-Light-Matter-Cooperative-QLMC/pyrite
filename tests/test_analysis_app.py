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
