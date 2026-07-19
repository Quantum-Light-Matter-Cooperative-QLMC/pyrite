"""Stable contracts for shared notebook presentation primitives."""

from pathlib import Path

import pytest

from notebooks import _design


def test_design_tokens_are_complete_and_export_safe() -> None:
    assert set(_design.COLORS) == {
        "beamline_navy",
        "instrument_slate",
        "xray_ice",
        "detector_cyan",
        "tungsten_amber",
        "validation_rose",
    }
    assert all(
        token_group
        for token_group in (
            _design.SPACING,
            _design.TYPOGRAPHY,
            _design.BORDERS,
            _design.FOCUS,
            _design.WIDTHS,
        )
    )
    css = _design.notebook_css()
    assert "focus-visible" in css
    assert "min-height: 44px" in css
    assert "body:has(.cxr-shell)" in css
    assert "body:has(.cxr-shell) *::before" in css
    assert "prefers-reduced-motion" in css
    assert "https://" not in css and "http://" not in css


def test_status_vocabulary_is_unique_and_explicit() -> None:
    assert len(_design.STATUS_KINDS) == len(set(_design.STATUS_KINDS))
    assert "Completed—interpret" in _design.STATUS_KINDS
    assert {"Ready", "Running", "Cached", "Passed", "Failed", "Skipped"} <= set(
        _design.STATUS_KINDS
    )


@pytest.mark.parametrize("value", [None, ""])
def test_context_value_names_missing_state(value) -> None:
    assert _design._display(value) == "not selected"


def test_core_package_does_not_import_notebook_design() -> None:
    root = Path(__file__).parents[1] / "src" / "cxr_mc"
    assert all("notebooks._design" not in path.read_text() for path in root.rglob("*.py"))
