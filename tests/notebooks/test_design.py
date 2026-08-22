"""Stable contracts for shared notebook presentation primitives."""

import tomllib
from pathlib import Path

import altair as alt
import pytest

from pyrite.apps import _design
from pyrite.apps._widgets import ThemeSelect


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


def test_marimo_starts_in_pyrite_default_light_theme() -> None:
    root = Path(__file__).parents[2]
    project = root / "pyproject.toml"
    config = tomllib.loads(project.read_text())

    assert config["tool"]["marimo"]["display"]["theme"] == "light"
    for app in ("analysis_app.py", "scan_app.py", "trace_app.py", "validation_app.py"):
        source = (root / "src" / "pyrite" / "apps" / app).read_text()
        assert '# theme = "light"' in source
        assert '# theme = "system"' not in source


def test_theme_switch_updates_marimo_shadow_dom_controls() -> None:
    source = ThemeSelect._esm

    assert ".shadowRoot" in source
    assert '.querySelectorAll(".marimo > .contents")' in source
    assert 'classList.remove("light", "dark")' in source
    assert "classList.add(resolved)" in source


def test_altair_light_theme_overrides_axis_and_embedded_title_colors() -> None:
    chart = (
        alt.Chart(alt.Data(values=[{"energy": 1.0, "photons": 2.0}]))
        .mark_line()
        .encode(x="energy:Q", y="photons:Q")
        .properties(title=alt.TitleParams(text="Spectrum", color="#FFFFFF"))
    )

    spec = _design.apply_altair_theme(chart, "light").to_dict()
    light = _design.THEMES["light"]
    assert spec["config"]["axis"]["labelColor"] == light["text"]
    assert spec["config"]["axis"]["titleColor"] == light["text"]
    assert spec["config"]["text"]["color"] == light["text"]
    assert spec["title"]["color"] == light["text"]


def test_status_vocabulary_is_unique_and_explicit() -> None:
    assert len(_design.STATUS_KINDS) == len(set(_design.STATUS_KINDS))
    assert "Completed—interpret" in _design.STATUS_KINDS
    assert {"Ready", "Running", "Cached", "Passed", "Failed", "Skipped"} <= set(
        _design.STATUS_KINDS
    )


@pytest.mark.parametrize("value", [None, ""])
def test_context_value_names_missing_state(value) -> None:
    assert _design._display(value) == "not selected"


class _FakeMo:
    """Minimal stand-in for the marimo module: only ``mo.Html`` is exercised."""

    class Html:
        def __init__(self, html: str) -> None:
            self.text = html


def test_scan_grid_renders_export_safe_state_matrix() -> None:
    energies = [30.0, 60.0]
    rows = [
        {
            "label": "5°",
            "cells": [
                {"cached": 1.0, "weight": 1.0},  # fully done, heaviest cell
                {"excluded": 1.0, "weight": 0.4},  # penetration-excluded
            ],
        },
        {"label": "15°", "cells": [None, {"running": 0.5, "weight": 0.2}]},
    ]
    html = _design.scan_grid(_FakeMo, energies, rows).text

    # structure: a cell per (row, energy) plus the state + legend vocabulary
    assert html.count("cxr-grid__cell") == 4
    for state in ("cached", "excluded", "running"):
        assert f"cxr-grid__seg--{state}" in html
    assert "Penetration-excluded" in html and "Remaining" in html
    # cost sizing: the unit-weight cell fills the marker, the light cell shrinks
    assert "width:100%" in html
    # export safety: no network dependencies leak into the notebook HTML
    assert "https://" not in html and "http://" not in html


def test_scan_grid_states_map_to_palette_tokens() -> None:
    assert set(_design.SCAN_GRID_STATES) == {"cached", "done", "running", "excluded"}
    # every stacked state (plus the empty "remaining" track) has a legend entry
    assert {*_design.SCAN_GRID_STATES, "remaining"} <= set(_design._SCAN_GRID_LEGEND)


def test_core_package_does_not_import_notebook_design() -> None:
    root = Path(__file__).parents[1] / "src" / "pyrite"
    assert all("notebooks._design" not in path.read_text() for path in root.rglob("*.py"))
