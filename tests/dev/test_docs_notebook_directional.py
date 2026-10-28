"""Directional-source tutorial contracts and checked captured output (#758)."""

import ast
import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
NAME = "05-angular-radiation-distributions"


def test_directional_tutorial_uses_public_api_and_fetches_tables_first():
    path = DOCS / "notebooks" / f"{NAME}.py"
    text = path.read_text()
    tree = ast.parse(text)
    assert "from pyrite.radiation import directional_source" in text
    assert "pyrite.fetch_tables()" in text
    assert "pyrite.apps" not in text
    assert "endpoint=False" in text  # periodic angular boundary is not repeated
    fetch = text.index("pyrite.fetch_tables()")
    assert fetch < text.index("source = directional_source(")
    assert "molab-shield.svg" not in text
    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    group = (DOCS / "notebooks" / "designing" / "index.md").read_text()
    assert NAME in group
    assert f"../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page
    assert f"../outputs/{NAME}.json" in page
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "pyrite.radiation"
        for node in ast.walk(tree)
    )


def test_directional_tutorial_registration_and_capture(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS))
    spec = importlib.util.spec_from_file_location(
        "docs_tutorials_for_directional", DOCS / "_tutorials.py"
    )
    tutorials = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorials)
    notebooks = tutorials.configured_notebooks()
    assert notebooks.NOTEBOOK_FIGURES[NAME] == 1
    manifest = notebooks.checked_notebook(NAME)
    execution = manifest["execution"]
    assert execution["backend"] == "cpu"
    assert execution["component"] == "incoherent-pxr-cbs"
    assert execution["angles"] == 48
    assert execution["sampled_cxr_histories"] == 512
    assert execution["current_nA"] == pytest.approx(50)
    assert 0 < execution["supported"] <= 48
    assert execution["energy_bounds_eV"] == [100.0, 2000.0]
    _, capture = notebooks.read_session(
        DOCS / "notebooks" / "__marimo__" / "session" / f"{NAME}.py.json"
    )
    assert len(capture.figures) >= 1
