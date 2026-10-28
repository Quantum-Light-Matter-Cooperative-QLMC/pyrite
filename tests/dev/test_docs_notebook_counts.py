"""Detector-count tutorial navigation, public imports and executed capture."""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
NAME = "06-detector-counts"


def test_count_tutorial_capture_and_navigation(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS))
    spec = importlib.util.spec_from_file_location(
        "docs_tutorials_for_counts", DOCS / "_tutorials.py"
    )
    tutorials = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorials)
    notebooks = tutorials.configured_notebooks()

    def forbidden(*args, **kwargs):
        pytest.fail("capture validation must not execute transport or download tables")

    monkeypatch.setattr(notebooks.subprocess, "run", forbidden)
    monkeypatch.setattr(notebooks.subprocess, "check_output", forbidden)
    manifest = notebooks.checked_notebook(NAME)  # Checks governed imports and every artifact hash.
    execution = manifest["execution"]
    assert execution["backend"] == "cpu"
    assert execution["seed"] == 1
    assert execution["acquisition_seed"] == 23
    assert execution["current_nA"] == pytest.approx(10)
    assert execution["exposure_s"] == 1
    assert execution["pixel_shape"] == [16, 16]
    assert execution["angular_tiles"] == [4, 4]
    assert execution["measured_edges_eV"] == [500, 4000]
    assert set(execution["components"]) == {"cxr", "brem", "characteristic"}
    assert execution["expected_total"] > 0
    assert isinstance(execution["realized_total"], int)
    assert execution["realized_total"] > 0
    assert all(value >= 0 for value in execution["outside_window_expected"].values())
    assert len(execution["source_identity"]) == len(execution["observation_identity"]) == 64
    _, capture = notebooks.read_session(
        DOCS / "notebooks" / "__marimo__" / "session" / f"{NAME}.py.json"
    )
    assert notebooks.NOTEBOOK_FIGURES[NAME] == len(capture.figures) == 1

    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    assert NAME in (DOCS / "notebooks" / "designing" / "index.md").read_text()
    assert notebooks.MOLAB + NAME + ".py" in page
    assert f"{{literalinclude}} ../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page
    assert f"../outputs/{NAME}.json" in page
    assert (
        f"../notebooks/designing/{NAME}.md"
        in (DOCS / "guides" / "detector-observations.md").read_text()
    )
