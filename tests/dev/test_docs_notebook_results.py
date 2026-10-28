"""Result tutorial capture distinguishes numerical histories from physical flux."""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"


def test_results_capture_has_physical_normalization_and_two_figures():
    spec = importlib.util.spec_from_file_location("docs_notebooks", DOCS / "_notebooks.py")
    notebooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebooks)
    manifest = notebooks.checked_notebook("02-understanding-results")
    provenance = manifest["execution"]
    assert provenance["mean_current_nA"] == pytest.approx(50)
    assert provenance["sampled_cxr_histories"] == 8_192
    assert provenance["accepted_solid_angle_sr"] > 0
    assert provenance["band_eV"] == [600.0, 2_000.0]
    session = DOCS / "notebooks" / "__marimo__" / "session" / "02-understanding-results.py.json"
    _, capture = notebooks.read_session(session)
    assert len(capture.figures) >= 2
