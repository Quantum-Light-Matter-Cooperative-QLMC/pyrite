"""Public API, convergence semantics and captured output of notebook 07 (#758)."""

import ast
import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
NAME = "07-sampling-convergence-performance"


def test_convergence_tutorial_structure():
    source = (DOCS / "notebooks" / f"{NAME}.py").read_text()
    tree = ast.parse(source)
    assert "pyrite.fetch_tables()" in source
    assert "from pyrite.config import AdaptiveSampling" in source
    assert "FixedSampling" in source
    assert 'brem_source="none"' in source
    assert "perf_counter" in source
    assert "not** an independent-sample uncertainty" in source
    assert "pyrite.apps" not in source
    assert "molab-shield.svg" not in source
    assert all(not isinstance(node, ast.ImportFrom) or node.level == 0 for node in ast.walk(tree))
    # The pinned release predates the checkout's Numerics(seed=...) addition.
    numerics_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Numerics"
    ]
    assert len(numerics_calls) == 3
    assert all("seed" not in {kw.arg for kw in call.keywords} for call in numerics_calls)
    group = (DOCS / "notebooks" / "designing" / "index.md").read_text()
    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    assert NAME in group
    assert f"../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page
    assert f"../outputs/{NAME}.json" in page


def test_convergence_capture_is_reproducible_and_linked(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS))
    spec = importlib.util.spec_from_file_location("tutorials_convergence", DOCS / "_tutorials.py")
    tutorials = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorials)
    notebooks = tutorials.configured_notebooks()
    assert notebooks.NOTEBOOK_FIGURES[NAME] == 1

    def forbid(*args, **kwargs):
        pytest.fail("checking a captured tutorial must not execute transport")

    monkeypatch.setattr(notebooks.subprocess, "run", forbid)
    monkeypatch.setattr(notebooks.subprocess, "check_output", forbid)
    manifest = notebooks.checked_notebook(NAME)
    execution = manifest["execution"]
    assert execution["backend"] == "cpu"
    assert execution["band_eV"] == [600, 2000]
    assert [r["histories"] for r in execution["fixed_runs"]] == [
        256,
        1024,
        4096,
        8192,
        16384,
        32768,
    ]
    assert {r["seed"] for r in execution["fixed_runs"]} == {1}
    adaptive = execution["adaptive_run"]
    stopping = execution["adaptive_stopping_diagnostics"]
    statistics = stopping["statistics"]["cxr"]
    assert adaptive["seed"] == 1
    assert adaptive["max_electrons"] == 32768
    assert 256 <= adaptive["histories"] <= adaptive["max_electrons"]
    assert adaptive["histories"] == stopping["realized_electrons"]
    assert adaptive["stop_reason"] == stopping["stop_reason"]
    assert adaptive["statistics_limited"] == stopping["statistics_limited"]
    for field, key in (
        ("rse", "relative_se"),
        ("effective_sample_size", "effective_sample_size"),
        ("max_electron_share", "max_electron_share"),
        ("stability", "stability"),
        ("guard_passes", "passes"),
    ):
        assert adaptive[field] == statistics[key]
    assert all(r["elapsed_s"] >= 0 for r in execution["fixed_runs"])
    assert all(r["yield_per_electron_sr"] >= 0 for r in execution["fixed_runs"])
    assert execution["adaptive_run"]["yield_per_electron_sr"] >= 0
    _, capture = notebooks.read_session(
        DOCS / "notebooks" / "__marimo__" / "session" / f"{NAME}.py.json"
    )
    assert len(capture.figures) == 1
