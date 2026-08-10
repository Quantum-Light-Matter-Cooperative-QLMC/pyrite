"""Focused checks for the strict documentation warning boundary."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def warning_baseline_module():
    path = Path(__file__).parents[2] / "docs" / "_warning_baseline.py"
    spec = importlib.util.spec_from_file_location("cxr_docs_warning_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(location: Path, *, warning_type: str = "docutils") -> logging.LogRecord:
    record = logging.LogRecord("sphinx.test", logging.WARNING, "", 0, "problem", (), None)
    record.location = f"{location}:docstring of example:3"
    record.type = warning_type
    record.subtype = ""
    return record


def test_warning_baseline_filters_only_source_docutils(
    warning_baseline_module, monkeypatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "src"
    generated_root = tmp_path / "docs" / "_autosummary"
    baseline = warning_baseline_module.AutodocWarningBaseline(source_root, generated_root)

    assert baseline.filter(_record(source_root / "cxr_mc" / "example.py")) is False
    generated = _record(generated_root / "cxr_mc.example.rst")
    generated.location = f"{generated_root / 'cxr_mc.example.rst'}:12:<autosummary>"
    assert baseline.filter(generated) is False
    assert baseline.filter(_record(tmp_path / "docs" / "guide.md")) is True
    assert (
        baseline.filter(_record(source_root / "cxr_mc" / "example.py", warning_type="ref")) is True
    )
    assert len(baseline.fingerprints) == 2

    monkeypatch.setenv("CXR_DOCS_SHOW_AUTODOC_WARNINGS", "1")
    assert baseline.filter(_record(source_root / "cxr_mc" / "visible.py")) is True


def test_changed_warning_fingerprint_emits_unsuppressed_failure(
    warning_baseline_module, monkeypatch, tmp_path: Path
) -> None:
    baseline = warning_baseline_module.AutodocWarningBaseline(tmp_path / "src")
    app = SimpleNamespace()
    warning_baseline_module._BASELINES[id(app)] = baseline
    calls = []
    logger = SimpleNamespace(warning=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(warning_baseline_module.sphinx_logging, "getLogger", lambda _: logger)

    warning_baseline_module._check_filter(app, None)

    assert len(calls) == 1
    assert calls[0][1] == {
        "type": "autodoc_baseline",
        "subtype": "changed",
    }
