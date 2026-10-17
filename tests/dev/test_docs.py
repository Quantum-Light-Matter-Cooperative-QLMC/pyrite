"""Focused checks for the strict documentation warning boundary."""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrite.devtools.docs_paths import StaleDocPath, check_doc_paths, find_stale_doc_paths


@pytest.fixture(scope="module")
def warning_baseline_module():
    path = Path(__file__).parents[2] / "docs" / "_warning_baseline.py"
    spec = importlib.util.spec_from_file_location("pyrite_docs_warning_baseline", path)
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

    assert baseline.filter(_record(source_root / "pyrite" / "example.py")) is False
    generated = _record(generated_root / "pyrite.example.rst")
    generated.location = f"{generated_root / 'pyrite.example.rst'}:12:<autosummary>"
    assert baseline.filter(generated) is False
    assert baseline.filter(_record(tmp_path / "docs" / "guide.md")) is True
    assert (
        baseline.filter(_record(source_root / "pyrite" / "example.py", warning_type="ref")) is True
    )
    assert len(baseline.fingerprints) == 2

    monkeypatch.setenv("PYRITE_DOCS_SHOW_AUTODOC_WARNINGS", "1")
    assert baseline.filter(_record(source_root / "pyrite" / "visible.py")) is True


def test_changed_warning_fingerprint_emits_unsuppressed_failure(
    warning_baseline_module, monkeypatch, tmp_path: Path
) -> None:
    baseline = warning_baseline_module.AutodocWarningBaseline(tmp_path / "src")
    baseline.filter(_record(tmp_path / "src" / "pyrite" / "example.py"))
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


def test_literal_documentation_path_check_reports_missing_target(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "current.md").write_text("current", encoding="utf-8")
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text(
        '"""See ' + "docs/" + "current.md and " + "docs/" + 'retired-note.md."""\n',
        encoding="utf-8",
    )

    assert find_stale_doc_paths(tmp_path) == [
        StaleDocPath(Path("src/example.py"), 1, "docs/" + "retired-note.md")
    ]
    with pytest.raises(RuntimeError, match=r"src/example\.py:1: docs/" + r"retired-note\.md"):
        check_doc_paths(tmp_path)


def test_literal_documentation_path_check_ignores_agentdocs_substring(tmp_path: Path) -> None:
    source = tmp_path / "example.md"
    source.write_text("See agentdocs/README.md.\n", encoding="utf-8")

    check_doc_paths(tmp_path)


def test_literal_documentation_path_check_ignores_remember_notes(tmp_path: Path) -> None:
    source = tmp_path / ".remember" / "today.done.md"
    source.parent.mkdir()
    source.write_text("See " + "docs/" + "retired-note.md.\n", encoding="utf-8")

    check_doc_paths(tmp_path)


def test_validation_writeups_use_rendering_math_delimiters() -> None:
    """`\\(...\\)` and `\\[...\\]` reach the HTML build as literal text, with no warning."""
    root = Path(__file__).parents[2] / "docs" / "validation"
    offenders: list[str] = []
    for page in sorted(root.rglob("*.md")):
        if page.name == "formatting-style.md":
            continue
        fence: str | None = None
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            marker = line.lstrip()[:3]
            if marker in {"```", "~~~"}:
                fence = None if fence == marker else fence or marker
                continue
            if fence is None and (r"\(" in line or (r"\[" in line and r"\\[" not in line)):
                offenders.append(f"{page.relative_to(root)}:{number}")
    assert not offenders, f"non-rendering math delimiters: {offenders}"


def test_browser_demo_imports_only_pyodide_safe_modules() -> None:
    """The docs demo runs under Pyodide, which has no numba build for pyrite."""
    import ast

    path = Path(__file__).parents[2] / "docs" / "demo" / "results_app.py"
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"json", "pathlib", "marimo", "numpy", "plotly", "pyodide"}


def test_demo_export_joins_interleaved_segments_per_track() -> None:
    path = Path(__file__).parents[2] / "scripts" / "export_docs_demo.py"
    spec = importlib.util.spec_from_file_location("pyrite_export_docs_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nan = float("nan")
    # Two tracks, steps interleaved as transport records them: a0 b0 a1 b1.
    segments = [((0, 0, 0), (0, 0, 1), 0), ((5, 0, 0), (5, 0, 1), 1)]
    segments += [((0, 0, 1), (0, 0, 2), 0), ((5, 0, 1), (5, 0, 2), 1)]
    trace = {"x": [], "y": [], "z": [], "line": {"color": []}, "customdata": []}
    for start, end, track in segments:
        for point in (start, end, (nan, nan, nan)):
            for axis, value in zip("xyz", point, strict=True):
                trace[axis].append(value)
            trace["line"]["color"].append(float(track))
            trace["customdata"].append([0.0, 0.0, 0.0, float(track), -1.0])

    joined = module._polyline(trace)

    assert joined["z"] == [0.0, 1.0, 2.0, None, 0.0, 1.0, 2.0]
    assert joined["x"] == [0.0, 0.0, 0.0, None, 5.0, 5.0, 5.0]
