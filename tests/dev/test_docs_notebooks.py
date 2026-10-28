"""Executable tutorial sources and captured outputs stay one scientific example."""

import ast
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).parents[2] / "docs"


def test_released_tutorial_environment_pins_compatible_endf_parser():
    for path in (DOCS / "notebooks").glob("*.py"):
        metadata = path.read_text().split("# /// script\n", 1)[1].split("# ///", 1)[0]
        script = tomllib.loads("\n".join(line.removeprefix("# ") for line in metadata.splitlines()))
        # The published 0.22.5 loader expects the pre-0.18 MF26 dictionary layout.
        # Revisit this pin when the notebooks move to a release with the adapter.
        assert "endf-parserpy==0.17.0" in script["dependencies"], path.name


def test_tutorials_use_molab_package_manager_instructions():
    for path in (DOCS / "notebooks").glob("*.py"):
        assert "pip install" not in path.read_text(), path.name
    setup = (DOCS / "notebooks" / "01-first-spectrum.py").read_text()
    assert "built-in package manager" in setup
    assert "enter `endf-parserpy==0.17.0`" in setup


def test_prose_and_boilerplate_cells_hide_code():
    for path in (DOCS / "notebooks").glob("*.py"):
        for cell in ast.parse(path.read_text()).body:
            if not isinstance(cell, ast.FunctionDef):
                continue
            markdown = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mo"
                and node.func.attr == "md"
                for node in ast.walk(cell)
            )
            imports_only = all(
                isinstance(node, (ast.Import, ast.ImportFrom, ast.Return)) for node in cell.body
            )
            bootstrap = any(
                isinstance(node, ast.Import) and any(a.name == "pyrite" for a in node.names)
                for node in cell.body
            )
            if markdown or imports_only or bootstrap:
                decorator = cell.decorator_list[0]
                assert isinstance(decorator, ast.Call), (path.name, cell.lineno)
                assert any(
                    keyword.arg == "hide_code"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in decorator.keywords
                ), (path.name, cell.lineno)


def test_transport_depends_on_completed_table_fetch():
    for path in (DOCS / "notebooks").glob("*.py"):
        cells = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)]
        for cell in cells:
            transport = [
                node
                for node in ast.walk(cell)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"simulate", "directional_source"}
            ]
            if not transport:
                continue
            assert "tables_ready" in {arg.arg for arg in cell.args.args}, path.name
            assert any(
                isinstance(statement, ast.Assert)
                and isinstance(statement.test, ast.Name)
                and statement.test.id == "tables_ready"
                and statement.lineno < min(call.lineno for call in transport)
                for statement in cell.body
            ), path.name
            bootstrap = cells[0]
            ready = next(
                n
                for n in bootstrap.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "tables_ready" for t in n.targets)
            )
            fetch = next(
                n
                for n in ast.walk(bootstrap)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "fetch_tables"
            )
            assert fetch.lineno < ready.lineno, path.name
            assert any(
                isinstance(n, ast.Name) and n.id == "tables_ready"
                for n in ast.walk(bootstrap.body[-1])
            ), path.name


@pytest.fixture
def notebooks():
    spec = importlib.util.spec_from_file_location("docs_notebooks", DOCS / "_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_captures_render_without_execution(notebooks, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("checking captures must not execute notebooks, transport, or downloads")

    monkeypatch.setattr(notebooks.subprocess, "run", forbidden)
    monkeypatch.setattr(notebooks.subprocess, "check_output", forbidden)
    notebooks.check_all()
    for name, count in notebooks.NOTEBOOK_FIGURES.items():
        data = notebooks.checked_notebook(name)
        assert len(data["git_revision"]) == 40
        assert data["execution"]["backend"] == "cpu"
        _, session = notebooks.read_session(
            DOCS / "notebooks" / "__marimo__" / "session" / f"{name}.py.json"
        )
        assert len(session.figures) == count
        if count:
            assert session.figures[0].startswith(b"\x89PNG\r\n\x1a\n")


def test_combined_setup_capture_shows_environment_and_figure_with_hidden_provenance(notebooks):
    session, capture = notebooks.read_session(
        DOCS / "notebooks" / "__marimo__" / "session" / "01-first-spectrum.py.json"
    )
    assert capture.provenance["tutorial"] == "01-first-spectrum"
    assert len(capture.figures) == 1
    html = [
        value
        for cell in session["cells"]
        for output in cell["outputs"]
        for mime, value in output.get("data", {}).items()
        if mime == "text/html"
    ]
    summary = next(value for value in html if "selected_backend" in value)
    provenance = next(value for value in html if "source_identity" in value)
    assert "cupy_installed" in summary
    assert "display:none" not in summary.replace(" ", "")
    assert "display:none" in provenance.replace(" ", "")
    assert not (DOCS / "notebooks" / "00-setup.py").exists()
    legacy_page = (DOCS / "notebooks" / "fundamentals" / "00-setup.md").read_text()
    assert "(01-first-spectrum.md)" in legacy_page


def test_changed_source_and_capture_are_rejected(notebooks, monkeypatch, tmp_path):
    import shutil

    shutil.copytree(DOCS / "notebooks", tmp_path / "docs" / "notebooks")
    shutil.copytree(DOCS / "_static" / "notebooks", tmp_path / "docs" / "_static" / "notebooks")
    shutil.copyfile(DOCS.parent / "api-surface.toml", tmp_path / "api-surface.toml")
    shutil.copyfile(DOCS.parent / "pyproject.toml", tmp_path / "pyproject.toml")
    monkeypatch.setattr(notebooks, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(notebooks, "NOTEBOOKS", tmp_path / "docs" / "notebooks")
    source = tmp_path / "docs" / "notebooks" / "01-first-spectrum.py"
    original = source.read_text()
    source.write_text(original + "\n# changed source\n")
    with pytest.raises(ValueError, match="stale notebook output"):
        notebooks.checked_notebook("01-first-spectrum")
    source.write_text(original)
    capture = tmp_path / "docs" / "_static" / "notebooks" / "01-first-spectrum.html"
    capture.write_text("corrupt output")
    with pytest.raises(ValueError, match="stale notebook artifact"):
        notebooks.checked_notebook("01-first-spectrum")


@pytest.mark.parametrize(
    "import_statement",
    [
        "from pyrite.apps import analysis_app",
        "import pyrite._backend",
        "from pyrite.beams import nonexistent",
        "from pyrite import *",
        "from .helpers import simulate",
    ],
)
def test_unsupported_imports_fail_before_execution(notebooks, tmp_path, import_statement):
    source = tmp_path / "01-first-spectrum.py"
    source.write_text(
        import_statement
        + f"\n# {notebooks.MOLAB}{source.name}\n# https://marimo.io/molab-shield.svg\n"
    )
    with pytest.raises(ValueError, match="import"):
        notebooks.validate_source(source)


def test_notebook_links_source_and_navigation_agree(notebooks):
    index = (DOCS / "index.md").read_text()
    # Check stable Sphinx destinations instead of presentation labels, whose
    # capitalization and display names are owned by the #759 docs overhaul.
    assert (
        index.index("<guides/getting-started>")
        < index.index("<notebooks/index>")
        < index.index("<guides/index>")
    )
    assert "fundamentals/index" in (DOCS / "notebooks" / "index.md").read_text()
    # Raw iframe URLs bypass Sphinx's download-link rewriting.
    assert 'html_static_path = ["_static"]' in (DOCS / "conf.py").read_text()
    group = (DOCS / "notebooks" / "fundamentals" / "index.md").read_text()
    for name in notebooks.NOTEBOOK_FIGURES:
        source = DOCS / "notebooks" / f"{name}.py"
        page = (DOCS / "notebooks" / "fundamentals" / f"{name}.md").read_text()
        assert name in group
        assert "molab-shield.svg" not in source.read_text()
        assert notebooks.MOLAB + source.name in page
        assert f"{{literalinclude}} ../{source.name}" in page
        assert f"../../_static/notebooks/{name}.html" in page
        assert f"../outputs/{name}.json" in page


def test_all_badges_follow_rtd_titles_only(notebooks):
    for source in (DOCS / "notebooks").glob("*.py"):
        notebooks.validate_source(source)
        assert "molab-shield.svg" not in source.read_text()


@pytest.mark.parametrize("placement", ["missing", "after_description", "inside_notebook"])
def test_badge_placement_is_validated(notebooks, monkeypatch, tmp_path, placement):
    monkeypatch.setattr(notebooks, "NOTEBOOKS", tmp_path)
    page = tmp_path / "fundamentals" / "example.md"
    page.parent.mkdir()
    badge = f"[![Open in Molab](https://marimo.io/molab-shield.svg)]({notebooks.MOLAB}example.py)"
    source = tmp_path / "example.py"
    source.write_text("# Standalone notebook\n")
    if placement == "missing":
        page.write_text("# Example\n\nDescription.\n")
    elif placement == "after_description":
        page.write_text(f"# Example\n\nDescription.\n\n{badge}\n")
    else:
        page.write_text(f"# Example\n\n{badge}\n")
        source.write_text(f"# {badge}\n")
    with pytest.raises(ValueError, match="Molab badge"):
        notebooks.validate_source(source)


def test_failed_or_incomplete_sessions_are_rejected(notebooks, tmp_path):
    path = tmp_path / "failed.py.json"
    path.write_text(json.dumps({"cells": [{"outputs": [{"type": "error", "evalue": "failed"}]}]}))
    with pytest.raises(ValueError, match="failed notebook cell"):
        notebooks.read_session(path)


def test_notebook_exports_explicitly_decline_sandbox_prompts(notebooks, monkeypatch):
    calls = []

    def export(command, **kwargs):
        calls.append(command)
        if "export" in command:
            assert "--no-sandbox" in command
        if "html" in command:
            raise RuntimeError("stop before promoting mocked captures")

    monkeypatch.setattr(notebooks.subprocess, "run", export)
    with pytest.raises(RuntimeError, match="stop before promoting"):
        notebooks.regenerate("03-pxr-line-energies")
    assert len(calls) == 3


def test_line_energy_notebook_cpu_smoke(monkeypatch):
    monkeypatch.setenv("PYRITE_MC_BACKEND", "cpu")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = DOCS / "notebooks" / "03-pxr-line-energies.py"
    spec = importlib.util.spec_from_file_location("line_energy_notebook", path)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    # Notebook startup fetches data for users, but unit tests must stay offline.
    import pyrite

    monkeypatch.setattr(pyrite, "fetch_tables", lambda: [])
    _, definitions = notebook.app.run()
    energies = definitions["energy_lines_eV"]
    assert energies.shape == (3, 131)
    assert np.isfinite(energies).all() and np.all(energies > 0)
    # Same basal Friedel pair: doubling Miller order doubles the vacuum resonance.
    first_order = definitions["tilting_lines_eV"][1]
    assert definitions["second_order_eV"] == pytest.approx(2 * first_order)
    assert definitions["tutorial_provenance"]["seed"] is None
    import matplotlib.pyplot as plt

    plt.close("all")
