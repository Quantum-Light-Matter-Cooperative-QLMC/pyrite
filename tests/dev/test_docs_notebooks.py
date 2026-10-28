"""Tutorial sources, public imports, navigation and disposable HTML export."""

import ast
import importlib.util
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
    assert 'nb_execution_mode = "off"' in (DOCS / "conf.py").read_text()
    for source in sorted(notebooks.NOTEBOOKS.glob("*.py")):
        name = source.stem
        page = notebooks.page_location(source)
        assert page.suffix == ".ipynb", name
        assert name in (page.parent / "index.md").read_text()
        notebook = __import__("json").loads(page.read_text())
        assert notebooks.MOLAB + source.name in str(notebook["cells"])
        assert "iframe" not in str(notebook["cells"])


def test_all_badges_follow_rtd_titles_only(notebooks):
    for source in (DOCS / "notebooks").glob("*.py"):
        notebooks.validate_source(source)
        assert "molab-shield.svg" not in source.read_text()


def test_missing_title_rejected_by_native_export(notebooks, tmp_path):
    import nbformat

    source = tmp_path / "example.py"
    source.write_text("# source\n")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("No heading")])
    with pytest.raises(ValueError, match="lost the tutorial figures"):
        notebooks.captured_notebook(source, nb, "Example", "Next")


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
    import matplotlib.pyplot as plt

    plt.close("all")


# Native capture/regeneration behavior is tested in test_docs_native_notebooks.py.
