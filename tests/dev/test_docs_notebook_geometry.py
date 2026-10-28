"""Geometry notebook capture, registration and explicit pin synchronization."""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"
NAME = "04-beams-crystals-target-geometry"


def _load_tutorials():
    spec = importlib.util.spec_from_file_location("docs_tutorials_for_test", DOCS / "_tutorials.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_version_updates_base_and_extras(tmp_path, monkeypatch):
    tutorials = _load_tutorials()
    docs = tmp_path / "docs"
    notebook_dir = docs / "notebooks"
    notebook_dir.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.22.5"\n')
    source = notebook_dir / "01-first-spectrum.py"
    source.write_text(
        '# dependencies = ["pyrite-mc==0.22.4", "numpy>=2"]\n'
        "uv pip install 'pyrite-mc[apps]==0.22.3'\n"
        "uv pip install 'pyrite-mc[nvidia]==0.22.3'\n"
        'unrelated_version = "0.22.3"\n'
    )
    monkeypatch.setattr(tutorials, "DOCS", docs)
    assert tutorials.sync_notebook_version("01-first-spectrum") is True
    assert '"pyrite-mc==0.22.5"' in source.read_text()
    assert "pyrite-mc[apps]==0.22.5" in source.read_text()
    assert "pyrite-mc[nvidia]==0.22.5" in source.read_text()
    assert 'unrelated_version = "0.22.3"' in source.read_text()
    assert tutorials.sync_notebook_version("01-first-spectrum") is False


def test_sync_version_rejects_notebook_without_inline_pin(tmp_path, monkeypatch):
    tutorials = _load_tutorials()
    docs = tmp_path / "docs"
    (docs / "notebooks").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.22.5"\n')
    (docs / "notebooks" / "missing.py").write_text("# no pin\n")
    monkeypatch.setattr(tutorials, "DOCS", docs)
    with pytest.raises(ValueError, match="inline pyrite-mc pin"):
        tutorials.sync_notebook_version("missing")


def test_geometry_notebook_registers_and_has_checked_capture(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS))
    tutorials = _load_tutorials()
    notebooks = tutorials.configured_notebooks()
    assert notebooks.NOTEBOOK_FIGURES[NAME] == 2
    manifest = notebooks.checked_notebook(NAME)
    data = manifest["execution"]
    assert data["backend"] == "cpu"
    assert data["beam_energy_keV"] == pytest.approx(30)
    assert data["spot_fwhm_x_mm"] == pytest.approx(0.45)
    assert data["material"] == "hopg"
    assert data["thickness_ang"] == pytest.approx(1_000)
    assert data["cut_beam_uvw"] == [0, 0, 2]
    assert data["sampled_cxr_histories"] == 2_048
    _, capture = notebooks.read_session(
        DOCS / "notebooks" / "__marimo__" / "session" / f"{NAME}.py.json"
    )
    assert len(capture.figures) >= 2


def test_designing_navigation():
    assert "designing/index" in (DOCS / "notebooks" / "index.md").read_text()
    assert NAME in (DOCS / "notebooks" / "designing" / "index.md").read_text()
    notebook = (DOCS / "notebooks" / f"{NAME}.py").read_text()
    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    assert "molab-shield.svg" not in notebook
    assert f"docs/notebooks/{NAME}.py" in page
    assert f"{{literalinclude}} ../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page


def test_all_notebooks_bootstrap_tables_before_simulation():
    import ast

    names = (
        "01-first-spectrum",
        "02-understanding-results",
        "03-pxr-line-energies",
        NAME,
        "05-angular-radiation-distributions",
        "06-detector-counts",
    )
    for name in names:
        source = (DOCS / "notebooks" / f"{name}.py").read_text()
        tree = ast.parse(source)
        first_cell = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(dec.func if isinstance(dec, ast.Call) else dec, ast.Attribute)
                and (dec.func if isinstance(dec, ast.Call) else dec).attr == "cell"
                for dec in node.decorator_list
            )
        )
        imports = [
            stmt
            for stmt in first_cell.body
            if isinstance(stmt, ast.Import) and any(alias.name == "pyrite" for alias in stmt.names)
        ]
        fetches = [
            stmt
            for stmt in first_cell.body
            if isinstance(stmt, (ast.Expr, ast.Assign))
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "pyrite"
            and stmt.value.func.attr == "fetch_tables"
        ]
        assert len(imports) == len(fetches) == 1, name
        assert imports[0].lineno < fetches[0].lineno, name
