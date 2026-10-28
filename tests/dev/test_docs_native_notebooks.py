"""Focused tests for publishing checked native notebooks without RTD execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest


def test_capture_builder_never_executes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_notebooks", Path(__file__).parents[2] / "docs/_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = tmp_path / "docs"
    sources = docs / "notebooks"
    (sources / "fundamentals").mkdir(parents=True)
    (sources / "first.py").write_text("# first\n")
    monkeypatch.setattr(module, "DOCS", docs)
    monkeypatch.setattr(module, "NOTEBOOKS", sources)
    monkeypatch.setattr(module, "validate_source", lambda source: None)
    monkeypatch.setattr(module, "project_version", lambda: "0.22.5")
    source = sources / "first.py"
    captured = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# First\n\n[![Open in Molab](https://marimo.io/molab-shield.svg)]"
                f"({module.MOLAB}first.py)"
            ),
            nbformat.v4.new_code_cell(
                "1 + 1",
                outputs=[
                    nbformat.v4.new_output("display_data", data={"image/png": "iVBORw0KGgo="})
                ],
            ),
        ],
        metadata={
            "pyrite_docs": {
                "schema": module.SCHEMA,
                "source_sha256": module.digest(source),
                "pyrite_version": "0.22.5",
            }
        },
    )
    nbformat.write(captured, sources / "fundamentals/first.ipynb")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("build ran marimo"))
    app = SimpleNamespace(builder=SimpleNamespace(format="html"))
    module.check_notebooks(app)


def test_export_migrates_wrapper_and_retains_hidden_input(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_notebooks", Path(__file__).parents[2] / "docs/_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    docs = tmp_path / "docs"
    sources = docs / "notebooks"
    page = sources / "fundamentals/first.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# First\n\n[badge]\n\n## Continue learning\n\nNext: [Second](02-second.md).\n\n## Canonical source\n"
    )
    (sources / "first.py").write_text("# marimo source\n")
    monkeypatch.setattr(module, "DOCS", docs)
    monkeypatch.setattr(module, "NOTEBOOKS", sources)
    monkeypatch.setattr(module, "validate_source", lambda source: None)
    monkeypatch.setattr(module, "project_version", lambda: "0.22.5")
    calls = []

    def mock_run(command, **kwargs):
        calls.append(command)
        if "export" in command:
            assert "ipynb" in command and "--no-sandbox" in command
            assert "--include-outputs" in command and "--sort=top-down" in command
            exported = Path(command[command.index("-o") + 1])
            nbformat.write(
                nbformat.v4.new_notebook(
                    cells=[
                        nbformat.v4.new_code_cell(
                            "import marimo", metadata={"tags": ["remove-input"]}
                        ),
                        nbformat.v4.new_markdown_cell("# First\n\nNotebook prose"),
                        nbformat.v4.new_code_cell(
                            "figure",
                            outputs=[
                                nbformat.v4.new_output(
                                    "display_data", data={"image/png": "iVBORw0KGgo="}
                                )
                            ],
                        ),
                    ]
                ),
                exported,
            )

    monkeypatch.setattr(module.subprocess, "run", mock_run)
    module.regenerate(["first"])
    assert not page.exists()
    exported = page.with_suffix(".ipynb")
    nb = nbformat.read(exported, as_version=4)
    assert nb.cells[0]["metadata"]["tags"] == ["remove-input"]
    assert f"{module.MOLAB}first.py" in nb.cells[1].source
    assert "02-second.ipynb" in nb.cells[-1].source
    assert nb.metadata["pyrite_docs"]["source_sha256"] == module.digest(sources / "first.py")
    assert len(calls) == 2  # marimo check, then a single captured ipynb export
    module.check_notebooks()


def test_failed_export_leaves_old_pages_intact(tmp_path, monkeypatch):
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "_notebooks", Path(__file__).parents[2] / "docs/_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = tmp_path / "notebooks"
    page = sources / "fundamentals/first.md"
    page.parent.mkdir(parents=True)
    page.write_text("# First\n\n[badge]\n\n## Continue learning\n\nNext\n\n## Canonical source\n")
    (sources / "first.py").write_text("# source\n")
    monkeypatch.setattr(module, "DOCS", tmp_path)
    monkeypatch.setattr(module, "NOTEBOOKS", sources)
    monkeypatch.setattr(module, "validate_source", lambda source: None)

    def fail(command, **kwargs):
        if "export" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.regenerate(["first"])
    assert page.exists()
    assert not page.with_suffix(".ipynb").exists()
