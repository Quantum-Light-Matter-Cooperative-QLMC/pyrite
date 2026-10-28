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
                                    "stream",
                                    name="stderr",
                                    text=(
                                        "before\n"
                                        "/usr/lib/python3.14/contextlib.py:148: "
                                        "LineOpticalDataWarning: unavailable optical data\n"
                                        "  next(self.gen)\n"
                                        "example.py:7: UserWarning: keep this warning\n"
                                        "  warn('keep this warning')\n"
                                    ),
                                ),
                                nbformat.v4.new_output(
                                    "stream",
                                    name="stderr",
                                    text=(
                                        "C:/project/spectrum.py:328: "
                                        "LineOpticalDataWarning: unavailable optical data\n"
                                        "  publish_line_data_coverage(coverage)\n"
                                        "C:/project/transition_emission.py:134: "
                                        "CoherentSamplingWarning: unresolved bunch excess\n"
                                        "  spectrum = crossing_transition_radiation()\n"
                                    ),
                                ),
                                nbformat.v4.new_output(
                                    "stream", name="stdout", text="result: 42\n"
                                ),
                                nbformat.v4.new_output(
                                    "display_data", data={"image/png": "iVBORw0KGgo="}
                                ),
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
    assert "Captured results" not in nb.cells[1].source
    outputs = nb.cells[2].outputs
    assert len(outputs) == 3
    assert outputs[0].text == (
        "before\nexample.py:7: UserWarning: keep this warning\n  warn('keep this warning')\n"
    )
    assert outputs[1].text == "result: 42\n"
    assert "image/png" in outputs[2].data
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


def test_docs_links_target_exported_notebooks_not_deleted_md_wrappers():
    """Avoid Sphinx warnings for .md links after MyST-NB takes over the pages."""
    import re

    docs = Path(__file__).parents[2] / "docs"
    notebook_names = {source.stem for source in (docs / "notebooks").glob("*.py")}
    assert notebook_names
    for source in docs.rglob("*.md"):
        if "_build" in source.parts or "_autosummary" in source.parts:
            continue
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
            path = target.split("#", 1)[0]
            basename = path.rsplit("/", 1)[-1]
            if basename.endswith(".md") and basename[:-3] in notebook_names:
                raise AssertionError(
                    f"{source.relative_to(docs)} links to deleted notebook wrapper "
                    f"{target}; use .ipynb instead"
                )
            if basename.endswith(".ipynb") and basename[:-6] in notebook_names:
                assert (source.parent / path).is_file(), (
                    f"{source.relative_to(docs)}: missing notebook target {target}"
                )


def test_published_tutorial_outputs_have_no_json_dump_widgets():
    notebooks = Path(__file__).parents[2] / "docs/notebooks"
    for path in notebooks.glob("*/*.ipynb"):
        captured = nbformat.read(path, as_version=4)
        for cell in captured.cells:
            for output in cell.get("outputs", []):
                html = output.get("data", {}).get("text/html", "")
                assert "marimo-json-output" not in html, path.name


def test_environment_healthcheck_is_a_rendered_table():
    path = Path(__file__).parents[2] / "docs/notebooks/fundamentals/01-first-spectrum.ipynb"
    captured = nbformat.read(path, as_version=4)
    tables = [
        output.get("data", {}).get("text/html", "")
        for cell in captured.cells
        for output in cell.get("outputs", [])
    ]
    assert any("<table>" in html and "selected backend" in html for html in tables)
