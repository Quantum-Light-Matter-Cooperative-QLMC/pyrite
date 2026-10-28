"""Publish cached marimo tutorials as native MyST-NB documents.

The explicit regeneration CLI is the *only* place where PyRITE simulations run.
Sphinx validates committed captures and never invokes marimo or downloads tables.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
MOLAB = (
    "https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/"
    "blob/main/docs/notebooks/"
)
SCHEMA = "pyrite.myst-nb-tutorial.v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_version():
    return tomllib.loads((DOCS.parent / "pyproject.toml").read_text())["project"]["version"]


def validate_source(path):
    """Reject app/private imports and imports outside the governed public surface."""
    source = path.read_text()
    inventory = tomllib.loads((DOCS.parent / "api-surface.toml").read_text())
    exports = {module: dict(names) for module, names in inventory["package_exports"].items()}
    exports["pyrite"] = inventory["root_exports"]
    for name, entry in inventory["root_exports"].items():
        domain = entry.get("domain")
        if domain in inventory["canonical_facades"]:
            exports.setdefault(f"pyrite.{domain}", {})[name] = entry
    for node in ast.walk(ast.parse(source)):
        imports = []
        if isinstance(node, ast.Import):
            imports = [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError(f"{path.name}: relative imports are not standalone")
            imports = [(node.module or "", alias.name) for alias in node.names]
        for module, name in imports:
            if module == "pyrite" or module.startswith("pyrite."):
                if module not in exports or (name is not None and name not in exports[module]):
                    raise ValueError(f"{path.name}: unsupported public API import {module}:{name}")
    if "molab-shield.svg" in source:
        raise ValueError(f"{path.name}: Molab badge belongs on the RTD page only")


def page_location(source):
    """Find either the legacy wrapper or its generated notebook successor."""
    candidates = [
        *NOTEBOOKS.glob(f"*/{source.stem}.md"),
        *NOTEBOOKS.glob(f"*/{source.stem}.ipynb"),
    ]
    if len(candidates) != 1:
        raise ValueError(f"{source.name}: expected exactly one tutorial page, got {candidates}")
    return candidates[0]


def page_context(page):
    """Preserve authored cross-links while migrating the old iframe wrapper."""
    if page.suffix == ".ipynb":
        metadata = json.loads(page.read_text())["metadata"]["pyrite_docs"]
        return metadata["title"], metadata["footer"]
    content = page.read_text()
    title = content.splitlines()[0].removeprefix("# ")
    if "## Continue learning" not in content or "## Canonical source" not in content:
        raise ValueError(f"{page}: expected Continue learning and Canonical source sections")
    footer = content.split("## Continue learning", 1)[1].split("## Canonical source", 1)[0].strip()
    # Only notebook-to-notebook URLs change suffix; links to guides and references stay .md.
    footer = re.sub(r"(\d\d-[a-z0-9-]+)\.md(?=[)#])", r"\1.ipynb", footer)
    return title, footer


def output_contains_figure(notebook):
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data or "image/svg+xml" in data:
                return True
            if "<img" in str(data.get("text/html", "")) or "<svg" in str(data.get("text/html", "")):
                return True
    return False


def suppress_model_warnings(notebook):
    """Omit known model-warning tracebacks explained in the tutorial prose."""
    pattern = re.compile(
        r"^[^\n]*:\d+: (?:LineOpticalDataWarning|CoherentSamplingWarning):"
        r"[^\n]*(?:\n[ \t]+[^\n]*)?\n?",
        re.MULTILINE,
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                output["text"] = pattern.sub("", output["text"])
                if not output["text"].strip():
                    continue
            outputs.append(output)
        cell["outputs"] = outputs


def captured_notebook(source, notebook, title, footer):
    """Add site-only material without changing the editable marimo source."""
    import nbformat

    if not output_contains_figure(notebook):
        raise ValueError(f"{source.stem}: Jupyter export lost the tutorial figures")
    if any(
        output.get("output_type") == "error"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ):
        raise ValueError(f"{source.stem}: exported notebook contains an execution error")

    badge = f"[![Open in Molab](https://marimo.io/molab-shield.svg)]({MOLAB}{source.name})"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        markdown = str(cell["source"])
        match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
        if not match:
            continue
        if match.group(1).strip() != title:
            raise ValueError(f"{source.name}: Marimo and documentation titles disagree")
        heading = f"# {title}\n\n{badge}"
        cell["source"] = markdown[: match.start()] + heading + markdown[match.end() :]
        break
    else:
        raise ValueError(f"{source.stem}: missing Markdown title cell")

    suppress_model_warnings(notebook)
    notebook["cells"].append(
        nbformat.v4.new_markdown_cell(
            "## Continue learning\n\n"
            + footer
            + f"\n\n{{download}}`Download canonical Marimo notebook <../{source.name}>`"
        )
    )
    notebook["metadata"]["pyrite_docs"] = {
        "schema": SCHEMA,
        "title": title,
        "footer": footer,
        "source_sha256": digest(source),
        "pyrite_version": project_version(),
        "backend": "cpu",
    }
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


def check_notebooks(app=None):
    """Validate captures during every Sphinx build; do not execute anything."""
    sources = sorted(NOTEBOOKS.glob("*.py"))
    if not sources:
        raise ValueError("No tutorial sources found")
    for source in sources:
        validate_source(source)
        page = page_location(source)
        if page.suffix != ".ipynb":
            raise ValueError(
                f"{source.stem}: old iframe page has not been migrated; run "
                "uv run --group docs python docs/_notebooks.py --regenerate-all"
            )
        notebook = json.loads(page.read_text())
        metadata = notebook.get("metadata", {}).get("pyrite_docs", {})
        if (
            metadata.get("schema") != SCHEMA
            or metadata.get("source_sha256") != digest(source)
            or metadata.get("pyrite_version") != project_version()
        ):
            raise ValueError(f"stale tutorial capture: {source.stem}; regenerate its .ipynb")
        if not output_contains_figure(notebook):
            raise ValueError(f"{source.stem}: missing captured notebook figure")
        if any(
            output.get("output_type") == "error"
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        ):
            raise ValueError(f"{source.stem}: captured execution error")
        badge = f"{MOLAB}{source.name}"
        markdown = [
            str(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        ]
        if not markdown or badge not in markdown[0]:
            raise ValueError(f"{source.stem}: missing Molab badge beneath tutorial title")


def regenerate(names):
    """Explicitly execute tutorials, then atomically promote checked ipynb pages."""
    import nbformat

    sources = [NOTEBOOKS / f"{name}.py" for name in names]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"unknown tutorial: {source.stem}")
        validate_source(source)
    env = dict(os.environ, PYRITE_MC_BACKEND="cpu", MPLBACKEND="Agg")
    env.setdefault("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "128")
    env.setdefault("MPLCONFIGDIR", "/tmp/pyrite-mpl-cache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/pyrite-notebook-numba")
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", *map(str, sources)],
        check=True,
        env=env,
    )
    # Stage *all* exports before touching tracked documentation pages. Failures
    # cannot leave a partially migrated published curriculum.
    with tempfile.TemporaryDirectory(prefix="pyrite-notebooks-") as temporary:
        directory = Path(temporary)
        completed = []
        for source in sources:
            page = page_location(source)
            title, footer = page_context(page)
            staged_source = directory / source.name
            shutil.copyfile(source, staged_source)
            exported = directory / f"{source.stem}.ipynb"
            print(f"Capturing tutorial: {source.stem}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "ipynb",
                    str(staged_source),
                    "--include-outputs",
                    "--sort=top-down",
                    "--no-sandbox",
                    "-o",
                    str(exported),
                ],
                check=True,
                env=env,
                cwd=DOCS.parent,
            )
            notebook = captured_notebook(
                source, nbformat.read(exported, as_version=4), title, footer
            )
            nbformat.validate(notebook)
            nbformat.write(notebook, exported)
            completed.append((page, exported))
        for old_page, exported in completed:
            destination = old_page.with_suffix(".ipynb")
            # The exported document is the canonical Sphinx page, with the same docname.
            local_stage = destination.with_suffix(".ipynb.tmp")
            shutil.copyfile(exported, local_stage)
            os.replace(local_stage, destination)
            if old_page != destination:
                old_page.unlink()
            print(destination)


def setup(app):
    app.connect("builder-inited", check_notebooks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regenerate", metavar="NAME")
    group.add_argument("--regenerate-all", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_notebooks()
    elif args.regenerate_all:
        regenerate([source.stem for source in sorted(NOTEBOOKS.glob("*.py"))])
    else:
        regenerate([args.regenerate])
