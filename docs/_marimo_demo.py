"""Export the browser-only results demo (``docs/demo/``) into the HTML build.

``marimo export html-wasm`` writes the page plus marimo's frontend assets;
Pyodide and packages load from a CDN when viewed. The directory export (not
``--single-file``) is required: Pyodide's worker must be served from
``demo/assets/`` for ``mo.notebook_location()`` to resolve to ``demo/``, where
the bundled ``public/`` data sits. A ``--single-file`` page runs its worker
from a ``blob:`` URL, so the notebook cannot locate its data. The README
preview still is copied beside it as the no-JS fallback.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

DOCS = Path(__file__).resolve().parent
DEMO = DOCS / "demo"
# README still written by scripts/export_docs_demo.py; the overview page's
# no-JS fallback. Copied by hand because the raw-HTML embed is not tracked.
PREVIEW = DOCS / "_static" / "demo" / "results-demo-preview.png"


def _export(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    out = Path(app.outdir) / "demo"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(DEMO / "results_app.py"),
            "--output",
            str(out),
            "--mode",
            "run",
            "--no-show-code",
            "--force",
            "--no-sandbox",
        ],
        check=True,
        # WASM export resolves the head file against its short notebook name.
        # Keep relative notebook assets valid in both run and export modes.
        cwd=DEMO,
    )
    # marimo drops an assistant prompt file into exports; it is not site content.
    (out / "CLAUDE.md").unlink(missing_ok=True)
    if (DEMO / "public").is_dir():
        shutil.copytree(DEMO / "public", out / "public", dirs_exist_ok=True)
    branding = DOCS.parent / "src" / "pyrite" / "_assets" / "branding"
    (out / "public").mkdir(exist_ok=True)
    shutil.copyfile(branding / "pyrite-wordmark.png", out / "public" / "pyrite-wordmark.png")
    if PREVIEW.is_file():
        shutil.copyfile(PREVIEW, out / "preview.png")


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("build-finished", _export)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
