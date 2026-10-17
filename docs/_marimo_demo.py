"""Export the browser-only results demo (``docs/demo/``) into the HTML build.

``marimo export html-wasm --single-file`` writes one small page that fetches
marimo and Pyodide from a CDN when viewed. The bundled data in
``docs/demo/public/`` is copied beside it, where ``mo.notebook_location()``
resolves it at view time, along with the README preview still.
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
            str(out / "index.html"),
            "--single-file",
            "--mode",
            "run",
            "--no-show-code",
            "--force",
        ],
        check=True,
    )
    if (DEMO / "public").is_dir():
        shutil.copytree(DEMO / "public", out / "public")
    if PREVIEW.is_file():
        shutil.copyfile(PREVIEW, out / "preview.png")


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("build-finished", _export)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
