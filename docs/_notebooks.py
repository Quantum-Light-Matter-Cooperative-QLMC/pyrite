"""Build disposable tutorial HTML from the canonical marimo sources."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
MOLAB = "https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/"


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
    pages = list(NOTEBOOKS.glob(f"*/{path.stem}.md"))
    if len(pages) != 1:
        raise ValueError(f"{path.name}: expected one notebook-specific RTD page")
    lines = [line for line in pages[0].read_text().splitlines() if line.strip()]
    badge = f"[![Open in Molab](https://marimo.io/molab-shield.svg)]({MOLAB}{path.name})"
    if len(lines) < 2 or not lines[0].startswith("# ") or lines[1] != badge:
        raise ValueError(f"{path.name}: Molab badge must immediately follow the RTD title")


def export_notebooks(app):
    """Execute each tutorial once, writing only inside the build and a tempdir."""
    sources = sorted(NOTEBOOKS.glob("*.py"))
    for source in sources:
        validate_source(source)
    if app.builder.format != "html":
        return
    output = Path(app.outdir) / "_static" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYRITE_MC_BACKEND="cpu", MPLBACKEND="Agg")
    # Transport reserves Ne * max_steps rows, even for thin tutorial slabs.
    # Begin with 128 steps; longer trajectories trigger deterministic retries.
    # The electron count and physical stopping criteria are unchanged.
    env["PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS"] = "128"
    env.setdefault("MPLCONFIGDIR", "/tmp/pyrite-mpl-cache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/pyrite-notebook-numba")
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", *map(str, sources)],
        check=True,
        env=env,
    )
    with tempfile.TemporaryDirectory(prefix="pyrite-notebooks-") as temporary:
        for source in sources:
            staged = Path(temporary) / source.name
            shutil.copyfile(source, staged)
            print(f"Exporting tutorial: {source.stem}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "html",
                    str(staged),
                    "--include-code",
                    "--no-sandbox",
                    "-o",
                    str(output / f"{source.stem}.html"),
                ],
                check=True,
                env=env,
                cwd=DOCS.parent,
            )


def setup(app):
    app.connect("builder-inited", export_notebooks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
