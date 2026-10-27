"""Checked marimo artifacts for ``_tutorials``; never execute during Sphinx builds."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from html.parser import HTMLParser
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
STATIC = DOCS / "_static" / "notebooks"
# Number of primary figures required in each captured session.
NOTEBOOK_FIGURES = {"00-setup": 0, "01-first-spectrum": 1, "03-pxr-line-energies": 1}
MOLAB = "https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if MOLAB + path.name not in source:
        raise ValueError(f"{path.name}: missing notebook-specific Molab URL")
    if "https://marimo.io/molab-shield.svg" not in source:
        raise ValueError(f"{path.name}: missing Molab badge")


class _Capture(HTMLParser):
    def __init__(self):
        super().__init__()
        self.provenance = None
        self.figures = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "marimo-json-output":
            value = json.loads(attributes["data-json-data"])
            if isinstance(value, dict) and "tutorial" in value:
                self.provenance = value
        if tag == "img" and attributes.get("src", "").startswith("data:image/png;base64,"):
            self.figures.append(base64.b64decode(attributes["src"].split(",", 1)[1]))


def read_session(path):
    session = json.loads(path.read_text())
    capture = _Capture()
    for cell in session["cells"]:
        for output in cell["outputs"]:
            if output["type"] == "error":
                raise ValueError(f"failed notebook cell in {path.name}: {output.get('evalue')}")
            for mime, value in output.get("data", {}).items():
                if mime == "application/vnd.marimo+mimebundle":
                    bundle = json.loads(value)
                    if "image/png" in bundle:
                        capture.figures.append(
                            base64.b64decode(bundle["image/png"].split(",", 1)[1])
                        )
                elif mime.startswith("text/"):
                    capture.feed(value)
    return session, capture


def checked_notebook(name):
    source = NOTEBOOKS / f"{name}.py"
    validate_source(source)
    manifest = json.loads((NOTEBOOKS / "outputs" / f"{name}.json").read_text())
    if manifest["schema"] != "pyrite.marimo-tutorial.v1":
        raise ValueError(f"unsupported notebook capture schema: {name}")
    if manifest["source_sha256"] != digest(source):
        raise ValueError(f"stale notebook output: {name}; regenerate its captured outputs")
    if manifest["exporter_sha256"] != digest(Path(__file__)):
        raise ValueError(f"stale notebook exporter: {name}; regenerate its captured outputs")
    required_artifacts = {
        f"notebooks/__marimo__/session/{name}.py.json",
        f"_static/notebooks/{name}.html",
        *(f"_static/notebooks/{name}-{index + 1}.png" for index in range(NOTEBOOK_FIGURES[name])),
    }
    if not required_artifacts <= manifest["artifacts"].keys():
        raise ValueError(f"missing notebook artifacts: {name}")
    for relative, expected in manifest["artifacts"].items():
        path = DOCS / relative
        if not path.resolve().is_relative_to(DOCS.resolve()) or digest(path) != expected:
            raise ValueError(f"stale notebook artifact: {relative}")
    _, capture = read_session(NOTEBOOKS / "__marimo__" / "session" / f"{name}.py.json")
    if len(capture.figures) < NOTEBOOK_FIGURES[name]:
        raise ValueError(f"missing notebook figure: {name}")
    provenance = capture.provenance
    required = {"tutorial", "pyrite_version", "backend", "seed", "table_identity"}
    if not provenance or not required <= provenance.keys() or provenance["tutorial"] != name:
        raise ValueError(f"missing notebook provenance: {name}")
    version = tomllib.loads((DOCS.parent / "pyproject.toml").read_text())["project"]["version"]
    if (
        provenance["pyrite_version"] != version
        or f'"pyrite-mc=={version}"' not in source.read_text()
    ):
        raise ValueError(f"stale notebook PyRITE version: {name}; update inline pin and recapture")
    if provenance != manifest["execution"]:
        raise ValueError(f"inconsistent notebook provenance: {name}")
    return manifest


def check_all():
    for name in NOTEBOOK_FIGURES:
        checked_notebook(name)


def regenerate(name):
    """Explicit CPU execution, with artifacts promoted only after successful exports."""
    source = NOTEBOOKS / f"{name}.py"
    validate_source(source)
    source_sha256 = digest(source)
    env = dict(os.environ, PYRITE_MC_BACKEND="cpu", MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/pyrite-mpl-cache")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/pyrite-notebook-numba")
    with tempfile.TemporaryDirectory(prefix="pyrite-notebook-") as temporary:
        temporary = Path(temporary)
        staged = temporary / source.name
        staged.write_bytes(source.read_bytes())
        for arguments in (
            ["check", str(staged)],
            ["export", "session", str(staged), "--no-sandbox", "--no-continue-on-error"],
            [
                "export",
                "html",
                str(staged),
                "--include-code",
                "-o",
                str(staged.with_suffix(".html")),
            ],
        ):
            subprocess.run([sys.executable, "-m", "marimo", *arguments], check=True, env=env)
        session_path = temporary / "__marimo__" / "session" / f"{name}.py.json"
        _, capture = read_session(session_path)
        if capture.provenance is None or len(capture.figures) < NOTEBOOK_FIGURES[name]:
            raise ValueError(f"incomplete notebook capture: {name}")
        if digest(source) != source_sha256:
            raise ValueError(f"notebook source changed during export: {name}")
        artifacts = {
            NOTEBOOKS / "__marimo__" / "session" / session_path.name: session_path.read_bytes(),
            STATIC / f"{name}.html": (
                "\n".join(
                    line.rstrip() for line in staged.with_suffix(".html").read_text().splitlines()
                )
                + "\n"
            ).encode(),
        }
        for index, figure in enumerate(capture.figures):
            artifacts[STATIC / f"{name}-{index + 1}.png"] = figure
        for target, content in artifacts.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    manifest = {
        "schema": "pyrite.marimo-tutorial.v1",
        "source_sha256": source_sha256,
        "exporter_sha256": digest(Path(__file__)),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DOCS, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=DOCS)),
        "python_version": platform.python_version(),
        "execution": capture.provenance,
        "artifacts": {str(path.relative_to(DOCS)): digest(path) for path in artifacts},
    }
    destination = NOTEBOOKS / "outputs" / f"{name}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    checked_notebook(name)
    print(destination)
