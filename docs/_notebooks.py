"""Publish cached marimo tutorials as native MyST-NB documents.

The explicit regeneration CLI is the *only* place where PyRITE simulations run.
Sphinx validates committed captures and never invokes marimo or downloads tables.
A tutorial whose script metadata declares ``[tool.pyrite] backend = "cuda"`` is
captured on a CUDA device (the lab GPU through ``checks/tutorial_capture_cuda.sbatch``);
because a GPU is not always available, its stale capture warns instead of failing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
MOLAB = (
    "https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/"
    "blob/main/docs/notebooks/"
)
LEGACY_SCHEMA = "pyrite.myst-nb-tutorial.v1"
SCHEMA = "pyrite.myst-nb-tutorial.v2"
LEGACY_FINGERPRINT = "python-ast-pep723-v1"
FINGERPRINT = "python-ast-pep723-v2"
CAPTURE_BACKENDS = ("cpu", "cuda")
# PyRITE's own release pin; the version is provenance, like `pyrite_version`.
PYRITE_PIN = re.compile(r"^(pyrite-mc(?:\[([^\]]*)\])?)\s*==\s*[^;\s]+")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_metadata(content):
    """Read script metadata from real comments, not lookalikes inside strings."""
    metadata = []
    block = None
    for token in tokenize.generate_tokens(io.StringIO(content).readline):
        if token.type != tokenize.COMMENT or token.start[1] != 0:
            continue
        comment = token.string
        if comment == "# /// script":
            if block is not None or metadata:
                raise ValueError("duplicate or nested script metadata")
            block = []
        elif block is not None:
            if comment == "# ///":
                metadata.append(tomllib.loads("\n".join(block)))
                block = None
            else:
                block.append(comment.removeprefix("#").removeprefix(" "))
    if block is not None:
        raise ValueError("unterminated script metadata")
    return metadata


def unpinned_metadata(content):
    """Script metadata with PyRITE's own release pin reduced to its name and extras."""
    metadata = script_metadata(content)
    for block in metadata:
        if "dependencies" in block:
            block["dependencies"] = [PYRITE_PIN.sub(r"\1", dep) for dep in block["dependencies"]]
    return metadata


def source_fingerprint(content, *, legacy=False):
    """Ignore Python formatting/comments, retaining prose and script dependencies.

    Releases rewrite every tutorial's ``pyrite-mc==X.Y.Z`` pin, so a version
    change alone never stales a capture; the legacy v1 form hashed the pin.
    """
    metadata = script_metadata(content) if legacy else unpinned_metadata(content)
    canonical = json.dumps(
        [ast.dump(ast.parse(content), include_attributes=False), metadata],
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def fingerprint(path):
    return source_fingerprint(path.read_text())


def legacy_source(source, expected):
    """Recover the exact captured source, never infer provenance from current code."""
    if digest(source) == expected:
        return source.read_text()
    relative = source.relative_to(DOCS.parent).as_posix()
    revisions = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", relative],
        cwd=DOCS.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for revision in revisions:
        content = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=DOCS.parent,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(content).hexdigest() == expected:
            return content.decode()
    raise ValueError("captured source cannot be recovered from version control")


def migrate_hashes():
    """Upgrade proven equivalent legacy captures without running notebooks."""
    import nbformat

    rejected = []
    for source in sorted(NOTEBOOKS.glob("*.py")):
        page = page_location(source)
        notebook = nbformat.read(page, as_version=4)
        metadata = notebook.metadata.get("pyrite_docs", {})
        schema = metadata.get("schema")
        if schema == SCHEMA and metadata.get("source_fingerprint") == FINGERPRINT:
            continue
        try:
            canonical = fingerprint(source)
            if schema == SCHEMA and metadata.get("source_fingerprint") == LEGACY_FINGERPRINT:
                # The v1 hash covers the current source exactly, pin included.
                current = source_fingerprint(source.read_text(), legacy=True)
                if metadata.get("source_fingerprint_sha256") != current:
                    raise ValueError(
                        "code, prose or script dependencies changed; regenerate capture"
                    )
            elif schema == LEGACY_SCHEMA:
                original = legacy_source(source, metadata.get("source_sha256"))
                if source_fingerprint(original) != canonical:
                    raise ValueError(
                        "code, prose or script dependencies changed; regenerate capture"
                    )
            else:
                raise ValueError("capture schema is incompatible")
            metadata.update(
                schema=SCHEMA, source_fingerprint=FINGERPRINT, source_fingerprint_sha256=canonical
            )
            stage = page.with_suffix(".ipynb.tmp")
            nbformat.write(notebook, stage)
            os.replace(stage, page)
            print(f"Migrated capture fingerprint: {source.stem}")
        except (ValueError, subprocess.CalledProcessError) as error:
            rejected.append(source.stem)
            print(f"Cannot migrate {source.stem}: {error}", file=sys.stderr)
    return not rejected


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


def capture_backend(source):
    """The backend a tutorial is captured on: ``[tool.pyrite] backend``, else CPU."""
    backend = "cpu"
    for block in script_metadata(source.read_text()):
        backend = block.get("tool", {}).get("pyrite", {}).get("backend", backend)
    if backend not in CAPTURE_BACKENDS:
        raise ValueError(f"{source.name}: capture backend must be one of {CAPTURE_BACKENDS}")
    return backend


def checks_freshness(source):
    """False when script metadata sets ``[tool.pyrite] freshness = false``."""
    fresh = True
    for block in script_metadata(source.read_text()):
        fresh = block.get("tool", {}).get("pyrite", {}).get("freshness", fresh)
    if not isinstance(fresh, bool):
        raise ValueError(f"{source.name}: [tool.pyrite] freshness must be true or false")
    return fresh


def has_cuda(env):
    """True when a CUDA device is visible to cupy."""
    probe = "import cupy; assert cupy.cuda.runtime.getDeviceCount() > 0"
    return not subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True
    ).returncode


def require_cuda(env):
    """Fail before any export when a CUDA tutorial has no device to run on."""
    if not has_cuda(env):
        raise RuntimeError(
            "CUDA tutorial capture needs a CUDA device and the nvidia extra; run it on the "
            "lab GPU with checks/tutorial_capture_cuda.sbatch"
        )


def regenerate_command(sources):
    """The full regeneration command, including extras the tutorials declare."""
    extras = sorted(
        {
            extra.strip()
            for source in sources
            for block in script_metadata(source.read_text())
            for dep in block.get("dependencies", [])
            if (match := PYRITE_PIN.match(dep)) and match.group(2)
            for extra in match.group(2).split(",")
        }
    )
    if any(capture_backend(source) == "cuda" for source in sources):
        extras = sorted({*extras, "nvidia"})
    flags = "".join(f" --extra {extra}" for extra in extras)
    target = f"--regenerate {sources[0].stem}" if len(sources) == 1 else "--regenerate-all"
    return f"uv run --group docs{flags} python docs/_notebooks.py {target}"


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
    """Omit known model-warning tracebacks and matplotlib font-fallback noise.

    ``findfont`` lines come from the capture host lacking STIX/Computer Modern
    fonts; they say nothing about the tutorial.
    """
    pattern = re.compile(
        r"^[^\n]*:\d+: (?:LineOpticalDataWarning|CoherentSamplingWarning):"
        r"[^\n]*(?:\n[ \t]+[^\n]*)?\n?"
        r"|^findfont: [^\n]*\n?",
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
        "source_fingerprint": FINGERPRINT,
        "source_fingerprint_sha256": fingerprint(source),
        "pyrite_version": project_version(),
        "backend": capture_backend(source),
    }
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


def format_capture(path):
    """Ruff-format exported code cells as published; fingerprints ignore formatting."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--quiet",
            "--config",
            str(DOCS.parent / "pyproject.toml"),
            str(path),
        ],
        check=True,
    )


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
                + regenerate_command(sources)
            )
        notebook = json.loads(page.read_text())
        metadata = notebook.get("metadata", {}).get("pyrite_docs", {})
        if (
            metadata.get("schema") == LEGACY_SCHEMA
            or metadata.get("source_fingerprint") == LEGACY_FINGERPRINT
        ):
            raise ValueError(
                f"{source.stem}: legacy capture fingerprint; run "
                "uv run --group docs python docs/_notebooks.py --migrate-hashes"
            )
        # `pyrite_version` is provenance only: releases do not re-execute tutorials.
        if checks_freshness(source) and (
            metadata.get("schema") != SCHEMA
            or metadata.get("source_fingerprint") != FINGERPRINT
            or metadata.get("source_fingerprint_sha256") != fingerprint(source)
        ):
            message = f"stale tutorial capture: {source.stem}; run " + regenerate_command([source])
            if capture_backend(source) != "cuda":
                raise ValueError(message)
            # Not a Sphinx warning: -W builds must still pass without a GPU.
            print(f"warning: {message} on a CUDA host; publishing the old capture", file=sys.stderr)
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
    base = dict(os.environ, MPLBACKEND="Agg")
    base.setdefault("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "128")
    base.setdefault("MPLCONFIGDIR", "/tmp/pyrite-mpl-cache")
    base.setdefault("NUMBA_CACHE_DIR", "/tmp/pyrite-notebook-numba")
    envs = {source: dict(base, PYRITE_MC_BACKEND=capture_backend(source)) for source in sources}
    if any(env["PYRITE_MC_BACKEND"] == "cuda" for env in envs.values()):
        require_cuda(dict(base, PYRITE_MC_BACKEND="cuda"))
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", *map(str, sources)],
        check=True,
        env=dict(base, PYRITE_MC_BACKEND="cpu"),
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
                env=envs[source],
                cwd=DOCS.parent,
            )
            notebook = captured_notebook(
                source, nbformat.read(exported, as_version=4), title, footer
            )
            nbformat.validate(notebook)
            nbformat.write(notebook, exported)
            format_capture(exported)
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


def runtime_structure(content):
    """Retain runtime/module/cell options while excluding literal Markdown prose."""
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "app"
            and decorator.attr == "cell"
            or isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
            and decorator.func.attr == "cell"
            for decorator in node.decorator_list
        ):
            # Marimo injects these dependency parameters by name, not position.
            if (
                not node.args.defaults
                and not node.args.posonlyargs
                and not node.args.kwonlyargs
                and node.args.vararg is None
                and node.args.kwarg is None
            ):
                node.args.args.sort(key=lambda argument: argument.arg)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "mo"
            and node.func.attr == "md"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            node.args[0] = ast.Constant(value="<literal Markdown>")
    return ast.dump(tree, include_attributes=False)


def refreshed_notebook(source, exported, captured):
    """Reuse outputs only when every executable cell and dependency is unchanged."""
    metadata = captured.metadata["pyrite_docs"]
    if metadata.get("schema") not in {SCHEMA, LEGACY_SCHEMA}:
        raise ValueError("capture schema is incompatible")
    original = legacy_source(source, metadata.get("source_sha256"))
    if unpinned_metadata(original) != unpinned_metadata(source.read_text()):
        raise ValueError("script dependencies changed; regenerate capture")
    if runtime_structure(original) != runtime_structure(source.read_text()):
        raise ValueError("runtime or cell configuration changed; regenerate capture")
    old_cells = [cell for cell in captured.cells if cell.cell_type == "code"]
    new_cells = [cell for cell in exported.cells if cell.cell_type == "code"]
    old_code = [ast.dump(ast.parse(cell.source), include_attributes=False) for cell in old_cells]
    new_code = [ast.dump(ast.parse(cell.source), include_attributes=False) for cell in new_cells]
    if old_code != new_code:
        raise ValueError("executable cells changed; regenerate capture")
    for old, new in zip(old_cells, new_cells, strict=True):
        new.outputs = old.outputs
        new.execution_count = old.execution_count
    return captured_notebook(source, exported, metadata["title"], metadata["footer"])


def refresh_page(name):
    """Export presentation only, retaining proven unchanged captured outputs."""
    import nbformat

    source = NOTEBOOKS / f"{name}.py"
    validate_source(source)
    page = page_location(source)
    captured = nbformat.read(page, as_version=4)
    with tempfile.TemporaryDirectory(prefix="pyrite-notebook-page-") as temporary:
        exported = Path(temporary) / page.name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(source),
                "--sort=top-down",
                "--no-sandbox",
                "-o",
                str(exported),
            ],
            cwd=DOCS.parent,
            check=True,
        )
        notebook = refreshed_notebook(source, nbformat.read(exported, as_version=4), captured)
        nbformat.validate(notebook)
        nbformat.write(notebook, exported)
        format_capture(exported)
        stage = page.with_suffix(".ipynb.tmp")
        shutil.copyfile(exported, stage)
        os.replace(stage, page)


def setup(app):
    app.connect("builder-inited", check_notebooks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regenerate", metavar="NAME")
    group.add_argument("--regenerate-all", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--migrate-hashes", action="store_true")
    group.add_argument("--refresh-page", metavar="NAME")
    args = parser.parse_args()
    if args.refresh_page:
        refresh_page(args.refresh_page)
    elif args.migrate_hashes:
        sys.exit(0 if migrate_hashes() else 1)
    elif args.check:
        check_notebooks()
    elif args.regenerate_all:
        names = [source.stem for source in sorted(NOTEBOOKS.glob("*.py"))]
        cuda_names = [name for name in names if capture_backend(NOTEBOOKS / f"{name}.py") == "cuda"]
        if cuda_names and not has_cuda(dict(os.environ, PYRITE_MC_BACKEND="cuda")):
            print(
                "warning: no CUDA device; skipping CUDA tutorial(s) "
                + ", ".join(cuda_names)
                + " (keeping their existing captures)",
                file=sys.stderr,
            )
            names = [name for name in names if name not in cuda_names]
        regenerate(names)
    else:
        regenerate([args.regenerate])
