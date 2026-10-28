"""Render checked tutorial outputs; regenerate simulations only on explicit request."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

DOCS = Path(__file__).resolve().parent
EXAMPLES = DOCS / "examples"
NAMES = (
    "python_api_workflow",
    "getting_started",
    "working_with_results",
    "configuration_cookbook",
    "directional_source_heatmap",
    "detector_observations",
    "gpt_gdf_beams",
    "transition_radiation_demo",
    "radiation_diagnostics",
    "experimental_planar_channeling",
    "experimental_axial_channeling",
    "optimization_studies",
    "adding_materials",
)

# Register additional notebooks from the Sphinx/CLI wrapper rather than changing
# _notebooks.py: that exporter's exact file digest is in existing capture manifests.
# Appending a tutorial must not invalidate unrelated, already captured notebooks.
ADDITIONAL_NOTEBOOK_FIGURES = {
    "04-beams-crystals-target-geometry": 2,
    "05-angular-radiation-distributions": 1,
    "06-detector-counts": 1,
    "07-sampling-convergence-performance": 1,
    "08-parameter-sweeps": 1,
}
_PIN = re.compile(r"(?P<prefix>\bpyrite-mc(?:\[[A-Za-z0-9,_.-]+\])?==)\d[\w.+-]*")


def configured_notebooks():
    """Reuse the sole marimo exporter with the complete published curriculum."""
    import _notebooks

    _notebooks.NOTEBOOK_FIGURES.update(ADDITIONAL_NOTEBOOK_FIGURES)
    return _notebooks


def sync_notebook_version(name):
    """Update an explicit tutorial's PyPI pins from pyproject before recapture.

    Also correct any install-command extras in the setup tutorial. Read-only docs
    checks never call this function, and no source changes occur without an
    explicit --regenerate-notebook action.
    """
    source = DOCS / "notebooks" / f"{name}.py"
    version = tomllib.loads((DOCS.parent / "pyproject.toml").read_text())["project"]["version"]
    previous = source.read_text()
    updated = _PIN.sub(lambda match: match.group("prefix") + version, previous)
    if f'"pyrite-mc=={version}"' not in updated:
        raise ValueError(f"{name}: missing required inline pyrite-mc pin")
    if updated == previous:
        return False
    source.write_text(updated)
    print(f"Synchronized {source} to pyrite-mc=={version}")
    return True


def load_example(name):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tutorial_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_digest(name):
    return hashlib.sha256((EXAMPLES / f"{name}.py").read_bytes()).hexdigest()


def checked_output(name):
    payload = json.loads((EXAMPLES / "data" / f"{name}.json").read_text())
    if payload["source_sha256"] != source_digest(name):
        raise ValueError(
            f"stale tutorial output: {name}; regenerate with "
            f"uv run python docs/_tutorials.py --regenerate {name}"
        )
    output = payload["output"]
    for filename, digest in output.get("input_sha256", {}).items():
        if hashlib.sha256((EXAMPLES / "data" / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"stale tutorial input: {name}/{filename}; regenerate its output")
    for filename, digest in output.get("repository_input_sha256", {}).items():
        if hashlib.sha256((DOCS.parent / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"stale tutorial repository input: {name}/{filename}")
    if "script_sha256" in output:
        script = EXAMPLES / f"{name}.sh"
        if hashlib.sha256(script.read_bytes()).hexdigest() != output["script_sha256"]:
            raise ValueError(f"stale tutorial script: {name}; regenerate its output")
    if "display_text" in output:
        displayed = (EXAMPLES / "data" / f"{name}.txt").read_text()
        if displayed != output["display_text"]:
            raise ValueError(f"stale tutorial transcript: {name}")
    return output


def _check_outputs(app):
    for name in NAMES:
        checked_output(name)


def python_api_workflow():
    load_example("python_api_workflow").plot(checked_output("python_api_workflow"))


def working_with_results():
    load_example("working_with_results").plot(checked_output("working_with_results"))


def directional_source_heatmap():
    load_example("directional_source_heatmap").plot(checked_output("directional_source_heatmap"))


def detector_observations():
    load_example("detector_observations").plot(checked_output("detector_observations"))


def gpt_gdf_beams():
    load_example("gpt_gdf_beams").plot(checked_output("gpt_gdf_beams"))


def transition_radiation_demo(view="analytic"):
    load_example("transition_radiation_demo").plot(
        checked_output("transition_radiation_demo"), view=view
    )


def radiation_diagnostics(view="signal"):
    load_example("radiation_diagnostics").plot(checked_output("radiation_diagnostics"), view=view)


def experimental_planar_channeling():
    load_example("experimental_planar_channeling").plot(
        checked_output("experimental_planar_channeling")
    )


def experimental_axial_channeling(view="density"):
    load_example("experimental_axial_channeling").plot(
        checked_output("experimental_axial_channeling"), view=view
    )


def optimization_studies(view="finalists"):
    load_example("optimization_studies").plot(checked_output("optimization_studies"), view=view)


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported tutorial metadata: {type(value).__name__}")


def regenerate(name, input_path=None):
    import pyrite

    example = load_example(name)
    output = example.compute() if input_path is None else example.compute(input_path)
    payload = {
        "schema": "pyrite.tutorial.v1",
        "source_sha256": source_digest(name),
        "pyrite_version": pyrite.__version__,
        "python_version": platform.python_version(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DOCS, text=True
        ).strip(),
        "output": output,
    }
    path = EXAMPLES / "data" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, indent=2, allow_nan=False) + "\n")
    if "display_text" in output:
        path.with_suffix(".txt").write_text(output["display_text"])
    print(path)


def _check_notebooks(app):
    configured_notebooks().check_all()


def regenerate_notebooks():
    """Refresh every registered CPU tutorial through the canonical exporter."""
    notebooks = configured_notebooks()
    for name in notebooks.NOTEBOOK_FIGURES:
        sync_notebook_version(name)
        notebooks.regenerate(name)


def setup(app):
    app.connect("builder-inited", _check_outputs)
    app.connect("builder-inited", _check_notebooks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    notebooks = configured_notebooks()

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--regenerate", choices=NAMES)
    actions.add_argument("--regenerate-notebook", choices=notebooks.NOTEBOOK_FIGURES)
    actions.add_argument("--regenerate-notebooks", action="store_true")
    actions.add_argument("--check-notebooks", action="store_true")
    parser.add_argument("--input", type=Path, help="saved input required by the example")
    args = parser.parse_args()
    if args.regenerate_notebooks:
        regenerate_notebooks()
    elif args.regenerate_notebook:
        sync_notebook_version(args.regenerate_notebook)
        notebooks.regenerate(args.regenerate_notebook)
    elif args.check_notebooks:
        notebooks.check_all()
    else:
        regenerate(args.regenerate, args.input)
