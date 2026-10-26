"""Render checked tutorial outputs; regenerate simulations only on explicit request."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
import subprocess
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
)


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


def setup(app):
    app.connect("builder-inited", _check_outputs)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", required=True, choices=NAMES)
    parser.add_argument("--input", type=Path, help="saved input required by the example")
    args = parser.parse_args()
    regenerate(args.regenerate, args.input)
