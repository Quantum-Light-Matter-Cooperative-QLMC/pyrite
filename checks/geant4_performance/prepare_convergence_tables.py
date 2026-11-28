"""Stage exact ELSEPA refinement requests for #1041; no generator or histories."""

import argparse
import json
from pathlib import Path

from pyrite._util.hashing import file_sha256
from pyrite.perf import _benchmark_tables
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables

HERE = Path(__file__).resolve().parent


def make_plan(levels=(1, 2, 3)):
    spec = json.loads((HERE / "scenes.json").read_text())
    entries, scenes = [], []
    for scene in spec["scenes"]:
        material = spec["materials"][scene["material"]]
        composition = [(scene["material"], material["number_density_ang3"])]
        layer = resolve_layer_tables(composition)
        entries.extend(layer)
        coverage = [
            _benchmark_tables.table_coverage(e.arrays["energy_eV"], 1.25, scene["energy_keV"])
            for e in layer
        ]
        scenes.append(
            dict(
                scene=scene,
                composition=composition,
                native_keys=[t.key for e in layer for t in e.tables],
                coverage_at_L3_cutoff=coverage,
            )
        )
    result = _benchmark_tables.elastic_refinement_plan(entries, levels)
    result.update(
        schema="pyrite.geant4-performance.elsepa-refinement-plan.v1",
        issue=1041,
        purpose="accuracy-only",
        timing_comparison_eligible=False,
        scene_family="six-scene-W-19.3; not the 44-case-W-19.261 matrix",
        scenes=scenes,
        materials=spec["materials"],
        submission_ready=False,
        inputs_sha256=dict(
            scenes=file_sha256(HERE / "scenes.json"),
            planner=file_sha256(Path(__file__)),
            admission=file_sha256(Path(_benchmark_tables.__file__)),
        ),
        requirements=[
            "Evaluate every requested energy with the pinned independent generator",
            "Use an isolated output store; do not change catalogs or released bundles",
            "Retain source/build/deck/raw-panel inputs and outputs with SHA-256s",
            "Keep source/target/physics options, endpoints and native angular grid fixed",
            "Verify exact request keys and payloads with benchmark admission before transport",
            "No resampling/interpolation of existing knots can supply generator evidence",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True, help="new read-only planning record")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; compare/amend plans explicitly")
    result = make_plan(args.levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=1)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                native_tables=len(result["baselines"]),
                requests=len(result["requests"]),
                generation_performed=False,
                submission_ready=False,
            )
        )
    )


if __name__ == "__main__":
    main()
