"""Forecast #1041 confirmation costs; never submit jobs or qualify a gate."""

import argparse
import json
from pathlib import Path

from pyrite._util.hashing import file_sha256
from pyrite.perf import _convergence_resources

HERE = Path(__file__).resolve().parent


def make_plan(design_path, pilot_path):
    design_hash = file_sha256(design_path)
    pilots = json.loads(pilot_path.read_text())
    if (
        pilots["schema"] != "pyrite.geant4-performance.convergence-pilot-summary.v1"
        or pilots["design_sha256"] != design_hash
    ):
        raise ValueError("pilot summary does not match frozen design")
    # Hash verification is retention integrity, not certification of the summaries.
    evidence = pilots["evidence_files_sha256"]
    for name, expected in evidence.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence paths must be relative to the pilot summary")
        if file_sha256(pilot_path.parent / path) != expected:
            raise ValueError(f"pilot evidence checksum mismatch: {name}")
    for row in pilots["rows"]:
        if not row["evidence_files"] or not set(row["evidence_files"]) <= evidence.keys():
            raise ValueError("every pilot row needs retained hashed evidence")
    result = _convergence_resources.resource_plan(json.loads(design_path.read_text()), pilots)
    result.update(
        design_sha256=design_hash,
        pilot_summary_sha256=file_sha256(pilot_path),
        evidence_files_sha256=evidence,
        planner_sha256=file_sha256(Path(_convergence_resources.__file__)),
        script_sha256=file_sha256(Path(__file__)),
        pilot_row_evidence={r["id"]: r["evidence_files"] for r in pilots["rows"]},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=HERE / "convergence-design-v1.json")
    parser.add_argument("--pilots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new planning JSON")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new; compare/amend plans explicitly")
    result = make_plan(args.design, args.pilots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=1)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "multiplicity",
                    "budget_status",
                    "cost_complete",
                    "submission_ready",
                    "headline",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
