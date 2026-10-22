"""Summarize paired PenRed/PyRITE comparison records per scene and observable.

Reports each engine's mean and 1-sigma SE, the difference and its z score
from the combined independent SEs. A z score is a plausibility screen only:
no equivalence margin, multiple-comparison policy or numerical gate is
applied, so the output never states equivalence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def load(directory):
    out, failures = {}, []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "pyrite.equal-precision.comparison.v1":
            continue
        if record["status"] == "ok":
            out[record["case_id"]] = (record, hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            failures.append({"case_id": record["case_id"], "status": record["status"]})
    return out, failures


def compare(a, b):
    rows = {}
    for name in sorted(set(a["moments"]) & set(b["moments"])):
        ma, mb = a["moments"][name], b["moments"][name]
        se = math.hypot(ma["se"] or 0.0, mb["se"] or 0.0)
        diff = ma["mean"] - mb["mean"]
        rows[name] = {
            "penred": [ma["mean"], ma["se"]],
            "pyrite": [mb["mean"], mb["se"]],
            "difference": diff,
            "z": diff / se if se > 0.0 else None,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("penred", type=Path)
    parser.add_argument("pyrite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    penred, penred_failures = load(args.penred)
    pyrite, pyrite_failures = load(args.pyrite)
    scenes = {}
    for case_id in sorted(set(penred) & set(pyrite)):
        (a, sha_a), (b, sha_b) = penred[case_id], pyrite[case_id]
        scenes[case_id] = {
            "n_primaries": {"penred": a["n_primaries"], "pyrite": b["n_primaries"]},
            "pyrite_straggling": b["provenance"]["scene"]["straggling"],
            "record_sha256": {"penred": sha_a, "pyrite": sha_b},
            "timings_s": {
                "penred": a["provenance"]["timings_s"],
                "pyrite": b["provenance"]["timings_s"],
            },
            "observables": compare(a, b),
        }
    summary = {
        "schema": "pyrite.equal-precision.penred-pyrite-screen.v1",
        "scenes": scenes,
        "unpaired": sorted(set(penred) ^ set(pyrite)),
        "failures": {"penred": penred_failures, "pyrite": pyrite_failures},
        "interpretation": "plausibility screen; z uses independent 1-sigma SEs; no "
        "equivalence, numerical or timing claim",
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for case_id, scene in scenes.items():
        flagged = {
            k: round(v["z"], 1)
            for k, v in scene["observables"].items()
            if v["z"] and abs(v["z"]) > 3
        }
        print(case_id, "|z|>3:", flagged or "none")


if __name__ == "__main__":
    main()
