"""Replay saved #1252 primary scores under decision-point stopping (#1253).

Usage: uv run python checks/adaptive_decision_cadence_1253.py CAPTURE_DIR... --out REPORT

Selects the longest #1252 capture per (material, energy, tilt), hashes it, and
replays the stopping rule (CPU, no transport):

* at chunk sizes 1, 1000, 7168 and 48000 and one chunk per decision point;
  every record must be identical (decisions independent of transport chunks);
* against a dense grid (``decision_growth = 0.01`` with the stability window
  kept at the same share of the sample, k = 29): a converged default stop may
  exceed the dense stop by at most the growth ``delta = 0.1`` relative, plus
  one point of the dense grid;
* beside the block-rule stops recorded in ``adaptive_tail_1254_results.json``.

A capture ending before a decision is reported as exhausted. Full-capture CV
is a finite-reference proxy, not the population truth. Exit 1 on a failed
assertion.

Validation: adaptive-sample-size-stopping
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyrite._precision import ADAPTIVE_SAMPLING_MODEL, Precision
from pyrite.montecarlo.runner.adaptive import run_stopping_rule

CHUNKS = (None, 1, 1000, 7168, 48000)
DENSE = {"decision_growth": 0.01, "stability_checks": 29}


def _select(directories):
    selected = {}
    for directory in directories:
        for path in sorted(Path(directory).rglob("*.values.npz")):
            meta = json.loads(path.with_name(path.name.replace(".values.npz", ".json")).read_text())
            key = meta["material"], meta["E0_keV"], meta["tilt_deg"]
            with np.load(path) as data:
                size = data["cxr"].size
            if key not in selected or size > selected[key][0]:
                selected[key] = size, path, meta
    return selected


def _record(policy, population, chunk=None):
    def measure(start, stop, _block):
        return {name: values[start:stop] for name, values in population.items()}

    return run_stopping_rule(policy, measure, chunk=chunk)


def _summary(record, captured):
    reason = record["stop_reason"]
    if record["realized_electrons"] == captured and reason == "max_electrons":
        reason = "capture_exhausted"
    return {
        "realized": record["realized_electrons"],
        "reason": reason,
        "checks": record["checks"],
        "cxr_relative_se": record["statistics"]["cxr"]["relative_se"],
    }


def analyse(size, path, meta, block_rule, maximum):
    with np.load(path) as data:
        population = {name: data[name] for name in ("cxr", "brem")}
    base = Precision.from_dict(meta.get("profile_precision", meta["precision"]))
    policy = replace(
        base,
        min_histories=3000,
        pilot_histories=3000,
        max_histories=min(maximum, size),
        block_histories=None,
    )
    records = [_record(policy, population, chunk) for chunk in CHUNKS]
    independent = all(record == records[0] for record in records[1:])
    default = _summary(records[0], size)
    dense = _summary(_record(replace(policy, **DENSE), population), size)
    values = population["cxr"]
    cv = float(values.std(ddof=1) / values.mean())
    ratio = None
    if default["reason"] == "converged":
        ratio = default["cxr_relative_se"] / (cv / math.sqrt(default["realized"]))
    overshoot = None
    if default["reason"] == dense["reason"] == "converged":
        overshoot = default["realized"] / dense["realized"] - 1.0
    key = (meta["material"], meta["E0_keV"], meta["tilt_deg"])
    return {
        "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_code_sha": meta.get("code_sha"),
        "material": key[0],
        "energy_keV": key[1],
        "tilt_deg": key[2],
        "captured": size,
        "chunk_independent": independent,
        "decision_points": default,
        "dense_grid": dense,
        "overshoot_vs_dense": overshoot,
        "reported_to_reference_rse": ratio,
        "block_rule_v2": block_rule.get(key, {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=1_000_000)
    parser.add_argument(
        "--block-rule",
        type=Path,
        default=Path(__file__).with_name("adaptive_tail_1254_results.json"),
    )
    args = parser.parse_args()
    block_rule = {
        (case["material"], case["energy_keV"], case["tilt_deg"]): {
            str(row["block"]): [row["current"]["realized"], row["current"]["reason"]]
            for row in case["replays"]
        }
        for case in json.loads(args.block_rule.read_text())["cases"]
    }
    selected = _select(args.directories)
    if not selected:
        parser.error("no captures found")
    cases = [analyse(*selected[key], block_rule, args.maximum) for key in sorted(selected)]
    growth = Precision(0.1, 2, 3).decision_growth
    failures = [c["source"] for c in cases if not c["chunk_independent"]]
    failures += [
        c["source"]
        for c in cases
        if c["overshoot_vs_dense"] is not None and c["overshoot_vs_dense"] > growth + 0.01
    ]
    report = {
        "sampling_model": ADAPTIVE_SAMPLING_MODEL,
        "chunks": [c or "per-decision-point" for c in CHUNKS],
        "dense_grid": DENSE,
        "maximum": args.maximum,
        "cases": cases,
        "failures": failures,
    }
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    for case in cases:
        point, dense = case["decision_points"], case["dense_grid"]
        print(
            f"{case['material']} {case['energy_keV']:g} keV {case['tilt_deg']:g} deg: "
            f"{point['realized']} {point['reason']} (dense {dense['realized']} {dense['reason']}); "
            f"chunk-independent {case['chunk_independent']}; block rule {case['block_rule_v2']}"
        )
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
