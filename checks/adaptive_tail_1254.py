"""Replay captured primary scores against the CXR variance guard (#1254).

Usage: uv run python checks/adaptive_tail_1254.py CAPTURE_DIR... --out REPORT

Selects the longest #1252 capture for each (material, energy, tilt), hashes
its bytes, and replays old and new policies on that identical population.
A capture ending before a decision is reported as exhausted, never as a
converged or maximum-budget result. Full-capture CV is a finite-reference
proxy, not the population truth. No transport runs here.

Since #1253 the monitor decides at policy decision points, not at block ends;
``adaptive_tail_1254_results.json`` records the v2 block rule and reproduces only at the
revision it names. ``adaptive_decision_cadence_1253.py`` replays v3.

Validation: adaptive-sample-size-stopping
"""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyrite._precision import ADAPTIVE_SAMPLING_MODEL, Precision
from pyrite.montecarlo.runner.adaptive import RunningMoments, StoppingMonitor


class _PreviousMonitor(StoppingMonitor):
    def observable_status(self, name):
        status = super().observable_status(name)
        status["passes"].pop("variance_sampling")
        status["converged"] = all(status["passes"].values())
        return status


def _replay(monitor_type, policy, population):
    monitor = monitor_type(
        policy, lambda start, stop, _: {k: v[start:stop] for k, v in population.items()}
    )
    captured = min(v.size for v in population.values())
    block = policy.block_histories
    for start in range(0, min(captured, policy.max_histories) - block + 1, block):
        stop = start + block
        monitor.on_block(start, stop, None)
        if monitor.should_stop(stop):
            break
    record = monitor.record()
    return {
        "realized": record["realized_electrons"],
        "reason": record["stop_reason"] or "capture_exhausted",
        "statistics": record["statistics"],
    }


def _tail(values):
    mean = float(values.mean())
    cv = float(values.std(ddof=1) / mean)
    centered = values - mean
    top = np.argsort(values)[-10:][::-1]
    variance_top = np.sort(centered**2)[-10:]
    m2 = float(np.sum(centered**2))
    moments = RunningMoments()
    prefixes = []
    for start in range(0, len(values), 1000):
        moments.add_block(values[start : start + 1000])
        prefixes.append((moments.n, moments.relative_se() * math.sqrt(moments.n) / cv))
    last_outside = max(
        (i for i, (_, ratio) in enumerate(prefixes) if not 0.8 <= ratio <= 1.2), default=-1
    )
    return {
        "n": len(values),
        "full_rse": cv / math.sqrt(len(values)),
        "full_cv": cv,
        "quantiles_over_mean": (np.quantile(values, [0, 0.5, 0.99, 0.999, 1]) / mean).tolist(),
        "largest_primary_ids_zero_based": top.tolist(),
        "largest_scores_over_mean": (values[top] / mean).tolist(),
        "largest_variance_share": float(variance_top[-1] / m2),
        "top10_variance_share": float(variance_top.sum() / m2),
        "first_prefix_with_all_later_cv_within_20_percent": (
            prefixes[last_outside + 1][0] if last_outside + 1 < len(prefixes) else None
        ),
    }


def analyse(path, blocks, maximum):
    source = json.loads(path.with_name(path.name.replace(".values.npz", ".json")).read_text())
    with np.load(path) as data:
        population = {name: data[name] for name in data.files if name != "block"}
    base = Precision.from_dict(source.get("profile_precision", source["precision"]))
    tail = _tail(population["cxr"])
    rows = []
    for block in blocks:

        def up(n, block=block):
            return math.ceil(n / block) * block

        policy = replace(
            base,
            min_histories=up(3000),
            pilot_histories=up(3000),
            max_histories=up(maximum),
            block_histories=block,
        )
        old = _replay(_PreviousMonitor, policy, population)
        new = _replay(StoppingMonitor, policy, population)
        ratio = None
        if new["reason"] == "converged":
            n = new["realized"]
            ratio = new["statistics"]["cxr"]["relative_se"] / (tail["full_cv"] / math.sqrt(n))
        rows.append(
            {"block": block, "previous": old, "current": new, "reported_to_reference_rse": ratio}
        )
    return {
        "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_code_sha": source.get("code_sha"),
        "material": source["material"],
        "energy_keV": source["E0_keV"],
        "tilt_deg": source["tilt_deg"],
        "tail": tail,
        "replays": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=[1000, 3000, 6000, 12000, 24000, 48000]
    )
    parser.add_argument("--maximum", type=int, default=1_000_000)
    args = parser.parse_args()
    selected = {}
    for directory in args.directories:
        for path in sorted(directory.glob("*.values.npz")):
            source = json.loads(
                path.with_name(path.name.replace(".values.npz", ".json")).read_text()
            )
            key = source["material"], source["E0_keV"], source["tilt_deg"]
            with np.load(path) as data:
                size = data["cxr"].size
            if key not in selected or size > selected[key][0]:
                selected[key] = size, path
    if not selected:
        parser.error("no captures found")
    cases = [analyse(path, args.blocks, args.maximum) for _, path in selected.values()]
    ratios = [
        r["reported_to_reference_rse"]
        for c in cases
        for r in c["replays"]
        if r["reported_to_reference_rse"] is not None
    ]
    high_energy_ratios = [
        r["current"]["realized"] / r["previous"]["realized"]
        for c in cases
        if c["energy_keV"] == 30000
        for r in c["replays"]
        if r["current"]["reason"] == r["previous"]["reason"] == "converged"
    ]
    report = {
        "sampling_model": ADAPTIVE_SAMPLING_MODEL,
        "maximum": args.maximum,
        "minimum_and_pilot": 3000,
        "cases": cases,
        "minimum_reported_to_reference_rse": min(ratios),
        "maximum_30MeV_stop_ratio": max(high_energy_ratios),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    return 0 if min(ratios) >= 0.5 and max(high_energy_ratios) <= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
