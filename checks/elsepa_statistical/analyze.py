"""Reduce matched #895 arms; report timing spread and #838 epsilon=2% intervals.

No adoption verdict: this reducer rejects incomplete seed/arm sets and keeps
unresolved intervals distinct from equivalence. Histories are paired by primary
index within seed, despite the changed draw-to-angle map.
"""

import argparse
import hashlib
import json
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np

GATE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "fp32_speed_of_light/gate838.py"))


def reduce(directory, scenes):
    stamp = dict(line.split(": ", 1) for line in (directory / "sync.txt").read_text().splitlines())
    if stamp["code_dirty"] != "False":
        raise ValueError("timing evidence requires a clean source revision")
    groups = defaultdict(dict)
    for path in directory.glob("*.json"):
        record = json.loads(path.read_text())
        study = record.get("elsepa_interpolation_study") if isinstance(record, dict) else None
        if study is None:
            continue
        key = (record["controls"]["transport_precision"], record["scene"]["id"], record["seed"])
        law = study["law"]
        if law in groups[key]:
            raise ValueError(f"duplicate {key} {law}")
        groups[key][law] = (record, path.with_suffix(".npz"))
    results = []
    for precision in ("mixed", "fp64"):
        for scene in scenes:
            pairs = [
                (seed, pair)
                for (p, s, seed), pair in sorted(groups.items())
                if p == precision and s == scene
            ]
            if len(pairs) < 3 or any(set(pair) != {"quantile", "mixture"} for _, pair in pairs):
                raise ValueError(f"{precision}/{scene}: need >=3 complete seed pairs")
            timings, scores = [], {"quantile": [], "mixture": []}
            for seed, pair in pairs:
                base, candidate = (pair[law][0] for law in ("quantile", "mixture"))
                for key in (
                    "histories",
                    "warmup_histories",
                    "block",
                    "block_rows",
                    "controls",
                    "scene",
                    "estimator",
                    "cone_scoring",
                    "tables",
                ):
                    if base[key] != candidate[key]:
                        raise ValueError(f"unmatched {key}: {precision}/{scene}/{seed}")
                entry = {"seed": seed}
                for law in scores:
                    record, path = pair[law]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != record["raw_scores_sha256"]:
                        raise ValueError(f"primary score digest mismatch: {path}")
                    with np.load(path) as arrays:
                        data = {name: arrays[name] for name in arrays.files}
                    if any(value.shape != (record["histories"],) for value in data.values()):
                        raise ValueError(f"unpaired primary arrays: {path}")
                    scores[law].append(GATE["observables"](data))
                    work = record["work"]["warm"]
                    entry[law] = {
                        "kernel_ns_per_row": 1e6 * work["kernel_ms"] / work["retained_rows"],
                        "warm_request_s": record["timings_s"]["warm_request"],
                        "rows": work["retained_rows"],
                        "kernel_launches": work["kernel_launches"],
                    }
                timings.append(entry)
            headline, diagnostics = [], []
            for name in sorted(scores["quantile"][0]):
                a, b = [
                    np.concatenate([s[name] for s in scores[law]])
                    for law in ("quantile", "mixture")
                ]
                if not (np.any(a) or np.any(b)):
                    continue
                result = GATE["bound"](a, b)
                result["baseline_mean"] = result.pop("fp64_mean")
                result.update(observable=name, histories=int(a.size))
                if GATE["headline"](name, a.mean(), base["scene"], base["estimator"]):
                    result["equivalence_epsilon_2percent"] = GATE["outcome"](result, 0.02)
                    headline.append(result)
                else:
                    diagnostics.append(result)
            ratios = {}
            for metric in ("kernel_ns_per_row", "warm_request_s"):
                values = [t["mixture"][metric] / t["quantile"][metric] for t in timings]
                ratios[metric] = dict(
                    median=float(np.median(values)),
                    min=min(values),
                    max=max(values),
                    candidate_over_baseline=values,
                )
            results.append(
                dict(
                    precision=precision,
                    scene=scene,
                    seeds=[s for s, _ in pairs],
                    timings=timings,
                    ratios=ratios,
                    headline=headline,
                    diagnostics=diagnostics,
                )
            )
    return {
        "schema": "pyrite.elsepa895.study.v1",
        "code_revision": stamp["code_revision"],
        "code_digest": stamp["code_digest"],
        "epsilon": 0.02,
        "relative_margin": 0.02 / 3,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--scenes", nargs="+", default=["w-1MeV-100um"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.directory, args.scenes)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for entry in result["results"]:
        print(entry["precision"], entry["scene"], entry["ratios"])
        for row in entry["headline"]:
            print(
                row["observable"],
                row["rel"],
                row["resolution"],
                row["equivalence_epsilon_2percent"],
            )


if __name__ == "__main__":
    main()
