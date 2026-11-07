"""Apply the frozen #894 design to held-out samples; incomplete data stays unresolved."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "equal_precision"))
from paired_equivalence import paired_interval  # noqa: E402


def samples(root, arm, scene, seeds, histories, keep=None):
    """Concatenated per-primary scores of one arm, in seed and primary order."""
    parts, timing, rows = [], [], 0
    for seed in seeds:
        path = root / f"{arm}-{scene}-{seed}.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record["raw_scores_sha256"]:
            raise ValueError(f"score digest mismatch: {raw}")
        if record["histories"] != histories or record["seed"] != seed:
            raise ValueError(f"sample outside the frozen design: {path}")
        with np.load(raw) as arrays:
            parts.append({k: arrays[k][:keep] for k in arrays.files})
        timing.append(record["timings_s"]["warm_request"])
        rows += record["work"]["warm"]["retained_rows"]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return dict(scores=merged, warm_s=timing, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seeds = protocol["seeds"]
    comparisons, cost, missing = [], [], []
    for family_name, family in protocol["families"].items():
        for scene, cell in family["scenes"].items():
            histories = cell["histories_per_seed"]
            ref = samples(args.evidence, family["reference"], scene, seeds, histories)
            if ref is None:
                missing.append([family["reference"], scene])
                continue
            diagnostic = family.get("diagnostic_arms", {})
            for arm in [*family["arms"], *diagnostic]:
                is_diagnostic = arm in diagnostic
                if is_diagnostic:
                    n = max(1000, round(family["diagnostic_fraction"] * histories))
                    mix = samples(args.evidence, arm, scene, seeds, n)
                    pair = samples(args.evidence, family["reference"], scene, seeds, histories, n)
                    total = n * len(seeds)
                else:
                    pair, total = ref, histories * len(seeds)
                    mix = samples(args.evidence, arm, scene, seeds, histories)
                if mix is None or pair is None:
                    missing.append([arm, scene])
                    continue
                # Cost per history, so a quarter-size diagnostic arm compares fairly.
                ref_n = histories * len(seeds)
                ref_rows, mix_rows = ref["rows"] / ref_n, mix["rows"] / total
                ref_us = 1e6 * np.asarray(ref["warm_s"]) / histories
                mix_us = 1e6 * np.asarray(mix["warm_s"]) / (total / len(seeds))
                cost.append(
                    dict(
                        diagnostic=is_diagnostic,
                        family=family_name,
                        scene=scene,
                        arm=arm,
                        histories=total,
                        rows_per_history_reference=ref_rows,
                        rows_per_history=mix_rows,
                        row_reduction=ref_rows / mix_rows,
                        warm_us_per_history_reference=ref_us.tolist(),
                        warm_us_per_history=mix_us.tolist(),
                        warm_reduction_median=float(np.median(ref_us / mix_us)),
                    )
                )
                for name, spec in cell["observables"].items():
                    a, b = pair["scores"][name], mix["scores"][name]
                    base = dict(family=family_name, scene=scene, arm=arm, observable=name)
                    z = (
                        (b.mean() - a.mean())
                        / np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
                        if max(a.var(), b.var()) > 0
                        else 0.0
                    )
                    base |= dict(
                        relative_difference=float(b.mean() / a.mean() - 1) if a.mean() else None,
                        z=float(z),
                    )
                    if not spec["headline"] or is_diagnostic:
                        comparisons.append(
                            base | dict(status="diagnostic", reference_mean=float(a.mean()))
                        )
                        continue
                    result = paired_interval(
                        a,
                        b,
                        bound=spec["bound"],
                        margin=spec["margin"],
                        comparisons=protocol["comparisons"],
                        alpha=protocol["alpha"],
                    )
                    comparisons.append(base | result)
    if missing:
        for c in comparisons:
            if c["status"] != "diagnostic":
                c["status"] = "unresolved"
                c["reason"] = "incomplete fixed design"
    output = dict(
        schema="pyrite.mixed-elastic.equivalence.v1",
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        complete=not missing,
        missing=missing,
        cost=cost,
        comparisons=comparisons,
    )
    args.output.write_text(json.dumps(output, indent=1) + "\n")
    print("complete", not missing, "missing", len(missing))
    for family_name, family in protocol["families"].items():
        for arm in family["arms"]:
            statuses = {}
            for c in comparisons:
                if c["family"] == family_name and c["arm"] == arm:
                    statuses[c["status"]] = statuses.get(c["status"], 0) + 1
            print(family_name, arm, statuses)
    print("max |z| per arm")
    for arm in sorted({c["arm"] for c in comparisons}):
        worst = max((c for c in comparisons if c["arm"] == arm), key=lambda c: abs(c["z"]))
        print(arm, round(worst["z"], 2), worst["scene"], worst["observable"])


if __name__ == "__main__":
    main()
