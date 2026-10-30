"""Summarize ``probe_components_803.py`` records into per-component costs.

Records are grouped by scene, core, controls and seed. The ``K = 0`` record of
a group (any component: the source is unchanged at ``K = 0``) is the
reference; every amplified record must share its trajectory digest, or the
group is reported as broken rather than used. For each component and ``K``:

    cost = (T_K - T_0) / K

where ``T`` is the median warm CUDA-event kernel time on CUDA and the median
warm request wall on CPU (the CPU kernel is the request's ``launch`` stage).
``cost`` is the time of one evaluation of the component at every execution of
its site: a zero-cost-replacement ceiling at that site, not a realized saving.
Shares are of the ``K = 0`` kernel time and of the warm request wall. Seeds are
summarized by mean and range; with several ``K`` the per-``K`` costs are
reported separately so nonlinearity is visible. ``unattributed`` is one minus
the sum of the largest-``K`` component kernel shares, a rough remainder: the
marginal costs are not additive by construction.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCHEMA = "pyrite.lut-probe-803.sample.v1"
GROUP_CONTROLS = (
    "core",
    "elastic",
    "radiative",
    "inelastic",
    "lut_requested",
    "moments_requested",
    "schedule_requested",
)


def load(directories):
    records = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("schema") == SCHEMA and record.get("status") == "ok":
                records.append(record)
    return records


def kernel_time(record):
    """Median warm kernel seconds (CUDA events) or warm request wall (CPU)."""
    splits = record.get("split_warm") or []
    kernels = [s["kernel"]["kernel_s"] for s in splits if "kernel" in s]
    if kernels:
        return float(np.median(kernels))
    return float(record["timings_s"]["warm_median"])


def group_key(record):
    c = record["controls"]
    precision = record["probe"].get("precision", "exact")
    return (
        record["scene"]["id"],
        *(c[name] for name in GROUP_CONTROLS),
        precision,
        record["histories"],
    )


def per_seed(records):
    """``{(group, seed): {"ref": record, (component, K): record}}``."""
    table = defaultdict(dict)
    for r in records:
        probe = r["probe"]
        slot = table[(group_key(r), r["seed"])]
        if probe["amplification"] == 0:
            slot.setdefault("ref", r)
        else:
            slot[(probe["component"], probe["amplification"])] = r
    return table


def summarize(records):
    table = per_seed(records)
    costs = defaultdict(list)
    broken, missing_ref = [], []
    for (group, seed), slot in sorted(table.items(), key=lambda item: str(item[0])):
        ref = slot.get("ref")
        if ref is None:
            missing_ref.append({"group": list(group), "seed": seed})
            continue
        t0 = kernel_time(ref)
        wall0 = float(ref["timings_s"]["warm_median"])
        for (component, k), r in _probes(slot):
            if r["trajectory_sha256"] != ref["trajectory_sha256"]:
                broken.append({"group": list(group), "seed": seed, "component": component, "K": k})
                continue
            cost = (kernel_time(r) - t0) / k
            costs[(group, component, k)].append(
                {
                    "seed": seed,
                    "cost_s": cost,
                    "kernel_share": cost / t0,
                    "wall_share": cost / wall0,
                }
            )
    rows = []
    for (group, component, k), samples in sorted(costs.items(), key=lambda item: str(item[0])):
        shares = np.array([s["kernel_share"] for s in samples])
        walls = np.array([s["wall_share"] for s in samples])
        rows.append(
            {
                "scene": group[0],
                "controls": dict(zip(GROUP_CONTROLS, group[1:-2], strict=True)),
                "precision": group[-2],
                "histories": group[-1],
                "component": component,
                "K": k,
                "seeds": [s["seed"] for s in samples],
                "kernel_share_mean": float(shares.mean()),
                "kernel_share_range": [float(shares.min()), float(shares.max())],
                "wall_share_mean": float(walls.mean()),
                "cost_s_mean": float(np.mean([s["cost_s"] for s in samples])),
            }
        )
    return rows, broken, missing_ref, _references(table)


def _probes(slot):
    return ((key, r) for key, r in slot.items() if key != "ref")


def _references(table):
    refs = defaultdict(list)
    for (group, _), slot in table.items():
        ref = slot.get("ref")
        if ref is None:
            continue
        wall = float(ref["timings_s"]["warm_median"])
        refs[group].append({"kernel_s": kernel_time(ref), "wall_s": wall})
    return {
        group: {
            "kernel_s": float(np.median([x["kernel_s"] for x in v])),
            "wall_s": float(np.median([x["wall_s"] for x in v])),
        }
        for group, v in refs.items()
    }


def remainders(rows):
    """Per group: one minus the summed largest-``K`` component kernel shares."""
    best = {}
    for row in rows:
        key = (row["scene"], (*row["controls"].values(), row["precision"]), row["component"])
        if key not in best or row["K"] > best[key]["K"]:
            best[key] = row
    totals = defaultdict(float)
    for (scene, controls, _), row in best.items():
        totals[(scene, controls)] += row["kernel_share_mean"]
    return [
        {
            "scene": scene,
            "controls": dict(zip(GROUP_CONTROLS, controls[:-1], strict=True)),
            "precision": controls[-1],
            "attributed_kernel_share": total,
            "unattributed": 1.0 - total,
        }
        for (scene, controls), total in sorted(totals.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load(args.directories)
    rows, broken, missing_ref, refs = summarize(records)
    summary = {
        "schema": "pyrite.lut-probe-803.analysis.v1",
        "sources": sorted({r["source"]["sha"] for r in records}),
        "records": len(records),
        "broken_digests": broken,
        "missing_reference": missing_ref,
        "references": [
            {"group": list(group), **value} for group, value in sorted(refs.items(), key=str)
        ],
        "components": rows,
        "remainders": remainders(rows),
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for row in rows:
        c = row["controls"]
        print(
            f"{row['scene']:<22}{c['core']:<13}{c['inelastic']:<16}{row['precision']:<11}"
            f"{row['component']:<10}"
            f"K={row['K']:<3}kernel {100 * row['kernel_share_mean']:6.2f}% "
            f"[{100 * row['kernel_share_range'][0]:.2f},{100 * row['kernel_share_range'][1]:.2f}] "
            f"wall {100 * row['wall_share_mean']:6.2f}% n={len(row['seeds'])}"
        )
    for rem in summary["remainders"]:
        print(
            f"{rem['scene']:<22}{rem['controls']['core']:<13}{rem['controls']['inelastic']:<16}"
            f"unattributed {100 * rem['unattributed']:.1f}%"
        )
    print(f"records={len(records)} broken={len(broken)} missing_ref={len(missing_ref)}")


if __name__ == "__main__":
    main()
