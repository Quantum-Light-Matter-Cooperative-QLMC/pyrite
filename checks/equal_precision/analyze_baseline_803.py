"""Summarize a ``run_baseline_803.py`` directory into matched comparisons.

Each comparison pairs records that differ in exactly one control, at the same
scene, core and seed. Timing ratios are reference/candidate warm medians per
seed (>1 means the candidate is faster), summarized by their geometric mean
and range over seeds. Schedule comparisons must be bitwise: equal trajectory
digests. Interpolating comparisons (LUT, moments) get a paired per-primary
screen: same seed means the same primaries, so the per-primary difference has
a valid standard error. That error is tiny, so a test against zero flags
harmless interpolation shifts; the screen instead compares the 95% interval of
the difference with a recorded practical tolerance (``--rel-tol`` of the
reference mean, absolute for a zero reference): ``equivalent`` when the whole
interval is inside, ``different`` when it is wholly outside, else
``unresolved``. This is a screen, not the predeclared qualification gate; an
unresolved observable does not pass.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# (name, varied control, reference value, candidate value, fixed controls)
COMPARISONS = (
    ("lut", "lut_requested", "off", "on", {"elastic": "sr"}),
    ("moments", "moments_requested", "exact", "table", {"elastic": "elsepa"}),
    ("shell-cost", "inelastic", "continuous", "shell-soft-hard", {"moments_requested": "table"}),
    ("continuation", "schedule_requested", "full-history", "continuation", {}),
    ("auto", "schedule_requested", "full-history", "auto", {}),
)
BITWISE = {"continuation", "auto"}
# v2 adds the kernel-event and NVTX stage split; v1 records still compare.
SAMPLE_SCHEMAS = ("pyrite.lut-baseline-803.sample.v1", "pyrite.lut-baseline-803.sample.v2")
# A model change: cost only; equal observables are not expected.
COST_ONLY = {"shell-cost"}
KEY_CONTROLS = (
    "core",
    "elastic",
    "radiative",
    "inelastic",
    "lut_requested",
    "moments_requested",
    "schedule_requested",
)


def load(directory):
    records, failures = [], []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".failed.json"):
            failures.append(json.loads(path.read_text()))
            continue
        record = json.loads(path.read_text())
        if record.get("schema") in SAMPLE_SCHEMAS:
            record["_path"] = path
            records.append(record)
    return records, failures


def key(record, drop=None):
    c = record["controls"]
    return (record["scene"]["id"], record["seed"]) + tuple(
        (name, c[name]) for name in KEY_CONTROLS if name != drop
    )


def warm_fraction(record, site):
    """Median over warm requests of one observed site's share of request wall."""
    shares = [
        sum(c["wall_s"] for c in calls if c["site"] == site) / wall
        for calls, wall in zip(
            record["observed_warm"], record["timings_s"]["warm_requests"], strict=True
        )
    ]
    return float(np.median(shares))


def kernel_share(record):
    """Median warm CUDA-event kernel time over request wall (None off CUDA/v1)."""
    splits = record.get("split_warm")
    if not splits:
        return None
    shares = [
        s["kernel"]["kernel_s"] / wall
        for s, wall in zip(splits, record["timings_s"]["warm_requests"], strict=True)
        if "kernel" in s
    ]
    return float(np.median(shares)) if shares else None


def stage_split(record):
    """Exclusive host seconds per NVTX range in the stage-synchronized request.

    Shares are of that request's own wall, so stages plus ``unranged`` sum to 1.
    """
    synced = record.get("split_stage_synchronized")
    if not synced:
        return None
    wall = synced["wall_s"]
    stages = {name: v["exclusive_s"] / wall for name, v in sorted(synced["stages"].items())}
    stages["unranged"] = 1.0 - sum(stages.values())
    out = {"wall_s": wall, "exclusive_share": stages}
    if "kernel" in synced:
        out["kernel_share"] = synced["kernel"]["kernel_s"] / wall
        out["launches"] = synced["kernel"]["launches"]
    return out


def paired_screen(ref, cand, rel_tol):
    a, b = np.load(ref["_path"].with_suffix(".npz")), np.load(cand["_path"].with_suffix(".npz"))
    out = {}
    for name in sorted(set(a.files) & set(b.files)):
        d = b[name] - a[name]
        se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
        mean = float(d.mean())
        reference = float(a[name].mean())
        tol = rel_tol * abs(reference) if reference != 0 else rel_tol
        half = 1.96 * se
        status = "unresolved"
        if abs(mean) + half <= tol:
            status = "equivalent"
        elif abs(mean) - half > tol:
            status = "different"
        out[name] = {
            "reference_mean": reference,
            "mean_difference": mean,
            "se": se,
            "tolerance": tol,
            "status": status,
        }
    return out


def compare(records, rel_tol):
    by_key = {key(r): r for r in records if r["status"] == "ok"}
    results = []
    for name, control, ref_value, cand_value, fixed in COMPARISONS:
        groups = defaultdict(list)
        for r in records:
            c = r["controls"]
            if r["status"] != "ok" or c[control] != cand_value:
                continue
            if any(c[k] != v for k, v in fixed.items()):
                continue
            ref_key = tuple(
                (item[0], ref_value) if isinstance(item, tuple) and item[0] == control else item
                for item in key(r)
            )
            ref = by_key.get(ref_key)
            if ref is None:
                continue
            other = "moments_requested" if control == "inelastic" else "inelastic"
            group = (r["scene"]["id"], c["core"], c[other], c["elastic"])
            if control != "schedule_requested":
                group += (c["schedule_requested"],)
            groups[group].append((ref, r))
        for group, pairs in sorted(groups.items()):
            ratios = [
                ref["timings_s"]["warm_median"] / cand["timings_s"]["warm_median"]
                for ref, cand in pairs
            ]
            entry = {
                "comparison": name,
                "scene": group[0],
                "core": group[1],
                "fixed": list(group[2:]),
                "seeds": [ref["seed"] for ref, _ in pairs],
                "speedup_geomean": float(np.exp(np.mean(np.log(ratios)))),
                "speedup_range": [float(min(ratios)), float(max(ratios))],
                "reference_warm_per_history_s": float(
                    np.median([ref["timings_s"]["warm_per_history"] for ref, _ in pairs])
                ),
                "cold_request_s": {
                    "reference": float(
                        np.median([p[0]["timings_s"]["cold_request"] for p in pairs])
                    ),
                    "candidate": float(
                        np.median([p[1]["timings_s"]["cold_request"] for p in pairs])
                    ),
                },
                "resolved_candidate": pairs[0][1]["resolved"],
            }
            if name in BITWISE:
                entry["bitwise_equal"] = all(
                    ref["trajectory_sha256"] == cand["trajectory_sha256"] for ref, cand in pairs
                )
            elif name not in COST_ONLY:
                screens = [paired_screen(ref, cand, rel_tol) for ref, cand in pairs]
                entry["paired_screens"] = screens
                statuses = defaultdict(set)
                for screen in screens:
                    for obs, result in screen.items():
                        statuses[result["status"]].add(obs)
                entry["screen"] = {k: sorted(v) for k, v in sorted(statuses.items())}
            results.append(entry)
    return results


def preparation(records):
    rows = []
    for r in records:
        if r["status"] != "ok":
            continue
        rows.append(
            {
                "scene": r["scene"]["id"],
                "seed": r["seed"],
                **{k: r["controls"][k] for k in KEY_CONTROLS},
                "warm_median_s": r["timings_s"]["warm_median"],
                "cold_request_s": r["timings_s"]["cold_request"],
                "warm_moment_prep_fraction": warm_fraction(r, "radiative_core_args"),
                "warm_lut_build_fraction": warm_fraction(r, "_transport_lut"),
                "warm_kernel_share": kernel_share(r),
                "stage_synchronized": stage_split(r),
                "retained_output_bytes": r["memory"].get("retained_output_bytes"),
                "peak_rss_kib": r["memory"]["peak_rss_kib"],
                "device_pool_bytes": r["memory"]["device_pool_reserved_bytes_after_warm"],
                "segments": r["segments"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    args = parser.parse_args()
    records, failures = [], []
    for directory in args.directories:
        r, f = load(directory)
        records += r
        failures += f
    sources = sorted({r["source"]["sha"] for r in records})
    summary = {
        "schema": "pyrite.lut-baseline-803.analysis.v1",
        "sources": sources,
        "records": len(records),
        "unsupported": [
            {"scene": r["scene"]["id"], "controls": r["controls"], "refusal": r["refusal"]}
            for r in records
            if r["status"] == "unsupported"
        ],
        "failures": failures,
        "screen_rel_tol": args.rel_tol,
        "comparisons": compare(records, args.rel_tol),
        "preparation": preparation(records),
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for c in summary["comparisons"]:
        if "bitwise_equal" in c:
            tail = f"bitwise={c['bitwise_equal']}"
        elif "screen" in c:
            tail = (
                f"different={c['screen'].get('different', [])} "
                f"unresolved={c['screen'].get('unresolved', [])}"
            )
        else:
            tail = "cost only (model change)"
        print(
            f"{c['comparison']:<13}{c['scene']:<22}{c['core']:<13}{','.join(c['fixed']):<34}"
            f"x{c['speedup_geomean']:.3f} [{c['speedup_range'][0]:.3f},{c['speedup_range'][1]:.3f}] "
            f"n={len(c['seeds'])} {tail}"
        )
    print(f"sources={sources} unsupported={len(summary['unsupported'])} failed={len(failures)}")


if __name__ == "__main__":
    main()
