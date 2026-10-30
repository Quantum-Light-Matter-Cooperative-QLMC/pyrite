"""Summarize ``qualify_schedule.py`` records (#799).

Groups samples by (scene, histories, schedule), reports median [min, max]
transport-request and runner transport seconds, peak pool and RSS, the resolved
schedule, and whether trajectory arrays and exit tallies match across schedules for each
(scene, histories, seed). Writes a JSON summary and prints a Markdown table.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def fmt(stat, scale=1.0, digits=2):
    if stat is None:
        return "—"
    med, lo, hi = (stat[k] / scale for k in ("median", "min", "max"))
    if stat["n"] == 1:
        return f"{med:.{digits}f}"
    return f"{med:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Exact factorial design of qualify-schedule-799.sbatch: 30 samples.
    # Do not infer the expected denominator from successful output files.
    expected = {
        (scene, histories, seed, schedule)
        for histories in (5000, 20000, 100000)
        for seed in ((79901,) if histories == 100000 else (79901, 79902))
        for scene in ("silicon", "tungsten", "pbwo4")
        for schedule in ("full-history", "auto")
    }
    records = []
    failures = []
    seen = set()
    for scene, histories, seed, schedule in sorted(expected):
        stem = f"{scene}-{histories}-{seed}-{schedule}"
        sample_path = args.evidence / f"{stem}.json"
        status_path = args.evidence / f"{stem}.status.json"
        if not status_path.exists():
            failures.append(f"{stem}: missing status marker")
            continue
        try:
            status = json.loads(status_path.read_text())
        except (ValueError, OSError) as exc:
            failures.append(f"{stem}: unreadable status ({exc})")
            continue
        if status.get("status") != "ok" or status.get("exit_code") != 0:
            failures.append(
                f"{stem}: {status.get('status', 'invalid')} rc={status.get('exit_code')}"
            )
            continue
        try:
            r = json.loads(sample_path.read_text())
            if r.get("measurement_scope") != "runner-transport-only":
                raise ValueError("wrong measurement scope")
            if (r.get("scene"), r.get("histories"), r.get("seed"), r.get("schedule")) != (
                scene,
                histories,
                seed,
                schedule,
            ):
                raise ValueError("sample identity disagrees with requested design")
            if not r.get("measured", {}).get("digest") or r.get("warmup_wall_s") is None:
                raise ValueError("incomplete warmup/measured output")
            if not (r["measured"]["wall_s"] > 0):
                raise ValueError("invalid measured wall time")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{stem}: missing/invalid result ({exc})")
            continue
        records.append(r)
        seen.add((scene, histories, seed, schedule))
    # Files not belonging to the declared experimental design are not evidence.
    for path in args.evidence.glob("*.json"):
        if path.resolve() != args.output.resolve() and not path.name.endswith(".status.json"):
            if path.stem not in {f"{sc}-{h}-{sd}-{sch}" for sc, h, sd, sch in expected}:
                failures.append(f"unexpected result: {path.name}")
    groups = defaultdict(list)
    digests = defaultdict(dict)
    for r in records:
        groups[(r["scene"], r["histories"], r["schedule"])].append(r)
        digests[(r["scene"], r["histories"], r["seed"])][r["schedule"]] = r["measured"]["digest"]

    identical = {
        f"{scene}-{histories}-{seed}": len({json.dumps(d, sort_keys=True) for d in by.values()})
        == 1
        and len(by) > 1
        for (scene, histories, seed), by in sorted(digests.items())
    }
    for key, equal in identical.items():
        if not equal:
            failures.append(f"{key}: missing/mismatched schedule digest")
    qualified = not failures and len(seen) == len(expected)
    summary = []
    for (scene, histories, schedule), rs in sorted(groups.items()):
        summary.append(
            {
                "scene": scene,
                "histories": histories,
                "schedule": schedule,
                "seeds": sorted(r["seed"] for r in rs),
                "wall_s": spread([r["measured"]["wall_s"] for r in rs]),
                "transport_s": spread([r["measured"]["transport_s"] for r in rs]),
                "peak_pool_bytes": spread([r["measured"]["peak_pool_bytes"] for r in rs]),
                "peak_rss_bytes": spread([r["peak_rss_bytes"] for r in rs]),
                "schedule_log": sorted({m for r in rs for m in r["measured"]["schedule_log"]}),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "measurement_scope": "runner-transport-only",
                "status": "qualified" if qualified else "incomplete_or_failed",
                "qualified": qualified,
                "expected_samples": len(expected),
                "valid_samples": len(records),
                "failed_samples": failures,
                "groups": summary,
                "identical": identical,
            },
            indent=1,
        )
    )

    mib, gib = 1024.0**2, 1024.0**3
    print(
        "| Scene | Histories | Schedule | Transport request (s) | Runner transport (s) | Peak pool (MiB) | Peak RSS (GiB) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for g in summary:
        print(
            f"| {g['scene']} | {g['histories']} | {g['schedule']} | {fmt(g['wall_s'])} | "
            f"{fmt(g['transport_s'])} | {fmt(g['peak_pool_bytes'], mib, 0)} | "
            f"{fmt(g['peak_rss_bytes'], gib, 1)} |"
        )
    print()
    print("Identical outputs across schedules:", identical)
    print(
        f"Qualification: {'PASS' if qualified else 'FAIL'} ({len(records)}/{len(expected)} valid)"
    )
    if failures:
        print("Failed or missing samples:", *failures, sep="\n  - ")
        sys.exit(1)


if __name__ == "__main__":
    main()
