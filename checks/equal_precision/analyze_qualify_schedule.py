"""Summarize ``qualify_schedule.py`` records (#799).

Groups samples by (scene, histories, schedule), reports median [min, max]
transport-request and runner transport seconds, peak pool and RSS, the resolved
schedule, and whether trajectory arrays and exit tallies match across schedules for each
(scene, histories, seed). Writes a JSON summary and prints a Markdown table.
"""

import argparse
import json
import statistics
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

    records = [
        json.loads(p.read_text())
        for p in sorted(args.evidence.glob("*.json"))
        if p.resolve() != args.output.resolve()
    ]
    if not records or any(r.get("measurement_scope") != "runner-transport-only" for r in records):
        raise ValueError("expected transport-only samples; do not mix spectral request evidence")
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


if __name__ == "__main__":
    main()
