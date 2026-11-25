"""Device-idle attribution of one ``gpu_host.py --nsys`` trace (#1432).

Usage: nsys_gaps.py TRACE.sqlite OUT_JSON

Input is ``nsys export --type sqlite`` of the trace. Device busy time is the
union of CUPTI kernel, memcpy and memset intervals; device idle is its
complement inside the harness ``gpuhost.run`` range. Each idle nanosecond is
charged to the innermost NVTX range open on the main (range-emitting) thread
at that instant, else ``<none>``. Ranges come from production
``_nsys_push`` stages and the harness ``gpuhost.*`` spans. Also reported:
per-range wall and device-busy time, the idle-gap size histogram, the largest
gaps with their range stacks, the top kernels and CUDA API calls, and the idle
split by (innermost range at the end of the gap, first device activity that
ends it): the launch that waited, which names the eager operation the host was
issuing when the gap closed.
Engineering evidence; no physics claim.
"""

from __future__ import annotations

import bisect
import collections
import json
import sqlite3
import sys
from pathlib import Path

BUCKETS_NS = (10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)
ENDED_BY_LIMIT = 120


def _tables(db):
    return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def _strings(db):
    return dict(db.execute("select id, value from StringIds"))


def _busy(db, tables):
    intervals = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if table in tables:
            intervals += db.execute(f"select start, end from {table}").fetchall()
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _activities(db, tables, strings):
    """``(start, name)`` of every device activity, sorted by start."""
    out = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        out += [
            (start, strings.get(short, str(short)))
            for start, short in db.execute(
                "select start, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
            )
        ]
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "<memcpy>"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "<memset>"),
    ):
        if table in tables:
            out += [(start, label) for (start,) in db.execute(f"select start from {table}")]
    out.sort()
    return out


def _ranges(db, strings):
    rows = db.execute(
        "select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS "
        "where end is not null and end > start"
    ).fetchall()
    out = []
    for start, end, text, text_id, tid in rows:
        name = text or strings.get(text_id, "")
        if name.startswith(("pyrite.", "gpuhost.")):
            out.append((start, end, name, tid))
    return out


def _overlap(busy, starts, lo, hi):
    """Busy nanoseconds of ``busy`` (sorted, disjoint) inside ``[lo, hi)``."""
    total = 0
    i = max(0, bisect.bisect_right(starts, lo) - 1)
    while i < len(busy) and busy[i][0] < hi:
        total += max(0, min(busy[i][1], hi) - max(busy[i][0], lo))
        i += 1
    return total


def _sweep(ranges, idle, lo, activities, activity_starts):
    """Idle ns by innermost open range, and by (range at gap end | ending activity).

    Boundaries of ranges and idle gaps are swept in time order; the innermost
    open range is the latest-started one still open (ranges on one thread
    nest; with several threads it is the latest-started across them).
    """
    events = []
    for index, (s, e, _, _) in enumerate(ranges):
        events.append((s, 1, index))
        events.append((e, 0, index))
    for s, e in idle:
        events.append((s, 2, -1))
        events.append((e, 3, -1))
    events.sort()
    open_ranges: list[int] = []
    in_gap = False
    gap_start = lo
    previous = lo
    idle_by_range: dict[str, int] = collections.Counter()
    ended_by: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # ns, gaps
    for t, kind, index in events:
        if in_gap and t > previous:
            name = ranges[open_ranges[-1]][2] if open_ranges else "<none>"
            idle_by_range[name] += t - previous
        previous = t
        if kind == 1:
            open_ranges.append(index)
        elif kind == 0:
            if index in open_ranges:
                open_ranges.remove(index)
        elif kind == 2:
            in_gap = True
            gap_start = t
        else:
            in_gap = False
            where = ranges[open_ranges[-1]][2] if open_ranges else "<none>"
            at = bisect.bisect_left(activity_starts, t)
            nxt = activities[at][1] if at < len(activities) else "<end>"
            row = ended_by[f"{where} | {nxt}"]
            row[0] += t - gap_start
            row[1] += 1
    return idle_by_range, ended_by


def reduce(path):
    db = sqlite3.connect(path)
    tables = _tables(db)
    strings = _strings(db)
    ranges = _ranges(db, strings)
    run = [r for r in ranges if r[2] == "gpuhost.run"]
    if len(run) != 1:
        raise SystemExit(f"expected one gpuhost.run range, found {len(run)}")
    lo, hi, _, main_tid = run[0]
    ranges_all = [r for r in ranges if r[0] < hi and r[1] > lo]
    ranges = [r for r in ranges_all if r[3] == main_tid]
    busy = [[max(s, lo), min(e, hi)] for s, e in _busy(db, tables) if e > lo and s < hi]
    starts = [s for s, _ in busy]
    activities = _activities(db, tables, strings)
    activity_starts = [a[0] for a in activities]
    idle = []
    cursor = lo
    for s, e in busy:
        if s > cursor:
            idle.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        idle.append((cursor, hi))

    idle_by_range, ended_by = _sweep(ranges, idle, lo, activities, activity_starts)
    # Same sweep over the ranges of every thread: scoring and other work on
    # worker threads (e.g. ``pyrite-stream-score``) is invisible to the
    # main-thread view, which then charges its idle to whatever the main
    # thread has open.
    idle_all_threads, _ = _sweep(ranges_all, idle, lo, activities, activity_starts)

    per_range: dict[str, dict] = collections.defaultdict(
        lambda: {"calls": 0, "wall_s": 0.0, "device_busy_s": 0.0}
    )
    for s, e, name, _ in ranges:
        row = per_range[name]
        row["calls"] += 1
        row["wall_s"] += (e - s) / 1e9
        row["device_busy_s"] += _overlap(busy, starts, s, e) / 1e9

    histogram = collections.Counter()
    for s, e in idle:
        d = e - s
        label = next((f"<{b / 1e6:g}ms" for b in BUCKETS_NS if d < b), ">=100ms")
        histogram[label] += d

    def stack_at(t):
        return [r[2] for r in sorted(ranges, key=lambda r: r[0]) if r[0] <= t < r[1]]

    largest = sorted(idle, key=lambda g: g[1] - g[0], reverse=True)[:25]
    kernels = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        kernels = db.execute(
            "select coalesce(s.value, k.shortName), count(*), sum(k.end - k.start) "
            "from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.shortName "
            "group by 1 order by 3 desc limit 25"
        ).fetchall()
    api = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api = db.execute(
            "select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on s.id = r.nameId group by 1 order by 3 desc limit 25"
        ).fetchall()
    off_main: dict[str, dict] = collections.defaultdict(lambda: {"calls": 0, "wall_s": 0.0})
    for s_, e_, name, tid in ranges_all:
        if tid != main_tid:
            off_main[name]["calls"] += 1
            off_main[name]["wall_s"] += (e_ - s_) / 1e9
    wall = (hi - lo) / 1e9
    busy_s = sum(e - s for s, e in busy) / 1e9
    return {
        "schema": "pyrite.native-cpu-audit.nsys-gaps.v1",
        "run_wall_s": wall,
        "device_busy_s": busy_s,
        "device_idle_s": wall - busy_s,
        "device_busy_fraction": busy_s / wall if wall else None,
        "idle_gaps": len(idle),
        "idle_by_gap_size_s": {k: v / 1e9 for k, v in sorted(histogram.items())},
        "idle_by_innermost_range_s": {
            k: v / 1e9 for k, v in sorted(idle_by_range.items(), key=lambda kv: -kv[1])
        },
        "idle_by_innermost_range_all_threads_s": {
            k: v / 1e9 for k, v in sorted(idle_all_threads.items(), key=lambda kv: -kv[1])
        },
        "ranges_off_main_thread": off_main,
        "idle_by_range_and_ending_activity": [
            {"where": k, "idle_s": v[0] / 1e9, "gaps": v[1]}
            for k, v in sorted(ended_by.items(), key=lambda kv: -kv[1][0])[:ENDED_BY_LIMIT]
        ],
        "per_range": dict(sorted(per_range.items(), key=lambda kv: -kv[1]["wall_s"])),
        "largest_idle_gaps": [
            {
                "offset_s": (s - lo) / 1e9,
                "duration_s": (e - s) / 1e9,
                "stack": stack_at((s + e) // 2),
            }
            for s, e in largest
        ],
        "top_kernels": [{"name": n, "calls": c, "total_s": t / 1e9} for n, c, t in kernels],
        "top_cuda_api": [{"name": n, "calls": c, "total_s": t / 1e9} for n, c, t in api],
    }


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    report = reduce(argv[1])
    Path(argv[2]).write_text(json.dumps(report, indent=1) + "\n")
    brief = {k: report[k] for k in ("run_wall_s", "device_busy_s", "device_busy_fraction")}
    brief["top_idle"] = list(report["idle_by_innermost_range_s"].items())[:5]
    print(json.dumps(brief))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
