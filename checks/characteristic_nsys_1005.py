"""Kernel launches and copies inside the #1005 ``char.*`` NVTX ranges.

Usage: characteristic_nsys_1005.py REPORT.sqlite

Reads an ``nsys export --type sqlite`` of a ``characteristic_scoring_1005.py``
run. A CUDA runtime call belongs to a range when it starts inside that range
on the same thread; its kernel or copy is attributed through the correlation
ID. ``setup`` is ``total`` less ``clip``, ``escape`` and ``accumulate``
(nested ranges are counted in their parents too). Prints one JSON object of
launches, GPU kernel time, and H2D/D2H count, bytes and time per range.
"""

import bisect
import json
import sqlite3
import sys
from collections import defaultdict

NAMES = ("total", "clip", "escape", "accumulate", "interp")
COPY_KIND = {1: "h2d", 2: "d2h"}


def _ranges(db):
    has_ids = (
        db.execute("SELECT 1 FROM sqlite_master WHERE name='StringIds'").fetchone() is not None
    )
    text = "COALESCE(e.text, s.value)" if has_ids else "e.text"
    join = "LEFT JOIN StringIds s ON e.textId = s.id" if has_ids else ""
    rows = db.execute(
        f"SELECT {text}, e.start, e.end, e.globalTid FROM NVTX_EVENTS e {join} "
        "WHERE e.end IS NOT NULL"
    ).fetchall()
    out = defaultdict(list)
    for name, start, end, tid in rows:
        if name and name.startswith("char."):
            out[name[5:]].append((start, end, tid))
    return out


def summarize(path):
    db = sqlite3.connect(path)
    ranges = _ranges(db)
    runtime = db.execute(
        "SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    kernels = dict(
        db.execute("SELECT correlationId, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    )
    copies = {
        cid: (kind, size, dur)
        for cid, kind, size, dur in db.execute(
            "SELECT correlationId, copyKind, bytes, end - start FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    }
    stats = {}
    for name in NAMES:
        spans = sorted(ranges.get(name, ()))
        s = defaultdict(float)
        s["ranges"] = len(spans)
        starts = [span[0] for span in spans]
        for start, tid, cid in runtime:
            i = bisect.bisect_right(starts, start) - 1
            if i < 0 or start > spans[i][1] or spans[i][2] != tid:
                continue
            if cid in kernels:
                s["launches"] += 1
                s["kernel_s"] += kernels[cid] * 1e-9
            if cid in copies:
                kind, size, dur = copies[cid]
                key = COPY_KIND.get(kind, "other")
                s[f"{key}_n"] += 1
                s[f"{key}_MB"] += size / 1e6
                s[f"{key}_s"] += dur * 1e-9
        stats[name] = dict(s)
    keys = set().union(*(stats[n].keys() for n in NAMES)) - {"ranges"}
    stats["setup"] = {
        k: stats["total"].get(k, 0.0)
        - sum(stats[n].get(k, 0.0) for n in ("clip", "escape", "accumulate"))
        for k in sorted(keys)
    }
    return stats


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=1))
