"""Summarize a Linux slow-test audit directory; retain raw artifacts separately."""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = [
    json.loads(line)
    for path in (root / "rss").glob("*.jsonl")
    for line in path.read_text().splitlines()
]
phases = defaultdict(dict)
for duration, phase, node in re.findall(
    r"^\s*([\d.]+)s (setup|call|teardown)\s+(.+)$", (root / "pytest.log").read_text(), re.M
):
    phases[node][phase] = float(duration)
workers = defaultdict(list)
for row in rows:
    workers[row["worker"]].append(row)
wall = [row["wall_s"] for row in rows]
xml = ET.parse(root / "junit.xml").getroot()
suites = list(xml.iter("testsuite"))
result = {
    "environment": json.loads((root / "environment.json").read_text()),
    "sync_stamp": (root / "sync-stamp").read_text(),
    "tables": json.loads((root / "tables.json").read_text()),
    "junit": [
        {key: suite.get(key) for key in ("tests", "errors", "failures", "skipped", "time")}
        for suite in suites
    ],
    "per_test_wall_percentiles_s": dict(
        zip(
            ("p50", "p90", "p95", "p99", "max"),
            np.percentile(wall, [50, 90, 95, 99, 100]).tolist(),
            strict=True,
        )
    ),
    "top_wall": [
        {
            "nodeid": row["nodeid"],
            "wall_s": row["wall_s"],
            "phases_s": phases.get(row["nodeid"], {}),
        }
        for row in sorted(rows, key=lambda row: row["wall_s"], reverse=True)[:20]
    ],
    "top_rss": [
        {
            key: row[key]
            for key in ("nodeid", "worker", "start_rss", "peak_rss", "peak_tree_rss", "wall_s")
        }
        for row in sorted(rows, key=lambda row: row["peak_rss"], reverse=True)[:20]
    ],
    "workers": {
        worker: {
            "tests": len(items),
            "busy_s": sum(item["wall_s"] for item in items),
            "last_test_end": max(item["t_end"] for item in items),
        }
        for worker, items in workers.items()
    },
}
with (root / "tree.csv").open() as stream:
    tree = list(csv.DictReader(stream))
result["sampled_tree_peak_bytes"] = max(int(row["total_rss"]) for row in tree)
print(json.dumps(result, indent=2))
