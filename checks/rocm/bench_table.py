"""Summarize ``evaluate.sbatch`` bench runs: warm transport per history.

Reads ``bench-<precision>-<scene>-<seed>.json`` from one or more evidence
directories and prints a Markdown table of median warm-transport time per
history and launched rows per second, per directory, scene and precision.

    uv run python checks/rocm/bench_table.py evidence/gpu-eval888/amd/<job> \\
        [evidence/gpu-eval888/nvidia/<job>]
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^bench-(fp64|mixed)-(.+)-(\d+)\.json$")


def collect(directory: Path):
    rows = defaultdict(list)
    for path in sorted(directory.glob("bench-*.json")):
        match = NAME.match(path.name)
        if not match:
            continue
        precision, scene, _seed = match.groups()
        record = json.loads(path.read_text())
        warm = record["timings_s"]["warm_transport"]
        histories = record["histories"]
        rows[(scene, precision)].append(
            (warm / histories, record["work"]["warm"]["launched_rows"] / warm)
        )
    return rows


def main(argv: list[str]) -> int:
    print("| run | scene | precision | n | warm us/history | Mrows/s | mixed speedup |")
    print("|---|---|---|---:|---:|---:|---:|")
    for arg in argv:
        directory = Path(arg)
        rows = collect(directory)
        label = "/".join(directory.parts[-2:])
        for scene in sorted({scene for scene, _ in rows}):
            per = {}
            for precision in ("fp64", "mixed"):
                values = rows.get((scene, precision))
                if not values:
                    continue
                per[precision] = statistics.median(v[0] for v in values)
                speed = ""
                if precision == "mixed" and "fp64" in per:
                    speed = f"{per['fp64'] / per['mixed']:.2f}x"
                rate = statistics.median(v[1] for v in values) / 1e6
                print(
                    f"| {label} | {scene} | {precision} | {len(values)} | "
                    f"{per[precision] * 1e6:.1f} | {rate:.1f} | {speed} |"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
