"""Summarize diagnostic timing, replay work and same-seed trajectory hashes."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    grouped = defaultdict(list)
    for directory in args.directories:
        for path in sorted(directory.glob("*-audit.json")):
            stem = path.name.removesuffix("-audit.json")
            if (directory / f"{stem}.failed").exists():
                continue
            audit = json.loads(path.read_text())
            sample = json.loads(path.with_name(path.name.replace("-audit", "")).read_text())
            histories = sample["histories"]
            warmup_blocks = (histories + sample["batch_histories"] - 1) // sample["batch_histories"]
            split = audit["blocks"][warmup_blocks - 1]["launch_end"]
            launches = audit["launches"][split:]
            row = {
                "file": f"evidence/{directory.parent.name}/{directory.name}/{path.name}",
                "scene": sample["scene"]["id"],
                "histories": histories,
                "seed": sample["seed"],
                "variant": audit["variant"],
                "transport_s": sample["timings_s"]["transport_and_emission"],
                "launch_s": sum(x["synchronized_wall_s"] for x in launches),
                "launches": len(launches),
                "replayed_launches": sum(x["replayed"] for x in launches),
                "executed_segments": sum(x["segments"] for x in launches),
                "retained_segments": sum(x["segments"] for x in audit["blocks"][warmup_blocks:]),
                "measured_hashes": [x["arrays_sha256"] for x in audit["blocks"][warmup_blocks:]],
                "warmup_hashes": [x["arrays_sha256"] for x in audit["blocks"][:warmup_blocks]],
                "max_launch_electrons": max(x["electrons"] for x in launches),
            }
            rows.append(row)
            grouped[row["scene"], histories, row["variant"]].append(row)
    baseline = {
        (r["scene"], r["histories"], r["seed"]): r for r in rows if r["variant"] == "baseline"
    }
    for row in rows:
        reference = baseline.get((row["scene"], row["histories"], row["seed"]))
        row["identical_to_baseline"] = None
        if reference and row["variant"] != "radiation-off":
            row["identical_to_baseline"] = all(
                row[key] == reference[key] for key in ["warmup_hashes", "measured_hashes"]
            )
            if not row["identical_to_baseline"]:
                raise ValueError(f"trajectory mismatch: {row['file']}")
    groups = []
    for (scene, histories, variant), samples in sorted(grouped.items()):
        times = [r["transport_s"] for r in samples]
        groups.append(
            {
                "scene": scene,
                "histories": histories,
                "variant": variant,
                "repeats": len(times),
                "median_s": statistics.median(times),
                "min_s": min(times),
                "max_s": max(times),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.kernel-summary.v1",
                "groups": groups,
                "samples": rows,
                "interpretation": "Instrumented diagnostic, not a qualified cross-engine speed comparison. Radiation-off changes physical workload. No pooling across population sizes.",
            },
            indent=2,
        )
        + "\n"
    )
    for group in groups:
        print(group)


if __name__ == "__main__":
    main()
