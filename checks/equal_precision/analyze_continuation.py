"""Summarize ``benchmark_continuation.py`` evidence folders (#789).

Usage: ``python analyze_continuation.py EVIDENCE_DIR [...] --output summary.json``.

Groups measured samples by (scene, histories, policy, code revision) and
reports median [min, max] request and launch times, launch counts,
executed/retained rows, peak pool bytes, largest launch scratch and two
schedule diagnostics per request:

- ``tail_s``: synchronized launch time of launches whose population is below
  half the request's largest launch population.
- ``lane_step_efficiency``: retained rows over the sum, across launches, of
  launch population times that launch's longest lane row count. It is the
  fraction of lane-steps doing work if every lane waited for its launch's
  longest; a schedule proxy, not a measured warp or SM occupancy.

Every sample's warmup and measured array hashes are compared against the same
scene/histories/seed's ``reference`` policy on the same revision (or, without
one, the record's first policy), and
``main`` vs branch ``reference`` hashes are compared when both are present.
Failed markers are listed and their outputs never parsed.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def spread(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def schedule(measured):
    launches = measured["launches"]
    peak = max(launch["electrons"] for launch in launches)
    tail = sum(
        launch["synchronized_wall_s"] for launch in launches if launch["electrons"] < 0.5 * peak
    )
    lane_steps = sum(launch["electrons"] * launch["max_rows"] for launch in launches)
    return tail, measured["retained_rows"] / lane_steps if lane_steps else None


def load(folders):
    records, failed = [], []
    for folder in folders:
        for path in sorted(Path(folder).glob("*.json")):
            if path.with_suffix(".failed.json").exists() or path.name.endswith(".failed.json"):
                failed.append(str(path))
                continue
            record = json.loads(path.read_text())
            if record.get("schema") != "pyrite.transport-continuation.benchmark.v1":
                continue
            if record.get("status") != "ok":
                failed.append(str(path))
                continue
            # Synced lab checkouts are not git work trees; job 2640 names them.
            record["git_revision"] = record.get("git_revision") or path.name.split("-")[0]
            records.append(record)
    return records, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, failed = load(args.folders)
    reference = {}
    for record in records:
        for sample in record["samples"]:
            if sample["policy"]["name"] == "reference":
                key = (record["git_revision"], record["scene"], record["histories"], record["seed"])
                reference[key] = (
                    sample["warmup"]["arrays_sha256"],
                    sample["measured"]["arrays_sha256"],
                )
    groups = defaultdict(list)
    mismatches = []
    resources = {}
    for record in records:
        resources[record["git_revision"]] = record["compiled_kernel_resources"]
        key = (record["git_revision"], record["scene"], record["histories"], record["seed"])
        first = record["samples"][0]
        expected = reference.get(key) or (
            first["warmup"]["arrays_sha256"],
            first["measured"]["arrays_sha256"],
        )
        for sample in record["samples"]:
            name = sample["policy"]["name"]
            hashes = (sample["warmup"]["arrays_sha256"], sample["measured"]["arrays_sha256"])
            if hashes != expected:
                mismatches.append({"key": list(key), "policy": name})
            group = (record["git_revision"], record["scene"], record["histories"])
            groups[(*group, record["scratch_mib"], name)].append(sample["measured"])
    revisions = sorted(resources)
    cross = []
    for (revision, scene, histories, seed), hashes in sorted(reference.items()):
        for other in revisions:
            peer = reference.get((other, scene, histories, seed))
            if other != revision and peer is not None and revision < other:
                cross.append(
                    {
                        "scene": scene,
                        "histories": histories,
                        "seed": seed,
                        "revisions": [revision, other],
                        "identical": peer == hashes,
                    }
                )
    summary = []
    for (revision, scene, histories, scratch_mib, name), samples in sorted(groups.items()):
        tails, efficiencies = zip(*(schedule(m) for m in samples), strict=True)
        summary.append(
            {
                "revision": revision,
                "scene": scene,
                "histories": histories,
                "scratch_mib": scratch_mib,
                "policy": name,
                "request_s": spread([m["request_s"] for m in samples]),
                "launch_s": spread([m["launch_s"] for m in samples]),
                "tail_s": spread(list(tails)),
                "lane_step_efficiency": spread(list(efficiencies)),
                "launches": spread([len(m["launches"]) for m in samples]),
                "executed_rows": spread([m["executed_rows"] for m in samples]),
                "retained_rows": spread([m["retained_rows"] for m in samples]),
                "peak_pool_above_request_mib": spread(
                    [m["peak_pool_above_request_bytes"] / 2**20 for m in samples]
                ),
                "max_launch_scratch_mib": spread(
                    [m["max_launch_scratch_bytes"] / 2**20 for m in samples]
                ),
                "retained_host_mib": spread([m["retained_host_bytes"] / 2**20 for m in samples]),
            }
        )
    out = {
        "schema": "pyrite.transport-continuation.summary.v1",
        "groups": summary,
        "hash_mismatches_vs_reference": mismatches,
        "reference_hashes_across_revisions": cross,
        "compiled_kernel_resources": resources,
        "failed_or_partial": failed,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for g in summary:
        print(
            g["revision"][:8],
            g["scene"],
            g["histories"],
            f"{g['scratch_mib']}MiB",
            g["policy"],
            f"req {g['request_s']['median']:.3f}",
            f"launch {g['launch_s']['median']:.3f}",
            f"tail {g['tail_s']['median']:.3f}",
            f"eff {g['lane_step_efficiency']['median']:.2f}",
            f"n={g['request_s']['n']}",
            f"peak {g['peak_pool_above_request_mib']['median']:.0f}MiB",
        )
    print("mismatches:", mismatches)
    print("cross-revision:", cross)
    print("failed:", failed)


if __name__ == "__main__":
    main()
