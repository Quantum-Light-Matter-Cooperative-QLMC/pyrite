"""Shard the two over-budget #1041 cones; never produce timing comparisons."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONE = "cone_source_photons_ge10kev"
SCENES = {"si-300keV-100um", "si-3MeV-1mm"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_plan(source, shard_histories=1_000_000, threads=8):
    if shard_histories < 2 or threads < 1:
        raise ValueError("need at least two histories per shard and positive threads")
    frozen = json.loads(source.read_text())
    items = [
        i
        for i in frozen["items"]
        if i["scene"] in SCENES
        and i["config"] == "geant4-32t"
        and i["set"] == "cone"
        and i["target_rse"] == 0.02
    ]
    if len(items) != 2 or {i["scene"] for i in items} != SCENES:
        raise ValueError("need both frozen Si cone items at epsilon=2%")
    tasks = []
    for item in sorted(items, key=lambda i: i["scene"]):
        total = item["histories"]
        if item["status"] != "over-budget" or not isinstance(total, int) or total < 2:
            raise ValueError("reference must derive from a positive over-budget frozen count")
        # Balanced shards avoid a final one-history sample (undefined sample variance).
        count = math.ceil(total / shard_histories)
        n, remainder = divmod(total, count)
        if n < 2:
            raise ValueError("shard size is too small to retain two primaries per shard")
        for shard in range(count):
            index = len(tasks)
            tasks.append(
                dict(
                    index=index,
                    scene=item["scene"],
                    histories=n + (shard < remainder),
                    seed=200001 + index,
                )
            )
    return dict(
        schema="pyrite.geant4-performance.accuracy-reference-plan.v1",
        purpose="accuracy-only",
        source_plan_sha256=digest(source),
        target_rse=0.02,
        threads=threads,
        tasks=tasks,
        numerical_convergence="unverified",
        timing_comparison_eligible=False,
    )


def run_task(plan_path, index, executable, output):
    plan = json.loads(plan_path.read_text())
    if plan["purpose"] != "accuracy-only" or plan["timing_comparison_eligible"]:
        raise ValueError("an accuracy-only plan is required")
    task = plan["tasks"][index]
    if index < 0 or task["index"] != index:
        raise ValueError("invalid shard index")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"shard-{index:04d}.json"
    if path.exists():
        raise FileExistsError(path)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "geant4_bench.py"),
            task["scene"],
            "--histories",
            str(task["histories"]),
            "--warmup",
            "100",
            "--seed",
            str(task["seed"]),
            "--threads",
            str(plan["threads"]),
            "--executable",
            str(executable),
            "--output",
            str(path),
            "--accuracy-only",
        ],
        check=True,
    )
    record = json.loads(path.read_text())
    record.update(
        reference_plan_sha256=digest(plan_path),
        reference_shard=index,
        reference_harness_sha256=digest(Path(__file__)),
    )
    path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def summarize(plan_path, output):
    plan = json.loads(plan_path.read_text())
    expected = digest(plan_path)
    rows, binaries = {}, set()
    for task in plan["tasks"]:
        path = output / f"shard-{task['index']:04d}.json"
        record = json.loads(path.read_text())
        if (
            record.get("purpose") != "accuracy-only"
            or record.get("reference_plan_sha256") != expected
            or record.get("reference_shard") != task["index"]
            or record["scene"]["id"] != task["scene"]
            or record["seed"] != task["seed"]
            or record["histories"] != task["histories"]
            or record["threads"] != plan["threads"]
            or digest(path.with_suffix(".npz")) != record["raw_scores_sha256"]
        ):
            raise ValueError(f"shard identity or raw-score digest mismatch: {path}")
        binaries.add(record["executable_sha256"])
        moments = record["observables"][CONE]
        if moments["n"] != task["histories"]:
            raise ValueError(f"wrong incident-primary count: {path}")
        row = rows.setdefault(task["scene"], dict(n=0, sum=0.0, sum_squares=0.0, nonzero=0))
        for key in row:
            row[key] += moments[key]
    if len(binaries) != 1:
        raise ValueError("all shards must use the same Geant4 executable")
    for row in rows.values():
        n = row["n"]
        mean = row["sum"] / n
        variance = max(0.0, (row["sum_squares"] - row["sum"] ** 2 / n) / (n - 1))
        se = math.sqrt(variance / n)
        rse = se / mean if mean > 0 and se > 0 else None
        row.update(
            mean=mean, se=se, rse=rse, precision_met=rse is not None and rse <= plan["target_rse"]
        )
    return dict(
        purpose="accuracy-only",
        reference_plan_sha256=expected,
        executable_sha256=next(iter(binaries)),
        numerical_convergence="unverified",
        timing_comparison_eligible=False,
        scenes=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("plan")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--shard-histories", type=int, default=1_000_000)
    freeze.add_argument("--threads", type=int, default=8)
    run = commands.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("index", type=int)
    run.add_argument("--executable", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    collect = commands.add_parser("summarize")
    collect.add_argument("plan", type=Path)
    collect.add_argument("evidence", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_task(args.plan, args.index, args.executable, args.output)
        return
    result = (
        make_plan(args.source, args.shard_histories, args.threads)
        if args.command == "plan"
        else summarize(args.plan, args.evidence)
    )
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
