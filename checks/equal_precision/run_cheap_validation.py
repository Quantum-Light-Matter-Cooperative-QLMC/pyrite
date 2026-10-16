"""Run a fixed, capped production-FP64 numerical-validation pilot remotely."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def requests(protocol, stage):
    rows = []
    for seed in protocol["convergence"]["seeds"] if stage == "gpu" else []:
        for scene in protocol["scenes"]:
            arms = list(protocol["arms"])
            if seed % 2:
                arms.reverse()
            rows.extend(
                dict(
                    scene=scene,
                    seed=seed,
                    histories=protocol["convergence"]["histories"],
                    arm=arm,
                    core="cuda",
                )
                for arm in arms
            )
    for seed in protocol["backend"]["seeds"]:
        for scene in protocol["scenes"]:
            rows.append(
                dict(
                    scene=scene,
                    seed=seed,
                    histories=protocol["backend"]["histories"],
                    arm="baseline",
                    core="cuda" if stage == "gpu" else "per-electron",
                )
            )
    if stage == "cpu":
        rows.extend(
            dict(
                scene=scene,
                seed=protocol["geant4"]["seed"],
                histories=protocol["geant4"]["histories"],
                arm="baseline",
                core="geant4",
            )
            for scene in protocol["scenes"]
        )
    return rows


def main():
    from run_pilot import run_command

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cpu", "gpu"])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("cheap-validation.json")
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = requests(protocol, args.stage)
    deadline = time.monotonic() + (540 if args.stage == "gpu" else 240)
    root = Path(__file__).parent
    incomplete = False
    for row in rows:
        name = "{scene}-{histories}-{seed}-{core}-{arm}".format(**row)
        path = args.output / (name + ".json")
        if path.exists() or path.with_suffix(".failed.json").exists():
            raise ValueError("immutable output directory must be fresh")
        if row["core"] == "geant4":
            cmd = [
                sys.executable,
                str(root / "geant4_adapter.py"),
                row["scene"],
                "--histories",
                str(row["histories"]),
                "--seed",
                str(row["seed"]),
                "--output",
                str(path),
                "--executable",
                protocol["geant4"]["executable"],
            ]
        else:
            cmd = [
                sys.executable,
                str(root / "benchmark_validation.py"),
                "--protocol",
                str(args.protocol),
                "--arm",
                row["arm"],
                row["scene"],
                "--histories",
                str(row["histories"]),
                "--seed",
                str(row["seed"]),
                "--core",
                row["core"],
                "--output",
                str(path),
                "--batch-histories",
                "500",
                "--scratch-mib",
                "4096",
                "--probe-electrons",
                "0",
            ]
        remaining = deadline - time.monotonic()
        if remaining < 10:
            path.with_suffix(".deferred.json").write_text(
                json.dumps(dict(request=row, reason="allocation budget"))
            )
            incomplete = True
            continue
        started = time.monotonic()
        env = dict(os.environ, PYRITE_MC_BACKEND="cuda" if row["core"] == "cuda" else "cpu")
        try:
            with path.with_suffix(".log").open("w") as log:
                run_command(cmd, env, log, min(90, remaining))
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            path.with_suffix(".failed.json").write_text(
                json.dumps(dict(request=row, reason=str(error)))
            )
            print("FAILED", name, flush=True)
            incomplete = True
        else:
            path.with_suffix(".wall.json").write_text(
                json.dumps(dict(wall_s=time.monotonic() - started))
            )
            print("OK", name, flush=True)
    if incomplete:
        raise RuntimeError(
            "fixed cheap-validation stage incomplete; inspect failure/deferred evidence"
        )


if __name__ == "__main__":
    main()
