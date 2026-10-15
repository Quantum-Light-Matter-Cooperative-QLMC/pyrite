"""Interleave fresh-process adapters within one bounded SLURM allocation."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def run_command(cmd, env, log, timeout):
    """Enforce a sample deadline on its complete process tree, including uv."""
    with subprocess.Popen(
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    ) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f"adapter exit {returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "pilot", "measurement", "scaling"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads(Path(__file__).with_name("manifest.json").read_text())
    adapter_digest = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in ["manifest.json", "pyrite_adapter.py", "geant4_adapter.py"]
        )
    ).hexdigest()[:12]
    scaling = (
        json.loads(Path(__file__).with_name("scaling.json").read_text())
        if args.stage == "scaling"
        else None
    )
    if scaling:
        adapter_digest = hashlib.sha256(
            (adapter_digest + json.dumps(scaling, sort_keys=True)).encode()
        ).hexdigest()[:12]
    output = root / "evidence" / args.stage / adapter_digest
    output.mkdir(parents=True, exist_ok=True)
    uv = str(Path.home() / ".local/bin/uv")
    started = time.monotonic()
    # Leave time to flush evidence before the scheduler's ten-minute limit.
    deadline = started + (540 if args.stage != "smoke" else 240)
    sizes = [20] if args.stage == "smoke" else list(reversed(protocol["histories"]))
    scenes = protocol["scenes"][:2] if args.stage == "smoke" else protocol["scenes"]
    seeds = (
        protocol["measurement_seeds"] if args.stage == "measurement" else protocol["pilot_seeds"]
    )
    if args.stage == "smoke":
        seeds = seeds[:1]
    if scaling:
        sizes = [scaling["histories"]]
        seeds = scaling["seeds"]
        scenes = [s for s in protocol["scenes"] if s["id"] in scaling["scenes"]]
    for seed in seeds:
        for n in sizes:
            for scene in scenes:
                arms = list(scaling["arms"]) if scaling else ["per-electron", "geant4", "cuda"]
                if seed % 2:
                    arms.reverse()
                for arm in arms:
                    path = output / f"{scene['id']}-{n}-{seed}-{arm}.json"
                    if scaling and path.with_suffix(".failed.json").exists():
                        # A fresh allocation may have enough time for a request
                        # that hit the previous allocation's remaining budget.
                        attempts = list(output.glob(path.stem + "-attempt-*.json"))
                        if any(
                            json.loads(p.read_text()).get("schema")
                            == "pyrite.equal-precision.sample.v1"
                            for p in attempts
                        ):
                            continue
                        path = path.with_name(
                            path.stem + "-attempt-" + os.environ["SLURM_JOB_ID"] + ".json"
                        )
                    if path.exists() or path.with_suffix(".failed.json").exists():
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining < 30:
                        print(
                            "BOUNDED_STOP: resume with the same stage; existing records remain immutable",
                            flush=True,
                        )
                        return
                    if scaling and arm == "cuda":
                        # Full-request warmup plus measurement needs another
                        # allocation when the observed request cost cannot fit.
                        minimum_start = 150 if scene["element"] == "Si" else 480
                        if remaining < minimum_start:
                            print("BUDGET_DEFERRED", path.name, flush=True)
                            continue
                    adapter = "geant4_adapter.py" if arm == "geant4" else "pyrite_adapter.py"
                    cmd = [
                        uv,
                        "run",
                        "--no-sync",
                        "python",
                        str(Path(__file__).with_name(adapter)),
                        scene["id"],
                        "--histories",
                        str(n),
                        "--seed",
                        str(seed),
                        "--output",
                        str(path),
                    ]
                    if arm == "geant4":
                        cmd += ["--executable", str(root / "benchmark-geant4-v2/build/TestEm5")]
                    else:
                        cmd += ["--core", arm]
                        if scaling:
                            cmd += [
                                "--batch-histories",
                                str(scaling["batch_histories"]),
                                "--scratch-mib",
                                str(scaling["scratch_mib"]),
                                "--probe-electrons",
                                str(scaling["probe_electrons"]),
                            ]
                        if args.stage == "smoke":
                            cmd += ["--audit-cxr"]
                    env = dict(os.environ, PYRITE_MC_BACKEND="cuda" if arm == "cuda" else "cpu")
                    print("RUN", path.name, flush=True)
                    begin = time.monotonic()
                    try:
                        with path.with_suffix(".log").open("w") as log:
                            run_command(cmd, env, log, min(480 if scaling else 120, remaining - 10))
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        path.with_suffix(".failed.json").write_text(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "command": cmd,
                                    "reason": str(exc),
                                    "wall_s": time.monotonic() - begin,
                                }
                            )
                            + "\n"
                        )
                        print("FAILED", path.name, str(exc), flush=True)
                        # Smoke failures must be resolved before matrix work.
                        if args.stage == "smoke":
                            raise
                    wall = {
                        "command": cmd,
                        "wall_s": time.monotonic() - begin,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    }
                    path.with_suffix(".wall.json").write_text(json.dumps(wall) + "\n")


if __name__ == "__main__":
    main()
