"""Bounded single-scattering follow-up using the unchanged Geant4 adapter."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_pilot import run_command


def main():
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).parent
    config = json.loads((here / "scaling.json").read_text())
    digest = hashlib.sha256(
        b"".join(
            (here / name).read_bytes()
            for name in [
                "manifest.json",
                "scaling.json",
                "geant4_adapter.py",
                "run_single_scattering.py",
            ]
        )
    ).hexdigest()[:12]
    output = root / "evidence" / "single-scattering" / digest
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 540
    uv = str(Path.home() / ".local/bin/uv")
    # Small preflights retain startup/table costs, like the previous Geant4 arm.
    requests = [(1000, 53000, scene) for scene in config["scenes"]]
    requests += [
        (config["histories"], seed, scene) for seed in config["seeds"] for scene in config["scenes"]
    ]
    for histories, seed, scene in requests:
        path = output / f"{scene}-{histories}-{seed}-geant4-SS.json"
        if path.exists() or path.with_suffix(".failed.json").exists():
            continue
        remaining = deadline - time.monotonic()
        if remaining < 60:
            print("BOUNDED_STOP", flush=True)
            return
        cmd = [
            uv,
            "run",
            "--no-sync",
            "python",
            str(here / "geant4_adapter.py"),
            scene,
            "--histories",
            str(histories),
            "--seed",
            str(seed),
            "--physics",
            "emstandardSS",
            "--executable",
            str(root / "benchmark-geant4-v2/build/TestEm5"),
            "--output",
            str(path),
        ]
        print("RUN", path.name, flush=True)
        started = time.monotonic()
        try:
            with path.with_suffix(".log").open("w") as log:
                run_command(
                    cmd, dict(os.environ, PYRITE_MC_BACKEND="cpu"), log, min(420, remaining - 10)
                )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            path.with_suffix(".failed.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "command": cmd,
                        "reason": str(exc),
                        "wall_s": time.monotonic() - started,
                    }
                )
                + "\n"
            )
            print("FAILED", path.name, str(exc), flush=True)
            if histories == 1000:
                raise
        path.with_suffix(".wall.json").write_text(
            json.dumps(
                {
                    "command": cmd,
                    "wall_s": time.monotonic() - started,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
            )
            + "\n"
        )
        print("DONE", path.name, time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
