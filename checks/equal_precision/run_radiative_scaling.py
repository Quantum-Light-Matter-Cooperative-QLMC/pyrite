"""Execute one frozen large-population stage under a process-group deadline."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_pilot import run_command


def main():
    stage, destination = sys.argv[1:]
    output = Path(destination)
    manifest = json.loads(Path(__file__).with_name("radiative-scaling.json").read_text())
    deadline = time.monotonic() + 565
    for scene, variant in manifest["stages"][stage]:
        stem = output / f"{scene}-{manifest['seed']}-{variant}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_radiative.py")),
            variant,
            str(stem) + "-audit.json",
            "--baseline-source",
            str(Path(__file__).with_name("radiative_baseline.py")),
            scene,
            "--histories",
            str(manifest["histories"]),
            "--seed",
            str(manifest["seed"]),
            "--core",
            "cuda",
            "--batch-histories",
            str(manifest["batch_histories"]),
            "--scratch-mib",
            str(manifest["scratch_mib"]),
            "--probe-electrons",
            "0",
            "--output",
            str(stem) + ".json",
        ]
        remaining = deadline - time.monotonic()
        if remaining < 5:
            Path(str(stem) + ".deferred").write_text("allocation budget exhausted\n")
            continue
        with Path(str(stem) + ".log").open("w") as log:
            try:
                run_command(command, os.environ.copy(), log, remaining)
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                Path(str(stem) + ".failed").write_text(str(error) + "\n")


if __name__ == "__main__":
    main()
