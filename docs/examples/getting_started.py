"""Capture a real CLI quickstart and its checkpoint inventory on CPU."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def compute():
    # Reuse installed tables; isolate checkpoint writes from existing campaigns.
    with tempfile.TemporaryDirectory(prefix="pyrite-quickstart-") as workspace:
        checkpoints = Path(workspace) / "pyrite-output" / "checkpoints"
        command = [
            "pyrite",
            "run",
            "quickstart",
            "--checkpoint-dir",
            str(checkpoints),
            "--workers",
            "0",
            "--no-progress",
            "-o",
            "json",
        ]
        env = dict(os.environ, PYRITE_MC_BACKEND="cpu", NO_COLOR="1")
        run = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
        # Inventory discovery needs only the checkpoint workspace, not tables.
        inventory = subprocess.run(
            ["pyrite", "checkpoint", "list"],
            env=dict(env, PYRITE_HOME=workspace),
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "command": [part.replace(workspace, "<workspace>") for part in command],
            "run": json.loads(run.stdout.replace(workspace, "<workspace>")),
            "display_text": inventory.stdout,
        }


if __name__ == "__main__":
    print(compute()["display_text"], end="")
