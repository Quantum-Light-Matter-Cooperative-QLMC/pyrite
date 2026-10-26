"""Capture the disposable configuration exercise; no transport or table access."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_suffix(".sh")


def compute():
    # The script isolates the user store, catalog and workspace before any edit.
    run = subprocess.run(
        ["sh", str(SCRIPT)],
        env=dict(os.environ, PYRITE_MC_BACKEND="cpu"),
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "exit_code": run.returncode,
        "display_text": run.stdout,
    }


if __name__ == "__main__":
    print(json.dumps(compute(), indent=2))
