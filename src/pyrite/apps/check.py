"""Persisted defaults and remote-Zhai orchestration for the marimo validation
app (``src/pyrite/apps/validation_app.py``).

The validation app reproduces fixed literature figures (e.g. Zhai et al.)
rather than sweeping a chosen material, so the only state it persists is the
supplementary TMD azimuth. The remote helpers below drive the Zhai
reproduction sweep from inside the notebook.

The launcher and cached-figure export are in
:mod:`pyrite.cli.commands.app_validation`; this module stays free of CLI
imports so the notebook can load it on its own.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..paths import app_dir, atomic_write_text, state_dir
from ..remote.config import remote_host

DEFAULTS_PATH = state_dir() / "validation-defaults.json"
_PACKAGED_DEFAULTS_PATH = app_dir() / "validation_defaults.json"


def load_default_azimuth(path=DEFAULTS_PATH):
    """Load the user override or packaged exploratory TMD azimuth default."""
    resolved = Path(path)
    if resolved == DEFAULTS_PATH and not resolved.is_file():
        resolved = _PACKAGED_DEFAULTS_PATH
    data = json.loads(resolved.read_text(encoding="utf-8"))
    value = float(data["tmd_exploratory_azimuth_deg"])
    if not 0.0 <= value <= 180.0:
        raise ValueError("default azimuth must be between 0 and 180 degrees")
    return value


def save_default_azimuth(value, path=DEFAULTS_PATH):
    """Atomically update the tracked exploratory TMD azimuth default."""
    value = float(value)
    if not 0.0 <= value <= 180.0:
        raise ValueError("default azimuth must be between 0 and 180 degrees")
    path = Path(path)
    atomic_write_text(
        path,
        json.dumps({"tmd_exploratory_azimuth_deg": value}, indent=2) + "\n",
    )


def _cli(*args):
    return [sys.executable, "-m", "pyrite.cli", *args]


def _remote_cli(*args):
    return _cli("remote", *args)


def probe_remote_zhai(timeout=5):
    """Return whether the optional SSH-backed GPU runner is reachable."""
    missing = [command for command in ("ssh", "scp") if shutil.which(command) is None]
    if missing:
        return False, f"missing remote transport command(s): {', '.join(missing)}"
    try:
        completed = subprocess.run(
            [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={int(timeout)}",
                remote_host(),
                "true",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout + 1,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"remote GPU probe failed: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"ssh exited {completed.returncode}"
        return False, f"remote GPU unavailable: {detail}"
    return True, "remote GPU available"


_JOB_ID_RE = re.compile(r"JOB (\d{8}-\d{6}-[0-9a-f]{8}) · SUBMITTED")


def start_remote_zhai(*, ne, ne_brem, ne_supp, tmd_azimuth, refresh=False):
    """Launch the existing detached remote Zhai job and return its job id."""
    # `pyrite run --preset zhai --remote --detach` is the canonical launch path
    # (the retired `remote check`/`validate` spelling this used to call is no
    # longer mounted on the CLI tree, see commit fa3e0aa1). It prints a
    # `JOB <jobid> · SUBMITTED` banner (`remote/queue.py::start_zhai_queue`) on
    # success, not the "started zhai job <id>" text this used to look for.
    command = _cli(
        "run",
        "--preset",
        "zhai",
        "--remote",
        "--detach",
        "--ne",
        str(int(ne)),
        "--ne-brem",
        str(int(ne_brem)),
        "--ne-supp",
        str(int(ne_supp)),
        "--tmd-azimuth",
        str(float(tmd_azimuth)),
        *(["--refresh"] if refresh else []),
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    match = _JOB_ID_RE.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"remote launch did not report a job id:\n{completed.stdout.strip()}")
    return match.group(1)


def remote_zhai_status(jobid):
    """Return a normalized state plus the complete remote status report."""
    # `pyrite remote status` is retired (commit fa3e0aa1); job status now
    # lives under `pyrite job status <jobid>`.
    completed = subprocess.run(
        _cli("job", "status", jobid),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).rstrip()
    if completed.returncode != 0:
        return "error", output
    match = re.search(r"-- state --\s*\n([^\r\n]+)", completed.stdout)
    raw_state = match.group(1).strip().lower() if match else "unknown"
    state = next(
        (prefix for prefix in ("running", "done", "failed") if raw_state.startswith(prefix)),
        raw_state,
    )
    return state, output


def pull_remote_zhai():
    """Pull completed remote Zhai caches into the local repository."""
    # `pyrite remote check --pull` is retired (commit fa3e0aa1); the live
    # equivalent is `pyrite remote pull --preset zhai`.
    completed = subprocess.run(
        _remote_cli("pull", "--preset", "zhai"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).rstrip()
    if completed.returncode != 0:
        raise RuntimeError(output)
    return output
