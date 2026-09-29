"""Whole-run Python sampling with py-spy, shared by local and remote ``--py-spy``.

py-spy launches the scan as its own child, so it needs no ptrace permission on
an already-running process (lab hosts set ``kernel.yama.ptrace_scope=1`` and
grant no sudo). It samples the production run -- the real grid, backend and
worker tree -- unlike the bounded ``--quick`` cProfile pass. It is resolved
through ``uv tool run`` when no ``py-spy`` executable is on PATH, so it is not a
project dependency.
"""

import shutil
import subprocess
import sys
from pathlib import Path

#: Requirement ``uv tool run`` resolves when py-spy is not installed.
PY_SPY_REQUIREMENT = "py-spy>=0.4"
#: Samples per second. ``--nonblocking`` keeps the target running while it is
#: read; on the lab host 25 Hz already fell seconds behind a multi-GiB scan.
PY_SPY_RATE_HZ = 10
#: Output suffix next to the performance NDJSON; open it in https://speedscope.app.
PY_SPY_SUFFIX = ".py-spy.json"
#: Suffix of the file holding the sampled command's exit status. ``py-spy record
#: --subprocesses`` exits 0 even when its child fails, so the status travels
#: through :func:`main` instead.
PY_SPY_STATUS_SUFFIX = ".py-spy.status"


def status_wrapped(status_file: str, command: list[str], python: str = sys.executable) -> list[str]:
    """``command`` run through :func:`main`, which records its exit status."""
    return [python, "-m", "pyrite.perf.py_spy", status_file, "--", *command]


def read_status(status_file: str | Path) -> int:
    """Exit status :func:`main` recorded, or 1 when the command never reported."""
    try:
        return int(Path(status_file).read_text().strip())
    except OSError, ValueError:
        return 1


def py_spy_record_args(output: str) -> list[str]:
    """``record`` arguments that end with ``--``; append the command to sample."""
    return [
        "record",
        "--subprocesses",
        "--nonblocking",
        "--idle",
        "--rate",
        str(PY_SPY_RATE_HZ),
        "--format",
        "speedscope",
        "--output",
        output,
        "--",
    ]


def py_spy_launcher() -> list[str] | None:
    """Local py-spy executable argv, or ``None`` when neither it nor uv is on PATH."""
    if shutil.which("py-spy") is not None:
        return ["py-spy"]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "tool", "run", "--from", PY_SPY_REQUIREMENT, "py-spy"]
    return None


def main(argv: list[str] | None = None) -> int:
    """``python -m pyrite.perf.py_spy STATUS_FILE -- COMMAND...``.

    Runs COMMAND, writes its exit status to STATUS_FILE, and exits with it, so a
    caller can recover the status py-spy discards.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 3 or args[1] != "--":
        print("usage: python -m pyrite.perf.py_spy STATUS_FILE -- COMMAND...", file=sys.stderr)
        return 2
    status_file, command = args[0], args[2:]
    try:
        status = subprocess.call(command)
    except OSError as error:
        print(f"cannot run {command[0]!r}: {error}", file=sys.stderr)
        status = 127
    Path(status_file).write_text(f"{status}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
