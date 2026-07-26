"""``cxr validate`` -- launch the marimo validation app (``notebooks/validation_app.py``),
or render its figures in batch from cache.

Unlike ``cxr analyze``, this command takes no material argument -- the
validation app reproduces fixed literature figures (e.g. Zhai et al.) rather
than sweeping a chosen material, so there's no initial-material selection to
resolve or persist.

    cxr validate                   # `marimo run` the validation app
    cxr validate --watch           # add marimo's --watch
    cxr validate --edit            # `marimo edit` instead of `marimo run`
    cxr validate --export          # skip marimo; render the full Zhai figure
                                    # set from checkpoints/zhai_reproduction/
                                    # (see `cxr remote check`) to figures/
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click

from ._acp import running_acp
from ._remote.config import HOST as REMOTE_HOST
from .cli import _core as _cli_core

NOTEBOOK = "notebooks/validation_app.py"
TUNNEL_PORT = 2718
DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "validation_defaults.json"


def load_default_azimuth(path=DEFAULTS_PATH):
    """Load the repository-wide exploratory TMD azimuth default."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps({"tmd_exploratory_azimuth_deg": value}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remote_cli(*args):
    return [sys.executable, "-m", "cxr_mc.cli", "remote", *args]


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
                REMOTE_HOST,
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


def start_remote_zhai(*, ne, ne_brem, ne_supp, tmd_azimuth, refresh=False):
    """Launch the existing detached remote Zhai job and return its job id."""
    command = _remote_cli(
        "check",
        "--detached",
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
    match = re.search(r"started zhai job (\d{8}-\d{6})", completed.stdout)
    if match is None:
        raise RuntimeError(f"remote launch did not report a job id:\n{completed.stdout.strip()}")
    return match.group(1)


def remote_zhai_status(jobid):
    """Return a normalized state plus the complete remote status report."""
    completed = subprocess.run(
        _remote_cli("status", jobid),
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
    completed = subprocess.run(
        _remote_cli("check", "--pull"),
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


def _command(*, edit=False, watch=False, tunnel=False):
    """The marimo argv for one launch (module-run through the current
    interpreter so the venv's marimo is the one that runs). Marimo's own flags
    go before the notebook path; app args go after ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        *(["--port", str(TUNNEL_PORT)] if tunnel else []),
        NOTEBOOK,
        "--",
    ]


def _launch(*, edit=False, watch=False, acp=False, tunnel=False):
    cmd = _command(edit=edit, watch=watch, tunnel=tunnel)
    if tunnel:
        print(f"launching {NOTEBOOK} ({'edit' if edit else 'run'})")
        print(f"ssh -L {TUNNEL_PORT}:127.0.0.1:{TUNNEL_PORT} <your-pi-ssh-host>")
        print(f"http://127.0.0.1:{TUNNEL_PORT}")
    env = {**os.environ}
    try:
        if acp:
            with running_acp():
                subprocess.run(cmd, check=True, env=env)
        else:
            subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        # Ctrl+C is delivered to the marimo child and this parent on Windows.
        # Once marimo has handled its interactive exit, avoid a second traceback.
        return


def _export(outdir="figures", ne=20_000, ne_brem=200, ne_supp=200):
    checks_dir = Path(__file__).resolve().parents[2] / "checks"
    if str(checks_dir) not in sys.path:
        sys.path.insert(0, str(checks_dir))
    import anchor_figures as af  # ty: ignore[unresolved-import]

    written = af.export_all_figures(outdir, ne=ne, ne_brem=ne_brem, ne_supp=ne_supp)
    for path in written:
        print(f"wrote {path}")


def _cli(args):
    if args.export:
        _export(args.outdir, ne=args.ne, ne_brem=args.ne_brem, ne_supp=args.ne_supp)
        return
    launch_args = {"edit": args.edit, "watch": args.watch}
    if args.acp:
        launch_args["acp"] = True
    if args.tunnel:
        launch_args["tunnel"] = True
    _launch(**launch_args)


@click.command(
    "check",
    help=(
        f"Launch {NOTEBOOK}, or export its cached validation figures.\n\n"
        "--export skips marimo and writes figures to --outdir. Electron-count "
        "options affect export mode only."
    ),
)
@click.option("--watch", is_flag=True, help="Pass marimo's --watch.")
@click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
@click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
@click.option("--tunnel", is_flag=True, help="Use fixed port for SSH tunneling.")
@click.option("--export", "export_", is_flag=True, help="Render cached figures instead of marimo.")
@click.option(
    "--outdir",
    default="figures",
    show_default=True,
    metavar="DIR",
    help="With --export, output directory.",
)
@click.option(
    "--ne",
    type=_cli_core.POSITIVE_INT,
    default=20_000,
    show_default=True,
    help="With --export, Fig. 1c line electrons per energy.",
)
@click.option(
    "--ne-brem",
    type=_cli_core.POSITIVE_INT,
    default=200,
    show_default=True,
    help="With --export, Fig. 1c bremsstrahlung electrons per energy.",
)
@click.option(
    "--ne-supp",
    type=_cli_core.POSITIVE_INT,
    default=200,
    show_default=True,
    help="With --export, supplementary electrons per polar-tilt spectrum.",
)
def command(watch, edit, acp, tunnel, export_, outdir, ne, ne_brem, ne_supp):
    return _cli_core.invoke_legacy(
        _cli,
        watch=watch,
        edit=edit,
        acp=acp,
        tunnel=tunnel,
        export=export_,
        outdir=outdir,
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
    )


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="cxr-check")


if __name__ == "__main__":
    raise SystemExit(main())
