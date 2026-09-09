"""Click wiring and launcher for ``pyrite app validation``.

Starts the marimo validation app (``src/pyrite/apps/validation_app.py``), or
renders its figures in batch from cache.

Unlike ``pyrite app analysis``, this command takes no material argument -- the
validation app reproduces fixed literature figures (e.g. Zhai et al.) rather
than sweeping a chosen material, so there's no initial-material selection to
resolve or persist.

    pyrite app validation             # `marimo run` the validation app
    pyrite app validation --watch     # add marimo's --watch
    pyrite app validation --edit      # `marimo edit` instead of `marimo run`
    pyrite app validation export      # skip marimo; render the full Zhai figure
                                    # set from checkpoints/zhai_reproduction/
                                    # (see `pyrite run --preset zhai --remote`) to figures/
"""

import importlib
import os
import subprocess
import sys

import click

from ..._acp import running_acp
from ...paths import app_dir
from .. import _core as _cli_core

NOTEBOOK = str(app_dir() / "validation_app.py")
TUNNEL_PORT = 2718


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
    af = importlib.import_module("pyrite.validation.anchor_figures")

    try:
        written = af.export_all_figures(
            outdir,
            ne=ne,
            ne_brem=ne_brem,
            ne_supp=ne_supp,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
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
        "Launch the interactive validation application, or export its cached figures.\n\n"
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
    return _cli_core.run(command, argv, prog_name="pyrite-check")


if __name__ == "__main__":
    raise SystemExit(main())
