"""``cxr check`` -- launch the marimo validation app (``notebooks/validation_app.py``).

Unlike ``cxr analyze``, this command takes no material argument -- the
validation app reproduces fixed literature figures (e.g. Zhai et al.) rather
than sweeping a chosen material, so there's no initial-material selection to
resolve or persist.

    cxr check                      # `marimo run` the validation app
    cxr check --watch              # add marimo's --watch
    cxr check --edit               # `marimo edit` instead of `marimo run`
"""

import argparse
import os
import subprocess
import sys

NOTEBOOK = "notebooks/validation_app.py"


def _command(*, edit=False, watch=False):
    """The marimo argv for one launch (module-run through the current
    interpreter so the venv's marimo is the one that runs). Marimo's own flags
    go before the notebook path; app args go after ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        NOTEBOOK,
        "--",
    ]


def _launch(*, edit=False, watch=False):
    cmd = _command(edit=edit, watch=watch)
    env = {**os.environ}
    subprocess.run(cmd, check=True, env=env)


def _cli(args):

    _launch(edit=args.edit, watch=args.watch)


def add_subparser(sub):
    """Register the ``check`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser("check", help=f"launch {NOTEBOOK} (marimo run/edit)")
    ap.add_argument("--watch", action="store_true", help="pass marimo's --watch")
    ap.add_argument("--edit", action="store_true", help="use `marimo edit` instead of `marimo run`")
    ap.set_defaults(func=_cli)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-check", description="launch the validation app")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
