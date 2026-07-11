"""``cxr check`` -- launch the marimo validation app (``notebooks/validation_app.py``),
or render its figures in batch from cache.

Unlike ``cxr analyze``, this command takes no material argument -- the
validation app reproduces fixed literature figures (e.g. Zhai et al.) rather
than sweeping a chosen material, so there's no initial-material selection to
resolve or persist.

    cxr check                      # `marimo run` the validation app
    cxr check --watch              # add marimo's --watch
    cxr check --edit               # `marimo edit` instead of `marimo run`
    cxr check --export             # skip marimo; render the full Zhai figure
                                    # set from checkpoints/zhai_reproduction/
                                    # (see `cxr remote check`) to figures/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

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


def _export(outdir="figures", ne=20_000, ne_brem=200, ne_supp=200):
    checks_dir = Path(__file__).resolve().parents[2] / "checks"
    if str(checks_dir) not in sys.path:
        sys.path.insert(0, str(checks_dir))
    import anchor_figures as af  # type: ignore[reportMissingImports]

    written = af.export_all_figures(outdir, ne=ne, ne_brem=ne_brem, ne_supp=ne_supp)
    for path in written:
        print(f"wrote {path}")


def _cli(args):
    if args.export:
        _export(args.outdir, ne=args.ne, ne_brem=args.ne_brem, ne_supp=args.ne_supp)
        return
    _launch(edit=args.edit, watch=args.watch)


def add_subparser(sub):
    """Register the ``check`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser(
        "check", help=f"launch {NOTEBOOK} (marimo run/edit), or --export its figures"
    )
    ap.add_argument("--watch", action="store_true", help="pass marimo's --watch")
    ap.add_argument("--edit", action="store_true", help="use `marimo edit` instead of `marimo run`")
    ap.add_argument(
        "--export",
        action="store_true",
        help="skip marimo; render the full Zhai figure set from cache to --outdir",
    )
    ap.add_argument("--outdir", default="figures", help="with --export: output directory")
    ap.add_argument("--ne", type=int, default=20_000, help="with --export: Fig.1c line electrons")
    ap.add_argument("--ne-brem", type=int, default=200, help="with --export: Fig.1c brem electrons")
    ap.add_argument(
        "--ne-supp", type=int, default=200, help="with --export: supplementary electrons"
    )
    ap.set_defaults(func=_cli)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-check", description="launch the validation app")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
