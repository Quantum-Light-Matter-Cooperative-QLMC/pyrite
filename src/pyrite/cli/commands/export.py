"""Click wiring and driver for static analysis-app export.

Runs ``pyrite app analysis export`` via ``marimo export``.

The Jupyter ``analysis.ipynb`` this used to render (via nbconvert webpdf) was
replaced by the marimo app ``src/pyrite/apps/analysis_app.py``; ``marimo export html``
executes the app headlessly and writes a static HTML snapshot -- the shareable
artifact that replaces the old PDF. The default output name carries today's
date so successive exports are self-describing; pass an explicit stem to
override: ``pyrite app analysis export my_custom_name``.

Run from the repo root (it reads ``src/pyrite/apps/analysis_app.py`` and writes into
``results/``). The app's material dropdown defaults to hopg; the export renders
whatever the app computes headlessly.
"""

import datetime
import subprocess
import sys

import click

from ...console import output as _cli_core
from ...paths import app_dir

NOTEBOOK = str(app_dir() / "analysis_app.py")


def _default_stem():
    return f"pyrite_analysis_{datetime.date.today():%Y-%m-%d}"


def _command(stem):
    """The ``marimo export html`` argv for one output stem (module-run through
    the current interpreter so the venv's marimo is the one that runs)."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        NOTEBOOK,
        "-o",
        f"results/{stem}.html",
    ]


def _export(stem=None):
    stem = stem or _default_stem()
    print(f"exporting {NOTEBOOK} -> results/{stem}.html")
    subprocess.run(_command(stem), check=True)


@click.command(
    "export",
    help=(
        "Render the analysis application to static HTML.\n\n"
        "Writes results/<stem>.html; STEM defaults to analysis."
    ),
)
@click.argument("stem", required=False)
def command(stem):
    _export(stem)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-export")


if __name__ == "__main__":
    raise SystemExit(main())
