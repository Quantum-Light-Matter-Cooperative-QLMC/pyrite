"""Click wiring and driver for static analysis-app export.

Runs ``pyrite app analysis export`` via ``marimo export``.

The Jupyter ``analysis.ipynb`` this used to render (via nbconvert webpdf) was
replaced by the marimo app ``src/pyrite/apps/analysis_app.py``; ``marimo export html``
executes the app headlessly and writes a static HTML snapshot -- the shareable
artifact that replaces the old PDF. The default output name carries today's
date so successive exports are self-describing; pass an explicit stem to
override: ``pyrite app analysis export my_custom_name``.

Exports use the packaged application and write into
``pyrite-output/results/`` in the selected workspace. The app's material
dropdown defaults to hopg; the export renders whatever it computes headlessly.
"""

import datetime
import subprocess
import sys

import click

from ...console import output as _cli_core
from ...console.outputs import output_dir
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
        str(output_dir("results") / f"{stem}.html"),
    ]


def _export(stem=None):
    stem = stem or _default_stem()
    output = output_dir("results") / f"{stem}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting {NOTEBOOK} -> {output}")
    subprocess.run(_command(stem), check=True)


@click.command(
    "export",
    help=(
        "Render the analysis application to static HTML.\n\n"
        "Writes pyrite-output/results/<stem>.html; STEM defaults to a dated analysis name."
    ),
)
@click.argument("stem", required=False)
def command(stem):
    _export(stem)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-export")


if __name__ == "__main__":
    raise SystemExit(main())
