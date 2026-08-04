"""Export the analysis app via ``cxr app analysis export`` and ``marimo export``.

The Jupyter ``analysis.ipynb`` this used to render (via nbconvert webpdf) was
replaced by the marimo app ``notebooks/analysis_app.py``; ``marimo export html``
executes the app headlessly and writes a static HTML snapshot -- the shareable
artifact that replaces the old PDF. The default output name carries today's
date so successive exports are self-describing; pass an explicit stem to
override: ``cxr app analysis export my_custom_name``.

Run from the repo root (it reads ``notebooks/analysis_app.py`` and writes into
``results/``). The app's material dropdown defaults to hopg; the export renders
whatever the app computes headlessly.
"""

import datetime
import subprocess
import sys

from .cli import _core as _cli_core

NOTEBOOK = "notebooks/analysis_app.py"


def _default_stem():
    return f"cxr_analysis_{datetime.date.today():%Y-%m-%d}"


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


def main(argv=None):
    from .cli.commands.export import command

    return _cli_core.run(command, argv, prog_name="cxr-export")


def __getattr__(name: str):
    if name == "command":
        from .cli.commands.export import command

        return command
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
