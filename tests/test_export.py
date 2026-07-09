"""``cxr export`` targets the marimo analysis app, not the retired Jupyter
notebook (analysis.ipynb was replaced by notebooks/analysis_app.py in the
marimo migration -- exporting must follow, via ``marimo export html``)."""

import datetime
import sys
from pathlib import Path

from cxr_mc import export


def test_export_targets_existing_notebook():
    # the export source must exist in the repo (run from the repo root,
    # which is the documented contract for `cxr export`)
    assert Path(export.NOTEBOOK).exists()


def test_command_invokes_marimo_html_export():
    cmd = export._command("mystem")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "marimo", "export"]
    assert "html" in cmd
    assert export.NOTEBOOK in cmd
    assert "results/mystem.html" in cmd


def test_default_stem_is_dated():
    assert export._default_stem() == f"cxr_analysis_{datetime.date.today():%Y-%m-%d}"
