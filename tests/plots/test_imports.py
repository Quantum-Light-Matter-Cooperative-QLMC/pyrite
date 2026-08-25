# tests/plots/test_imports.py

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "pyrite.plots.altair.trajectories",
        "pyrite.plots.plotly.trajectories",
    ],
)
def test_interactive_backend_import_does_not_pull_in_matplotlib(module: str) -> None:
    """Rendering an interactive Altair or Plotly chart must not import matplotlib.

    ``pyrite.plots.altair`` / ``pyrite.plots.plotly`` reuse renderer-neutral
    helpers from :mod:`pyrite.plots._common` / :mod:`pyrite.plots._frames` --
    NOT from :mod:`pyrite.plots.mpl`, and neither neutral module (nor importing
    the ``pyrite.plots`` package itself, which every dotted submodule import
    executes first) imports matplotlib. Runs in a fresh interpreter subprocess
    (not in-process), since an earlier test in the same session may already
    have imported matplotlib for an unrelated reason and mask a regression
    here."""
    script = textwrap.dedent(
        f"""
        import sys

        import {module}

        loaded = sorted(m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib."))
        assert not loaded, f"importing {module!r} pulled matplotlib into sys.modules: {{loaded}}"
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"importing {module} unexpectedly pulled in matplotlib:\n{result.stdout}\n{result.stderr}"
    )
