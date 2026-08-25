"""
P3 #3 -- CLI/remote output noise.

Regression coverage for silencing:
  - the GPU/CPU backend-probe banner (pyrite._backend), previously
    `print()`ed at import time on every `pyrite` invocation.
  - the "no Mott transport table for 'X'" notice (montecarlo.transport),
    previously `print()`ed the first time an element without a NIST table
    was requested.

Both are now module-level `logging.getLogger(__name__).debug(...)` calls,
silent unless the caller opts in (PYRITE_MC_DEBUG=1 or their own logging
config), so a plain import / CLI run stays quiet.
"""

import logging
import subprocess
import sys

import numpy as np

from pyrite.montecarlo import transport


def test_cli_help_has_no_gpu_banner():
    """A plain `pyrite --help` must not print the GPU/CPU backend-probe banner."""
    result = subprocess.run(
        [sys.executable, "-m", "pyrite.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    combined = result.stdout + result.stderr
    assert "No GPU found" not in combined
    assert "Using GPU" not in combined


def test_mott_missing_table_logs_debug_once(caplog):
    """An element with no NIST Mott table logs once at DEBUG, then is cached
    in _NO_MOTT and stays silent on repeat calls (no re-stat, no re-log)."""
    element = "__not_a_real_element__"
    transport._NO_MOTT.discard(element)

    rng = np.random.default_rng(0)
    E_keV = np.array([10.0, 20.0])

    with caplog.at_level(logging.DEBUG, logger="pyrite.montecarlo.transport"):
        transport._sample_cos_theta(6, E_keV, rng, "mott", element)
        transport._sample_cos_theta(6, E_keV, rng, "mott", element)

    matches = [r for r in caplog.records if "no Mott transport table" in r.getMessage()]
    assert len(matches) == 1
    assert matches[0].levelno == logging.DEBUG
    assert element in transport._NO_MOTT
