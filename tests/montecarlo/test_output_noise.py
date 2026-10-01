"""
P3 #3 -- CLI/remote output noise.

Regression coverage for silencing:
  - the GPU/CPU backend-probe banner (pyrite._backend), previously
    `print()`ed at import time on every `pyrite` invocation.

It is now a module-level `logging.getLogger(__name__).debug(...)` call, silent
unless the caller opts in (PYRITE_MC_DEBUG=1 or their own logging config), so a
plain import / CLI run stays quiet. (The former "no Mott transport table" notice
is gone: a missing SRD 64 table is now an error, #263.)
"""

import subprocess
import sys


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
