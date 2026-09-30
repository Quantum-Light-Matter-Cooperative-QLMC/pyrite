"""Guard: remote tests never reach a real box.

Every remote test stubs its own transport, but a missed stub is invisible on a
developer machine whose ssh to the lab box works -- it opens a real connection,
passes, and fails only in CI with ``ssh command failed (exit 255)``. That is how
the ``_materials_needing_pull`` call added in #119 shipped with
``tests/remote/test_click.py`` unstubbed. Fail loudly and locally instead.
"""

import subprocess

import pytest

from pyrite.remote import transport

#: Commands that would leave the machine. Everything else (``bash -n`` script
#: syntax checks, local tar) passes straight through to the real runner.
_NETWORK_COMMANDS = frozenset({"ssh", "scp", "rsync"})


@pytest.fixture(autouse=True)
def _forbid_real_ssh(monkeypatch):
    real_run = subprocess.run

    def guarded_run(cmd, *args, **kwargs):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        head = str(argv[0]) if argv else ""
        if head in _NETWORK_COMMANDS:
            raise AssertionError(
                f"remote test reached real {head}; stub the transport call instead: {argv}"
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(transport.subprocess, "run", guarded_run)


@pytest.fixture(autouse=True)
def _no_local_xsgen_tables(monkeypatch):
    """Sync tests must not depend on the developer's real table store."""
    monkeypatch.setattr(
        transport, "_real_local_xsgen_tables", transport._local_xsgen_tables, raising=False
    )
    monkeypatch.setattr(transport, "_local_xsgen_tables", lambda: {})
