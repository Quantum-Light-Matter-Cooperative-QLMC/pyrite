"""Guard: remote tests never reach a real box.

Every remote test stubs its own transport, but a missed stub is invisible on a
developer machine whose ssh to the lab box works -- it opens a real connection,
passes, and fails only in CI with ``ssh command failed (exit 255)``. That is how
the ``_materials_needing_pull`` call added in #119 shipped with
``tests/remote/test_click.py`` unstubbed. Fail loudly and locally instead.
"""

import contextlib
import shutil
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
    """Sync tests must not depend on the developer's real table or dataset store."""
    monkeypatch.setattr(
        transport, "_real_local_xsgen_tables", transport._local_xsgen_tables, raising=False
    )
    monkeypatch.setattr(transport, "_local_xsgen_tables", lambda: {})
    monkeypatch.setattr(transport, "_real_local_datasets", transport._local_datasets, raising=False)
    monkeypatch.setattr(transport, "_local_datasets", lambda: {})
    monkeypatch.setattr(transport, "_real_local_sdbase", transport._local_sdbase, raising=False)
    monkeypatch.setattr(transport, "_local_sdbase", lambda: {})


@pytest.fixture(autouse=True)
def _no_remote_sync_lock(monkeypatch):
    """Sync tests stub the box; the lock's own tests exercise the real context."""
    monkeypatch.setattr(
        transport, "_real_remote_sync_lock", transport._remote_sync_lock, raising=False
    )
    monkeypatch.setattr(transport, "_remote_sync_lock", contextlib.nullcontext)


@pytest.fixture
def empty_user_catalog(monkeypatch, tmp_path_factory):
    """An empty user layer, for sync tests that pin the exact shipped payload."""
    layer = tmp_path_factory.mktemp("empty-user-catalog")
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(layer))
    return layer


@pytest.fixture
def multi_detector_standard(tmp_path, monkeypatch):
    """Bundled-catalog copy whose ``standard`` profile also declares a physical
    detector, so it resolves the two detector IDs ``physical`` and ``default``."""
    from pyrite import DATA_DIR, materials
    from pyrite.campaign import config as campaign_config
    from pyrite.materials import load_material_catalog

    root = tmp_path / "catalog"
    shutil.copytree(DATA_DIR / "catalog", root)
    profile = root / "profiles" / "standard.toml"
    with profile.open("a") as handle:
        handle.write("\n[physical_detector]\ndistance_mm = 300.0\nshape = [2, 3]\n")
        handle.write("pitch_mm = [0.1, 0.2]\n")
    catalog = load_material_catalog(root)
    monkeypatch.setenv("PYRITE_CATALOG", str(root))
    monkeypatch.setattr(materials, "CATALOG", catalog)
    monkeypatch.setattr(campaign_config, "CATALOG", catalog)
    assert len(catalog.profile_detector_set("standard")) == 2
    return catalog
