"""Import pyarrow FIRST: on Windows, importing pyrite (numpy/scipy DLLs) before
pyarrow leaves pyarrow's arrow DLLs binding against the wrong runtime, and the
altair/pandas tests then hard-crash (native fault in DataFrame construction).
Loading pyarrow here, before any test module imports pyrite, fixes the order.
Harmless when pyarrow is absent (the altair tests skip without it)."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Pin the test session to the CPU/NumPy backend (see pyproject.toml: "Tests
# stay CPU-only/fast"). Much of the suite asserts bit-exact fp64 numerics and
# CPU resource-policy behavior that accelerator fp32 paths (CUDA/ROCm/SYCL)
# cannot reproduce, so an ambient PYRITE_MC_BACKEND from a GPU-equipped dev
# box (e.g. sycl on Intel) must not leak into the session. Set
# PYRITE_TEST_BACKEND to run the suite against another backend on purpose;
# hardware-gated tests scrub this pin from subprocesses they spawn.
_TEST_BACKEND = os.environ.get("PYRITE_TEST_BACKEND") or "cpu"
os.environ["PYRITE_MC_BACKEND"] = _TEST_BACKEND
# A developer's PYRITE_HOME names their user workspace; tests must neither
# depend on it nor write generated tables into it. Drop it before any module
# resolves import-time workspace defaults; tests that exercise it set their own.
os.environ.pop("PYRITE_HOME", None)
# Likewise a developer's PYRITE_CATALOG names their real catalog. CLI editors
# honor the test-patched path only while the bundled catalog is selected, so an
# ambient value would let profile/detector tests rewrite the real catalog.
os.environ.pop("PYRITE_CATALOG", None)
# The user catalog layer (``catalog.user``) defaults to the developer's config
# directory and is read over the bundled catalog. Point it at the tracked
# fixture layer -- the research campaign profiles the suite pins -- before any
# module loads the catalog singleton; ``_isolate_user_catalog`` then gives each
# test a writable copy.
USER_CATALOG_FIXTURE = Path(__file__).parent / "data" / "user_catalog"
os.environ["PYRITE_USER_CATALOG"] = str(USER_CATALOG_FIXTURE)
# PyRITE ships no NIST SRD 64 tables (#263); a real user configures
# ``mott.tables_dir``. Point ``elastic_model="mott"`` at the synthetic
# SRD 64-format fixture, whose numbers reproduce the analytic Joy/Bishop
# screening (``tests/helpers/mott_synthetic.py``). Set at import so module- and
# session-scoped fixtures see it too; tests of the missing-table error unset it.
MOTT_SYNTHETIC_DIR = Path(__file__).parent / "data" / "mott_srd64_synthetic"
os.environ["PYRITE_MOTT_TABLES_DIR"] = str(MOTT_SYNTHETIC_DIR)

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

# Likewise the developer's config store (``workspace.root``, ``catalog.path``,
# ...): point it at an absent file before any module resolves import-time
# defaults. ``_isolate_config_store`` then gives each test its own empty store.
from pyrite.console import config as _config  # noqa: E402

_config.CONFIG_PATH = Path(tempfile.gettempdir()) / "pyrite-tests-absent-store" / "config.toml"

# Opt-in ``slow`` tier (#381): pytest finds the hooks by name in this namespace.
from tests.helpers.slow_marker import (  # noqa: E402, F401
    pytest_collection_modifyitems,
    pytest_terminal_summary,
)


@pytest.fixture(autouse=True)
def _isolate_config_store(monkeypatch, tmp_path):
    """Give every test an empty config store.

    The developer's store may set ``workspace.root`` (redirecting generated
    xsgen tables into a checkout) or other keys that CI never sees. Tests that
    exercise the store point ``CONFIG_PATH`` at their own file.
    """
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "pyrite-config" / "config.toml")


@pytest.fixture(autouse=True)
def _isolate_user_catalog(monkeypatch, tmp_path_factory):
    """Give every test a writable copy of the fixture user catalog layer."""
    import shutil

    # Not under tmp_path: many tests build a catalog directly in it.
    layer = tmp_path_factory.mktemp("user-catalog")
    shutil.copytree(USER_CATALOG_FIXTURE, layer, dirs_exist_ok=True)
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(layer))
    return layer


@pytest.fixture(autouse=True)
def _isolate_output_workspace(monkeypatch, tmp_path):
    """Resolve default outputs (``pyrite-output/``) under each test's tmp_path.

    Without an explicit workspace the output root follows the working
    directory, which under pytest is the checkout itself. Tests that exercise
    cwd discovery patch ``outputs._cwd`` themselves.
    """
    from pyrite.console import outputs

    monkeypatch.setattr(outputs, "_cwd", lambda: tmp_path.resolve())
    # The root ``pyrite`` callback opts its process into legacy moves; keep
    # one CLI test from moving another test's fixture ``checkpoints/``.
    monkeypatch.setattr(outputs, "_migration_enabled", False)
    monkeypatch.setattr(outputs, "_migrated", set())


@pytest.fixture(autouse=True)
def _pin_remote_paths(monkeypatch):
    """Give remote tests a fixed host and fixed absolute paths.

    There is no built-in remote host, so an unpinned test would resolve whatever
    ``remote.target`` the developer's own config store holds.

    The real defaults are ``~/pyrite`` and ``~/.local/bin/uv``, which resolve
    against the remote login home over ssh; tests must never do that. Tests that
    exercise the ``~`` expansion or the real defaults set their own values.
    """
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")
    config = sys.modules.get("pyrite.remote.config")
    if config is None:
        return
    monkeypatch.setattr(config, "HOST", "remote-host")
    monkeypatch.setattr(config, "REMOTE_DIR", "/path/to/pyrite")
    monkeypatch.setattr(config, "REMOTE_UV", "uv")
    # The SLURM target profile likewise resolves through the developer's store.
    monkeypatch.setattr(config, "REMOTE_GPU_VENDOR", "nvidia")
    monkeypatch.setattr(config, "SLURM_PARTITION", "gpu")
    monkeypatch.setattr(config, "SLURM_NODELIST", "any")
    monkeypatch.setattr(config, "SLURM_GRES", "gpu:1")
    package = sys.modules.get("pyrite.remote")
    if package is not None:
        monkeypatch.setattr(package, "HOST", "remote-host")
        monkeypatch.setattr(package, "REMOTE_DIR", "/path/to/pyrite")
        monkeypatch.setattr(package, "REMOTE_UV", "uv")


@pytest.fixture(scope="session")
def _lab_catalog_dir(tmp_path_factory):
    """Bundled catalog plus the campaign profiles that live in the lab catalog.

    The bundled catalog keeps only small examples; profiles and beams that moved
    to a lab catalog are stored under ``tests/data/lab_overlay`` and layered onto
    a copy of the bundled catalog so their digests stay pinned bit-for-bit.
    """
    import shutil
    from pathlib import Path

    from pyrite import DATA_DIR

    root = tmp_path_factory.mktemp("lab_catalog")
    shutil.copytree(DATA_DIR / "catalog", root, dirs_exist_ok=True)
    shutil.copytree(USER_CATALOG_FIXTURE, root, dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parent / "data" / "lab_overlay", root, dirs_exist_ok=True)
    return root


@pytest.fixture
def lab_catalog(_lab_catalog_dir, monkeypatch):
    """Select the bundled-plus-lab-profiles catalog for one test."""
    from pyrite import materials
    from pyrite.materials import load_material_catalog

    catalog = load_material_catalog(_lab_catalog_dir)
    monkeypatch.setenv("PYRITE_CATALOG", str(_lab_catalog_dir))
    monkeypatch.setattr(materials, "CATALOG", catalog)
    return catalog
