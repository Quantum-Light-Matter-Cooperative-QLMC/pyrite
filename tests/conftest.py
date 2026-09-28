"""Import pyarrow FIRST: on Windows, importing pyrite (numpy/scipy DLLs) before
pyarrow leaves pyarrow's arrow DLLs binding against the wrong runtime, and the
altair/pandas tests then hard-crash (native fault in DataFrame construction).
Loading pyarrow here, before any test module imports pyrite, fixes the order.
Harmless when pyarrow is absent (the altair tests skip without it)."""

import os
import sys

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

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _pin_remote_paths(monkeypatch):
    """Give remote tests a fixed host and fixed absolute paths.

    There is no built-in remote host, so an unpinned test would resolve whatever
    ``remote.target`` the developer's own config store holds.

    The real defaults are ``~/pyrite`` and ``~/.local/bin/uv``, which resolve
    against the remote login home over ssh; tests must never do that. Tests that
    exercise the ``~`` expansion or the real defaults set their own values.
    """
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
