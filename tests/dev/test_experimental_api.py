"""Issue #706: exact, lazy, identity-preserving experimental interfaces."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from pyrite.devtools import api_packages

ROOT = Path(__file__).resolve().parents[2]
FACADES = ("radiation", "observations", "quantum")


def _inventory():
    return tomllib.loads((ROOT / "api-surface.toml").read_text(encoding="utf-8"))


def test_experimental_facades_have_exact_reviewed_inventory():
    registered = _inventory()["package_exports"]
    for domain in FACADES:
        module = importlib.import_module(f"pyrite.experimental.{domain}")
        names = registered[f"pyrite.experimental.{domain}"]
        assert set(module.__all__) == set(names)
        assert len(module.__all__) == len(set(module.__all__))
        assert all(row["status"] == "experimental" for row in names.values())
        assert set(module.__all__) <= set(dir(module))
        with pytest.raises(AttributeError):
            _ = module.UnreviewedPrivateKernel


@pytest.mark.parametrize("domain", FACADES)
def test_experimental_exports_preserve_implementation_identity(domain: str):
    namespace = importlib.import_module(f"pyrite.experimental.{domain}")
    for name, row in _inventory()["package_exports"][f"pyrite.experimental.{domain}"].items():
        owner, attribute = row["target"].split(":", 1)
        assert getattr(namespace, name) is getattr(importlib.import_module(owner), attribute)


def test_experimental_facade_imports_are_lightweight():
    code = """import sys
import importlib.abc
import warnings
blocked = {"numpy", "scipy", "cupy", "matplotlib", "dask", "abtem", "torch"}
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("attempted heavy import: " + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
warnings.simplefilter("error")
import pyrite
before = set(sys.modules)
import pyrite.experimental
from pyrite.experimental import radiation, observations, quantum
assert set(pyrite.experimental.__all__) == {"radiation", "observations", "quantum"}
new = set(sys.modules) - before
for package in blocked:
    assert not any(name.split(".")[0] == package for name in new), package
for module in ("pyrite.directional", "pyrite.photon_events", "pyrite.montecarlo.channeling.axial_mcwf"):
    assert module not in new, module
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr


def test_experimental_manifest_rejects_stable_entries(tmp_path: Path):
    manifest = tmp_path / "api-surface.toml"
    manifest.write_text("""[package_exports."pyrite.experimental.radiation"]
Dummy = {target="pyrite.directional:directional_source", status="advanced-stable"}
""")
    _, _, errors = api_packages.read_package_inventory(manifest)
    assert any("experimental facade requires experimental status" in e for e in errors)


def test_uninventoried_experimental_modules_are_rejected(tmp_path: Path):
    source = tmp_path / "src" / "pyrite" / "experimental"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# namespace\n")
    (source / "surprise.py").write_text("__all__ = ['Leak']\n")
    (tmp_path / "api-surface.toml").write_text("schema_version = 1\n")
    errors = api_packages.check_package_contracts(tmp_path, {})
    assert any("experimental facade missing API inventory" in e for e in errors)


def test_unreviewed_experimental_export_is_rejected(tmp_path: Path):
    source = tmp_path / "src" / "pyrite" / "experimental"
    source.mkdir(parents=True)
    (source / "radiation.py").write_text("__all__ = ['Approved', 'Leak']\n")
    (tmp_path / "api-surface.toml").write_text(
        '[package_exports."pyrite.experimental.radiation"]\n'
        'Approved = {target="pyrite.directional:directional_source", status="experimental"}\n'
    )
    errors = api_packages.check_package_contracts(tmp_path, {})
    assert any("unreviewed experimental exports: ['Leak']" in e for e in errors)
