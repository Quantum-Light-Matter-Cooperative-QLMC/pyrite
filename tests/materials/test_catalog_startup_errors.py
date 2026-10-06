"""Regression tests for catalog failures during ordinary CLI startup."""

import subprocess
import sys

import pytest


def test_missing_gemmi_dependency_is_one_actionable_cli_error() -> None:
    script = r"""
import builtins

real_import = builtins.__import__

def import_without_gemmi(name, *args, **kwargs):
    if name == "gemmi":
        raise ModuleNotFoundError("No module named 'gemmi'", name="gemmi")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_gemmi
from pyrite import cli
cli.main(["run", "standard", "-m", "hopg", "--quick"])
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "required dependency 'gemmi' is not installed" in result.stderr
    assert "uv sync" in result.stderr
    assert result.stderr.count("required dependency 'gemmi'") == 1
    assert "materials." not in result.stderr
    assert "Traceback" not in result.stderr


def test_scan_entry_shim_reports_missing_gemmi_without_traceback() -> None:
    script = r"""
import builtins
import runpy
import sys

real_import = builtins.__import__

def import_without_gemmi(name, *args, **kwargs):
    if name == "gemmi":
        raise ModuleNotFoundError("No module named 'gemmi'", name="gemmi")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_gemmi
sys.argv = ["pyrite._entry.scan", "standard", "-m", "hopg", "--quick"]
runpy.run_module("pyrite._entry.scan", run_name="__main__")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "required dependency 'gemmi' is not installed" in result.stderr
    assert "uv sync" in result.stderr
    assert result.stderr.count("required dependency 'gemmi'") == 1
    assert "materials." not in result.stderr
    assert "Traceback" not in result.stderr


def test_transitive_module_not_found_is_not_reported_as_bad_catalog(monkeypatch) -> None:
    from pyrite.materials import _parse, catalog

    def missing_transitive_dependency(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'spglib'", name="spglib")

    monkeypatch.setattr(_parse, "load_crystal_from_cif", missing_transitive_dependency)
    catalog.load_material_catalog.cache_clear()

    with pytest.raises(ModuleNotFoundError, match="spglib"):
        catalog.load_material_catalog()
