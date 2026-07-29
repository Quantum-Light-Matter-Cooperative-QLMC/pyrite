"""Regression tests for catalog failures during ordinary CLI startup."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_missing_crystals_dependency_is_one_actionable_cli_error() -> None:
    script = r"""
import builtins

real_import = builtins.__import__

def import_without_crystals(name, *args, **kwargs):
    if name == "crystals":
        raise ModuleNotFoundError("No module named 'crystals'", name="crystals")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_crystals
from cxr_mc import cli
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
    assert "required dependency 'crystals' is not installed" in result.stderr
    assert "uv sync" in result.stderr
    assert result.stderr.count("required dependency 'crystals'") == 1
    assert "materials." not in result.stderr
    assert "Traceback" not in result.stderr


def test_scan_entry_shim_reports_missing_crystals_without_traceback() -> None:
    script = r"""
import builtins
import runpy
import sys

real_import = builtins.__import__

def import_without_crystals(name, *args, **kwargs):
    if name == "crystals":
        raise ModuleNotFoundError("No module named 'crystals'", name="crystals")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_crystals
sys.argv = ["cxr_mc._entry.scan", "standard", "-m", "hopg", "--quick"]
runpy.run_module("cxr_mc._entry.scan", run_name="__main__")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "required dependency 'crystals' is not installed" in result.stderr
    assert "uv sync" in result.stderr
    assert result.stderr.count("required dependency 'crystals'") == 1
    assert "materials." not in result.stderr
    assert "Traceback" not in result.stderr


def test_transitive_module_not_found_is_not_reported_as_bad_catalog(monkeypatch) -> None:
    from cxr_mc.materials import catalog

    def missing_transitive_dependency(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'spglib'", name="spglib")

    monkeypatch.setattr(catalog, "load_crystal_from_cif", missing_transitive_dependency)
    catalog.load_material_catalog.cache_clear()

    with pytest.raises(ModuleNotFoundError, match="spglib"):
        catalog.load_material_catalog()
