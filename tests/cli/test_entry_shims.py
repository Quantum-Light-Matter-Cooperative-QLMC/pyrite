"""Smoke coverage for Python module entry shims."""

from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType

import pytest


def test_energy_grid_compatibility_module_is_the_command_implementation(monkeypatch):
    monkeypatch.delitem(sys.modules, "cxr_mc.cli.energy_grid", raising=False)

    alias = importlib.import_module("cxr_mc.cli.energy_grid")
    implementation = importlib.import_module("cxr_mc.cli.commands.energy_grid")

    assert alias is implementation


def test_cli_module_entry_point_delegates_to_main(monkeypatch):
    from cxr_mc import cli

    monkeypatch.setattr(cli, "main", lambda: 23)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("cxr_mc.cli.__main__", run_name="__main__")

    assert error.value.code == 23


def test_scan_module_entry_point_delegates_to_scan_main(monkeypatch):
    scan = ModuleType("cxr_mc.scan")
    scan.main = lambda: 23
    monkeypatch.delitem(sys.modules, "cxr_mc._entry.scan", raising=False)
    monkeypatch.setitem(sys.modules, "cxr_mc.scan", scan)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("cxr_mc._entry.scan", run_name="__main__")

    assert error.value.code == 23
