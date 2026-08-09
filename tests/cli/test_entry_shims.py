"""Smoke coverage for Python module entry shims."""

from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType

import pytest


def test_energy_grid_command_adapter_exports_domain_command():
    adapter = importlib.import_module("cxr_mc.cli.commands.energy_grid")
    implementation = importlib.import_module("cxr_mc.energy_grid")

    assert adapter.command is implementation.command


def test_cli_module_entry_point_delegates_to_main(monkeypatch):
    from cxr_mc import cli

    monkeypatch.setattr(cli, "main", lambda: 23)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("cxr_mc.cli.__main__", run_name="__main__")

    assert error.value.code == 23


def test_scan_module_entry_point_delegates_to_scan_main(monkeypatch):
    scan = ModuleType("cxr_mc.runs.scan")
    scan.main = lambda: 23
    monkeypatch.delitem(sys.modules, "cxr_mc._entry.scan", raising=False)
    monkeypatch.setitem(sys.modules, "cxr_mc.runs.scan", scan)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("cxr_mc._entry.scan", run_name="__main__")

    assert error.value.code == 23
