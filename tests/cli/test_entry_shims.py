"""Smoke coverage for Python module entry shims."""

from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType

import pytest


def test_energy_grid_command_lives_in_cli_and_not_in_the_domain_package():
    """The command group moved out of ``energy_grid`` to break its cycle with ``cli``.

    ``energy_grid`` must not grow a Click surface again: it is imported on the
    Monte Carlo hot path, and re-exporting the group is what put the two
    packages in an import cycle.
    """
    import click

    from pyrite.cli import _COMMANDS

    command = importlib.import_module("pyrite.cli.commands.energy_grid").command
    domain = importlib.import_module("pyrite.energy_grid")

    assert isinstance(command, click.Group)
    assert _COMMANDS["energy-grid"] == "pyrite.cli.commands.energy_grid.command"
    assert not hasattr(domain, "command")
    assert not any(isinstance(value, click.Command) for value in vars(domain).values())


def test_cli_module_entry_point_delegates_to_main(monkeypatch):
    from pyrite import cli

    monkeypatch.setattr(cli, "main", lambda: 23)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("pyrite.cli.__main__", run_name="__main__")

    assert error.value.code == 23


def test_scan_module_entry_point_delegates_to_scan_main(monkeypatch):
    scan = ModuleType("pyrite.runs.scan")
    scan.main = lambda: 23
    monkeypatch.delitem(sys.modules, "pyrite._entry.scan", raising=False)
    monkeypatch.setitem(sys.modules, "pyrite.runs.scan", scan)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("pyrite._entry.scan", run_name="__main__")

    assert error.value.code == 23
