"""Smoke coverage for Python module entry shims."""

import importlib
import runpy
import sys

import pytest


def test_energy_grid_command_lives_in_cli_and_not_in_the_domain_package():
    """The command group moved out of ``energy_grid`` to break its cycle with ``cli``.

    ``energy_grid`` must not grow a Click surface again: it is imported on the
    Monte Carlo hot path, and re-exporting the group is what put the two
    packages in an import cycle.
    """
    import click

    from pyrite.cli import _COMMANDS

    module = importlib.import_module("pyrite.cli.commands.energy_grid")
    domain = importlib.import_module("pyrite.energy_grid")

    assert isinstance(module.add_command, click.Command)
    assert "energy-grid" not in _COMMANDS
    assert not hasattr(domain, "command")
    assert not any(isinstance(value, click.Command) for value in vars(domain).values())


def test_cli_module_entry_point_delegates_to_main(monkeypatch):
    from pyrite import cli

    monkeypatch.setattr(cli, "main", lambda: 23)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("pyrite.cli.__main__", run_name="__main__")

    assert error.value.code == 23


def test_scan_module_entry_point_runs_the_scan_command(monkeypatch):
    """The entry shim binds the Click command to the shared runner.

    ``runs.scan`` owns no Click surface: the shim is what joins the command in
    ``cli.commands.scan`` to ``console.output.run`` (issue #64, finding 2), so
    the remote box's ``python -m pyrite._entry.scan`` keeps its exit contract
    without ``runs`` importing ``cli``.
    """
    from pyrite.cli.commands import scan as scan_cli
    from pyrite.console import output

    seen = {}

    def fake_run(command, argv=None, *, prog_name):
        seen["command"] = command
        seen["prog_name"] = prog_name
        return 23

    monkeypatch.setattr(output, "run", fake_run)
    monkeypatch.delitem(sys.modules, "pyrite._entry.scan", raising=False)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("pyrite._entry.scan", run_name="__main__")

    assert error.value.code == 23
    # The full surface: remote perf jobs pass the hidden performance flags.
    assert seen["command"] is scan_cli._command
    assert seen["prog_name"] == "pyrite run"
