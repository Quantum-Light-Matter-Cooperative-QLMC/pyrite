"""Terminal-color contract for human and machine CLI output."""

import json

import pytest
from click.testing import CliRunner

from pyrite import cli


def test_color_always_styles_help_and_human_result():
    runner = CliRunner()

    help_result = runner.invoke(cli.command, ["--color", "always", "--help"])
    check_result = runner.invoke(cli.command, ["--color", "always", "material", "validate"])

    assert help_result.exit_code == 0
    assert "\033[38;2;92;207;230mOptions\033[0m:" in help_result.stdout
    assert "\033[38;2;92;207;230mCommands\033[0m:" in help_result.stdout
    assert check_result.exit_code == 0
    assert "\033[38;2;170;217;76mvalid\033[0m material catalog" in check_result.stdout


def test_color_never_and_no_color_keep_output_plain(monkeypatch):
    runner = CliRunner()

    explicit = runner.invoke(cli.command, ["--color", "never", "--help"], color=True)
    monkeypatch.setenv("NO_COLOR", "1")
    automatic = runner.invoke(cli.command, ["--color", "auto", "--help"], color=True)

    assert explicit.exit_code == 0
    assert automatic.exit_code == 0
    assert "\033[" not in explicit.output
    assert "\033[" not in automatic.output


def test_color_always_styles_usage_error_but_not_json(capsys):
    runner = CliRunner()

    with pytest.raises(SystemExit) as exc:
        cli.main(["--color", "always", "not-a-command"])
    error = capsys.readouterr()
    machine = runner.invoke(cli.command, ["--color", "always", "checkpoint", "list", "-o", "json"])

    assert exc.value.code == 2
    assert "\033[38;2;240;113;120mError:\033[0m" in error.err
    assert machine.exit_code == 0
    assert "\033[" not in machine.stdout
    assert json.loads(machine.stdout)["schema"] == "cxr.archives"
