"""Post-P0 argparse contract frozen for Click migration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cxr_mc import cli
from scripts.freeze_cli_contract import main as freeze_main

CONTRACT = Path(__file__).with_name("data") / "cli_contract.json"


def _help_cases(node):
    yield tuple(node["path"].split()), node["help"]
    for child in node["subcommands"]:
        yield from _help_cases(child)


_FROZEN = json.loads(CONTRACT.read_text(encoding="utf-8"))


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cxr_mc.cli", *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_argparse_contract_snapshot_is_current():
    assert freeze_main(["--check", str(CONTRACT)]) == 0


@pytest.mark.parametrize(
    ("path", "expected"),
    list(_help_cases(_FROZEN["root"])),
    ids=lambda value: "root" if value == () else None,
)
def test_every_help_path_uses_stdout(path, expected, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([*path, "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == expected
    assert captured.err == ""


def test_version_uses_stdout():
    completed = _run("--version")
    assert completed.returncode == 0
    assert completed.stdout.startswith("cxr-mc ")
    assert completed.stdout.endswith("\n")
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("argv", "diagnostic"),
    [
        ((), "the following arguments are required: command"),
        (("not-a-command",), "invalid choice: 'not-a-command'"),
        (("remote",), "the following arguments are required: remote_command"),
        (("line-grid",), "the following arguments are required: lg_command"),
    ],
)
def test_usage_errors_use_stderr_and_exit_two(argv, diagnostic):
    completed = _run(*argv)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("usage: ")
    assert diagnostic in completed.stderr
    assert "Traceback" not in completed.stderr
