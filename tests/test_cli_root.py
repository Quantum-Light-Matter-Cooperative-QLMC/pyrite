from __future__ import annotations

import importlib

import pytest

from cxr_mc import cli


def test_root_help_does_not_import_lazy_commands(monkeypatch, capsys):
    def unexpected_import(name):
        raise AssertionError(f"root help imported {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("Usage: cxr ")
    assert "scan" in captured.out
    assert captured.err == ""


def test_root_version_preserves_public_text(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"cxr-mc {cli.__version__}\n"
    assert captured.err == ""


def test_root_unknown_command_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["not-a-command"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such command 'not-a-command'" in captured.err
    assert "Traceback" not in captured.err


def test_root_help_prefers_grouped_checkpoint_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    command_lines = {
        line.split()[0]
        for line in lines
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert "checkpoint" in command_lines
    assert {"app", "catalog"}.issubset(command_lines)
    assert command_lines.isdisjoint(
        {
            "slim",
            "rebrem",
            "reline",
            "archive",
            "restore",
            "archives",
            "union",
            "check",
            "check-config",
        }
    )


@pytest.mark.parametrize(
    "alias",
    (
        "slim",
        "rebrem",
        "reline",
        "archive",
        "restore",
        "archives",
        "union",
        "check",
        "check-config",
    ),
)
def test_hidden_checkpoint_aliases_remain_callable(alias, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([alias, "--help"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith(f"Usage: cxr {alias} ")
