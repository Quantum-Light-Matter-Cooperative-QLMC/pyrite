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
