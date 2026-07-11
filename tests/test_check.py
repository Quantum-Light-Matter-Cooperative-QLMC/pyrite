"""Tests for cxr_mc.check: marimo launch args + the --export figure batch."""

import sys
from pathlib import Path

import pytest

from cxr_mc import check


def test_command_uses_run_by_default():
    cmd = check._command()
    assert cmd[1:4] == ["-m", "marimo", "run"]
    assert check.NOTEBOOK in cmd
    assert "--watch" not in cmd


def test_command_edit_and_watch_flags():
    cmd = check._command(edit=True, watch=True)
    assert "edit" in cmd and "run" not in cmd
    assert "--watch" in cmd


def test_export_cli_calls_export_all_figures_and_skips_marimo(monkeypatch, tmp_path):
    calls = []

    class _FakeAF:
        @staticmethod
        def export_all_figures(outdir, ne, ne_brem, ne_supp):
            calls.append((outdir, ne, ne_brem, ne_supp))
            return [Path(outdir) / "a.png"]

    monkeypatch.setitem(sys.modules, "anchor_figures", _FakeAF())
    monkeypatch.setattr(
        check, "_launch", lambda **kw: pytest.fail("--export must not launch marimo")
    )

    check.main(["check", "--export", "--outdir", str(tmp_path), "--ne", "11"])

    assert calls == [(str(tmp_path), 11, 200, 200)]


def test_default_cli_launches_marimo_not_export(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main(["check"])

    assert calls == [{"edit": False, "watch": False}]
