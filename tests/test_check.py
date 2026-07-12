"""Tests for cxr_mc.check: marimo launch args + the --export figure batch."""

import subprocess
import sys
from pathlib import Path

import pytest

from cxr_mc import check


def test_validation_default_azimuth_roundtrip(tmp_path):
    settings = tmp_path / "validation_defaults.json"
    settings.write_text('{"tmd_exploratory_azimuth_deg": 0.0}\n', encoding="utf-8")

    check.save_default_azimuth(35.0, settings)

    assert check.load_default_azimuth(settings) == 35.0


@pytest.mark.parametrize("value", [-0.1, 180.1])
def test_validation_default_azimuth_rejects_out_of_range(value, tmp_path):
    with pytest.raises(ValueError, match="between 0 and 180"):
        check.save_default_azimuth(value, tmp_path / "validation_defaults.json")


def test_remote_probe_reports_missing_transport(monkeypatch):
    monkeypatch.setattr(
        check.shutil, "which", lambda command: None if command == "ssh" else command
    )

    available, reason = check.probe_remote_zhai()

    assert not available
    assert "ssh" in reason


def test_remote_probe_checks_configured_host(monkeypatch):
    monkeypatch.setattr(check.shutil, "which", lambda command: command)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(check.subprocess, "run", fake_run)

    assert check.probe_remote_zhai() == (True, "remote GPU available")
    assert calls[0][0][-1] == "true"
    assert "BatchMode=yes" in calls[0][0]


def test_start_remote_zhai_returns_detached_job_id(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="started zhai job 20260711-123456 on qlmc\n",
            stderr="",
        )

    monkeypatch.setattr(check.subprocess, "run", fake_run)

    jobid = check.start_remote_zhai(ne=11, ne_brem=3, ne_supp=5, tmd_azimuth=35.0, refresh=True)

    assert jobid == "20260711-123456"


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ("-- state --\nrunning zhai reproduction since 2026-07-11\n", "running"),
        ("-- state --\ndone 2026-07-11T12:34:56\n", "done"),
        ("-- state --\nfailed (exit 1)\nprocess: not running\n", "failed"),
    ],
)
def test_remote_zhai_status_parses_job_state(monkeypatch, report, expected):
    monkeypatch.setattr(
        check.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, report, ""),
    )

    state, output = check.remote_zhai_status("20260711-123456")

    assert state == expected
    assert output == report.rstrip()


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


def test_acp_flag_starts_validation_with_bridge_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main(["check", "--acp"])

    assert calls == [{"edit": False, "watch": False, "acp": True}]


def test_launch_treats_keyboard_interrupt_as_normal_marimo_exit():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from cxr_mc import check; "
                "check.subprocess.run = lambda *args, **kwargs: "
                "(_ for _ in ()).throw(KeyboardInterrupt()); "
                "check._launch()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_validation_app_initializes_its_default_supplementary_study():
    repo_dir = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, check.NOTEBOOK],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_validation_app_centers_the_supplementary_figure():
    repo_dir = Path(__file__).resolve().parents[1]
    source = (repo_dir / check.NOTEBOOK).read_text(encoding="utf-8")

    assert "mo.center(figure)" in source
