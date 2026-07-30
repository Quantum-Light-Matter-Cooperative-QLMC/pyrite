"""Tests for cxr_mc.check: marimo launch args + the --export figure batch."""

import subprocess
import sys
from pathlib import Path

import pytest

from cxr_mc import check
from tests.cli_helpers import invoke


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
    assert "--port" not in cmd


def test_command_edit_and_watch_flags():
    cmd = check._command(edit=True, watch=True)
    assert "edit" in cmd and "run" not in cmd
    assert "--watch" in cmd


def test_command_tunnel_uses_fixed_marimo_port():
    command = check._command(tunnel=True)
    assert command[3:7] == ["run", "--port", "2718", check.NOTEBOOK]


def test_tunnel_launch_prints_forwarding_instructions_without_running_marimo(monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(
        check.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    check._launch(tunnel=True)

    output = capsys.readouterr().out
    assert "ssh -L 2718:127.0.0.1:2718 <your-pi-ssh-host>" in output
    assert "http://127.0.0.1:2718" in output
    assert len(launched) == 1


def test_default_launch_keeps_stdout_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(check.subprocess, "run", lambda *args, **kwargs: None)

    check._launch()

    assert capsys.readouterr().out == ""


def test_tunnel_flag_forwards_to_validation_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main(["--tunnel"])

    assert calls == [{"edit": False, "watch": False, "tunnel": True}]


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

    check.main(["--export", "--outdir", str(tmp_path), "--ne", "11"])

    assert calls == [(str(tmp_path), 11, 200, 200)]


def test_export_tunnel_cli_calls_export_all_figures_and_skips_marimo(monkeypatch, tmp_path):
    calls = []

    class _FakeAF:
        @staticmethod
        def export_all_figures(outdir, ne, ne_brem, ne_supp):
            calls.append((outdir, ne, ne_brem, ne_supp))
            return []

    monkeypatch.setitem(sys.modules, "anchor_figures", _FakeAF())
    monkeypatch.setattr(
        check, "_launch", lambda **kw: pytest.fail("--export --tunnel must not launch marimo")
    )

    check.main(["--export", "--tunnel", "--outdir", str(tmp_path), "--ne", "11"])

    assert calls == [(str(tmp_path), 11, 200, 200)]


def test_export_cache_miss_is_clean_cli_failure(monkeypatch, tmp_path):
    class _FakeAF:
        @staticmethod
        def export_all_figures(*_args, **_kwargs):
            raise FileNotFoundError(
                "Zhai cache missing; populate it with `cxr remote validate`"
            )

    monkeypatch.setitem(sys.modules, "anchor_figures", _FakeAF())

    result = invoke(check.command, ["--export", "--outdir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error: Zhai cache missing" in result.output
    assert "cxr remote validate" in result.output
    assert result.exception is not None


def test_default_cli_launches_marimo_not_export(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main([])

    assert calls == [{"edit": False, "watch": False}]


def test_acp_flag_starts_validation_with_bridge_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main(["--acp"])

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


def test_validation_app_never_falls_back_to_heavy_local_cache_population():
    source = Path(check.NOTEBOOK).read_text(encoding="utf-8")

    assert "falls back locally" not in source.lower()
    assert "af.reproduce_all(" not in source
    assert source.count("cache_only=True") >= 2
    assert "Heavy cache preparation was not started locally" in source
    assert "cxr remote validate" in source


def test_validation_app_declares_evidence_tasks_and_authorities():
    source = (Path(__file__).resolve().parents[1] / check.NOTEBOOK).read_text(encoding="utf-8")

    for task in ("Anchors", "Reproductions", "Supplementary", "Provenance"):
        assert f'"{task}"' in source
    for authority in ("Anchor", "Diagnostic", "Optional oracle", "Provenance"):
        assert authority in source

    task_navigation = source[source.rindex("mo.ui.tabs(") :]
    grouped_surfaces = {
        "Anchors": ("anchors_controls", "anchors_results"),
        "Reproductions": ("reproductions_controls", "reproductions_results"),
        "Supplementary": ("supplementary_controls", "supplementary_results", "remote_controls"),
        "Provenance": ("provenance_findings", "provenance_audit"),
    }
    task_offsets = {
        task: task_navigation.index(f'"{task}": mo.vstack(') for task in grouped_surfaces
    }
    ordered_tasks = list(grouped_surfaces)
    for index, task in enumerate(ordered_tasks):
        start = task_offsets[task]
        stop = task_offsets[ordered_tasks[index + 1]] if index + 1 < len(ordered_tasks) else None
        task_source = task_navigation[start:stop]
        for surface in grouped_surfaces[task]:
            assert surface in task_source


def test_validation_diagnostic_success_requires_interpretation():
    source = (Path(__file__).resolve().parents[1] / check.NOTEBOOK).read_text(encoding="utf-8")

    assert 'authority == "Diagnostic"' in source
    assert 'state = "Completed—interpret"' in source
    assert 'state = "Passed"' in source
    assert 'marker = "✓"' not in source


def test_validation_oracle_distinguishes_missing_dependency_from_failed_comparison():
    source = (Path(__file__).resolve().parents[1] / check.NOTEBOOK).read_text(encoding="utf-8")

    skip_branch = 'authority == "Optional oracle" and "result: skip"'
    assert skip_branch in source
    assert source.index(skip_branch) < source.index('report["returncode"] != 0')
    assert "uv run --group oracle" in source


def test_repository_default_save_names_mutated_setting_and_file():
    source = (Path(__file__).resolve().parents[1] / check.NOTEBOOK).read_text(encoding="utf-8")

    assert "Save repository default" in source
    assert "tmd_exploratory_azimuth_deg" in source
    assert "notebooks/validation_defaults.json" in source
