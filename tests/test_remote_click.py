"""Focused Click-contract tests for ``cxr remote``."""

from __future__ import annotations

import click
import pytest

from cxr_mc import remote
from cxr_mc._remote import lifecycle, viewer
from tests.cli_helpers import assert_clean_result, invoke

REMOTE_COMMANDS = (
    "scan",
    "rebrem",
    "reline",
    "submit",
    "start",
    "attach",
    "jobs",
    "status",
    "logs",
    "stop",
    "reap",
    "pull",
    "clear",
    "sync",
    "validate",
    "check",
)


def test_remote_exports_click_group():
    assert isinstance(remote.command, click.Group)
    assert set(remote.command.commands) == set(REMOTE_COMMANDS)


@pytest.mark.parametrize("name", REMOTE_COMMANDS)
def test_every_remote_click_help_path_is_offline(name):
    result = invoke(remote.command, [name, "--help"])

    assert_clean_result(result)
    assert f"Usage: remote {name} " in result.stdout


def test_start_click_defaults_and_zero_meanings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote.command, ["submit", "hopg", "--workers", "0"])

    assert_clean_result(result)
    assert calls == [
        (
            ["hopg"],
            {
                "quick": False,
                "fidelity": "full",
                "workers": 0,
                "parallel_materials": None,
                "chunk_minutes": 10.0,
                "no_sync": False,
                "dry_run": False,
                "high_energy_min_kev": None,
            },
        )
    ]


def test_hidden_remote_aliases_remain_callable():
    for alias in ("start", "check"):
        result = invoke(remote.command, [alias, "--help"])
        assert_clean_result(result)

    root_help = invoke(remote.command, ["--help"])
    command_lines = {
        line.split()[0]
        for line in root_help.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert {"submit", "validate"}.issubset(command_lines)
    assert command_lines.isdisjoint({"start", "check"})


@pytest.mark.parametrize(
    "argv, option",
    [
        (["start", "hopg", "--workers", "-1"], "--workers"),
        (["start", "hopg", "--chunk-minutes", "-1"], "--chunk-minutes"),
        (["rebrem", "hopg", "--ne-brem", "0"], "--ne-brem"),
        (["rebrem", "hopg", "--step", "nan"], "--step"),
        (["reline", "hopg", "--line-ne", "0"], "--line-ne"),
        (["reline", "hopg", "--line-step", "inf"], "--line-step"),
        (["reap", "--min-age-minutes", "-0.1"], "--min-age-minutes"),
        (["check", "--ne", "0"], "--ne"),
        (["check", "--tmd-azimuth", "nan"], "--tmd-azimuth"),
        (["check", "--tmd-azimuth", "-inf"], "--tmd-azimuth"),
    ],
)
def test_remote_numeric_domains_fail_at_click_boundary(argv, option):
    result = invoke(remote.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert option in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "argv, message",
    [
        (["start"], "needs material"),
        (["start", "hopg", "--all"], "--all does not take"),
        (["scan", "hopg", "--quick", "--grid"], "drop --grid"),
        (["pull", "hopg", "--brem-only", "--line-only"], "mutually exclusive"),
        (["stop"], "needs material"),
        (["clear"], "needs material"),
        (["check", "--follow"], "requires --detached"),
        (["check", "--pull", "--detached"], "mutually exclusive"),
    ],
)
def test_remote_incompatible_click_inputs_are_usage_errors(argv, message):
    result = invoke(remote.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize("command_name", ["scan", "rebrem", "reline", "submit"])
def test_fidelity_dispatches_cleanly(monkeypatch, command_name):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(
        lifecycle,
        "start_rebrem_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(
        lifecycle,
        "start_reline_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)

    # ``rebrem``/``reline`` support --dry-run, which returns before the
    # (mocked, always-disconnected) viewer.attach call; ``scan`` has no
    # --dry-run and always attaches, so a disconnected viewer is expected to
    # print its "skipping automatic pull" diagnostic; ``submit`` only
    # attaches with --follow (default off), so it stays clean either way.
    argv = [command_name, "hopg", "--fidelity", "survey", "--no-sync"]
    if command_name in ("rebrem", "reline"):
        argv.append("--dry-run")

    result = invoke(remote.command, argv)

    if command_name == "scan":
        assert result.exit_code == 0
        assert "skipping automatic pull" in result.stderr
    else:
        assert_clean_result(result)
    assert calls[0]["fidelity"] == "survey"


@pytest.mark.parametrize(
    ("flag", "dataset"),
    [("--brem-only", "brem"), ("--line-only", "line")],
)
def test_partial_pull_all_forwards_full_material_list(monkeypatch, tmp_path, flag, dataset):
    manifest = tmp_path / "materials.txt"
    manifest.write_text('materials = ["hopg", "hbn", "mos2"]\n')
    monkeypatch.setattr(remote.config, "MATS_FILE", manifest)
    seen = {}
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: seen.update(materials=materials, kwargs=kwargs),
    )

    result = invoke(remote.command, ["pull", "--all", flag])

    assert_clean_result(result)
    assert seen["materials"] == ["hopg", "hbn", "mos2"]
    assert seen["kwargs"]["dataset"] == dataset
    assert seen["kwargs"]["grid"] is False


@pytest.mark.parametrize("status", [1, 130])
def test_logs_click_propagates_follow_status(monkeypatch, status):
    monkeypatch.setattr(viewer, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(remote.command, ["logs", "--follow"])

    assert_clean_result(result, exit_code=status)


def test_remote_click_preserves_resumable_exit(monkeypatch):
    monkeypatch.setattr(viewer, "list_jobs", lambda: (_ for _ in ()).throw(SystemExit(75)))

    result = invoke(remote.command, ["jobs"])

    assert_clean_result(result, exit_code=75)


def test_stop_previews_by_default_and_yes_executes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "stop_jobs",
        lambda materials, all_jobs, *, yes: calls.append((materials, all_jobs, yes)),
    )

    preview = invoke(remote.command, ["stop", "hopg"])
    confirmed = invoke(remote.command, ["stop", "hopg", "--yes"])

    assert_clean_result(preview)
    assert_clean_result(confirmed)
    assert calls == [(["hopg"], False, False), (["hopg"], False, True)]
