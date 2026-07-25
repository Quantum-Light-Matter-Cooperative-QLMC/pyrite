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

    result = invoke(remote.command, ["start", "hopg", "--workers", "0"])

    assert_clean_result(result)
    assert calls == [
        (
            ["hopg"],
            {
                "quick": False,
                "workers": 0,
                "parallel_materials": None,
                "chunk_minutes": 10.0,
                "no_sync": False,
                "dry_run": False,
            },
        )
    ]


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


@pytest.mark.parametrize("status", [1, 130])
def test_logs_click_propagates_follow_status(monkeypatch, status):
    monkeypatch.setattr(viewer, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(remote.command, ["logs", "--follow"])

    assert_clean_result(result, exit_code=status)


def test_remote_click_preserves_resumable_exit(monkeypatch):
    monkeypatch.setattr(viewer, "list_jobs", lambda: (_ for _ in ()).throw(SystemExit(75)))

    result = invoke(remote.command, ["jobs"])

    assert_clean_result(result, exit_code=75)
