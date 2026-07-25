from __future__ import annotations

import subprocess
import sys

import pytest

from cxr_mc import (
    analyze,
    archive,
    blaze,
    check,
    check_config,
    export,
    rebrem,
    reline,
    scan,
    slim,
)
from tests.cli_helpers import assert_clean_result, invoke

LOCAL_COMMANDS = [
    scan.command,
    blaze.command,
    analyze.command,
    check.command,
    export.command,
    slim.command,
    rebrem.command,
    reline.command,
    archive.archive_command,
    archive.restore_command,
    archive.archives_command,
    archive.union_command,
    check_config.command,
]


@pytest.mark.parametrize("command", LOCAL_COMMANDS, ids=lambda command: command.name)
def test_every_local_click_help_path_is_clean(command):
    result = invoke(command, ["--help"])
    assert_clean_result(result)
    assert result.stdout.startswith("Usage:")
    assert result.stdout.endswith("\n")


def _capture(monkeypatch, module, handler_name):
    seen = {}

    def handler(args):
        seen.update(vars(args))

    monkeypatch.setattr(module, handler_name, handler)
    return seen


def test_scan_click_dispatches_defaults_and_zero_workers(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(
        scan.command,
        ["hopg", "--workers", "0", "--beam-uvw", "1", "0", "-1"],
    )
    assert_clean_result(result)
    assert seen == {
        "material": "hopg",
        "all": False,
        "workers": 0,
        "quick": False,
        "n_families": None,
        "beam_uvw": (1, 0, -1),
        "checkpoint_dir": "checkpoints",
        "max_minutes": None,
        "progress_file": None,
        "no_progress": False,
    }


def test_blaze_preserves_one_flag_many_values_syntax(monkeypatch):
    seen = _capture(monkeypatch, blaze, "run")
    result = invoke(
        blaze.command,
        [
            "hopg",
            "--energy",
            "30",
            "40",
            "--spacing",
            "1e-6",
            "--angles",
            "10",
            "20",
            "--workers",
            "0",
        ],
    )
    assert_clean_result(result)
    assert seen["energy"] == [30.0, 40.0]
    assert seen["spacing"] == [1e-6]
    assert seen["angles"] == [10.0, 20.0]
    assert seen["workers"] == 0


@pytest.mark.parametrize(
    ("module", "command", "handler", "argv", "expected"),
    [
        (
            analyze,
            analyze.command,
            "_cli",
            ["hopg", "--default", "--smoke"],
            {"material": "hopg", "default": True, "smoke": True},
        ),
        (
            check,
            check.command,
            "_cli",
            ["--export", "--ne", "11"],
            {"export": True, "ne": 11, "outdir": "figures"},
        ),
        (
            slim,
            slim.command,
            "_cli",
            ["in.pkl", "--line-only"],
            {"checkpoint": "in.pkl", "line_only": True, "compresslevel": 6},
        ),
        (
            rebrem,
            rebrem.command,
            "_cli",
            ["hopg", "w", "--ne-brem", "1000"],
            {"material": ["hopg", "w"], "all": False, "ne_brem": 1000},
        ),
        (
            reline,
            reline.command,
            "_cli",
            ["--all", "--line-step", "5"],
            {"material": [], "all": True, "line_step": 5.0},
        ),
        (
            check_config,
            check_config.command,
            "_run",
            ["catalog.toml"],
            {"manifest": "catalog.toml"},
        ),
    ],
)
def test_local_click_dispatches_legacy_handler(
    monkeypatch, module, command, handler, argv, expected
):
    seen = _capture(monkeypatch, module, handler)
    result = invoke(command, argv)
    assert_clean_result(result)
    for key, value in expected.items():
        actual = seen[key]
        if key == "manifest":
            actual = str(actual)
        assert actual == value


def test_export_click_dispatches_default_stem(monkeypatch):
    seen = []
    monkeypatch.setattr(export, "_export", seen.append)
    assert_clean_result(invoke(export.command))
    assert seen == [None]


@pytest.mark.parametrize(
    ("command", "handler_name", "argv", "expected"),
    [
        (
            archive.archive_command,
            "_cli_archive",
            ["hopg", "keeper", "--force"],
            {"stem": "hopg", "label": "keeper", "force": True},
        ),
        (
            archive.restore_command,
            "_cli_restore",
            ["keeper", "--as", "hopg"],
            {"label": "keeper", "stem": "hopg", "force": False},
        ),
        (archive.archives_command, "_cli_archives", [], {}),
        (
            archive.union_command,
            "_cli_union",
            ["hopg", "keeper", "--no-archive", "--delete-archive"],
            {
                "stem": "hopg",
                "label": "keeper",
                "no_archive": True,
                "delete_archive": True,
                "force": False,
            },
        ),
    ],
)
def test_archive_click_dispatch(command, handler_name, argv, expected, monkeypatch):
    seen = _capture(monkeypatch, archive, handler_name)
    assert_clean_result(invoke(command, argv))
    assert seen == expected


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        (scan.command, ["hopg", "--workers", "-1"]),
        (scan.command, ["hopg", "--beam-uvw", "0", "0", "0"]),
        (blaze.command, ["hopg", "--energy", "0", "--spacing", "1e-6"]),
        (blaze.command, ["hopg", "--energy", "30", "--spacing", "1e-6", "--angles", "90"]),
        (check.command, ["--ne", "0"]),
        (rebrem.command, ["hopg", "--save-every", "0"]),
        (reline.command, ["hopg", "--line-step", "nan"]),
    ],
)
def test_local_click_numeric_domains_are_usage_errors(command, argv):
    result = invoke(command, argv)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr


def test_slim_dataset_modes_are_mutually_exclusive():
    result = invoke(slim.command, ["in.pkl", "--brem-only", "--line-only"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "mutually exclusive" in result.stderr


def test_standalone_click_usage_error_preserves_exit_and_streams():
    completed = subprocess.run(
        [sys.executable, "-m", "cxr_mc.scan", "hopg", "--workers", "-1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Invalid value for '--workers'" in completed.stderr
    assert "Traceback" not in completed.stderr
