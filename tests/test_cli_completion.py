from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cxr_mc import _cli_completion


def _values(items):
    return [item.value for item in items]


def test_material_completion_reads_offline_catalog():
    values = _values(_cli_completion.complete_material(None, None, "di"))
    assert values == ["diamond"]


def test_comma_separated_material_completion_preserves_prefix_and_omits_duplicates():
    values = _values(_cli_completion.complete_material_csv(None, None, "hopg,di"))
    assert values == ["hopg,diamond"]
    assert "hopg,hopg" not in _values(_cli_completion.complete_material_csv(None, None, "hopg,"))


def test_material_completion_is_empty_when_catalog_read_fails(monkeypatch):
    _cli_completion._material_keys.cache_clear()
    monkeypatch.setattr(_cli_completion, "DATA_DIR", Path("/missing"))
    assert _cli_completion.complete_material(None, None, "") == []
    monkeypatch.undo()
    _cli_completion._material_keys.cache_clear()


def test_checkpoint_completion_is_nonrecursive_and_excludes_unsafe_files(tmp_path):
    (tmp_path / "hopg.pkl").touch()
    (tmp_path / "not a candidate.pkl").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.pkl").touch()
    (tmp_path / "linked.pkl").symlink_to(tmp_path / "hopg.pkl")
    ctx = SimpleNamespace(params={"checkpoint_dir": str(tmp_path)})

    assert _values(_cli_completion.complete_checkpoint(ctx, None, "ho")) == [f"{tmp_path}/hopg.pkl"]


def test_checkpoint_completion_honors_explicit_path(tmp_path):
    (tmp_path / "diamond.pkl").touch()
    ctx = SimpleNamespace(params={})
    incomplete = f"{tmp_path}/di"
    assert _values(_cli_completion.complete_checkpoint(ctx, None, incomplete)) == [
        f"{tmp_path}/diamond.pkl"
    ]


def test_archive_label_completion_reads_only_archive_shelf(tmp_path):
    (tmp_path / "active.pkl").touch()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "good-run.pkl").touch()
    old_root = _cli_completion._ARCHIVE_CHECKPOINT_ROOT
    _cli_completion._ARCHIVE_CHECKPOINT_ROOT = tmp_path
    try:
        labels = _values(_cli_completion.complete_archive_label(None, None, "good"))
        stems = _values(_cli_completion.complete_archive_stem(None, None, "act"))
    finally:
        _cli_completion._ARCHIVE_CHECKPOINT_ROOT = old_root

    assert labels == ["good-run"]
    assert stems == ["active"]


def test_local_completion_is_bounded(tmp_path):
    for index in range(_cli_completion.MAX_LOCAL_CANDIDATES + 5):
        (tmp_path / f"m{index:03}.pkl").touch()
    ctx = SimpleNamespace(params={"checkpoint_dir": str(tmp_path)})

    assert (
        len(_cli_completion.complete_checkpoint(ctx, None, ""))
        == _cli_completion.MAX_LOCAL_CANDIDATES
    )


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("ssh", 1),
        OSError("ssh missing"),
        SystemExit("bad config"),
    ],
)
def test_job_completion_silences_lookup_failures(monkeypatch, capsys, error):
    def fail():
        raise error

    monkeypatch.setattr(_cli_completion, "_query_remote_job_ids", fail)
    assert _cli_completion.complete_job_id(None, None, "") == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_job_completion_filters_unsafe_values_and_prefixes(monkeypatch):
    monkeypatch.setattr(
        _cli_completion,
        "_query_remote_job_ids",
        lambda: ["20260725-abc", "bad value", "-option", "20260724-old"],
    )
    assert _values(_cli_completion.complete_job_id(None, None, "20260725")) == ["20260725-abc"]


def test_remote_lookup_is_noninteractive_and_bounded(monkeypatch):
    recorded = {}

    def run(argv, **kwargs):
        recorded["argv"] = argv
        recorded["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="job-2\njob-1\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert _cli_completion._query_remote_job_ids() == ["job-2", "job-1"]
    assert "BatchMode=yes" in recorded["argv"]
    assert "ConnectTimeout=1" in recorded["argv"]
    assert recorded["kwargs"]["stdin"] is subprocess.DEVNULL
    assert recorded["kwargs"]["stderr"] is subprocess.DEVNULL
    assert recorded["kwargs"]["timeout"] == _cli_completion.REMOTE_COMPLETION_TIMEOUT_SECONDS


def test_choice_completer_returns_prefix_matches():
    complete = _cli_completion.choice_completer(range(1, 10))
    assert _values(complete(None, None, "1")) == ["1"]


def _parameter(command, name):
    return next(param for param in command.params if param.name == name)


def _callback(command, name):
    return _parameter(command, name)._custom_shell_complete


def test_local_commands_wire_material_checkpoint_archive_and_choice_completion():
    from cxr_mc import analyze, archive, blaze, rebrem, reline, scan, slim

    for command in (scan.command, blaze.command, analyze.command):
        assert _callback(command, "material") is _cli_completion.complete_material
    for command in (rebrem.command, reline.command):
        assert _callback(command, "materials") is _cli_completion.complete_checkpoint_stem

    assert _callback(slim.command, "checkpoint") is _cli_completion.complete_checkpoint
    assert _values(_parameter(slim.command, "compresslevel").shell_complete(None, "")) == [
        str(value) for value in range(1, 10)
    ]

    assert _callback(archive.archive_command, "stem") is _cli_completion.complete_archive_stem
    assert _callback(archive.archive_command, "label") is None
    assert _callback(archive.restore_command, "label") is _cli_completion.complete_archive_label
    assert _callback(archive.union_command, "stem") is _cli_completion.complete_archive_stem
    assert _callback(archive.union_command, "label") is _cli_completion.complete_archive_label


def test_remote_commands_wire_safe_completion_but_not_destructive_targets():
    from cxr_mc._remote import cli

    for name in ("scan", "start", "pull", "rebrem", "reline"):
        command = cli.command.commands[name]
        parameter = "materials" if name == "start" else "material"
        assert _callback(command, parameter) is _cli_completion.complete_material
    for name in ("attach", "status", "logs"):
        assert _callback(cli.command.commands[name], "jobid") is _cli_completion.complete_job_id
    for name in ("scan", "start"):
        values = _parameter(cli.command.commands[name], "parallel_materials").shell_complete(
            None, ""
        )
        assert _values(values) == ["1", "2", "3", "4"]

    assert _callback(cli.command.commands["stop"], "materials") is None
    assert _callback(cli.command.commands["clear"], "materials") is None
    assert all(
        param._custom_shell_complete is None for param in cli.command.commands["reap"].params
    )


def test_line_grid_wires_safe_completion_but_not_stop_target():
    from cxr_mc.line_grid import command

    for name in ("derive", "submit", "apply"):
        assert (
            _callback(command.commands[name], "materials") is _cli_completion.complete_material_csv
        )
    for name in ("set", "set-brem", "show"):
        assert _callback(command.commands[name], "material") is _cli_completion.complete_material
    for name in ("attach", "status", "logs"):
        assert _callback(command.commands[name], "jobid") is _cli_completion.complete_job_id
    assert _callback(command.commands["stop"], "jobid") is None
