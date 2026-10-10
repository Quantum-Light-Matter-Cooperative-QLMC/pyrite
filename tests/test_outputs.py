"""Workspace output contracts: precedence, migration, and public CLI defaults."""

import errno
import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from pyrite.console import config, outputs


@pytest.mark.parametrize("kind", sorted(outputs.OUTPUT_KINDS))
def test_output_kinds_follow_dynamic_workspace(monkeypatch, tmp_path, kind):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("PYRITE_HOME", str(first))
    assert config.output_dir(kind) == first / "pyrite-output" / kind
    monkeypatch.setenv("PYRITE_HOME", str(second))
    assert config.output_dir(kind) == second / "pyrite-output" / kind
    assert not first.exists() and not second.exists()


def test_output_precedence_and_nearest_ancestor(monkeypatch, tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    cwd = inner / "subdir"
    cwd.mkdir(parents=True)
    (outer / "pyrite-output").mkdir()
    (inner / "pyrite-output").mkdir()
    monkeypatch.setattr(outputs, "_cwd", lambda: cwd)
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    assert outputs.output_root() == inner / "pyrite-output"
    config.set_stored("workspace.root", str(tmp_path / "stored"))
    assert outputs.output_root() == tmp_path / "stored" / "pyrite-output"
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "env"))
    assert outputs.output_root() == tmp_path / "env" / "pyrite-output"
    assert outputs.output_root(tmp_path / "explicit") == tmp_path / "explicit" / "pyrite-output"


def test_output_falls_back_to_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    assert outputs.output_root() == tmp_path / "pyrite-output"
    with pytest.raises(ValueError, match="unknown output kind"):
        outputs.output_dir("../foreign")


def test_legacy_move_carves_cache_and_preserves_bytes(tmp_path, capsys):
    old = {
        "checkpoints/hopg/line.h5": b"durable",
        "checkpoints/archive/old/meta.json": b"archive",
        "checkpoints/zhai_reproduction/model.pkl": b"recomputable",
        "checkpoints/.analysis-cache/analysis.pkl": b"analysis",
        "observations/hopg/record.h5": b"observation",
        "performance-profiles/demo/hopg.ndjson": b"telemetry",
        "results/analysis.html": b"html",
    }
    for name, data in old.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    moved = outputs.migrate_legacy_outputs(tmp_path)
    for source, destination in moved:
        assert not source.exists()
        assert destination.is_dir()
    root = tmp_path / "pyrite-output"
    assert (root / "checkpoints/hopg/line.h5").read_bytes() == b"durable"
    assert (root / "checkpoints/archive/old/meta.json").read_bytes() == b"archive"
    assert (root / "cache/zhai_reproduction/model.pkl").read_bytes() == b"recomputable"
    assert (root / "cache/analysis/analysis.pkl").read_bytes() == b"analysis"
    assert (root / "observations/hopg/record.h5").read_bytes() == b"observation"
    assert (root / "performance/demo/hopg.ndjson").read_bytes() == b"telemetry"
    assert (root / "results/analysis.html").read_bytes() == b"html"
    output = capsys.readouterr()
    assert not output.out and "moved" in output.err


def test_legacy_collision_warns_once_without_merging(tmp_path, capsys):
    legacy = tmp_path / "checkpoints"
    current = tmp_path / "pyrite-output/checkpoints"
    legacy.mkdir()
    current.mkdir(parents=True)
    (legacy / "old.pkl").write_bytes(b"old")
    (current / "new.pkl").write_bytes(b"new")
    outputs.enable_legacy_migration()
    assert outputs.output_dir("checkpoints") == current
    outputs.output_dir("results")
    assert capsys.readouterr().err.count("warning:") == 1
    assert {p.name for p in legacy.iterdir()} == {"old.pkl"}
    assert {p.name for p in current.iterdir()} == {"new.pkl"}


def test_empty_placeholders_and_symlinks_are_not_moved(tmp_path):
    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    (legacy / ".gitkeep").touch()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "record.pkl").write_bytes(b"external")
    (tmp_path / "results").symlink_to(foreign, target_is_directory=True)
    assert outputs.migrate_legacy_outputs(tmp_path) == []
    assert not (tmp_path / "pyrite-output").exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_cross_filesystem_copy_is_verified(monkeypatch, tmp_path, corrupt):
    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    (legacy / "record.pkl").write_bytes(b"important")
    original_rename = Path.rename
    original_copy = outputs.shutil.copytree

    def cross_device(path, destination):
        if path == legacy:
            raise OSError(errno.EXDEV, "cross-device link")
        return original_rename(path, destination)

    def copy(source, destination, **kwargs):
        result = original_copy(source, destination, **kwargs)
        if corrupt:
            (Path(destination) / "record.pkl").write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(Path, "rename", cross_device)
    monkeypatch.setattr(outputs.shutil, "copytree", copy)
    if corrupt:
        with pytest.raises(click.ClickException, match="verification failed"):
            outputs.migrate_legacy_outputs(tmp_path)
        assert (legacy / "record.pkl").read_bytes() == b"important"
        assert not (tmp_path / "pyrite-output/checkpoints").exists()
    else:
        outputs.migrate_legacy_outputs(tmp_path)
        assert not legacy.exists()
        assert (tmp_path / "pyrite-output/checkpoints/record.pkl").read_bytes() == b"important"
    assert not list((tmp_path / "pyrite-output").glob(".pyrite-migrate-*"))


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"], ["checkpoint", "list", "--help"]])
def test_help_never_moves_legacy_outputs(tmp_path, argv):
    from pyrite.cli import command

    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    (legacy / "old.pkl").write_bytes(b"old")
    result = CliRunner().invoke(command, argv)
    assert result.exit_code == 0, result.output
    assert legacy.exists()
    assert not (tmp_path / "pyrite-output").exists()


@pytest.mark.parametrize("suffix", [[], ["--no-progress"]])
def test_bare_trajectories_uses_workspace_default(monkeypatch, tmp_path, suffix):
    from pyrite.cli import command
    from pyrite.runs import scan

    seen = {}
    monkeypatch.setattr(scan, "run", lambda args: seen.update(vars(args)))
    result = CliRunner().invoke(
        command, ["run", "standard", "-m", "hopg", "--trajectories", *suffix]
    )
    assert result.exit_code == 0, result.output
    assert seen["trajectories"] == tmp_path / "pyrite-output/trajectories"
    assert seen["checkpoint_dir"] == str(tmp_path / "pyrite-output/checkpoints")


def test_explicit_output_dirs_win(monkeypatch, tmp_path):
    from pyrite.cli import command
    from pyrite.runs import scan

    seen = {}
    monkeypatch.setattr(scan, "run", lambda args: seen.update(vars(args)))
    trajectory = tmp_path / "capture"
    checkpoint = tmp_path / "components"
    result = CliRunner().invoke(
        command,
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--trajectories",
            str(trajectory),
            "--checkpoint-dir",
            str(checkpoint),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["trajectories"] == trajectory
    assert seen["checkpoint_dir"] == str(checkpoint)


def test_remote_legacy_migration_and_collision(monkeypatch, tmp_path):
    from pyrite.remote import config as remote_config

    monkeypatch.setattr(remote_config, "REMOTE_DIR", str(tmp_path))
    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    (legacy / "old.pkl").write_bytes(b"old")
    performance = tmp_path / "jobs/job-1/performance/demo"
    performance.mkdir(parents=True)
    (performance / "hopg.ndjson").write_bytes(b"samples")
    command = remote_config.remote_output_migration_command()
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pyrite-output/checkpoints/old.pkl").read_bytes() == b"old"
    assert (
        tmp_path / "pyrite-output/performance/job-1/demo/hopg.ndjson"
    ).read_bytes() == b"samples"
    legacy.mkdir()
    (legacy / "foreign.pkl").write_bytes(b"foreign")
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and "warning:" in result.stderr
    assert (legacy / "foreign.pkl").read_bytes() == b"foreign"
    assert not (tmp_path / "pyrite-output/checkpoints/foreign.pkl").exists()


def test_explicit_trajectory_directory_may_name_a_profile(monkeypatch):
    from pyrite.cli import command
    from pyrite.runs import scan

    seen = {}
    monkeypatch.setattr(scan, "run", lambda args: seen.update(vars(args)))
    config.set_stored("profile.current", "standard")
    result = CliRunner().invoke(command, ["run", "--trajectories=standard", "-m", "hopg"])
    assert result.exit_code == 0, result.output
    assert seen["trajectories"] == Path("standard")


def test_cli_migration_preserves_json_stream(tmp_path):
    import json

    from pyrite.cli import command

    legacy = tmp_path / "checkpoints"
    legacy.mkdir()
    (legacy / "old.pkl").write_bytes(b"old")
    result = CliRunner().invoke(command, ["checkpoint", "list", "-o", "json"])
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    assert "moved" in result.stderr
    assert (tmp_path / "pyrite-output/checkpoints/old.pkl").read_bytes() == b"old"


@pytest.mark.parametrize("corrupt", [False, True])
def test_remote_cross_filesystem_copy_is_verified(tmp_path, corrupt):
    from pyrite.remote import config as remote_config

    source = tmp_path / "checkpoints"
    source.mkdir()
    (source / "record.pkl").write_bytes(b"important")
    stat = 'stat() { case "$3" in checkpoints) echo 1 ;; *) echo 2 ;; esac; }; '
    copy = 'cp() { command cp "$@" && echo corrupt > "${@: -1}/record.pkl"; }; ' if corrupt else ""
    result = subprocess.run(
        ["bash", "-c", stat + copy + remote_config.remote_output_migration_command()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    target = tmp_path / "pyrite-output/checkpoints"
    if corrupt:
        assert result.returncode != 0
        assert not target.exists()
        assert (source / "record.pkl").read_bytes() == b"important"
    else:
        assert result.returncode == 0, result.stderr
        assert not source.exists()
        assert (target / "record.pkl").read_bytes() == b"important"
    assert not list((tmp_path / "pyrite-output").glob(".pyrite-migrate.*"))
