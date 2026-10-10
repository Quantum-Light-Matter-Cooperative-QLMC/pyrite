"""Remote capture inventory, resumable verified pulls, and subset export."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyrite.cli.commands import remote_trajectories as cli
from pyrite.console import dashboard
from pyrite.console.dashboard import state as dashboard_state
from pyrite.console.json import job_kind
from pyrite.montecarlo.trajectories import (
    TrajectoryArtifactError,
    TrajectoryCapture,
    write_trajectory_artifact,
)
from pyrite.remote import _queue_scripts, config, jobs, trajectories, viewer
from pyrite.runs._trajectory_capture import CaptureSizeReport
from tests.helpers.cli import invoke
from tests.montecarlo.test_trajectory_capture import _read_vtp
from tests.montecarlo.test_trajectory_selection import _CASE, _secondaries


@pytest.fixture
def captures(tmp_path):
    root = tmp_path / "remote" / "captures"
    capture = TrajectoryCapture(str(root / "silicon"))
    cases = [dict(_CASE, Ne=3), dict(_CASE, name="other", Ne=3, E0_keV=30.0)]
    for case in cases:
        write_trajectory_artifact(capture.path_for(case), _secondaries(), case=case)
    return root, capture, cases


@pytest.fixture(autouse=True)
def remote_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path / "remote"))
    monkeypatch.setattr(config, "REMOTE_UV", "/bin/uv")
    monkeypatch.setattr(config, "HOST", "test-box")
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")


def _copy_rsync(monkeypatch):
    monkeypatch.setattr(trajectories.shutil, "which", lambda _name: "/bin/rsync")

    def copy(argv, **kwargs):
        assert all(flag in argv for flag in ("--partial", "--checksum", "--protect-args"))
        shutil.copyfile(argv[-2].split(":", 1)[1], argv[-1])

    monkeypatch.setattr(trajectories.transport, "_run", copy)
    return copy


def test_header_inventory_filters_without_reading_segments(captures, monkeypatch):
    root, capture, cases = captures
    monkeypatch.setattr(
        "pyrite.montecarlo.trajectories.read_trajectory_artifact",
        lambda *_a, **_k: pytest.fail("must read headers only"),
    )
    records = trajectories.inventory(root, "silicon", cases=["saved"], energies=[20.0])
    assert len(records) == 1 and records[0]["segments"] == 6 and records[0]["complete"]
    assert "sha256" not in records[0]
    assert records[0]["bytes"] == capture.path_for(cases[0]).stat().st_size
    assert trajectories.inventory(root, energies=[40.0]) == []
    capture.path_for(cases[0]).with_suffix(".h5.partial").write_text("interrupted")
    assert len(trajectories.inventory(root)) == 2


def test_inventory_refuses_incomplete_or_escaping_artifacts(captures, tmp_path):
    root, capture, cases = captures
    with h5py.File(capture.path_for(cases[0]), "r+") as handle:
        handle.attrs["complete"] = False
    with pytest.raises(TrajectoryArtifactError, match="incomplete"):
        trajectories.inventory(root)
    with pytest.raises(ValueError, match="one directory name"):
        trajectories.inventory(root, "../outside")
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        trajectories.inventory(root, "escape")


def test_pull_resumes_staging_and_installs_byte_identical_files(captures, tmp_path, monkeypatch):
    root, capture, cases = captures
    records = trajectories.inventory(root, "silicon", cases=["saved"], hashes=True)
    destination = tmp_path / "local"
    copy = _copy_rsync(monkeypatch)
    calls = []

    def interrupted(argv, **kwargs):
        staging = Path(argv[-1])
        calls.append(staging)
        if len(calls) == 1:
            staging.write_bytes(capture.path_for(cases[0]).read_bytes()[:100])
            raise subprocess.CalledProcessError(23, argv)
        assert staging.read_bytes() == capture.path_for(cases[0]).read_bytes()[:100]
        copy(argv, **kwargs)

    monkeypatch.setattr(trajectories.transport, "_run", interrupted)
    with pytest.raises(ValueError, match="rerun to resume"):
        trajectories.pull_files(records, root, destination)
    assert not (destination / records[0]["path"]).exists()
    paths = trajectories.pull_files(records, root, destination)
    assert calls[0] == calls[1] and not calls[0].exists()
    assert Path(paths[0]).read_bytes() == capture.path_for(cases[0]).read_bytes()
    assert len(list(destination.rglob("*.h5"))) == 1
    monkeypatch.setattr(
        trajectories.transport, "_run", lambda *_a, **_k: pytest.fail("already local")
    )
    assert trajectories.pull_files(records, root, destination) == paths


def test_pull_integrity_failure_preserves_existing_destination(captures, tmp_path, monkeypatch):
    root, _, _ = captures
    records = trajectories.inventory(root, hashes=True)[:1]
    destination = tmp_path / "local"
    target = destination / records[0]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous local file")
    _copy_rsync(monkeypatch)
    monkeypatch.setattr(
        trajectories.transport, "_run", lambda argv, **_k: Path(argv[-1]).write_bytes(b"broken")
    )
    with pytest.raises(ValueError, match="integrity check failed"):
        trajectories.pull_files(records, root, destination, overwrite=True)
    assert target.read_bytes() == b"previous local file" and not list(
        destination.rglob("*.partial")
    )


def test_pull_checks_all_conflicts_before_any_transfer(captures, tmp_path, monkeypatch):
    root, _, _ = captures
    records = trajectories.inventory(root, hashes=True)
    destination = tmp_path / "local"
    conflict = destination / records[-1]["path"]
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"conflict")
    _copy_rsync(monkeypatch)
    monkeypatch.setattr(
        trajectories.transport, "_run", lambda *_a, **_k: pytest.fail("must preflight all")
    )
    with pytest.raises(ValueError, match="pass --overwrite"):
        trajectories.pull_files(records, root, destination)
    assert not (destination / records[0]["path"]).exists()


def test_pull_checks_complete_header_even_with_matching_sha(captures, tmp_path, monkeypatch):
    root, capture, cases = captures
    path = capture.path_for(cases[0])
    with h5py.File(path, "r+") as handle:
        handle.attrs["complete"] = False
    record = dict(
        path=str(path.relative_to(root)),
        bytes=path.stat().st_size,
        sha256=trajectories.transport._local_sha256(path),
    )
    _copy_rsync(monkeypatch)
    with pytest.raises(TrajectoryArtifactError, match="incomplete"):
        trajectories.pull_files([record], root, tmp_path / "local")
    assert not (tmp_path / "local" / record["path"]).exists()


@pytest.mark.parametrize("relative", ["../escape.h5", "/absolute.h5"])
def test_pull_rejects_untrusted_inventory_paths(tmp_path, monkeypatch, relative):
    _copy_rsync(monkeypatch)
    with pytest.raises(ValueError, match="invalid remote artifact path"):
        trajectories.pull_files([dict(path=relative)], "/remote", tmp_path)


def test_capture_root_is_remote_and_shell_command_quotes_paths():
    base = config.remote_dir()
    root = trajectories.capture_root("captures/space ; ' name")
    assert root.startswith(base + "/")
    assert trajectories.capture_root() == base + "/pyrite-output/trajectories"
    cmd = trajectories._host_command("inventory", dict(root=root, cases=["a; touch /tmp/no"]))
    assert "PYRITE_REMOTE_DIR=" in cmd
    assert json.loads(shlex.split(cmd)[-1])["root"] == root
    for invalid in ("../../elsewhere", "/elsewhere"):
        with pytest.raises(ValueError, match="inside the remote checkout"):
            trajectories.capture_root(invalid)


@pytest.mark.parametrize("chunked", [False, True])
def test_generated_queue_captures_and_preflights_on_host(chunked):
    builder = _queue_scripts._chunked_queue_script if chunked else _queue_scripts._queue_script
    kwargs = {"chunk_minutes": 10} if chunked else {}
    root = trajectories.capture_root("captures/space ; ' name")
    script = builder(
        "job", ["hopg"], False, 0, trajectories=root, overwrite_trajectories=True, **kwargs
    )
    assert "--trajectories " in script and "--overwrite-trajectories" in script
    assert "export PYRITE_TRAJECTORY_SIZE_REPORT=1" in script
    if shutil.which("bash"):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert f"trajectories: {root}" in _queue_scripts._queue_metadata(
        "job", ["hopg"], False, 0, trajectories=root
    )


def test_size_report_warns_once_from_first_case(captures, capsys):
    _, capture, cases = captures
    report = CaptureSizeReport(capture, cases, threshold_bytes=1)
    for case in cases:
        report.completed(case)
    output = capsys.readouterr()
    assert output.out == "" and output.err.count("projected trajectory capture exceeds") == 1
    assert "B/segment" in output.err and "adaptive stopping" in output.err


def test_status_reports_capture_sizes():
    cmd = viewer._status_remote_command('JOB="job"', 0)
    assert "du -sb" in cmd and "emit TRAJECTORIES" in cmd
    sections = {"JOB": "job", "TRAJECTORIES": "100 /remote/captures/silicon/"}
    assert dashboard.marked_sections(dashboard_state._encode_sections(sections))[
        "TRAJECTORIES"
    ].startswith("100")
    assert "100 /remote/captures/silicon/" in dashboard.format_job_status(sections, 0)


def test_export_on_host_pulls_only_selected_vtp(captures, tmp_path, monkeypatch):
    root, _, _ = captures
    _copy_rsync(monkeypatch)
    monkeypatch.setattr(
        trajectories.transport,
        "_ssh_capture",
        lambda cmd: json.dumps(trajectories._export(json.loads(shlex.split(cmd)[-1]))),
    )
    paths = trajectories.remote_export(
        dict(
            root=str(root),
            stem="silicon",
            cases=["saved"],
            selection=dict(histories=[1], tracks=[], first=None, sample=None, seed=0),
        ),
        tmp_path / "vtk",
    )
    assert len(paths) == 1 and Path(paths[0]).suffix == ".vtp"
    piece, arrays = _read_vtp(Path(paths[0]))
    assert piece.get("NumberOfLines") == "3"
    np.testing.assert_array_equal(arrays["electron_id"], [1, 1, 1])
    assert not list((tmp_path / "vtk").rglob("*.h5"))


def test_score_submits_to_slurm_without_scoring_on_login_node(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        trajectories, "remote_inventory", lambda *_a, **_k: [dict(path="silicon/a.h5")]
    )
    monkeypatch.setattr(
        jobs,
        "_stage_job_script",
        lambda job, stems, upload, script: calls.update(job=job, stems=stems, script=script),
    )
    monkeypatch.setattr(jobs, "_submit_staged_job", lambda *_a: "123")
    job = trajectories.submit_score(dict(stem="silicon"))
    assert calls["job"] == job and calls["stems"] == ["silicon"]
    assert (
        "#SBATCH --gres=" in calls["script"]
        and "pyrite.remote.trajectories score" in calls["script"]
    )
    assert job_kind("trajectory-score") == "recompute"
    assert "trajectory spectrum replay" in dashboard.mode_summary("kind: trajectory-score")
    assert "completed: silicon" in calls["script"]


def test_pull_cli_previews_without_hashing_or_transfer(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        trajectories,
        "remote_inventory",
        lambda *a, **kw: (
            seen.update(args=a, kwargs=kw)
            or [dict(path="silicon/a.h5", bytes=99, segments=5, energy_keV=20)]
        ),
    )
    monkeypatch.setattr(
        trajectories, "pull_files", lambda *_a, **_k: pytest.fail("preview cannot transfer")
    )
    result = invoke(cli.command, ["pull", "silicon", "--case", "a", "--energy", "20"])
    assert result.exit_code == 0, result.output
    assert "99 bytes" in result.stdout and "Preview only" in result.stdout
    assert seen["kwargs"] == {"hashes": False} and seen["args"][2:] == (("a",), (20.0,))


@pytest.mark.parametrize(
    "args, message",
    [
        (["export"], "select --history"),
        (["export", "--first", "1", "--sample", "2"], "mutually exclusive"),
        (["export", "--first", "1", "--seed", "0"], "--seed requires --sample"),
        (["score"], "score requires one exact STEM"),
        (["pull", "../escape"], "STEM must be one directory name"),
        (["pull", "--energy", "nan"], "finite"),
    ],
)
def test_cli_rejects_invalid_selection_before_ssh(monkeypatch, args, message):
    monkeypatch.setattr(
        trajectories.transport,
        "_ssh_capture",
        lambda *_a: pytest.fail("invalid selection cannot reach SSH"),
    )
    result = invoke(cli.command, args)
    assert result.exit_code == 2 and message in result.stderr
