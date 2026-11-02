"""Additional remote submission tests."""

import os
import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as remote_cli
from pyrite.remote import config, lifecycle, scripts, state, transport, viewer
from pyrite.remote import queue as lifecycle_queue
from tests.remote._remote_test_helpers import (
    _bash_or_skip,
    _quiet_submission,
    _remote_main,
)


def test_zhai_start_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_zhai_queue()


def test_foreground_check_submits_then_attaches_and_pulls(monkeypatch):
    events = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: events.append("start") or "j")
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True, raising=False)
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: events.append("pull"))

    remote_cli.remote_check(no_sync=True)

    assert events == ["start", ("attach", "j"), "pull"]


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(scripts._queue_metadata("j", ["hopg"], False, None), id="scan"),
        pytest.param(scripts._zhai_queue_metadata("j", 1, 2, 3), id="zhai"),
        pytest.param(scripts._rebrem_queue_metadata("j", ["hopg"], 10, 25.0, False), id="rebrem"),
        pytest.param(scripts._recxr_queue_metadata("j", ["hopg"], 10, 25.0, False), id="reline"),
    ],
)
def test_submitted_job_meta_carries_the_synced_code_stamp(monkeypatch, tmp_path, metadata):
    """Every queue kind fixes its code identity when it is queued, read from the
    box's own stamp so a --no-sync submit records the code actually there."""
    bash = _bash_or_skip(tmp_path)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    (tmp_path / config.SYNC_STAMP_NAME).write_text(
        "code_digest: abc123\ncode_revision: feedface\ncode_dirty: False\n"
        "code_synced_at: 2026-09-19T14:14:00+00:00\ncode_source: laptop:/home/user/dev/pyrite\n"
    )
    jobdir = tmp_path / "jobs" / "j"
    jobdir.parent.mkdir(parents=True)

    command = scripts._write_job_script_command(jobdir.as_posix(), metadata)
    result = subprocess.run(
        [bash, "-c", command], input="#!/bin/sh\n", capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    written = (jobdir / "meta").read_text()
    assert written.startswith(metadata)
    assert "code_digest: abc123\n" in written
    assert "code_revision: feedface\n" in written


def test_submission_without_a_stamped_checkout_still_writes_job_meta(monkeypatch, tmp_path):
    """A checkout predating code stamping must submit, not fail; the missing
    stamp is what `sync` later reads as unverifiable."""
    bash = _bash_or_skip(tmp_path)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    jobdir = tmp_path / "jobs" / "j"
    jobdir.parent.mkdir(parents=True)
    metadata = scripts._queue_metadata("j", ["hopg"], False, None)

    command = scripts._write_job_script_command(jobdir.as_posix(), metadata)
    result = subprocess.run(
        [bash, "-c", command], input="#!/bin/sh\n", capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert (jobdir / "meta").read_text() == metadata


def test_cli_start_chunk_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        lifecycle,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    _remote_main(
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--dry-run",
            "--chunk-minutes",
            "0",
            "--parallel-materials",
            "3",
        ]
    )  # legal: monolithic
    assert (
        _remote_main(["run", "standard", "-m", "hopg", "--dry-run", "--parallel-materials", "3"])
        == 2
    )  # illegal: chunked default


def test_start_writes_static_metadata_before_sbatch(monkeypatch):
    uploads = []
    submissions = []
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: submissions.append(command))
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: submissions.append(["ssh", "-n", remote.HOST, command]) or "48291\n",
    )

    remote.start_queue(
        ["hopg"], quick=True, workers=3, parallel_materials=3, chunk_minutes=0, no_sync=True
    )

    upload_command = uploads[0][0][0][-1]
    assert "job: " in upload_command
    assert "materials: hopg" in upload_command
    assert "quick: True" in upload_command
    assert "workers: 3" in upload_command
    assert "parallel_materials: 3" in upload_command
    assert "chunk_minutes: 0" in upload_command
    assert "progress_dashboard: True" in upload_command
    assert "started:" not in upload_command
    assert "started: $(date -Is)" in uploads[0][1]["input"].decode()
    assert "materials: hopg" not in uploads[0][1]["input"].decode()
    assert 'mkdir "$R/$stem"' in submissions[0][-1]
    assert "sbatch --parsable" in submissions[1][-1]
    assert "--nice=10000" not in submissions[1][-1]  # monolithic: full priority


def test_start_queue_rejects_parallel_materials_above_four():
    with pytest.raises(SystemExit, match="between 1 and 4"):
        remote.start_queue(["hopg"], parallel_materials=5, chunk_minutes=0, dry_run=True)


def test_real_chunked_submission_carries_the_nice_flag(monkeypatch):
    """The LIVE submission path (not just dry-run) must emit --nice=10000:
    start_queue delegates to _submit_staged_job, which rebuilds the sbatch
    command itself, so the courtesy flag has to survive that hop too."""
    uploads = []
    submissions = []
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *args, **kwargs: uploads.append((args, kwargs))
    )
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: submissions.append(command) or "48291\n"
    )

    remote.start_queue(["hopg"], no_sync=True)  # chunked is the default

    assert "chunk_minutes: 10.0" in uploads[0][0][0][-1]
    sbatch_commands = [command for command in submissions if "sbatch" in command]
    assert sbatch_commands, "the live path must reach sbatch"
    assert all("sbatch --parsable --nice=10000" in command for command in sbatch_commands)


def test_ambiguous_submit_failure_inspects_queued_state_and_keeps_reservations(monkeypatch):
    commands = []
    captures = []
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)

    def capture(command):
        captures.append(command)
        if "sbatch --parsable" in command:
            raise SystemExit("ssh command failed (exit 255)")
        return "queued\n"

    monkeypatch.setattr(transport, "_ssh_capture", capture)

    with pytest.raises(SystemExit, match="ssh command failed"):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(captures) == 2
    assert "slurm_job_id" in captures[-1]
    assert 'cat "$D/state"' in captures[-1]
    assert len(commands) == 1  # reservation remains while queued submission is ambiguous


def test_profile_submit_names_job_after_profile(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_profile_jobdirs", lambda _profile: set())

    jobid = remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")

    assert jobid == "sub_100keV"


def test_profile_submit_suffixes_when_finished_run_holds_the_name(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(
        state,
        "_profile_jobdirs",
        lambda _profile: {"sub_100keV", "sub_100keV-2"},
    )

    jobid = remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")

    assert jobid == "sub_100keV-3"


def test_profile_submit_refuses_while_same_profile_is_live(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [("sub_100keV", False, ["hopg"])])
    monkeypatch.setattr(state, "_job_profiles", lambda jobids: {"sub_100keV": "sub_100keV"})

    with pytest.raises(SystemExit, match="already has a live job"):
        remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")


def test_submit_command_with_reservations_is_valid_bash(monkeypatch, tmp_path):
    """The owner-checked cleanup fragment must not break submission parsing."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    reservation = tmp_path / "jobs" / "reservations" / "hopg"
    jobdir.mkdir(parents=True)
    reservation.mkdir(parents=True)
    (reservation / "jobid").write_text("j\n")
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\nprintf '48291\\n'\n")
    fake_sbatch.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        [bash, "-c", remote._submit_slurm_command("j", ["hopg"])],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "48291"
    assert "slurm_job_id: 48291" in (jobdir / "meta").read_text()


def test_chunk_minutes_zero_emits_monolithic_script(monkeypatch, capsys):
    monkeypatch.setattr(
        lifecycle,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    remote.start_queue(["hopg"], dry_run=True, chunk_minutes=0)

    out = capsys.readouterr().out
    assert "#SBATCH --time=UNLIMITED" in out
    assert "parallel_materials=1" in out
    assert "--max-minutes" not in out


def test_parallel_materials_rejected_in_chunked_mode(monkeypatch):
    with pytest.raises(SystemExit, match="chunk-minutes 0"):
        remote.start_queue(["hopg"], dry_run=True, parallel_materials=2)
