"""Remote stop tests."""

import os
import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from pyrite.remote import jobs as lifecycle_jobs


def test_stop_waits_for_scheduler_cancellation_before_releasing_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(state, "_slurm_state", lambda _scheduler_id: "RUNNING")
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))

    remote._stop_jobid("j")

    command = commands[0][-1]
    assert "scancel 48291" in command
    assert "STATE=$(squeue -h -j 48291 -o '%T' 2>&1)" in command
    assert 'if [ "$STATUS" -ne 0 ]' in command
    assert '*"Invalid job id specified"*) break' in command
    assert 'cat "$d/jobid"' in command


def test_stop_retains_reservations_if_squeue_cancellation_query_fails(monkeypatch, tmp_path):
    """A failed scheduler query must not look like successful cancellation."""
    jobdir = tmp_path / "jobs" / "j"
    reservation = tmp_path / "jobs" / "reservations" / "hopg"
    jobdir.mkdir(parents=True)
    reservation.mkdir(parents=True)
    (jobdir / "state").write_text("running\n")
    (reservation / "jobid").write_text("j\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "scancel").write_text("#!/bin/sh\nexit 0\n")
    (commands / "squeue").write_text("#!/bin/sh\nexit 7\n")
    (commands / "scancel").chmod(0o755)
    (commands / "squeue").chmod(0o755)

    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(state, "_slurm_job_id", lambda _jobid: "48291")
    monkeypatch.setattr(state, "_slurm_state", lambda _scheduler_id: "RUNNING")

    def run_remote(command, **_kwargs):
        subprocess.run(
            ["bash", "-c", command[-1]],
            check=True,
            env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        )

    monkeypatch.setattr(transport, "_run", run_remote)

    with pytest.raises(subprocess.CalledProcessError):
        remote._stop_jobid("j")

    assert reservation.exists()
    assert (jobdir / "state").read_text().startswith("cancelling")


def test_stop_jobid_uses_scancel_not_kill(monkeypatch):
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))

    remote._stop_jobid("j")

    assert "scancel 48291" in commands[0][-1]
    assert "kill -TERM" not in commands[0][-1]
    assert "scancel 48291 || exit" in commands[0][-1]


def test_stop_jobid_rejects_legacy_job_without_scheduler_id(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("must not scancel")
    )

    with pytest.raises(SystemExit, match="not an active SLURM job"):
        remote._stop_jobid("j")


def test_stop_materials_resolve_unique_live_jobs(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(lifecycle_jobs, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(["wse2", "mose2"])

    assert stopped == ["job2"]


def test_stop_all_stops_every_live_job(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", True, ["mose2"])],
    )
    monkeypatch.setattr(lifecycle_jobs, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(all_jobs=True)

    assert stopped == ["job1", "job2"]


def test_stop_rejects_bad_material_before_live_job_lookup(monkeypatch):
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: pytest.fail("must validate before checking live jobs"),
    )

    with pytest.raises(SystemExit):
        remote.stop_jobs(["bad;material"])


def test_stop_profile_matches_live_job_metadata(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(
        state,
        "_job_profiles",
        lambda jobids: {"job1": "sub_100keV", "job2": "standard"},
    )
    monkeypatch.setattr(lifecycle_jobs, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(profile="sub_100keV")

    assert stopped == ["job1"]


def test_stop_profile_without_live_match_errors(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(state, "_job_profiles", lambda jobids: {"job1": "standard"})

    with pytest.raises(SystemExit, match="no live job found for profile"):
        remote.stop_jobs(profile="sub_100keV")


def test_stop_profile_rejects_materials_and_all(monkeypatch):
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: pytest.fail("must reject the combination before checking live jobs"),
    )
    with pytest.raises(SystemExit):
        remote.stop_jobs(["hopg"], profile="sub_100keV")
    with pytest.raises(SystemExit):
        remote.stop_jobs(all_jobs=True, profile="sub_100keV")


def test_stop_writes_stop_sentinel_before_scancel(monkeypatch):
    commands = []
    monkeypatch.setattr(state, "_slurm_job_id", lambda jobid: "123")
    monkeypatch.setattr(state, "_slurm_state", lambda sid: "RUNNING")
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: commands.append(cmd[-1]))
    remote._stop_jobid("20260717-abc")
    (cmd,) = commands
    assert cmd.index("STOP") < cmd.index("scancel")
