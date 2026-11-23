"""Generated SLURM batch scripts stay safe when SLURM requeues and reruns them."""

import os
import subprocess

import pytest

from pyrite.remote import config, scripts
from tests.remote._remote_test_helpers import _bash_or_skip


@pytest.fixture
def batch(monkeypatch, tmp_path):
    """Run the wrapped payload as SLURM job 101, restart ``restarts``."""
    bash = _bash_or_skip(tmp_path)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 101\n")
    reservations = tmp_path / "jobs" / "reservations"

    def run(payload='touch "$JOBDIR/ran"', *, restarts=0, stems=("hopg",)):
        script = scripts._slurm_batch_script(
            "j", payload, job_name="pyrite-j", reservation_stems=list(stems)
        )
        env = {**os.environ, "SLURM_JOB_ID": "101", "SLURM_RESTART_COUNT": str(restarts)}
        return subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    def reserve(stem, owner):
        (reservations / stem).mkdir(parents=True)
        (reservations / stem / "jobid").write_text(f"{owner}\n")

    return run, reserve, jobdir, reservations


def test_restart_reclaims_released_reservation_and_reruns(batch):
    run, _reserve, jobdir, reservations = batch
    payload = 'cat "$RESERVATIONS/hopg/jobid" > "$JOBDIR/owner_seen"; echo "done" > "$JOBDIR/state"'

    result = run(payload, restarts=1)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "owner_seen").read_text().strip() == "j"
    assert "requeued (restart 1)" in (jobdir / "log").read_text()
    assert "restarted: " in (jobdir / "meta").read_text()
    assert not (reservations / "hopg").exists()  # finish released it again


def test_restart_keeps_reservation_it_still_holds(batch):
    run, reserve, jobdir, _reservations = batch
    reserve("hopg", "j")

    result = run(restarts=2)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "ran").exists()


def test_restart_refuses_checkpoint_claimed_by_another_job(batch):
    run, reserve, jobdir, reservations = batch
    reserve("hopg", "other-job")

    result = run(restarts=1)

    assert result.returncode == 1
    assert not (jobdir / "ran").exists()
    state = (jobdir / "state").read_text()
    assert state.startswith("FAILED (requeued, but checkpoint hopg is now reserved by other-job)")
    assert (reservations / "hopg" / "jobid").read_text().strip() == "other-job"


def test_restart_reclaims_nothing_partially_on_conflict(batch):
    run, reserve, _jobdir, reservations = batch
    reserve("si", "other-job")

    result = run(restarts=1, stems=("hopg", "si"))

    assert result.returncode == 1
    assert not (reservations / "hopg").exists()  # reclaimed, then released by finish


def test_restart_honors_stop_request(batch):
    run, _reserve, jobdir, _reservations = batch
    (jobdir / "STOP").touch()

    result = run(restarts=1)

    assert result.returncode == 0
    assert not (jobdir / "ran").exists()
    assert (jobdir / "state").read_text().startswith("cancelled (stop requested)")


def test_restart_of_superseded_slice_leaves_successor_alone(batch):
    run, reserve, jobdir, reservations = batch
    (jobdir / "meta").write_text("slurm_job_id: 101\nslurm_job_id: 102\n")
    (jobdir / "state").write_text("running hopg [1/1]\n")
    reserve("hopg", "j")

    result = run(restarts=1)

    assert result.returncode == 0
    assert not (jobdir / "ran").exists()
    assert (jobdir / "state").read_text() == "running hopg [1/1]\n"
    assert (reservations / "hopg" / "jobid").read_text().strip() == "j"


def test_terminated_superseded_slice_keeps_successor_state_and_reservations(batch):
    run, reserve, jobdir, reservations = batch
    reserve("hopg", "j")
    payload = (
        'echo "queued slice 2" > "$JOBDIR/state"\n'
        'printf "slurm_job_id: 102\\n" >> "$JOBDIR/meta"\n'
        "kill -TERM $$"
    )

    result = run(payload)

    assert result.returncode == 143
    assert (jobdir / "state").read_text().startswith("queued slice 2")
    assert (reservations / "hopg" / "jobid").read_text().strip() == "j"


def test_first_run_termination_still_fails_and_releases(batch):
    run, reserve, jobdir, reservations = batch
    reserve("hopg", "j")

    result = run("kill -TERM $$")

    assert result.returncode == 143
    assert (jobdir / "state").read_text().startswith("FAILED (signal TERM)")
    assert not (reservations / "hopg").exists()
