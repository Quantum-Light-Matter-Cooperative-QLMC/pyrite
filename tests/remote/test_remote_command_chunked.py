"""Remote chunked tests."""

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
from tests.remote._remote_test_helpers import (
    _bash_or_skip,
)


def test_chunked_dry_run_emits_chain_script(monkeypatch, capsys):
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

    remote.start_queue(["hopg"], dry_run=True)  # chunked is the default

    out = capsys.readouterr().out
    assert "--max-minutes" in out
    assert "sbatch --parsable --nice=10000" in out  # resubmit + slice 0
    assert "queued slice" in out  # handoff state written BEFORE sbatch
    assert "FAILED (slice resubmission)" in out  # fail-closed resubmit
    assert '[ -f "$JOBDIR/STOP" ]' in out or '"$JOBDIR/STOP"' in out
    # STOP writes a terminal state so the finish trap releases reservations
    # instead of stamping FAILED over an operator-requested stop
    assert "cancelled (stop requested)" in out
    assert "failed: $m" in out  # hard-failure marker, never retried
    assert "#SBATCH --time=30" in out  # 3 x 10 min backstop
    assert '"queued slice"*' in out  # trap handoff case
    # handoff must not release reservations: the trap's release happens only in
    # terminal branches -- assert the handoff case body is empty (';;' right after)
    assert '"queued slice"*) ;;' in out


def test_chunked_script_hands_off_state_before_resubmitting(monkeypatch, tmp_path):
    """Chain state machine (design spec Component 2 step 4, state-first handoff
    ordering): the chain must write 'queued slice N+1' to state BEFORE calling
    sbatch, so a crash between the two still leaves a non-terminal state for the
    EXIT trap (a FAILED stamp there would race a next slice that really is
    pending). Runs the generated chunk payload under bash with a fake sbatch
    that snapshots state at the instant it is invoked."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 100\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        'case "$*" in\n'
        "  *hopg*) exit 75 ;;\n"  # stays unresolved: budget hit, not a failure
        "  *hbn*) exit 0 ;;\n"
        "esac\n"
        "exit 9\n"  # unexpected material: fail loudly rather than silently pass
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    witness = tmp_path / "witness"
    state_path = (jobdir / "state").as_posix()
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        f'printf \'argv: %s\\nstate: \' "$*" > "{witness.as_posix()}"\n'
        f'cat "{state_path}" >> "{witness.as_posix()}"\n'
        "echo 4242\n"
    )
    fake_sbatch.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg", "hbn"], quick=False, workers=None, chunk_minutes=10.0
    )

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("queued slice 2")
    witness_text = witness.read_text()
    assert "--parsable" in witness_text
    assert "--nice=10000" in witness_text
    assert "state: queued slice 2" in witness_text  # sbatch saw the handoff state first
    assert (jobdir / "meta").read_text().rstrip("\n").endswith("slurm_job_id: 4242")


def test_chunked_script_skips_failed_materials_and_terminates_without_resubmitting(
    monkeypatch, tmp_path
):
    """Persistent-failure regression (design spec Testing / 'Chain state
    machine'): a material already carrying a 'failed:' marker in the log (left
    by an earlier slice) must be skipped, not retried, and once every material
    is completed-or-failed the chain must reach a terminal state and NEVER call
    sbatch again -- otherwise a permanently-crashing material resubmits an
    endless chain of short SLURM jobs."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "log").write_text("failed: hopg\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        'case "$*" in\n'
        "  *hbn*) exit 0 ;;\n"
        "esac\n"
        "exit 9\n"  # hopg must never be re-scanned once it is marked failed
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "sbatch_called"
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\necho 4242\n")
    fake_sbatch.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg", "hbn"], quick=False, workers=None, chunk_minutes=10.0
    )

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("FAILED (1 of")
    assert not marker.exists(), "a fully-resolved chain must not resubmit"
    log = (jobdir / "log").read_text()
    assert log.count("failed: hopg") == 1
    assert "completed: hbn" in log


@pytest.mark.parametrize(
    "sbatch_body",
    ["#!/bin/sh\nexit 1\n", "#!/bin/sh\necho not-a-sid\n"],
    ids=["sbatch-fails", "sbatch-returns-garbage"],
)
def test_chunked_script_fails_closed_when_resubmission_fails(monkeypatch, tmp_path, sbatch_body):
    """Fail-closed regression (design spec Component 2 step 4): if sbatch
    itself fails, or 'succeeds' but prints something that isn't a numeric SID,
    the chain must not limp along with a dangling 'queued slice' state -- it
    must stamp FAILED (slice resubmission) and exit nonzero so the EXIT trap
    releases reservations and attach can terminate instead of polling forever."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        "exit 75\n"  # budget hit every slice: hopg stays unresolved
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(sbatch_body)
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    # Every retry fails the same way; skip the production backoff between them.
    monkeypatch.setattr(_queue_scripts, "_SBATCH_RETRY_SECONDS", 0)
    script = remote._chunked_queue_script(
        "j", ["hopg"], quick=False, workers=None, chunk_minutes=10.0
    )

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode != 0
    assert (jobdir / "state").read_text().startswith("FAILED (slice resubmission)")


def test_chunked_script_honors_stop_sentinel_without_resubmitting(monkeypatch, tmp_path):
    """Stop-must-not-zombie-the-chain regression (design spec 3b): once
    _stop_jobid drops a STOP sentinel into the jobdir, the running slice's
    handoff step must see it and cancel instead of resubmitting the next
    slice."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "STOP").write_text("")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\ncase "$*" in *"tables verify"*) exit 0 ;; esac\nexit 75\n'
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "sbatch_called"
    (bin_dir / "sbatch").write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg"], quick=False, workers=None, chunk_minutes=10.0
    )

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("cancelled (stop requested)")
    assert not marker.exists(), "STOP must prevent resubmission"


def test_chunked_script_marks_hard_failure_and_never_retries(monkeypatch, tmp_path):
    """Hard-failure marker regression (design spec Component 2 step 2): a
    material whose scan exits with a code other than 0 or 75 is a real crash,
    not a budget timeout -- the chain must log it as 'failed:' (so later slices
    skip it) instead of resubmitting the same doomed material forever."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\ncase "$*" in *"tables verify"*) exit 0 ;; esac\nexit 7\n'
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg"], quick=False, workers=None, chunk_minutes=10.0
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    log = (jobdir / "log").read_text()
    assert "WARNING: scan failed for hopg (exit 7); will not retry" in log
    assert "failed: hopg" in log
    assert (jobdir / "state").read_text().startswith("FAILED (1 of")


@pytest.mark.parametrize(
    "builder",
    [
        lambda: scripts._chunked_queue_script("j", ["hopg"], False, None, 10),
        lambda: scripts._rebrem_chunked_queue_script("j", ["hopg"], None, None, False, 10),
        lambda: scripts._recxr_chunked_queue_script("j", ["hopg"], None, None, False, 10),
    ],
    ids=["scan", "rebrem", "reline"],
)
@pytest.mark.parametrize(
    ("rc", "prelimit", "progress", "resumes"),
    [
        (143, True, True, True),
        (143, False, True, False),
        (143, True, False, False),
        (7, False, True, False),
    ],
)
def test_chunked_prelimit_signal_resumes_only_143_with_checkpoint_progress(
    monkeypatch, tmp_path, builder, rc, prelimit, progress, resumes
):
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 100\n")
    if not progress:
        (jobdir / "progress").mkdir()
        (jobdir / "progress" / "hopg.json").write_text("{")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        + (
            f"printf '%s\\n' '{{\"cached_cases\":0,\"completed_new_cases\":1}}' > '{jobdir}/progress/hopg.json'\n"
            if progress
            else ""
        )
        + ('kill -USR1 "$(ps -o ppid= -p "$PPID")"\n' if prelimit else "")
        + f"exit {rc}\n"
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "sbatch_called"
    (bin_dir / "sbatch").write_text(f"#!/bin/sh\ntouch '{marker}'\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REMOTE_UV", str(fake_uv))
    result = subprocess.run(
        [bash, "-c", builder()],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )
    assert marker.exists() is resumes, (result.stderr, (jobdir / "log").read_text())
    log = (jobdir / "log").read_text()
    if resumes:
        assert result.returncode == 0
        assert "resumable timeout: hopg" in log
        assert "failed: hopg" not in log
        assert (jobdir / "state").read_text().startswith("queued slice 2")
        assert (jobdir / "timeouts").read_text().startswith("resumable timeout: hopg")
    elif rc == 143 and prelimit:
        assert result.returncode != 0
        assert (jobdir / "state").read_text().startswith("FAILED (slice timeout without progress)")
    else:
        assert "failed: hopg" in log


def test_chunked_no_progress_slices_stop_after_three(monkeypatch, tmp_path):
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 100\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\ncase "$*" in *"tables verify"*) exit 0 ;; esac\nexit 75\n'
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "submissions"
    (bin_dir / "sbatch").write_text(f"#!/bin/sh\necho called >> '{marker}'\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REMOTE_UV", str(fake_uv))
    script = scripts._chunked_queue_script("j", ["hopg"], False, None, 10)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    results = [
        subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)
        for _ in range(3)
    ]
    assert [result.returncode for result in results] == [0, 0, 1]
    assert len(marker.read_text().splitlines()) == 2
    assert (
        (jobdir / "state").read_text().startswith("FAILED (3 slices without checkpoint progress)")
    )
