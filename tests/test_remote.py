"""cxr_mc.remote (``cxr remote``): material-name validation and the
detached-queue runner generation.

These are pure-string/logic checks (no ssh), so they run anywhere. The one
exception is the clear-listing regression test, which executes the box-side
shell snippet under a local bash (skipped when bash is unavailable)."""

import argparse
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from cxr_mc import remote


def test_check_materials_accepts_crystal_keys():
    remote._check_materials(["mose2", "hopg", "mos2-on-sio2-si", "silicon"])


def test_check_materials_rejects_shell_safe_unknown_catalog_keys():
    with pytest.raises(SystemExit, match="unknown material"):
        remote._check_materials(["not_in_catalog"])


@pytest.mark.parametrize("bad", ["rm -rf /", "a;b", "../etc", "a b", "", "m&n"])
def test_check_materials_rejects_injection(bad):
    with pytest.raises(SystemExit):
        remote._check_materials([bad])


def test_check_shell_tokens_preserves_checkpoint_and_synthetic_stems():
    remote._check_shell_tokens(["mose2_quick", "zhai", "20260101-000000"])


def test_queue_script_has_per_material_scan_calls():
    s = remote._queue_script("20260101-000000", ["mose2", "wse2"], quick=True, workers=8)
    assert "scan.py" in s
    assert "--quick" in s and "--workers 8" in s
    assert "mose2" in s and "wse2" in s
    assert "20260101-000000" in s  # job id is embedded
    assert "mats=(mose2 wse2)" in s  # bash array drives the loop
    assert "parallel_materials=2" in s
    assert 'mkdir -p "$JOBDIR/progress"' in s
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    assert "--no-progress" in s
    assert "/pid" not in s


def test_queue_script_accepts_three_parallel_materials():
    script = remote._queue_script(
        "j", ["hopg", "hbn", "hfse2"], quick=False, workers=4, parallel_materials=3
    )

    assert "parallel_materials=3" in script
    assert "wait -n" in script


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s


def test_queue_script_warns_and_continues_after_a_material_fails():
    script = remote._queue_script("j", ["hopg", "hbn"], quick=False, workers=None)

    assert "WARNING: scan failed for $m; continuing" in script
    assert "wait -n" in script
    assert "done with $failures warning(s)" in script


def test_slurm_batch_script_requests_the_lab_gpu_profile():
    script = remote._slurm_batch_script("j", "echo payload", job_name="cxr-j")

    assert "#SBATCH --job-name=cxr-j" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=UNLIMITED" in script
    assert "module purge 2>/dev/null || true" in script
    assert "module load cuda openmpi hdf5 2>/dev/null || true" in script
    assert "echo payload" in script


def test_submit_command_uses_sbatch_parsable_and_records_scheduler_id():
    command = remote._submit_slurm_command("20260715-120000")

    assert "sbatch --parsable" in command
    assert "slurm_job_id" in command
    assert "jobs/20260715-120000/run.sh" in command
    assert "nohup" not in command and "setsid" not in command


def test_same_second_starters_use_unique_exclusive_job_directories(monkeypatch):
    """Two submitters must not overwrite a shared second-resolution job dir."""

    real_datetime = remote.datetime.datetime

    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 7, 15, 12, 0, 0)

    suffixes = iter(["a" * 32, "b" * 32])
    uploads = []
    monkeypatch.setattr(remote.datetime, "datetime", FixedDatetime)
    monkeypatch.setattr(
        remote,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(suffixes))),
        raising=False,
    )
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda command, **_kwargs: uploads.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")

    first = remote.start_queue(["hopg"], no_sync=True)
    second = remote.start_queue(["hbn"], no_sync=True)

    assert first == "20260715-120000-aaaaaaaa"
    assert second == "20260715-120000-bbbbbbbb"
    assert first != second
    assert all("mkdir -p" not in upload[-1] for upload in uploads)
    assert all("mkdir '" in upload[-1] for upload in uploads)


def test_checkpoint_reservation_rejects_a_second_stager_for_the_same_stem(monkeypatch, tmp_path):
    """The remote ``mkdir`` lock closes the gap between a busy check and sbatch."""
    monkeypatch.setattr(remote, "REMOTE_DIR", str(tmp_path))

    first = subprocess.run(
        ["bash", "-c", remote._reserve_checkpoint_stems_command("first", ["hopg"])],
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        ["bash", "-c", remote._reserve_checkpoint_stems_command("second", ["hopg"])],
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0
    assert second.returncode != 0
    assert "refusing to stage" in second.stderr
    assert (tmp_path / "jobs" / "reservations" / "hopg" / "jobid").read_text().strip() == "first"


def test_slurm_batch_script_releases_its_checkpoint_reservations():
    script = remote._slurm_batch_script(
        "j", "echo payload", job_name="cxr-j", reservation_stems=["hopg"]
    )

    assert "release_reservations" in script
    assert '"$RESERVATIONS/hopg/jobid"' in script
    assert '"$JOBID"' in script


def test_submit_failure_releases_its_checkpoint_reservations():
    command = remote._submit_slurm_command("j", ["hopg"])

    assert "FAILED (sbatch submission)" in command
    assert '"$R/hopg/jobid"' in command


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
    monkeypatch.setattr(remote, "REMOTE_DIR", str(tmp_path))
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


def test_stop_waits_for_scheduler_cancellation_before_releasing_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(remote, "_slurm_state", lambda _scheduler_id: "RUNNING")
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: commands.append(command))

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

    monkeypatch.setattr(remote, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(remote, "_slurm_job_id", lambda _jobid: "48291")
    monkeypatch.setattr(remote, "_slurm_state", lambda _scheduler_id: "RUNNING")

    def run_remote(command, **_kwargs):
        subprocess.run(
            ["bash", "-c", command[-1]],
            check=True,
            env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        )

    monkeypatch.setattr(remote, "_run", run_remote)

    with pytest.raises(subprocess.CalledProcessError):
        remote._stop_jobid("j")

    assert reservation.exists()
    assert (jobdir / "state").read_text().startswith("cancelling")


def test_remote_dry_run_describes_sbatch_submission(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    remote.start_queue(["hopg"], dry_run=True)

    out = capsys.readouterr().out
    assert "#SBATCH --partition=gpu" in out
    assert "sbatch --parsable" in out
    assert "nohup" not in out and "setsid" not in out


def test_chunked_dry_run_emits_chain_script(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
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


def test_chunk_minutes_zero_emits_monolithic_script(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    remote.start_queue(["hopg"], dry_run=True, chunk_minutes=0)

    out = capsys.readouterr().out
    assert "#SBATCH --time=UNLIMITED" in out
    assert "parallel_materials=2" in out
    assert "--max-minutes" not in out


def test_parallel_materials_rejected_in_chunked_mode(monkeypatch):
    with pytest.raises(SystemExit, match="chunk-minutes 0"):
        remote.start_queue(["hopg"], dry_run=True, parallel_materials=2)


def test_cli_start_chunk_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        remote, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    remote.main(
        ["start", "hopg", "--dry-run", "--chunk-minutes", "0", "--parallel-materials", "3"]
    )  # legal: monolithic
    with pytest.raises(SystemExit):
        remote.main(
            ["start", "hopg", "--dry-run", "--parallel-materials", "3"]
        )  # illegal: chunked default


def test_start_writes_static_metadata_before_sbatch(monkeypatch):
    uploads = []
    submissions = []
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: submissions.append(command))
    monkeypatch.setattr(
        remote,
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


def test_real_chunked_submission_carries_the_nice_flag(monkeypatch):
    """The LIVE submission path (not just dry-run) must emit --nice=10000:
    start_queue delegates to _submit_staged_job, which rebuilds the sbatch
    command itself, so the courtesy flag has to survive that hop too."""
    uploads = []
    submissions = []
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *args, **kwargs: uploads.append((args, kwargs))
    )
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda command: submissions.append(command) or "48291\n"
    )

    remote.start_queue(["hopg"], no_sync=True)  # chunked is the default

    assert "chunk_minutes: 10.0" in uploads[0][0][0][-1]
    sbatch_commands = [command for command in submissions if "sbatch" in command]
    assert sbatch_commands, "the live path must reach sbatch"
    assert all("sbatch --parsable --nice=10000" in command for command in sbatch_commands)


def test_start_reports_the_submitted_slurm_job_id(monkeypatch, capsys):
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(remote.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg"], no_sync=True)

    output = capsys.readouterr().out
    assert "· SUBMITTED" in output
    assert "SLURM" in output
    assert "48291" in output
    assert "cxr remote attach" in output
    assert "cxr remote status" in output


def test_interrupted_job_upload_releases_its_checkpoint_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: pytest.fail("must not submit"))

    with pytest.raises(KeyboardInterrupt):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(commands) == 2
    assert 'J="' in commands[-1][-1]
    assert 'if [ "$(cat "$R/hopg/jobid"' in commands[-1][-1]


def test_ambiguous_submit_failure_inspects_queued_state_and_keeps_reservations(monkeypatch):
    commands = []
    captures = []
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(remote.subprocess, "run", lambda *_args, **_kwargs: None)

    def capture(command):
        captures.append(command)
        if "sbatch --parsable" in command:
            raise SystemExit("ssh command failed (exit 255)")
        return "queued\n"

    monkeypatch.setattr(remote, "_ssh_capture", capture)

    with pytest.raises(SystemExit, match="ssh command failed"):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(captures) == 2
    assert "slurm_job_id" in captures[-1]
    assert 'cat "$D/state"' in captures[-1]
    assert len(commands) == 1  # reservation remains while queued submission is ambiguous


def test_slurm_job_id_reads_recorded_scheduler_id(monkeypatch):
    commands = []
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda command: commands.append(command) or "48291\n"
    )

    assert remote._slurm_job_id("j") == "48291"
    assert "slurm_job_id" in commands[0]


def test_slurm_state_queries_squeue(monkeypatch):
    commands = []
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda command: commands.append(command) or "RUNNING\n"
    )

    assert remote._slurm_state("48291") == "RUNNING"
    assert "squeue -h -j 48291 -o '%T'" in commands[0]


def test_slurm_state_treats_retired_ids_as_not_queued_but_surfaces_other_failures(
    monkeypatch, tmp_path
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required for generated remote-command regression")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"

    def run_remote(command):
        result = subprocess.run(
            [bash, "-c", command],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        )
        if result.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {result.returncode})")
        return result.stdout

    monkeypatch.setattr(remote, "_ssh_capture", run_remote)
    squeue.write_text(
        "#!/bin/sh\necho 'slurm_load_jobs error: Invalid job id specified' >&2\nexit 1\n"
    )
    squeue.chmod(0o755)
    assert remote._slurm_state("48291") is None

    squeue.write_text("#!/bin/sh\necho 'controller unavailable' >&2\nexit 7\n")
    with pytest.raises(SystemExit, match="ssh command failed"):
        remote._slurm_state("48291")


def test_live_jobs_queries_squeue_for_recorded_scheduler_ids(monkeypatch):
    commands = []
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda command: commands.append(command) or "j\tFalse\thopg\n"
    )

    assert remote._live_jobs() == [("j", False, ["hopg"])]
    assert "squeue" in commands[0]
    assert "slurm_job_id" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_live_jobs_skips_retired_scheduler_ids_without_masking_other_query_failures(monkeypatch):
    commands = []
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda command: commands.append(command) or "j\tFalse\thopg\n"
    )

    remote._live_jobs()

    assert "STATE=$(squeue -h -j $SID -o '%T' 2>&1)" in commands[0]
    assert '*"Invalid job id specified"*) continue' in commands[0]
    assert 'echo "could not query SLURM job $SID" >&2; exit "$STATUS"' in commands[0]


def test_job_status_reports_scheduler_state_not_process_liveness(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or (
                "@@JOB\nj\n@@META\njob: j\nmaterials: hopg\nquick: False\n"
                "chunk_minutes: 10\nslurm_job_id: 48291\n@@STATE\nrunning hopg\n"
                "@@SQUEUE\njob_id=48291|state=RUNNING\n"
            )
        ),
    )

    remote.job_status("j")

    output = capsys.readouterr().out
    assert "JOB j" in output
    assert "48291 · RUNNING" in output
    assert "hopg" in output
    assert "standard · chunked into 10 min slices" in output
    assert "squeue" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_job_status_verbose_adds_scheduler_allocation_fields(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or (
                "@@JOB\nj\n@@META\njob: j\nmaterials: hopg\nworkers: None\n"
                "slurm_job_id: 48291\n@@STATE\nrunning hopg\n@@SQUEUE\n"
                "job_id=48291|state=RUNNING|name=cxr-j|partition=gpu|elapsed=1:02|"
                "left=UNLIMITED|nodes=1|reason=None\n"
            )
        ),
    )

    remote.job_status("j", detail=1)

    output = capsys.readouterr().out
    assert "ALLOCATION" in output
    assert "Partition  gpu" in output
    assert "Elapsed    1:02" in output
    assert "Reason     None" in output
    assert "elapsed=%M" in commands[0]
    assert "reason=%R" in commands[0]


def test_job_status_double_verbose_renders_case_progress(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or (
                "@@JOB\nj\n@@META\njob: j\nmaterials: hopg\nslurm_job_id: 48291\n"
                "@@STATE\nrunning hopg\n@@SQUEUE\njob_id=48291|state=RUNNING\n@@PROGRESS\n"
                '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":2,"state":"running"}\n'
                "@@LOG\nlast log line\n"
            )
        ),
    )

    remote.job_status("j", detail=2)

    output = capsys.readouterr().out
    assert "CASE PROGRESS" in output
    assert "MATERIAL" in output
    assert "hopg" in output
    assert "3/5" in output
    assert "60%" in output
    assert "RECENT LOG (diagnostics only)" in output
    assert '{"material"' not in output
    assert "last log line" in output
    assert '"$D"/progress/*.json' in commands[0]
    assert 'tail -c 32768 "$D/log"' in commands[0]


def test_status_collapses_legacy_tqdm_history_to_latest_material_bar(monkeypatch, capsys):
    legacy_log = (
        "setup complete\n"
        "cases:   0%|          | 0/10 [00:00<?, ?it/s]\r"
        "cases:  50%|█████     | 5/10 [00:01<00:01, 4.0it/s]\r"
        "cases: 100%|██████████| 10/10 [00:02<00:00, 5.0it/s]\n"
        "next sweep\n"
        "cases:   0%|          | 0/5 [00:00<?, ?it/s]\r"
        "cases:  40%|████      | 2/5 [00:03<00:04, 1.4s/it]\r"
    )
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda _command: (
            "@@JOB\nj\n@@META\njob: j\nmaterials: hopg\nslurm_job_id: 48291\n"
            "@@STATE\nrunning hopg [1/1]\n@@SQUEUE\njob_id=48291|state=RUNNING\n"
            f"@@PROGRESS\n@@LOG\n{legacy_log}"
        ),
    )

    remote.job_status("j", detail=2)

    output = capsys.readouterr().out
    assert "HOPG" in output
    assert "2/5" in output
    assert "40%" in output
    assert "5/10" not in output
    assert "10/10" not in output
    assert "0/5" not in output
    assert "setup complete" in output
    assert "next sweep" in output
    assert "cases:" not in output


def test_failed_legacy_status_marks_incomplete_bar_as_last_batch():
    output = remote._legacy_progress(
        "cases: 40%|████      | 2/5 [00:03<00:04, 1.4s/it]\r",
        "FAILED (exit 1) now",
    )

    assert output is not None
    assert "last case batch" in output
    assert "×" in output
    assert "2/5" in output


def test_line_grid_bounds_status_uses_diagnostic_metadata(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda _command: (
            "@@JOB\nj\n@@META\njob: j\nkind: line-grid-bounds\n"
            "slice_minutes: 10\nenergies: 200,250,300\nslurm_job_id: 227\n"
            "@@STATE\nFAILED (signal TERM) now\n"
            "@@SQUEUE\njob_id=227|state=NOT_QUEUED\n@@PROGRESS\n@@LOG\n"
            "cases: 40%|████      | 2/5 [17:31<26:18, 526.18s/it]\r"
        ),
    )

    remote.job_status("j", detail=2)

    output = capsys.readouterr().out
    assert "Kind      line-grid-bounds" in output
    assert "Energies  200,250,300 keV" in output
    assert "Slice     10 min soft / 30 min hard" in output
    assert "Materials" not in output
    assert "Mode" not in output
    assert "CASE PROGRESS" not in output
    assert "last case batch" not in output


def test_status_cli_repeats_verbose_for_case_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "job_status", lambda jobid, detail: calls.append((jobid, detail)))

    remote.main(["status", "j", "-vv"])

    assert calls == [("j", 2)]


def test_attach_uses_stdin_closed_ssh_for_live_view(monkeypatch):
    runs = []
    monkeypatch.setattr(remote, "_job_metadata", lambda _jobid: "materials: hopg\n")
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.attach("20260101-000000")

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4
    assert "squeue" in runs[0][-1]
    assert "slurm_job_id" in runs[0][-1]
    assert "kill -0" not in runs[0][-1]
    assert "/pid" not in runs[0][-1]


def test_attach_retries_until_a_queued_job_creates_its_log(monkeypatch):
    runs = []
    monkeypatch.setattr(remote, "_job_metadata", lambda _jobid: "materials: hopg\n")
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.attach("20260101-000000")

    assert 'tail -n 50 -F --retry "$D/log"' in runs[0][-1]


def test_implicit_job_selection_excludes_checkpoint_reservations(monkeypatch, tmp_path):
    """Bare queue commands must not mistake jobs/reservations for a job."""
    commands = []
    monkeypatch.setattr(remote, "_ssh_capture", lambda command: commands.append(command) or "")

    remote.list_jobs()
    remote.job_status()
    remote.tail_logs()
    remote._latest_jobid()

    jobs = tmp_path / "jobs"
    job = jobs / "20260715-113910-b0240c4f"
    job.mkdir(parents=True)
    (job / "meta").write_text("slurm_job_id: 48291\n")
    (job / "state").write_text("running\n")
    (jobs / "reservations" / "hopg").mkdir(parents=True)
    list_command = commands[0].replace(
        f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs}"'
    )
    result = subprocess.run(["bash", "-c", list_command], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "20260715-113910-b0240c4f\t48291\t?\t-\trunning \n"
    assert all('[ -f "$d/meta" ] || continue' in command for command in commands)


def test_jobs_report_identifiers_materials_and_last_event(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda _command: "20260719-120000-abcd1234\t48291\tFalse\thopg hbn\trunning hbn [2/2]\n",
    )

    remote.list_jobs()

    output = capsys.readouterr().out
    assert "JOB" in output
    assert "SLURM" in output
    assert "MODE" in output
    assert "MATERIALS" in output
    assert "LAST EVENT" in output
    assert "20260719-120000-abcd1234" in output
    assert "48291" in output
    assert "standard" in output
    assert "hopg, hbn" in output
    assert "running hbn [2/2]" in output


def test_logs_identify_resolved_job_and_host(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: commands.append(command) or "LOG j · qlmc\n\nline\n",
    )

    remote.tail_logs("j")

    assert capsys.readouterr().out == "LOG j · qlmc\n\nline\n"
    assert 'printf "LOG %s · qlmc' in commands[0]


def test_attach_returns_false_when_the_viewer_is_interrupted(monkeypatch):
    monkeypatch.setattr(remote, "_job_metadata", lambda _jobid: "materials: hopg\n")
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda _cmd: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert remote.attach("20260101-000000") is False


def test_parse_progress_records_ignores_malformed_snapshots():
    valid = {
        "material": "hopg",
        "total_cases": 5,
        "cached_cases": 1,
        "completed_new_cases": 2,
        "state": "running",
    }
    invalid_count = {**valid, "material": "hbn", "completed_new_cases": 9}
    invalid_bool = {**valid, "material": "hbn", "total_cases": True}
    payload = "\n".join(
        [
            json.dumps(valid),
            "{partial",
            json.dumps(invalid_count),
            json.dumps(invalid_bool),
            "[]",
        ]
    )

    assert remote._parse_progress_records(payload) == {"hopg": valid}


def test_parse_progress_records_accepts_paused_state():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"paused"}'
    )
    records = remote._parse_progress_records(payload)
    assert records["hopg"]["state"] == "paused"


class _FakeProgressBar:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = 0
        self.description = kwargs["desc"]
        self.postfix = {}
        self.refreshes = 0
        self.closed = False

    def set_description_str(self, description, refresh=False):
        self.description = description

    def set_postfix(self, **kwargs):
        kwargs.pop("refresh", None)
        self.postfix = kwargs

    def refresh(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


def test_progress_bars_update_materials_independently():
    bars = {
        "hopg": _FakeProgressBar(total=None, desc="hopg: pending"),
        "hbn": _FakeProgressBar(total=None, desc="hbn: pending"),
    }
    records = remote._parse_progress_records(
        json.dumps(
            {
                "material": "hopg",
                "total_cases": 5,
                "cached_cases": 1,
                "completed_new_cases": 2,
                "state": "running",
            }
        )
    )

    remote._update_progress_bars(bars, records)

    assert bars["hopg"].n == 3
    assert bars["hopg"].total == 5
    assert bars["hopg"].postfix == {"cached": 1, "new": 2}
    assert bars["hbn"].n == 0
    assert bars["hbn"].description == "hbn: pending"


def test_progress_bars_close_terminal_material_once():
    bars = {"hopg": _FakeProgressBar(total=None, desc="HOPG · pending")}
    records = remote._parse_progress_records(
        json.dumps(
            {
                "material": "hopg",
                "total_cases": 5,
                "cached_cases": 1,
                "completed_new_cases": 4,
                "state": "done",
            }
        )
    )

    finished = remote._update_progress_bars(bars, records)
    remote._update_progress_bars(bars, records, finished)

    assert finished == {"hopg"}
    assert bars["hopg"].description == "HOPG · done"
    assert bars["hopg"].refreshes == 1
    assert bars["hopg"].closed is True


def test_static_case_progress_uses_catalog_labels_and_one_row_per_material():
    records = remote._parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":4,"state":"done"}',
                '{"material":"hbn","total_cases":4,"cached_cases":1,'
                '"completed_new_cases":1,"state":"running"}',
            ]
        )
    )

    output = remote._format_case_progress(records, ["hopg", "hbn"])

    assert output.count("HOPG") == 1
    assert output.count("h-BN") == 1
    assert "████████████████" in output
    assert "████████░░░░░░░░" in output
    assert "5/5" in output
    assert "2/4" in output


def test_static_case_progress_colors_tracks_only_on_tty(monkeypatch):
    record = {
        "hopg": {
            "material": "hopg",
            "total_cases": 2,
            "cached_cases": 0,
            "completed_new_cases": 1,
            "state": "running",
        }
    }
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(remote.sys.stdout, "isatty", lambda: True)

    colored = remote._format_case_progress(record, ["hopg"])
    monkeypatch.setenv("NO_COLOR", "1")
    plain = remote._format_case_progress(record, ["hopg"])

    assert "\033[38;2;92;207;230m" in colored
    assert "\033[" not in plain


def test_attach_dashboard_allocates_every_material_row_without_dynamic_width(monkeypatch):
    created = []

    class Bar:
        def close(self):
            pass

    monkeypatch.setattr(remote, "tqdm", lambda **kwargs: created.append(kwargs) or Bar())
    monkeypatch.setattr(remote, "_poll_chain", lambda _jobid: ("done [3/3] now", False, {}))

    remote._attach_progress_dashboard("j", "materials: a b c\nslurm_job_id: 1\n")

    assert [kwargs["nrows"] for kwargs in created] == [4, 4, 4]
    assert all("dynamic_ncols" not in kwargs for kwargs in created)
    assert all(kwargs["colour"] == "#7f8c98" for kwargs in created)
    assert all("cases" in kwargs["bar_format"] for kwargs in created)


def test_dashboard_attach_closes_bars_and_prints_final_state(monkeypatch, capsys):
    metadata = "materials: hopg hbn\nprogress_dashboard: True\nslurm_job_id: 48291\n"
    payload = "\n".join(
        [
            json.dumps(
                {
                    "material": "hopg",
                    "total_cases": 4,
                    "cached_cases": 1,
                    "completed_new_cases": 3,
                    "state": "done",
                }
            ),
            json.dumps(
                {
                    "material": "hbn",
                    "total_cases": 2,
                    "cached_cases": 0,
                    "completed_new_cases": 1,
                    "state": "failed",
                }
            ),
        ]
    )
    bars = []
    records = remote._parse_progress_records(payload)
    polls = iter(
        [
            ("running hbn [2/2] since now", True, records),
            ("done with 1 warning(s) [2/2] now", False, records),
        ]
    )
    monkeypatch.setattr(remote, "_job_metadata", lambda _jobid: metadata)
    monkeypatch.setattr(remote, "_poll_chain", lambda _jobid: next(polls))
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        remote,
        "tqdm",
        lambda **kwargs: bars.append(_FakeProgressBar(**kwargs)) or bars[-1],
    )

    assert remote.attach("j") is True

    assert [bar.n for bar in bars] == [4, 1]
    assert all(bar.closed for bar in bars)
    output = capsys.readouterr().out
    assert "JOB j · FINISHED" in output
    assert "done with 1 warning(s)" in output


def test_dashboard_attach_ctrl_c_disconnects_viewer_only(monkeypatch):
    bars = []
    monkeypatch.setattr(
        remote,
        "_job_metadata",
        lambda _jobid: "materials: hopg\nprogress_dashboard: True\nslurm_job_id: 48291\n",
    )
    monkeypatch.setattr(
        remote,
        "_poll_chain",
        lambda _jobid: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(
        remote,
        "tqdm",
        lambda **kwargs: bars.append(_FakeProgressBar(**kwargs)) or bars[-1],
    )

    assert remote.attach("j") is False
    assert bars[0].closed is True


def test_stop_jobid_uses_scancel_not_kill(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: commands.append(command))

    remote._stop_jobid("j")

    assert "scancel 48291" in commands[0][-1]
    assert "kill -TERM" not in commands[0][-1]
    assert "scancel 48291 || exit" in commands[0][-1]


def test_stop_jobid_rejects_legacy_job_without_scheduler_id(monkeypatch):
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "")
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: pytest.fail("must not scancel"))

    with pytest.raises(SystemExit, match="not an active SLURM job"):
        remote._stop_jobid("j")


def test_stop_help_describes_slurm_cancellation(capsys):
    with pytest.raises(SystemExit):
        remote.main(["stop", "--help"])

    help_text = capsys.readouterr().out
    assert "cancel active SLURM job" in help_text
    assert "SIGTERM" not in help_text


def test_follow_logs_use_stdin_closed_ssh(monkeypatch):
    runs = []
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.tail_logs("20260101-000000", follow=True)

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4


def test_stems_quick_suffix():
    assert remote._stems(["mose2", "wse2"], True) == ["mose2_quick", "wse2_quick"]
    assert remote._stems(["mose2"], False) == ["mose2"]


def test_stop_materials_resolve_unique_live_jobs(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(remote, "_stop_jobid", stopped.append)

    remote.stop_jobs(["wse2", "mose2"])

    assert stopped == ["job2"]


def test_stop_all_stops_every_live_job(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", True, ["mose2"])],
    )
    monkeypatch.setattr(remote, "_stop_jobid", stopped.append)

    remote.stop_jobs(all_jobs=True)

    assert stopped == ["job1", "job2"]


def test_stop_rejects_bad_material_before_live_job_lookup(monkeypatch):
    monkeypatch.setattr(
        remote,
        "_live_jobs",
        lambda: pytest.fail("must validate before checking live jobs"),
    )

    with pytest.raises(SystemExit):
        remote.stop_jobs(["bad;material"])


def test_stop_cli_accepts_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "stop_jobs", lambda materials, all_jobs: calls.append((materials, all_jobs))
    )

    remote.main(["stop", "hopg", "mose2"])

    assert calls == [(["hopg", "mose2"], False)]


def test_stop_cli_accepts_all(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "stop_jobs", lambda materials, all_jobs: calls.append((materials, all_jobs))
    )

    remote.main(["stop", "--all"])

    assert calls == [([], True)]


# ---- clear <material> (checkpoint lifecycle, component 3) ----------------------
def _no_live_jobs(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])


def test_clear_refuses_when_a_live_job_produces_the_stem(monkeypatch):
    # a live job producing hopg -> clearing hopg must refuse before any ssh
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg", yes=True)


def test_clear_refuses_for_quick_stem_collision(monkeypatch):
    # a live --quick job producing hopg_quick still blocks a clear of hopg
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", True, ["hopg"])])
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg")


def test_clear_refuses_when_an_ambiguous_submission_holds_a_reservation(monkeypatch):
    """A retained pre-sbatch lock protects a possibly queued job without an ID."""
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *_args: "RESERVED\thopg\n")

    with pytest.raises(SystemExit, match="reservation"):
        remote.clear_remote("hopg", yes=True)


def test_clear_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl\nhopg_quick.pkl\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "hopg_quick.pkl" in out
    assert runs == []  # nothing deleted in a dry preview


def test_clear_yes_deletes_existing_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: commands.append(command) or "CLEARED\thopg.pkl\nCLEARED\thopg_quick.pkl\n",
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert runs == []
    assert 'mkdir "$R/$stem"' in commands[0]
    assert 'rm -f "$f"' in commands[0]
    assert "cleared on the box" in capsys.readouterr().out


def test_clear_reports_nothing_when_no_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "\n")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert "nothing to clear" in capsys.readouterr().out
    assert runs == []  # nothing to delete


def test_clear_rejects_bad_material(monkeypatch):
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must validate before touching jobs")
    )
    with pytest.raises(SystemExit):
        remote.clear_remote("rm -rf /")


def _bash_or_skip(tmp_path):
    """A bash that can reach tmp_path, or skip. Git Bash handles C:/-style posix
    paths; a WSL bash can't, and would hit the snippet's `|| exit 0` and silently
    return an empty listing -- so probe the cd before trusting the test."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    probe = subprocess.run([bash, "-c", f"cd '{tmp_path.as_posix()}'"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash cannot reach the pytest tmp dir")
    return bash


def test_parallel_queue_script_preserves_partial_failure_results(monkeypatch, tmp_path):
    """Run the generated payload: one failed child must not hide two successes."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *hbn*) exit 7 ;; esac\n'
        "exit 0\n"
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
    script = remote._queue_script(
        "j", ["hopg", "hbn", "hfse2"], quick=False, workers=4, parallel_materials=2
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("done with 1 warning(s)")
    log = (jobdir / "log").read_text()
    assert "completed: hopg" in log
    assert "completed: hfse2" in log
    assert "WARNING: scan failed for hbn; continuing" in log


@pytest.mark.parametrize("parallel_materials", [1, 2, 4])
def test_parallel_queue_script_enforces_process_limit(monkeypatch, tmp_path, parallel_materials):
    """Measure the generated payload's peak simultaneous scan processes."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    trace = tmp_path / "trace"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        f'echo "start $$" >> "{trace.as_posix()}"\n'
        "sleep 0.05\n"
        f'echo "end $$" >> "{trace.as_posix()}"\n'
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
    materials = ["hopg", "hbn", "hfse2", "v2o5"]
    script = remote._queue_script(
        "j",
        materials,
        quick=False,
        workers=4,
        parallel_materials=parallel_materials,
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    active = peak = 0
    for event in trace.read_text().splitlines():
        active += 1 if event.startswith("start ") else -1
        peak = max(peak, active)
    assert active == 0
    assert peak == parallel_materials


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
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
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
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg", "hbn"], quick=False, workers=None, chunk_minutes=10.0
    )

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert (jobdir / "state").read_text().startswith("done with 1 warning(s)")
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
        "exit 75\n"  # budget hit every slice: hopg stays unresolved
    )
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(sbatch_body)
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
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
    fake_uv.write_text('#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\nexit 75\n')
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "sbatch_called"
    (bin_dir / "sbatch").write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\necho 4242\n")
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
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
    fake_uv.write_text('#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\nexit 7\n')
    fake_uv.chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(remote, "REMOTE_UV", fake_uv.as_posix())
    script = remote._chunked_queue_script(
        "j", ["hopg"], quick=False, workers=None, chunk_minutes=10.0
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    log = (jobdir / "log").read_text()
    assert "WARNING: scan failed for hopg (exit 7); will not retry" in log
    assert "failed: hopg" in log
    assert (jobdir / "state").read_text().startswith("done with 1 warning(s)")


def test_clear_fails_closed_when_squeue_cannot_be_queried(monkeypatch, tmp_path):
    """A scheduler outage cannot be mistaken for an absent job before deletion."""
    bash = _bash_or_skip(tmp_path)
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    (job / "meta").write_text("slurm_job_id: 48291\nquick: False\nmaterials: hopg\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "squeue").write_text("#!/bin/sh\nexit 7\n")
    (commands / "squeue").chmod(0o755)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        env = {**os.environ, "PATH": f"{commands}:{os.environ.get('PATH', '')}"}
        result = subprocess.run([bash, "-c", remote_cmd], capture_output=True, env=env)
        if result.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {result.returncode})")
        return result.stdout.decode()

    monkeypatch.setattr(remote, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kw: pytest.fail("must not delete"))

    with pytest.raises(SystemExit, match="ssh command failed"):
        remote.clear_remote("hopg", yes=True)


def test_clear_listing_snippet_exits_zero_when_quick_pkl_missing(monkeypatch, tmp_path, capsys):
    """Regression (checkpoint-lifecycle verification, 2026-07-07): the box-side
    listing loop's last command is `[ -f "$f" ] && echo "$f"`, whose failure
    status leaks out as the loop's -- and hence ssh's -- exit status whenever the
    last stem (<material>_quick.pkl) is missing, making _ssh_capture abort the
    clear before the dry preview prints. Execute the real snippet under bash
    against a checkpoints/ dir holding only hopg.pkl."""
    bash = _bash_or_skip(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "hopg.pkl").write_bytes(b"x")  # no hopg_quick.pkl

    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(remote, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        # same semantics as _ssh_capture, but run the snippet locally
        r = subprocess.run([bash, "-c", remote_cmd], capture_output=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {r.returncode})")
        return r.stdout

    monkeypatch.setattr(remote, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(
        remote, "_run", lambda cmd, **kw: pytest.fail("dry preview must not delete")
    )
    remote.clear_remote("hopg", yes=False)  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "hopg.pkl" in out and "hopg_quick.pkl" not in out


# ---- scan --quick --grid must be rejected at parse time -------------------------
def test_scan_rejects_quick_plus_grid_before_any_work(monkeypatch):
    """--quick checkpoints aren't grid-filterable (cxr slim rejects _quick stems),
    so scan --quick --grid must fail up front -- not run the whole sweep and then
    traceback on the trailing pull."""
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must reject before the busy check")
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must reject before syncing"))
    monkeypatch.setattr(
        remote, "remote_scan", lambda *a, **kw: pytest.fail("must reject before scanning")
    )
    with pytest.raises(SystemExit, match="grid"):
        remote.main(["scan", "hopg", "--quick", "--grid"])


# ---- pull defaults to --grid; -f/--full opts into the plain whole-file pull -----
def test_pull_defaults_to_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg"])
    assert calls[0]["grid"] is True


def test_pull_full_flag_disables_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "--full"])
    assert calls[0]["grid"] is False


def test_pull_short_full_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "-f"])
    assert calls[0]["grid"] is False


def test_pull_warns_and_continues_when_one_checkpoint_is_missing(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "missing.pkl" in " ".join(cmd):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(remote, "_run", fake_run)
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: "remote")
    monkeypatch.setattr(remote, "_local_sha256", lambda _path: "local")

    remote.pull(["hopg", "missing"], no_sync=True)

    assert any("hopg.pkl" in " ".join(cmd) for cmd in calls)
    assert "warning: could not pull checkpoint 'missing'" in capsys.readouterr().out


def test_pull_skips_scp_when_remote_artifact_matches_local(monkeypatch, tmp_path, capsys):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "hopg.pkl").write_bytes(b"same")
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: "digest")
    monkeypatch.setattr(remote, "_local_sha256", lambda _path: "digest")
    monkeypatch.setattr(remote, "_run", lambda command, **_kw: calls.append(command))

    remote.pull(["hopg"], no_sync=True)

    assert not any(command[0] == "scp" for command in calls)
    assert "already current -> checkpoints/hopg.pkl" in capsys.readouterr().out


def test_full_pull_warns_without_scp_when_remote_checksum_fails(monkeypatch, tmp_path, capsys):
    """A failed remote hash must not fall through to an unchecked transfer."""
    calls = []
    commands = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: commands.append(command) or "sha256sum: missing file\\n",
    )
    monkeypatch.setattr(remote, "_run", lambda command, **_kw: calls.append(command))

    remote.pull(["hopg"], no_sync=True)

    assert len(commands) == 1
    assert commands[0].startswith(f"sha256sum {remote.REMOTE_DIR}/checkpoints/.hopg.pull.")
    assert not any(command[0] == "scp" for command in calls)
    assert "warning: could not pull checkpoint 'hopg'" in capsys.readouterr().out


def test_full_pull_transfers_stable_snapshot_once_when_artifacts_differ(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: "remote-digest")
    monkeypatch.setattr(remote, "_local_sha256", lambda _path: "local-digest")
    monkeypatch.setattr(remote, "_run", lambda command, **_kw: calls.append(command))

    remote.pull(["hopg"], no_sync=True)

    scp_calls = [command for command in calls if command[0] == "scp"]
    link_calls = [
        command
        for command in calls
        if command[:3] == ["ssh", "-n", remote.HOST]
        and len(command) == 4
        and command[3].startswith("ln ")
    ]
    assert len(scp_calls) == 1
    assert len(link_calls) == 1
    snapshot = link_calls[0][-1].split()[-1]
    assert scp_calls[0][1] == f"{remote.HOST}:{snapshot}"
    assert ["ssh", "-n", remote.HOST, f"rm -f {snapshot}"] in calls


def test_grid_pull_skips_scp_and_removes_matching_temp_artifact(monkeypatch, tmp_path, capsys):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "hopg.pkl").write_bytes(b"same")
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: "digest")
    monkeypatch.setattr(remote, "_local_sha256", lambda _path: "digest")
    monkeypatch.setattr(remote, "_run", lambda command, **_kw: calls.append(command))

    remote.pull(["hopg"], grid=True, no_sync=True)

    assert any("cxr slim" in " ".join(command) for command in calls)
    assert not any(command[0] == "scp" for command in calls)
    assert ["ssh", "-n", remote.HOST, "rm -f /tmp/hopg.grid.pkl"] in calls
    assert "already current -> checkpoints/hopg.pkl" in capsys.readouterr().out


def test_grid_pull_removes_temp_when_checksum_fails(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        remote,
        "_remote_sha256",
        lambda _path: (_ for _ in ()).throw(SystemExit("checksum failed")),
    )
    monkeypatch.setattr(remote, "_run", lambda command, **_kw: calls.append(command))

    remote.pull(["hopg"], grid=True, no_sync=True)

    assert ["ssh", "-n", remote.HOST, "rm -f /tmp/hopg.grid.pkl"] in calls
    assert "warning: could not pull checkpoint 'hopg'" in capsys.readouterr().out


def test_grid_pull_removes_temp_when_slim_fails(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)

    def run(command, **_kw):
        calls.append(command)
        if any("cxr slim" in part for part in command):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(remote, "_run", run)

    remote.pull(["hopg"], grid=True, no_sync=True)

    assert ["ssh", "-n", remote.HOST, "rm -f /tmp/hopg.grid.pkl"] in calls
    assert "warning: could not pull checkpoint 'hopg'" in capsys.readouterr().out


def test_remote_start_all_uses_toml_manifest(monkeypatch, tmp_path):
    manifest = tmp_path / "mats_to_sim.toml"
    manifest.write_text('materials = ["hopg", "hbn"]\n')
    monkeypatch.setattr(remote, "MATS_FILE", manifest)
    calls = []
    monkeypatch.setattr(
        remote, "start_queue", lambda materials, *args, **kwargs: calls.append(materials)
    )

    remote.main(["start", "--all", "--dry-run"])

    assert calls == [["hopg", "hbn"]]


def test_remote_start_defers_parallel_materials_default_to_start_queue(monkeypatch):
    """The CLI no longer bakes in a default: --parallel-materials is None
    unless given explicitly, so start_queue (mode-aware) resolves it. The
    monolithic default of 2 is exercised by
    test_chunk_minutes_zero_emits_monolithic_script."""
    calls = []
    monkeypatch.setattr(
        remote,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    remote.main(["start", "hopg", "--dry-run", "--chunk-minutes", "0"])

    assert calls[0][1]["parallel_materials"] is None
    assert calls[0][1]["chunk_minutes"] == 0.0


def test_remote_start_accepts_parallel_materials_three_and_four(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    remote.main(
        ["start", "hopg", "hbn", "--parallel-materials", "3", "--chunk-minutes", "0", "--dry-run"]
    )
    remote.main(
        ["start", "hopg", "hbn", "--parallel-materials", "4", "--chunk-minutes", "0", "--dry-run"]
    )

    assert [kwargs["parallel_materials"] for _materials, kwargs in calls] == [3, 4]


def test_remote_start_rejects_parallel_materials_above_four(monkeypatch):
    monkeypatch.setattr(
        remote, "start_queue", lambda *_args, **_kwargs: pytest.fail("must reject before start")
    )

    with pytest.raises(SystemExit):
        remote.main(["start", "hopg", "--parallel-materials", "5", "--dry-run"])


def test_start_queue_rejects_parallel_materials_above_four():
    with pytest.raises(SystemExit, match="between 1 and 4"):
        remote.start_queue(["hopg"], parallel_materials=5, chunk_minutes=0, dry_run=True)


def test_remote_scan_forwards_parallel_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )
    monkeypatch.setattr(remote, "attach", lambda _jobid: False)

    remote.main(["scan", "hopg", "--parallel-materials", "3", "--chunk-minutes", "0", "--no-sync"])

    assert calls[0][1]["parallel_materials"] == 3


def test_remote_scan_submits_then_attaches_and_pulls(monkeypatch):
    events = []
    monkeypatch.setattr(
        remote,
        "start_queue",
        lambda materials, **_kw: events.append(("start", materials)) or "j",
    )
    monkeypatch.setattr(remote, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(remote, "_completed_materials", lambda _jobid, materials: materials)
    monkeypatch.setattr(remote, "pull", lambda stems, **_kw: events.append(("pull", stems)))

    remote._cli_scan(
        argparse.Namespace(
            material="hopg",
            all=False,
            quick=False,
            workers=None,
            no_sync=False,
            grid=False,
            drop_wide_brem=False,
            downcast=False,
            remote_command="scan",
        )
    )

    assert events == [("start", ["hopg"]), ("attach", "j"), ("pull", ["hopg"])]


def test_interrupted_remote_scan_does_not_pull(monkeypatch):
    events = []
    monkeypatch.setattr(remote, "start_queue", lambda *_args, **_kwargs: "j")
    monkeypatch.setattr(remote, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        remote, "_completed_materials", lambda *_args: pytest.fail("must not inspect completion")
    )
    monkeypatch.setattr(remote, "pull", lambda *_args, **_kwargs: events.append("pull"))

    remote._cli_scan(
        argparse.Namespace(
            material="hopg",
            all=False,
            quick=False,
            workers=None,
            no_sync=False,
            grid=False,
            drop_wide_brem=False,
            downcast=False,
            remote_command="scan",
        )
    )

    assert events == []


def test_remote_scan_preserves_hyphenated_catalog_material(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "start_queue", lambda materials, **_kw: calls.append(materials) or "j"
    )
    monkeypatch.setattr(remote, "attach", lambda _jobid: None)
    monkeypatch.setattr(remote, "_completed_materials", lambda _jobid, materials: materials)
    monkeypatch.setattr(remote, "pull", lambda *_args, **_kwargs: None)

    remote.main(["scan", "mos2-on-sio2-si", "--no-sync"])

    assert calls == [["mos2-on-sio2-si"]]


def test_completed_materials_reads_success_markers_from_the_queue_log(monkeypatch):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: commands.append(command) or "hopg\nhbn\n",
    )

    assert remote._completed_materials("j", ["hopg", "wse2", "hbn"]) == ["hopg", "hbn"]
    assert "s/^completed: //p" in commands[0]


def test_remote_scan_rejects_unknown_before_busy_or_sync(monkeypatch):
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must validate before checking busy jobs")
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="unknown material"):
        remote.main(["scan", "not_in_catalog"])


def test_remote_start_accepts_hyphenated_catalog_material(capsys):
    remote.main(["start", "mos2-on-sio2-si", "--dry-run"])

    assert "mos2-on-sio2-si" in capsys.readouterr().out


def test_remote_start_rejects_unknown_before_busy_or_sync(monkeypatch):
    monkeypatch.setattr(
        remote, "_live_jobs", lambda: pytest.fail("must validate before checking busy jobs")
    )
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="unknown material"):
        remote.main(["start", "not_in_catalog"])


def test_pull_validates_safe_stems_without_requiring_catalog_membership(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(remote, "_remote_sha256", lambda _path: "remote")
    monkeypatch.setattr(remote, "_local_sha256", lambda _path: "local")

    remote.pull(["hopg_quick", "zhai"], no_sync=True)

    assert any("hopg_quick.pkl" in " ".join(command) for command in calls)
    assert any("zhai.pkl" in " ".join(command) for command in calls)


def test_pull_rejects_unsafe_stem_before_sync_or_local_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="invalid remote shell token"):
        remote.pull(["bad;stem"], grid=True)

    assert not (tmp_path / "checkpoints").exists()


def test_sync_paths_include_all_materials_manifest():
    assert "mats_to_sim.toml" in remote.SYNC_PATHS


# ---- cxr remote check (Zhai GPU reproduction) ------------------------------
def test_zhai_queue_script_has_ne_flags_and_meta():
    s = remote._zhai_queue_script(
        "20260101-000000",
        ne=11,
        ne_brem=3,
        ne_supp=5,
        tmd_azimuth=35.0,
        refresh=True,
    )
    assert "reproduce_zhai.py" in s
    assert "--ne 11" in s and "--ne-brem 3" in s and "--ne-supp 5" in s and "--refresh" in s
    assert "--tmd-azimuth 35.0" in s
    assert "20260101-000000" in s
    assert "materials: zhai" not in s and "quick: False" not in s
    assert "started: $(date -Is)" in s
    assert "/pid" not in s


def test_zhai_queue_script_no_refresh_flag_when_unset():
    s = remote._zhai_queue_script("j", ne=1, ne_brem=1, ne_supp=1, tmd_azimuth=0.0, refresh=False)
    assert "--refresh" not in s


def test_zhai_start_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_zhai_queue()


def test_zhai_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(remote, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_zhai_queue(dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "reproduce_zhai.py" in out


def test_remote_check_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.remote_check()


def test_foreground_check_submits_then_attaches_and_pulls(monkeypatch):
    events = []
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **_kw: events.append("start") or "j")
    monkeypatch.setattr(remote, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(remote, "_job_succeeded", lambda _jobid: True, raising=False)
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: events.append("pull"))

    remote.remote_check(no_sync=True)

    assert events == ["start", ("attach", "j"), "pull"]


def test_interrupted_foreground_check_does_not_pull(monkeypatch):
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(remote, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        remote, "pull_zhai_cache", lambda: pytest.fail("must not pull after interruption")
    )

    remote.remote_check(no_sync=True)


@pytest.mark.parametrize("state", ["FAILED (exit 1)", "cancelled [48291]"])
def test_failed_foreground_check_does_not_pull_stale_cache(monkeypatch, state):
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(remote, "attach", lambda _jobid: True)
    monkeypatch.setattr(remote, "_job_succeeded", lambda _jobid: False, raising=False)
    monkeypatch.setattr(remote, "_job_state", lambda _jobid: state, raising=False)
    monkeypatch.setattr(
        remote, "pull_zhai_cache", lambda: pytest.fail("failed job must not pull stale cache")
    )

    with pytest.raises(SystemExit, match="did not complete successfully"):
        remote.remote_check(no_sync=True)


def test_remote_check_no_sync_skips_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: calls.append(kw) or "j")
    monkeypatch.setattr(remote, "attach", lambda _jobid: calls.append("attach") or True)
    monkeypatch.setattr(remote, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))

    remote.remote_check(no_sync=True)

    assert calls == [
        {
            "ne": 20_000,
            "ne_brem": 200,
            "ne_supp": 200,
            "tmd_azimuth": 0.0,
            "refresh": False,
            "no_sync": True,
        },
        "attach",
        "pull",
    ]


def test_pull_zhai_cache_fetches_every_listed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda *a: (
            "/r/checkpoints/zhai_reproduction/zhai-a.pkl\n"
            "/r/checkpoints/zhai_reproduction/zhai-b.pkl\n"
        ),
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.pull_zhai_cache()

    assert len(runs) == 2
    assert runs[0][0] == "scp" and runs[0][1].endswith("zhai-a.pkl")
    assert (tmp_path / "checkpoints" / "zhai_reproduction").is_dir()


def test_pull_zhai_cache_reports_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(remote, "_ssh_capture", lambda command: commands.append(command) or "")
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: pytest.fail("nothing to pull"))

    remote.pull_zhai_cache()

    assert "no zhai cache files" in capsys.readouterr().out
    assert "find" in commands[0]


def test_check_cli_pull_flag_skips_run(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))
    monkeypatch.setattr(
        remote, "remote_check", lambda **kw: pytest.fail("--pull must not run the reproduction")
    )

    remote.main(["check", "--pull"])

    assert calls == ["pull"]


def test_check_cli_detached_starts_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: calls.append(kw) or "jid")
    monkeypatch.setattr(remote, "attach", lambda jobid: pytest.fail("no --follow: must not attach"))

    remote.main(["check", "--detached", "--ne", "11"])

    assert calls[0]["ne"] == 11


def test_check_cli_detached_follow_attaches(monkeypatch):
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: "jid")
    attached = []
    monkeypatch.setattr(remote, "attach", attached.append)

    remote.main(["check", "--detached", "--follow"])

    assert attached == ["jid"]


def test_check_cli_foreground_calls_remote_check(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "remote_check", lambda **kw: calls.append(kw))

    remote.main(["check", "--ne", "11", "--refresh"])

    assert calls == [
        {
            "ne": 11,
            "ne_brem": 200,
            "ne_supp": 200,
            "tmd_azimuth": 0.0,
            "refresh": True,
            "no_sync": False,
        }
    ]


def test_check_cli_rejects_follow_without_detached(monkeypatch, capsys):
    monkeypatch.setattr(
        remote, "remote_check", lambda **kw: pytest.fail("must reject before running")
    )

    with pytest.raises(SystemExit) as excinfo:
        remote.main(["check", "--follow"])

    assert excinfo.value.code == 2
    assert "--follow requires --detached" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--pull", "--detached"], ["--pull", "--detached", "--follow"]])
def test_check_cli_rejects_pull_with_detached(monkeypatch, capsys, args):
    monkeypatch.setattr(
        remote, "pull_zhai_cache", lambda: pytest.fail("must reject before pulling")
    )

    with pytest.raises(SystemExit) as excinfo:
        remote.main(["check", *args])

    assert excinfo.value.code == 2
    assert "not allowed with argument --pull" in capsys.readouterr().err


def test_sync_paths_ship_checks_and_zhai_shim():
    assert "checks" in remote.SYNC_PATHS


def _poll_payload(state, sid="123", squeue="RUNNING", progress=""):
    return f"@@STATE\n{state}\n@@SID\n{sid}\n@@SQUEUE\n{squeue}\n@@PROGRESS\n{progress}\n"


def test_poll_chain_parses_sections(monkeypatch):
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda cmd: _poll_payload(
            "running hopg [1/1] since now",
            progress='{"material":"hopg","total_cases":4,"cached_cases":1,'
            '"completed_new_cases":2,"state":"running"}',
        ),
    )
    state, live, records = remote._poll_chain("20260717-abc")
    assert state.startswith("running")
    assert live is True
    assert records["hopg"]["completed_new_cases"] == 2


def test_attach_dashboard_survives_slice_gap_and_ends_terminal(monkeypatch):
    responses = iter(
        [
            _poll_payload("running hopg [1/1] since now"),
            _poll_payload("queued slice 2 now", squeue=""),  # inter-slice gap: not live
            _poll_payload("running hopg [1/1] since now"),  # next slice picked up
            _poll_payload("done [1/1] now", squeue=""),
        ]
    )
    monkeypatch.setattr(remote, "_ssh_capture", lambda cmd: next(responses))
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    ok = remote._attach_progress_dashboard(
        "20260717-abc", "job: 20260717-abc\nmaterials: hopg\nslurm_job_id: 123\n"
    )
    assert ok is True


def test_attach_dashboard_watchdog_exits_on_broken_chain(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda cmd: _poll_payload("queued slice 2 now", squeue=""),
    )
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    ok = remote._attach_progress_dashboard(
        "20260717-abc", "job: 20260717-abc\nmaterials: hopg\nslurm_job_id: 123\n"
    )
    assert ok is False
    assert "CHAIN STALLED" in capsys.readouterr().out


def test_stop_writes_stop_sentinel_before_scancel(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_slurm_job_id", lambda jobid: "123")
    monkeypatch.setattr(remote, "_slurm_state", lambda sid: "RUNNING")
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: commands.append(cmd[-1]))
    remote._stop_jobid("20260717-abc")
    (cmd,) = commands
    assert cmd.index("STOP") < cmd.index("scancel")
    assert "reproduce_zhai.py" in remote.SYNC_PATHS


def test_is_reapable_only_when_dead_and_past_guard():
    guard = 300.0
    # dead (no SLURM allocation) AND old enough -> reap
    assert remote._is_reapable(None, 301, guard) is True
    assert remote._is_reapable(None, 300, guard) is True
    # dead but still inside the staging-race window -> keep
    assert remote._is_reapable(None, 299, guard) is False
    # live allocation is never reaped, however old the lock looks
    assert remote._is_reapable("RUNNING", 10_000, guard) is False
    assert remote._is_reapable("PENDING", 10_000, guard) is False


def test_orphaned_reservation_jobs_partitions_dead_stale_from_live_and_recent(monkeypatch):
    now = 10_000
    monkeypatch.setattr(
        remote,
        "_reservation_ledger",
        lambda: (
            now,
            [
                ("hopg", "dead", now - 600),  # dead + stale -> orphan
                ("hbn", "dead", now - 600),
                ("mose2", "live", now - 600),  # has SLURM allocation -> protected
                ("wse2", "fresh", now - 10),  # dead but mid-submit window -> protected
            ],
        ),
    )
    monkeypatch.setattr(
        remote, "_slurm_job_id", lambda jobid: "48291" if jobid in {"live", "fresh"} else None
    )
    monkeypatch.setattr(remote, "_slurm_state", lambda sid: "RUNNING" if sid == "48291" else None)

    orphans, protected = remote._orphaned_reservation_jobs(300.0)

    assert orphans == {"dead": ["hbn", "hopg"]}
    assert protected == {"live": ["mose2"], "fresh": ["wse2"]}


def test_reap_reservations_dry_run_previews_without_releasing(monkeypatch, capsys):
    monkeypatch.setattr(
        remote,
        "_orphaned_reservation_jobs",
        lambda _min_age: ({"dead": ["hopg", "hbn"]}, {"live": ["mose2"]}),
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.reap_reservations(yes=False)

    out = capsys.readouterr().out
    assert "would release dead" in out
    assert "keeping live" in out
    assert "re-run with --yes" in out
    assert runs == []


def test_reap_reservations_yes_releases_orphans_and_stamps_terminal_state(monkeypatch, capsys):
    monkeypatch.setattr(
        remote, "_orphaned_reservation_jobs", lambda _min_age: ({"dead": ["hopg"]}, {})
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd[-1]))

    remote.reap_reservations(yes=True)

    (command,) = runs
    assert 'J="dead"' in command  # releases only the orphan's own locks
    assert "rm -rf" in command
    assert "reaped (orphan reservations released)" in command
    assert "reaped 1 orphaned job(s)" in capsys.readouterr().out


def test_reap_job_command_releases_only_matching_owner(tmp_path):
    bash = _bash_or_skip(tmp_path)
    monkeypatch_dir = tmp_path
    reservations = monkeypatch_dir / "jobs" / "reservations"
    for stem, owner in (("hopg", "dead"), ("mose2", "dead"), ("wse2", "other")):
        d = reservations / stem
        d.mkdir(parents=True)
        (d / "jobid").write_text(f"{owner}\n")
    (monkeypatch_dir / "jobs" / "dead").mkdir(parents=True)

    import cxr_mc.remote as R

    old = R.REMOTE_DIR
    R.REMOTE_DIR = str(monkeypatch_dir)
    try:
        result = subprocess.run(
            [bash, "-c", R._reap_job_command("dead")], capture_output=True, text=True
        )
    finally:
        R.REMOTE_DIR = old

    assert result.returncode == 0, result.stderr
    assert not (reservations / "hopg").exists()
    assert not (reservations / "mose2").exists()
    assert (reservations / "wse2").exists()  # another owner's lock untouched
    assert "reaped" in (monkeypatch_dir / "jobs" / "dead" / "state").read_text()
