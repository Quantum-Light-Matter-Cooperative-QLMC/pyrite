"""cxr_mc.remote (``cxr remote``): material-name validation and the
detached-queue runner generation.

These are pure-string/logic checks (no ssh), so they run anywhere. The one
exception is the clear-listing regression test, which executes the box-side
shell snippet under a local bash (skipped when bash is unavailable)."""

import argparse
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
    assert "/pid" not in s


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s


def test_queue_script_warns_and_continues_after_a_material_fails():
    script = remote._queue_script("j", ["hopg", "hbn"], quick=False, workers=None)

    assert "WARNING: scan failed for $m; continuing" in script
    assert "continue" in script
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


def test_stop_waits_for_scheduler_cancellation_before_releasing_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(remote, "_slurm_state", lambda _scheduler_id: "RUNNING")
    monkeypatch.setattr(remote, "_run", lambda command, **_kwargs: commands.append(command))

    remote._stop_jobid("j")

    command = commands[0][-1]
    assert "scancel 48291" in command
    assert "S=$(squeue -h -j 48291 -o '%T')" in command
    assert 'if [ "$STATUS" -ne 0 ]' in command
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
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh"))
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh"))

    remote.start_queue(["hopg"], dry_run=True)

    out = capsys.readouterr().out
    assert "#SBATCH --partition=gpu" in out
    assert "sbatch --parsable" in out
    assert "nohup" not in out and "setsid" not in out


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

    remote.start_queue(["hopg"], quick=True, workers=3, no_sync=True)

    upload_command = uploads[0][0][0][-1]
    assert "job: " in upload_command
    assert "materials: hopg" in upload_command
    assert "quick: True" in upload_command
    assert "workers: 3" in upload_command
    assert "started:" not in upload_command
    assert "started: $(date -Is)" in uploads[0][1]["input"].decode()
    assert "materials: hopg" not in uploads[0][1]["input"].decode()
    assert "mkdir \"$R/$stem\"" in submissions[0][-1]
    assert "sbatch --parsable" in submissions[1][-1]


def test_start_reports_the_submitted_slurm_job_id(monkeypatch, capsys):
    monkeypatch.setattr(remote, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(remote.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg"], no_sync=True)

    assert "submitted SLURM job 48291" in capsys.readouterr().out


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


def test_job_status_reports_scheduler_state_not_process_liveness(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        remote,
        "_ssh_capture",
        lambda command: commands.append(command) or "slurm: 48291 RUNNING\n",
    )

    remote.job_status("j")

    assert "slurm: 48291 RUNNING" in capsys.readouterr().out
    assert "squeue" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_attach_uses_stdin_closed_ssh_for_live_view(monkeypatch):
    runs = []
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
    monkeypatch.setattr(remote.subprocess, "run", lambda cmd: runs.append(cmd))

    remote.attach("20260101-000000")

    assert 'tail -n 50 -F --retry "$D/log"' in runs[0][-1]


def test_attach_returns_false_when_the_viewer_is_interrupted(monkeypatch):
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda _cmd: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert remote.attach("20260101-000000") is False


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
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "hopg.pkl hopg_quick.pkl")
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert len(runs) == 1
    deletion = " ".join(runs[0])
    assert "rm -f" in deletion and "hopg.pkl" in deletion and "hopg_quick.pkl" in deletion
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

    remote.pull(["hopg", "missing"], no_sync=True)

    assert any("hopg.pkl" in " ".join(cmd) for cmd in calls)
    assert "warning: could not pull checkpoint 'missing'" in capsys.readouterr().out


def test_remote_start_all_uses_toml_manifest(monkeypatch, tmp_path):
    manifest = tmp_path / "mats_to_sim.toml"
    manifest.write_text('materials = ["hopg", "hbn"]\n')
    monkeypatch.setattr(remote, "MATS_FILE", manifest)
    calls = []
    monkeypatch.setattr(remote, "start_queue", lambda materials, *args: calls.append(materials))

    remote.main(["start", "--all", "--dry-run"])

    assert calls == [["hopg", "hbn"]]


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
    monkeypatch.setattr(remote, "start_queue", lambda materials, **_kw: calls.append(materials) or "j")
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
    assert 's/^completed: //p' in commands[0]


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
    assert "reproduce_zhai.py" in remote.SYNC_PATHS
