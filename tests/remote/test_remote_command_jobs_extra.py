"""Additional remote jobs tests."""

import os
import subprocess

import pytest

from pyrite import remote
from pyrite.console.dashboard import render as dashboard_render
from pyrite.console.dashboard import state as dashboard_state
from pyrite.remote import config, state, transport
from tests.remote._remote_test_helpers import _bash_or_skip, _queue_payload


def test_live_jobs_queries_squeue_for_recorded_scheduler_ids(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "j\tFalse\thopg\n"
    )

    assert remote._live_jobs() == [("j", False, ["hopg"])]
    assert "squeue" in commands[0]
    assert "slurm_job_id" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_live_jobs_batches_scheduler_query_and_fails_closed(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "j\tFalse\thopg\n"
    )

    remote._live_jobs()

    assert commands[0].count("squeue ") == 1
    assert "squeue -h -u \"$USER\" -o '%i'" in commands[0]
    assert 'case "$LIVE" in *" $SID "*)' in commands[0]
    assert 'echo "could not query SLURM jobs" >&2' in commands[0]


def test_live_jobs_joins_one_scheduler_snapshot_against_job_history(monkeypatch, tmp_path):
    bash = _bash_or_skip(tmp_path)
    jobs = tmp_path / "jobs"
    for jobid, scheduler_id, material in (
        ("old", "11", "hopg"),
        ("live", "12", "hbn"),
    ):
        jobdir = jobs / jobid
        jobdir.mkdir(parents=True)
        (jobdir / "meta").write_text(
            f"slurm_job_id: {scheduler_id}\nquick: False\nmaterials: {material}\n"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "squeue-calls"
    squeue = bin_dir / "squeue"
    squeue.write_text(f"#!/bin/sh\necho called >> '{calls}'\necho 12\n")
    squeue.chmod(0o755)

    def run_remote(command):
        result = subprocess.run(
            [bash, "-c", command],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(transport, "_ssh_capture", run_remote)

    assert remote._live_jobs() == [("live", False, ["hbn"])]
    assert calls.read_text().splitlines() == ["called"]


def test_live_job_code_reports_each_live_job_recorded_stamp(monkeypatch, tmp_path):
    """The recorded digest rides the existing squeue join -- no second round trip."""
    bash = _bash_or_skip(tmp_path)
    jobs = tmp_path / "jobs"
    for jobid, scheduler_id, stamp in (
        ("stamped", "12", "code_digest: abc123\ncode_revision: feedface\n"),
        ("legacy", "13", ""),
    ):
        jobdir = jobs / jobid
        jobdir.mkdir(parents=True)
        (jobdir / "meta").write_text(
            f"slurm_job_id: {scheduler_id}\nquick: False\nmaterials: hopg\n{stamp}"
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/bin/sh\necho 12\necho 13\n")
    squeue.chmod(0o755)

    def run_remote(command):
        result = subprocess.run(
            [bash, "-c", command],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(transport, "_ssh_capture", run_remote)

    assert sorted(state._live_job_code()) == [
        ("legacy", False, ["hopg"], "", ""),
        ("stamped", False, ["hopg"], "abc123", "feedface"),
    ]
    # Existing checkpoint-ownership callers keep the three-field shape.
    assert sorted(state._live_jobs()) == [
        ("legacy", False, ["hopg"]),
        ("stamped", False, ["hopg"]),
    ]


@pytest.mark.parametrize(
    ("target", "expected_rank"),
    [("10", 1), ("20", 2), ("30", 3)],
)
def test_pending_queue_rank_uses_priority_then_job_id(target, expected_rank):
    payload = _queue_payload(
        "job_id=30|state=PENDING|name=third|partition=gpu|reason=Resources|priority=50|user=c",
        "job_id=20|state=PENDING|name=second|partition=gpu|reason=Priority|priority=50|user=b",
        "job_id=10|state=PENDING|name=first|partition=gpu|reason=Priority|priority=60|user=a",
        "job_id=5|state=PENDING|name=other|partition=cpu|reason=Priority|priority=999|user=z",
        "job_id=40|state=RUNNING|name=active|partition=gpu|reason=None|priority=999|user=d",
    )

    context = dashboard_render._pending_queue_context(payload, target)

    assert context is not None
    assert (context["rank"], context["count"], context["partition"]) == (
        expected_rank,
        3,
        "gpu",
    )
    assert context["top"]["job_id"] == "10"


def test_pending_queue_context_changes_with_priority_and_rejects_retired_or_malformed():
    before = _queue_payload(
        "job_id=10|state=PENDING|name=a|partition=gpu|reason=Priority|priority=20|user=u",
        "job_id=20|state=PENDING|name=b|partition=gpu|reason=Priority|priority=10|user=u",
    )
    after = before.replace("priority=20", "priority=5").replace("priority=10", "priority=30")

    assert dashboard_render._pending_queue_context(before, "20")["rank"] == 2
    assert dashboard_render._pending_queue_context(after, "20")["rank"] == 1
    assert dashboard_render._pending_queue_context(before, "999") is None
    assert (
        dashboard_render._pending_queue_context(
            before + "\njob_id=bad|state=PENDING|partition=gpu|priority=nan", "bad"
        )
        is None
    )
    injected = _queue_payload(
        "job_id=20|state=PENDING|name=x|priority=999|partition=gpu|"
        "reason=Priority|priority=10|user=u"
    )
    assert dashboard_render._pending_queue_context(injected, "20") is None


def test_pending_queue_render_sanitizes_identity_reason_and_explains_backfill(capsys, monkeypatch):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 20",
                "STATE": "queued",
                "SQUEUE": "job_id=20|state=PENDING|partition=gpu",
                "QUEUE": _queue_payload(
                    "job_id=10|state=PENDING|name=top\x1b[2J|partition=gpu|"
                    "reason=Priority\u202e|priority=20|user=alice",
                    "job_id=20|state=PENDING|name=target|partition=gpu|"
                    "reason=Resources|priority=10|user=bob",
                ),
            }
        ),
    )

    remote.job_status("j")

    output = capsys.readouterr().out
    assert "Queue position  2/2 pending in gpu" in output
    assert "10 · alice/top?[2J · Priority?" in output
    assert "priority can change and backfill may run lower-ranked jobs first" in output
    assert "\x1b" not in output
    assert "\u202e" not in output


def test_list_jobs_sanitizes_hostile_remote_fields(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: "j\x1b[2J\t48291\tFalse\thopg\tRUNNING\u202e forged\n",
    )

    remote.list_jobs()

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "\u202e" not in output
    assert "j?[2J" in output
    assert "RUNNING? forged" in output


def test_implicit_job_selection_excludes_checkpoint_reservations(monkeypatch, tmp_path):
    """Bare queue commands must not mistake jobs/reservations for a job."""
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")

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
    assert result.stdout == "20260715-113910-b0240c4f\t48291\t?\tscan\t-\trunning \n"
    assert all('[ -f "$d/meta" ] || continue' in command for command in commands)


def test_latest_jobid_picks_newest_profile_job_not_bare_name(monkeypatch, tmp_path):
    """Profile jobs are name-suffixed, so name order is not chronological.

    The bare ``sub_100keV`` dir sorts after every ``sub_100keV-N`` (the glob
    adds a trailing ``/`` and ``-`` < ``/``), so a plain name sort would default
    ``attach``/``logs`` to the oldest, now-cancelled job. Selection must follow
    meta mtime instead: the live ``-7`` chain is the newest.
    """
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")

    remote._latest_jobid()

    jobs = tmp_path / "jobs"
    # Submission order: bare first (oldest), -7 last (newest, live). meta mtime
    # encodes that; name order does not.
    order = ["sub_100keV", "sub_100keV-2", "sub_100keV-7"]
    for age, name in enumerate(reversed(order)):
        job = jobs / name
        job.mkdir(parents=True)
        meta = job / "meta"
        meta.write_text("slurm_job_id: 900\n")
        stamp = 1_700_000_000 + (len(order) - age) * 60
        os.utime(meta, (stamp, stamp))

    command = commands[0].replace(
        f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs}"'
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sub_100keV-7"


def test_jobs_report_identifiers_materials_and_last_event(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
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


def test_jobs_reports_cpu_only_mode_and_phase_state(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: "cpu-job\t48291\tcpu-only\tscan\thopg\tprofiling CPU hopg [1/1]\n",
    )

    remote.list_jobs()

    output = capsys.readouterr().out
    assert "CPU-only" in output
    assert "profiling CPU hopg [1/1]" in output
