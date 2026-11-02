"""Remote status tests."""

import io
import json
import os
import shutil
import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.console import dashboard as presentation
from pyrite.console.dashboard import state as dashboard_state
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
    _remote_main,
)


def test_status_formats_latest_performance_profile():
    sections = {
        "JOB": "j",
        "META": (
            "job: j\nmaterials: hopg\nworkers: 4\n"
            "progress_dashboard: True\nperformance_profile: baseline\n"
        ),
        "STATE": "running hopg",
        "SQUEUE": "job_id=1|state=RUNNING",
        "PROGRESS": json.dumps(
            {
                "material": "hopg",
                "total_cases": 10,
                "cached_cases": 0,
                "completed_new_cases": 2,
                "state": "running",
            }
        ),
        "PERFORMANCE": json.dumps(
            {
                "schema": "cxr.performance.v1",
                "material": "hopg",
                "profile": "baseline",
                "cpu_percent": 80.0,
                "memory_percent": 40.0,
                "process_rss_bytes": 3 * 1024**3,
                "gpu_percent": 90.0,
                "vram_percent": 50.0,
                "effective_workers": 4,
                "spec_chunk": 2000,
                "brem_chunk": 4000,
            }
        ),
    }

    output = presentation.format_job_status(sections, 0)

    assert "PERFORMANCE PROFILE" in output
    assert "CPU  80%" in output
    assert "GPU  90%" in output
    assert "workers 4" in output
    assert "chunks 2000/4000" in output


def test_status_snapshot_uses_one_partition_bounded_squeue_query():
    command = viewer._status_remote_command('JOB="j"', 2)

    assert command.count("squeue ") == 1
    assert '--partition="$PART"' in command
    assert '#SBATCH --partition=//p" "$D/run.sh"' in command
    assert 'PART="gpu"' in command
    assert "--states=PENDING,RUNNING" in command
    assert "--sort=-p,i" in command


def test_status_snapshot_preserves_squeue_failure(monkeypatch, tmp_path):
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    (job / "meta").write_text("job: j\nslurm_job_id: 20\n")
    (job / "state").write_text("queued\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text("#!/bin/sh\necho scheduler-unavailable >&2\nexit 17\n")
    squeue.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))

    result = subprocess.run(
        ["bash", "-c", viewer._status_remote_command('JOB="j"', 0)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode == 17
    assert "scheduler-unavailable" in result.stderr


def test_status_snapshot_ranks_job_in_its_own_partition(monkeypatch, tmp_path):
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    (job / "meta").write_text("job: j\nslurm_job_id: 20\n")
    (job / "state").write_text("queued\n")
    (job / "run.sh").write_text("#!/usr/bin/env bash\n#SBATCH --partition=gpu-amd\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SQUEUE_ARGS"\n')
    squeue.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    args = tmp_path / "squeue-args"

    result = subprocess.run(
        ["bash", "-c", viewer._status_remote_command('JOB="j"', 0)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SQUEUE_ARGS": str(args),
        },
        check=True,
    )

    assert "--partition=gpu-amd" in args.read_text().splitlines()
    queue = presentation.marked_sections(result.stdout)["QUEUE"]
    assert queue.startswith("cohort_partition=gpu-amd|")


def test_status_framing_keeps_marker_like_payload_inside_original_section():
    payload = "running\n@@META\njob: forged\nCXR_REMOTE_V1\tJOB\tZm9yZ2Vk"
    wire = dashboard_state._encode_sections({"STATE": payload, "JOB": "real"})

    sections = presentation.marked_sections(wire)

    assert sections == {"STATE": payload, "JOB": "real"}


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("base64") is None,
    reason="bash and base64 required",
)
def test_status_shell_framing_round_trip_resists_payload_markers(monkeypatch, tmp_path):
    job = tmp_path / "jobs" / "j"
    (job / "progress").mkdir(parents=True)
    (job / "meta").write_text("job: j\nmaterials: hopg\n")
    hostile = "@@META\nCXR_REMOTE_V1\tJOB\tZm9yZ2Vk\nstate"
    (job / "state").write_text(hostile)
    (job / "log").write_text("log\n@@STATE\nforged")
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))

    result = subprocess.run(
        ["bash", "-c", viewer._status_remote_command('JOB="j"', 2)],
        check=True,
        capture_output=True,
        text=True,
    )

    sections = presentation.marked_sections(result.stdout)
    assert sections["JOB"] == "j"
    assert sections["STATE"] == hostile
    assert sections["LOG"] == "log\n@@STATE\nforged"


def test_status_resource_probes_are_highest_verbosity_only(monkeypatch):
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")

    remote.job_status("j", detail=1)

    assert "nvidia-smi" not in commands[0]
    assert "top -bn1" not in commands[0]
    assert "free -b" not in commands[0]


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
        transport,
        "_ssh_capture",
        lambda _command: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                "STATE": "running hopg [1/1]",
                "SQUEUE": "job_id=48291|state=RUNNING",
                "PROGRESS": "",
                "LOG": legacy_log,
            }
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


def test_status_cli_repeats_verbose_for_case_progress(monkeypatch):
    calls = []
    monkeypatch.setattr(viewer, "job_status", lambda jobid, detail: calls.append((jobid, detail)))

    _remote_main(["status", "j", "-vv"])

    assert calls == [("j", 2)]


def test_status_attach_rejects_json(monkeypatch):
    monkeypatch.setattr(viewer, "attach", lambda *_args: pytest.fail("must reject before attach"))

    assert _remote_main(["status", "--attach", "-o", "json"]) == 2


def test_status_stream_uses_one_ssh_process_for_multiple_frames(monkeypatch):
    processes = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                f"first\n{viewer._STATUS_FRAME_END}\nsecond\n{viewer._STATUS_FRAME_END}\n"
            )
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.terminated = True
            return 0

        def kill(self):
            self.terminated = True

    def popen(command, **kwargs):
        process = FakeProcess()
        processes.append((command, kwargs, process))
        return process

    monkeypatch.setattr(viewer.subprocess, "Popen", popen)
    stream = viewer._status_stream("printf snapshot")

    assert next(stream) == "first\n"
    assert next(stream) == "second\n"
    stream.close()

    assert len(processes) == 1
    command, kwargs, process = processes[0]
    assert command[:3] == ["ssh", "-n", remote.HOST]
    assert "while :; do printf snapshot" in command[3]
    assert "sleep 2" in command[3]
    assert kwargs["stdout"] is subprocess.PIPE
    assert process.terminated


def test_status_stream_reconnects_twice_before_disconnect(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self, output, returncode):
            self.stdout = io.StringIO(output)
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    processes = iter(
        [
            FakeProcess("", 255),
            FakeProcess(f"recovered\n{viewer._STATUS_FRAME_END}\n", 0),
        ]
    )
    monkeypatch.setattr(viewer.subprocess, "Popen", lambda *_args, **_kwargs: next(processes))
    stream = viewer._status_stream("printf snapshot")

    assert next(stream) == "recovered\n"
    stream.close()

    assert "reconnecting 1/2" in capsys.readouterr().err


def test_status_stream_explains_disconnect_after_retries(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("")

        def poll(self):
            return 255

        def wait(self, timeout=None):
            return 255

    monkeypatch.setattr(viewer.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    with pytest.raises(SystemExit, match="after 2 reconnect attempts.*job is unaffected"):
        next(viewer._status_stream("printf snapshot"))

    assert capsys.readouterr().err.count("reconnecting") == 2


def test_status_stream_composes_as_valid_bash():
    snapshot = viewer._status_remote_command('JOB="j"', 2)
    command = viewer._status_stream_command(snapshot)

    syntax = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)

    assert syntax.returncode == 0, syntax.stderr


def test_status_cpu_phase_uses_cpu_progress_without_hiding_primary():
    sections = {
        "JOB": "j",
        "META": (
            "job: j\nmaterials: hopg\nquick: False\nchunk_minutes: 0\n"
            "parallel_materials: 1\nperformance_profile: baseline\n"
            "nsys: False\ncpu: True\ncpu_only: False\n"
        ),
        "STATE": "profiling CPU hopg [1/1]",
        "SQUEUE": "job_id=1|state=RUNNING",
        "PROGRESS": "\n".join(
            [
                '{"material":"hopg","phase":"primary","total_cases":4,'
                '"cached_cases":0,"completed_new_cases":4,"state":"done"}',
                '{"material":"hopg","phase":"cpu","total_cases":4,'
                '"cached_cases":0,"completed_new_cases":1,"state":"running"}',
            ]
        ),
    }

    output = presentation.format_job_status(sections, 0)

    assert "CPU profile" in output
    assert "profiling CPU hopg" in output
    assert "25%" in output
    assert output.count("HOPG") == 2


def test_status_shows_one_compute_bar_at_base_verbosity_when_cost_data_present(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                "STATE": "running hopg",
                "SQUEUE": "job_id=48291|state=RUNNING",
                "PROGRESS": '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":2,"state":"running",'
                '"done_cost":30.0,"total_cost":100.0}',
            }
        ),
    )

    remote.job_status("j")  # detail 0

    out = capsys.readouterr().out
    assert "Progress " in out
    assert "Progress (compute)" not in out  # single row at base verbosity
    assert "Progress (cases)" not in out
    assert " 30%" in out  # compute bar, not the 3/5=60% case bar


def test_status_shows_both_bars_at_verbose_when_cost_data_present(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                "STATE": "running hopg",
                "SQUEUE": "job_id=48291|state=RUNNING",
                "PROGRESS": '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":2,"state":"running",'
                '"done_cost":30.0,"total_cost":100.0}',
            }
        ),
    )

    remote.job_status("j", 1)  # -v

    out = capsys.readouterr().out
    assert "Progress (compute)" in out
    assert "Progress (cases)" in out
    assert " 30%" in out  # compute row
    assert " 60%" in out  # cases row (3/5)


def test_status_pending_uses_one_state_and_pauses_job_progress(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": (
                    "job: j\nmaterials: hopg hbn\nslurm_job_id: 48291\n"
                    "catalog_profile: sub_100keV\nquick: False"
                ),
                "STATE": "queued slice 2",
                "SQUEUE": "job_id=48291|state=PENDING",
                "PROGRESS": '{"material":"hopg","total_cases":4,"cached_cases":4,'
                '"completed_new_cases":0,"state":"done"}',
            }
        ),
    )

    remote.job_status("j", 1)

    out = capsys.readouterr().out
    assert out.count("State") == 1
    assert "State      PENDING" in out
    assert "\n  SLURM state" not in out
    assert "Ⅱ" in out
    assert "Materials  1/2 complete" in out
    assert "Profile    profile=sub_100keV" in out
    assert "Mode       standard" in out
    assert "standard · profile=" not in out


def test_status_progress_row_unchanged_without_cost_data(monkeypatch, capsys):
    """No cost data (rebrem/reline/legacy scan) -> the pre-item-6 single
    "Progress" row, at every verbosity, byte-for-byte the same as before."""
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                "STATE": "running hopg",
                "SQUEUE": "job_id=48291|state=RUNNING",
                "PROGRESS": '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":2,"state":"running"}',
            }
        ),
    )

    remote.job_status("j", 1)  # -v

    out = capsys.readouterr().out
    assert "Progress (compute)" not in out
    assert "Progress (cases)" not in out
    assert "Progress " in out
    assert " 60%" in out


def test_status_renders_progress_bars_at_base_verbosity(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: dashboard_state._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                "STATE": "running hopg",
                "SQUEUE": "job_id=48291|state=RUNNING",
                "PROGRESS": '{"material":"hopg","total_cases":5,"cached_cases":1,'
                '"completed_new_cases":2,"state":"running"}',
            }
        ),
    )

    remote.job_status("j")  # detail 0

    out = capsys.readouterr().out
    assert "Progress" in out  # overall aggregate bar in the header
    assert "CASE PROGRESS" in out  # per-material block promoted to level 0
    assert "3/5" in out
    assert "ALLOCATION" not in out  # still -v only
    assert "RECENT LOG" not in out  # still -vv only


def test_status_always_fetches_progress_even_at_base_verbosity(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda cmd: commands.append(cmd) or "",
    )

    remote.job_status("j")  # detail 0

    assert '"$D"/progress/*.json' in commands[0]
    assert "tail -c 32768" not in commands[0]  # log tail stays -vv only
