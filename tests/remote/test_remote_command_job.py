"""Remote job tests."""

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.console.dashboard import render as dashboard_render
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


def test_job_script_waits_on_shared_sync_lock_before_payload():
    script = scripts._slurm_batch_script("j1", "echo payload", job_name="n")
    lock = script.index(f"flock -s -w {config.SYNC_LOCK_WAIT_SECONDS}")
    assert config.SYNC_LOCK_NAME in script
    assert lock < script.index("echo payload")


@pytest.mark.parametrize(
    ("job_state", "expected"),
    [
        ("done [3/3] 2026-09-19T00:00:00+00:00", True),
        ("done CPU profile [3/3] 2026-09-19T00:00:00+00:00", True),
        ("done with 2 warning(s) [3/3] 2026-09-19T00:00:00+00:00", False),
        ("FAILED (exit 1) 2026-09-19T00:00:00+00:00", False),
        ("cancelled [932] 2026-09-19T00:00:00+00:00", False),
        ("running hopg [1/3] since 2026-09-19T00:00:00+00:00", False),
        ("", False),
    ],
)
def test_job_succeeded_distinguishes_clean_done_from_step_warnings(
    monkeypatch, job_state, expected
):
    monkeypatch.setattr(state, "_job_state", lambda _jobid: job_state)

    assert state._job_succeeded("j") is expected


def test_job_status_reports_scheduler_state_not_process_liveness(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or dashboard_state._encode_sections(
                {
                    "JOB": "j",
                    "META": "job: j\nmaterials: hopg\nquick: False\n"
                    "chunk_minutes: 10\nslurm_job_id: 48291",
                    "STATE": "running hopg",
                    "SQUEUE": "job_id=48291|state=RUNNING",
                }
            )
        ),
    )

    remote.job_status("j")

    output = capsys.readouterr().out
    assert "JOB j" in output
    assert "State      RUNNING" in output
    assert output.count("State") == 1
    assert "hopg" in output
    assert "standard · chunked into 10 min slices" in output
    assert "squeue" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_job_status_sanitizes_hostile_remote_fields(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: dashboard_state._encode_sections(
            {
                "JOB": "j\x1b[2J",
                "META": "job: j\nmaterials: ho\u202epg\nslurm_job_id: 48291",
                "STATE": "running\x9b2J hopg",
                "SQUEUE": "job_id=48291|state=RUNNING|reason=\x1b[31mforged",
                "LOG": "diagnostic\x1b[Hline",
            }
        ),
    )

    remote.job_status("j", detail=2)

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "\x9b" not in output
    assert "\u202e" not in output
    assert "j?[2J" in output
    assert "diagnostic?[Hline" in output


def test_job_status_verbose_adds_scheduler_allocation_fields(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or dashboard_state._encode_sections(
                {
                    "JOB": "j",
                    "META": "job: j\nmaterials: hopg\nworkers: None\nslurm_job_id: 48291",
                    "STATE": "running hopg",
                    "SQUEUE": "job_id=48291|state=RUNNING|name=pyrite-j|partition=gpu|"
                    "elapsed=1:02|left=UNLIMITED|nodes=1|reason=None",
                }
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
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or dashboard_state._encode_sections(
                {
                    "JOB": "j",
                    "META": "job: j\nmaterials: hopg\nslurm_job_id: 48291",
                    "STATE": "running hopg",
                    "SQUEUE": "job_id=48291|state=RUNNING",
                    "PROGRESS": '{"material":"hopg","total_cases":5,"cached_cases":1,'
                    '"completed_new_cases":2,"state":"running"}',
                    "RESOURCES": (
                        "cpu_percent=25.0|memory_used_bytes=8589934592|"
                        "memory_total_bytes=17179869184|memory_percent=50.0|"
                        "gpu_percent=80.0|vram_used_mib=12000|"
                        "vram_total_mib=24000|vram_percent=50.0"
                    ),
                    "LOG": "last log line",
                }
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
    assert "COMPUTE USAGE" in output
    assert "CPU" in output and "25.0%" in output
    assert "Host memory" in output and "8.0/16.0 GiB" in output
    assert "GPU" in output and "80.0%" in output
    assert "GPU VRAM" in output and "12000/24000 MiB" in output
    assert "COUPLING PROVENANCE" not in output
    assert "χ_g/U_g" not in output
    assert '{"material"' not in output
    assert "last log line" in output
    assert '"$D"/progress/*.json' in commands[0]
    assert 'tail -c 32768 "$D/log"' in commands[0]
    assert "nvidia-smi --query-gpu" in commands[0]


def test_job_attach_repeats_verbose_for_the_live_report(monkeypatch):
    """`remote status -a` retired at 0.3.0; `pyrite remote job attach` is the door."""
    from pyrite.cli.commands.job import command as job_command
    from pyrite.console import output as core

    calls = []
    monkeypatch.setattr(viewer, "attach", lambda jobid, detail: calls.append((jobid, detail)))

    core.run(job_command, ["attach", "j", "-vv"], prog_name="pyrite job")

    assert calls == [("j", 2)]


def test_job_profiles_parses_tab_separated_metadata(monkeypatch):
    seen = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: seen.append(command) or "job1\tsub_100keV\njob2\tstandard\n",
    )

    profiles = state._job_profiles(["job1", "job2"])

    assert profiles == {"job1": "sub_100keV", "job2": "standard"}
    assert "catalog_profile" in seen[0]


def test_job_status_displays_resumable_timeout_event():
    output = dashboard_render.format_job_status(
        {
            "JOB": "j",
            "META": "job: j\nkind: scan\nmaterials: hopg\nslurm_job_id: 4242",
            "STATE": "queued slice 2 now",
            "SQUEUE": "job_id=4242|state=PENDING",
            "TIMEOUT": "resumable timeout: hopg [1/1] 2026-09-29T12:00:00Z",
        },
        detail=0,
    )
    assert "Last timeout" in output
    assert "resumable timeout: hopg [1/1]" in output


def test_job_reserved_stems_command_filters_by_owner_and_reports_written(monkeypatch):
    command = scripts._job_reserved_stems_command("job-1")
    assert 'cat "$d/jobid"' in command and '"job-1"' in command
    assert 'echo "written $s"' in command and 'echo "unwritten $s"' in command
    with pytest.raises(SystemExit, match="invalid remote shell token"):
        scripts._job_reserved_stems_command("bad id;")
