import os
import shlex
import subprocess

import pytest

from cxr_mc.energy_grid import job


def test_generated_slice_shell_has_fail_closed_handoff_contract():
    shell = job._slice_payload(
        "20260719-120000-deadbeef",
        slice_minutes=10.0,
        json_out="line_grid_bounds_rerun.json",
        energies="200,250,300",
        grid_stop=30000.0,
    )

    state_pos = shell.index("queued slice")
    submit_pos = shell.index("sbatch --parsable --nice=10000")
    assert state_pos < submit_pos
    assert "SID=${SID%%;*}" in shell
    assert "*[!0-9]*" in shell
    assert "FAILED (slice resubmission)" in shell
    assert '[ -f "$JOBDIR/STOP" ]' in shell
    assert "--max-minutes 10" in shell
    assert "CXR_MC_FREE_EVERY=40" in shell
    assert "CXR_MC_FREE_WATERMARK_MB=15000" in shell
    assert "CXR_MC_TIMING=1" in shell
    assert "python -m cxr_mc.energy_grid.derive" in shell
    assert "scripts/analyze_line_grid_bounds.py" not in shell


def test_batch_script_uses_three_times_slice_budget_backstop():
    script = job._job_script(
        "20260719-120000-deadbeef",
        slice_minutes=10.0,
        json_out="line_grid_bounds_rerun.json",
        energies="200,250,300",
        grid_stop=30000.0,
    )

    assert "#SBATCH --time=30" in script
    assert '"queued slice"*) ;;' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/result.json",
        "../result.json",
        "nested/result.json",
        "result.txt",
        "result\nforged.json",
    ],
)
def test_remote_json_output_is_constrained_to_json_basename(name):
    with pytest.raises(SystemExit, match="--json-out"):
        job._slice_payload(
            "20260719-120000-deadbeef",
            slice_minutes=10.0,
            json_out=name,
            energies="30",
            grid_stop=20_000.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jobid", "job\nkind: forged"),
        ("energies", "30\nmaterials: forged"),
        ("materials", "hopg\x1b[31m"),
    ],
)
def test_metadata_rejects_control_characters(field, value):
    kwargs = dict(
        jobid="20260719-120000-deadbeef",
        slice_minutes=10.0,
        json_out="result.json",
        energies="30",
        grid_stop=20_000.0,
        materials="hopg",
    )
    kwargs[field] = value

    label = field if field == "jobid" else f"--{field}"
    with pytest.raises(SystemExit, match=f"{label}.*control"):
        job._metadata(**kwargs)


def test_batch_script_records_signal_termination(tmp_path, monkeypatch):
    monkeypatch.setattr(job.remote.config, "REMOTE_DIR", str(tmp_path))
    jobid = "20260719-120000-deadbeef"
    jobdir = tmp_path / job.remote.JOBS_SUBDIR / jobid
    jobdir.mkdir(parents=True)
    script = job.remote._slurm_batch_script(
        jobid,
        "kill -TERM $$",
        job_name="line-grid-bounds",
        time_limit="30",
    )

    result = subprocess.run(["bash"], input=script, text=True, check=False)

    assert result.returncode == 143
    assert (jobdir / "state").read_text().startswith("FAILED (signal TERM)")


def test_start_submits_slice_zero_with_nice(monkeypatch):
    assert "scripts/analyze_line_grid_bounds.py" not in job.remote.SYNC_PATHS
    calls = []
    monkeypatch.setattr(job.remote, "sync_code", lambda: calls.append("sync"))
    monkeypatch.setattr(job.remote, "_new_jobid", lambda: "20260719-120000-deadbeef")
    monkeypatch.setattr(job.remote, "_stage_job_script", lambda *args: calls.append(args))
    monkeypatch.setattr(
        job.remote,
        "_submit_staged_job",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "220",
    )

    job.start(slice_minutes=10.0)

    assert calls[-1][1] == {"nice": True}


def test_start_dry_run_threads_geometry_and_set_default(monkeypatch, capsys):
    monkeypatch.setattr(job.remote, "_new_jobid", lambda: "20260719-120000-deadbeef")

    job.start(
        tilts="0,1.5",
        azimuths="45,90",
        thickness="1000,2000",
        set_default=True,
        brem_step=12.5,
        dry_run=True,
    )

    script = capsys.readouterr().out
    assert "--tilts 0,1.5" in script
    assert "--azimuths 45,90" in script
    assert "--thickness 1000,2000" in script
    assert "--set-default" in script
    assert "--brem-step 12.5" in script


def _run_payload(tmp_path, monkeypatch, analysis_exit):
    remote_root = tmp_path / "remote"
    jobid = "20260719-120000-deadbeef"
    jobdir = remote_root / "jobs" / jobid
    jobdir.mkdir(parents=True)
    (jobdir / "meta").write_text("slurm_job_id: 219\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(f"#!/bin/sh\nexit {analysis_exit}\n")
    fake_uv.chmod(0o755)
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\nexit 1\n")
    fake_sbatch.chmod(0o755)
    monkeypatch.setattr(job.remote.config, "REMOTE_DIR", str(remote_root))
    monkeypatch.setattr(job.remote.config, "REMOTE_UV", str(fake_uv))
    payload = job._slice_payload(
        jobid,
        slice_minutes=1.0,
        json_out="bounds.json",
        energies="200",
        grid_stop=30000.0,
    )
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = subprocess.run(["bash", "-c", payload], env=env, check=False)
    return result, (jobdir / "state").read_text()


def test_completed_analysis_writes_terminal_done_state(tmp_path, monkeypatch):
    result, state = _run_payload(tmp_path, monkeypatch, analysis_exit=0)
    assert result.returncode == 0
    assert state.startswith("done ")


def test_resubmit_failure_is_terminal_and_fail_closed(tmp_path, monkeypatch):
    result, state = _run_payload(tmp_path, monkeypatch, analysis_exit=75)
    assert result.returncode != 0
    assert state.startswith("FAILED (slice resubmission)")


def test_default_energies_span_all_seven_standard_beams():
    assert job.DEFAULT_ENERGIES == "30,50,100,150,200,250,300"


def test_slice_payload_threads_brem_grid_stop():
    shell = job._slice_payload(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies=job.DEFAULT_ENERGIES,
        grid_stop=20000.0,
        brem_grid_stop=40000.0,
    )
    assert "--brem-grid-stop 40000" in shell
    assert "--energies 30,50,100,150,200,250,300" in shell


def test_slice_payload_threads_geometry_and_set_default_to_derive():
    shell = job._slice_payload(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies="100,200",
        grid_stop=20000.0,
        materials="diamond,wse2",
        tilts="0,1.5; printf injected",
        azimuths="45,90",
        thickness="1000,2000",
        set_default=True,
    )

    derive_command = next(
        line for line in shell.splitlines() if "python -m cxr_mc.energy_grid.derive" in line
    )
    argv = shlex.split(derive_command)
    assert argv[argv.index("--tilts") + 1] == "0,1.5; printf injected"
    assert argv[argv.index("--azimuths") + 1] == "45,90"
    assert argv[argv.index("--thickness") + 1] == "1000,2000"
    assert "--set-default" in argv


def test_slice_payload_omits_unset_geometry_and_set_default():
    shell = job._slice_payload(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies="100,200",
        grid_stop=20000.0,
    )

    derive_command = next(
        line for line in shell.splitlines() if "python -m cxr_mc.energy_grid.derive" in line
    )
    assert "--tilts" not in derive_command
    assert "--azimuths" not in derive_command
    assert "--thickness" not in derive_command
    assert "--set-default" not in derive_command


def test_slice_payload_quotes_hostile_remote_dir(monkeypatch):
    monkeypatch.setattr(job.remote.config, "REMOTE_DIR", '/safe"; SENTINEL_LINE_GRID; #')

    shell = job._slice_payload(
        "j",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies="100",
        grid_stop=20000.0,
    )

    assert '\\"; SENTINEL_LINE_GRID' in shell
    subprocess.run(["bash", "-n"], input=shell, text=True, check=True)


def test_slice_payload_rejects_remote_uv_program_text(monkeypatch):
    monkeypatch.setattr(job.remote.config, "REMOTE_UV", "uv; SENTINEL_LINE_GRID #")

    with pytest.raises(SystemExit, match="CXR_REMOTE_UV"):
        job._slice_payload(
            "j",
            slice_minutes=10.0,
            json_out="bounds.json",
            energies="100",
            grid_stop=20000.0,
        )


def test_metadata_records_brem_grid_stop():
    meta = job._metadata(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies=job.DEFAULT_ENERGIES,
        grid_stop=20000.0,
        brem_grid_stop=40000.0,
    )
    assert "brem_grid_stop: 40000" in meta


def test_metadata_records_explicit_brem_step():
    meta = job._metadata(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies=job.DEFAULT_ENERGIES,
        grid_stop=20000.0,
        brem_step=12.5,
    )
    assert "brem_step: 12.5" in meta
