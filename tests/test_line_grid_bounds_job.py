import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _load_job_script():
    path = Path(__file__).parents[1] / "scripts" / "line_grid_bounds_job.py"
    spec = importlib.util.spec_from_file_location("line_grid_bounds_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_slice_shell_has_fail_closed_handoff_contract():
    job = _load_job_script()

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


def test_batch_script_uses_three_times_slice_budget_backstop():
    job = _load_job_script()
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


def test_batch_script_records_signal_termination(tmp_path, monkeypatch):
    job = _load_job_script()
    monkeypatch.setattr(job.remote, "REMOTE_DIR", str(tmp_path))
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
    job = _load_job_script()
    assert "scripts/analyze_line_grid_bounds.py" in job.remote.SYNC_PATHS
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


def _run_payload(tmp_path, monkeypatch, analysis_exit):
    job = _load_job_script()
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
    monkeypatch.setattr(job.remote, "REMOTE_DIR", str(remote_root))
    monkeypatch.setattr(job.remote, "REMOTE_UV", str(fake_uv))
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
    job = _load_job_script()
    assert job.DEFAULT_ENERGIES == "30,50,100,150,200,250,300"


def test_slice_payload_threads_brem_grid_stop():
    job = _load_job_script()
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


def test_metadata_records_brem_grid_stop():
    job = _load_job_script()
    meta = job._metadata(
        "20260720-000000-abcdef01",
        slice_minutes=10.0,
        json_out="bounds.json",
        energies=job.DEFAULT_ENERGIES,
        grid_stop=20000.0,
        brem_grid_stop=40000.0,
    )
    assert "brem_grid_stop: 40000" in meta
