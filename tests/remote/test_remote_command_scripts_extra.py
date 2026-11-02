"""Additional remote scripts tests."""

import os
import shutil
import subprocess

import pytest

from pyrite import remote
from pyrite.remote import config, scripts, transport
from pyrite.remote import queue as lifecycle_queue
from tests.remote._remote_test_helpers import (
    _bash_or_skip,
    _use_amd_target,
)


@pytest.mark.parametrize(
    ("state_text", "exit_code"),
    [("queued slice 2", 0), ("FAILED (3 slices without checkpoint progress)", 1)],
)
def test_batch_exit_trap_preserves_chunk_handoff_and_failure(
    monkeypatch, tmp_path, state_text, exit_code
):
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    payload = f'echo "{state_text}" > "$JOBDIR/state"\nexit {exit_code}'
    script = scripts._slurm_batch_script(
        "j", payload, job_name="pyrite-j", time_limit="30", chunked=True
    )
    assert "#SBATCH --signal=B:USR1@120" in script
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    assert result.returncode == exit_code
    assert (jobdir / "state").read_text().strip() == state_text


def test_start_reports_the_submitted_slurm_job_id(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg"], no_sync=True)

    output = capsys.readouterr().out
    assert "· SUBMITTED" in output
    assert "SLURM" in output
    assert "48291" in output
    assert "pyrite remote job status" in output
    assert "pyrite remote job attach" in output
    assert "pyrite remote status" not in output
    assert "pyrite remote attach" not in output


def test_intel_remote_vendor_fails_before_script_generation(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_GPU_VENDOR", "intel")

    with pytest.raises(ValueError, match="do not yet support intel"):
        scripts._slurm_batch_script("job1", "echo ok", job_name="pyrite-test")


def test_amd_slurm_batch_script_selects_rocm_target_without_nvidia_modules(monkeypatch):
    _use_amd_target(monkeypatch)

    script = scripts._slurm_batch_script("j", "echo payload", job_name="pyrite-j")

    assert "#SBATCH --partition=gpu-amd\n#SBATCH --nodelist=qlmc-ace\n#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:radeon8060s:1" in script
    assert "module" not in script
    assert "export PYRITE_MC_BACKEND=rocm" in script
    assert 'export UV_PROJECT_ENVIRONMENT="/path/to/pyrite/.venv-amd"' in script
    assert (
        '. "/path/to/pyrite/src/pyrite/remote/rocm_build_env.sh" && _pyrite_rocm_env "/path/to/pyrite"'
        in script
    )
    assert 'echo "partition: gpu-amd"' in script
    assert "rocminfo" in script
    assert "rocm-smi" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_amd_uv_sync_builds_cupy_against_hip(monkeypatch):
    _use_amd_target(monkeypatch)

    block = scripts._uv_sync_block()

    assert (
        'CUPY_INSTALL_USE_HIP=1 "uv" sync --package pyrite-mc --no-default-groups --extra amd'
        in block
    )


def test_nvidia_uv_sync_does_not_request_hip():
    block = scripts._uv_sync_block()

    assert "CUPY_INSTALL_USE_HIP" not in block
    assert "--extra nvidia" in block


@pytest.mark.parametrize(
    ("attribute", "value", "key"),
    [
        ("REMOTE_GPU_VENDOR", "arm", "remote.gpu_vendor"),
        ("SLURM_PARTITION", "gpu amd", "remote.partition"),
        ("SLURM_NODELIST", "ace;reboot", "remote.nodelist"),
        ("SLURM_GRES", "gpu:1\n#SBATCH --x", "remote.gres"),
    ],
)
def test_slurm_profile_rejects_unsafe_values(monkeypatch, attribute, value, key):
    monkeypatch.setattr(config, attribute, value)

    with pytest.raises(SystemExit, match=key):
        scripts._slurm_batch_script("j", "echo payload", job_name="pyrite-j")


def test_slurm_batch_script_requests_the_lab_gpu_profile():
    script = remote._slurm_batch_script("j", "echo payload", job_name="pyrite-j")

    assert "#SBATCH --job-name=pyrite-j" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=UNLIMITED" in script
    assert "module purge 2>/dev/null || true" in script
    assert "module load cuda openmpi hdf5 2>/dev/null || true" in script
    assert "echo payload" in script
    assert "--mem" not in script
    assert "--nodelist" not in script
    assert "PYRITE_MC_BACKEND" not in script
    assert "rocm" not in script


def test_slurm_batch_script_requests_optional_memory_per_cpu():
    script = remote._slurm_batch_script(
        "j", "echo payload", job_name="pyrite-j", mem_per_cpu="4000M"
    )

    assert "#SBATCH --mem-per-cpu=4000M" in script


def test_slurm_batch_script_releases_its_checkpoint_reservations():
    script = remote._slurm_batch_script(
        "j", "echo payload", job_name="pyrite-j", reservation_stems=["hopg"]
    )

    assert "release_reservations" in script
    assert '"$RESERVATIONS/hopg/jobid"' in script
    assert '"$JOBID"' in script


def test_slurm_job_id_reads_recorded_scheduler_id(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "48291\n"
    )

    assert remote._slurm_job_id("j") == "48291"
    assert "slurm_job_id" in commands[0]


def test_slurm_state_queries_squeue(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "RUNNING\n"
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

    monkeypatch.setattr(transport, "_ssh_capture", run_remote)
    squeue.write_text(
        "#!/bin/sh\necho 'slurm_load_jobs error: Invalid job id specified' >&2\nexit 1\n"
    )
    squeue.chmod(0o755)
    assert remote._slurm_state("48291") is None

    squeue.write_text("#!/bin/sh\necho 'controller unavailable' >&2\nexit 7\n")
    with pytest.raises(SystemExit, match="ssh command failed"):
        remote._slurm_state("48291")


@pytest.mark.parametrize("value", ["/safe path", "/safe#comment"])
def test_sbatch_rejects_remote_dir_unsupported_by_directives(monkeypatch, value):
    monkeypatch.setattr(config, "REMOTE_DIR", value)

    with pytest.raises(SystemExit, match="SBATCH"):
        remote._slurm_batch_script("j", ":", job_name="pyrite-j")


def test_submit_command_uses_sbatch_parsable_and_records_scheduler_id():
    command = remote._submit_slurm_command("20260715-120000")

    assert "sbatch --parsable" in command
    assert "slurm_job_id" in command
    assert "jobs/20260715-120000/run.sh" in command
    assert "nohup" not in command and "setsid" not in command


def test_parallel_queue_script_preserves_partial_failure_results(monkeypatch, tmp_path):
    """Run the generated payload: one failed child must not hide two successes."""
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        'case "$*" in *hbn*) exit 7 ;; esac\n'
        "exit 0\n"
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._queue_script(
        "j", ["hopg", "hbn", "hfse2"], quick=False, workers=4, parallel_materials=2
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 1, result.stderr
    assert (jobdir / "state").read_text().startswith("FAILED (1 of 3 material(s) failed)")
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
        'case "$*" in *"tables verify"*) exit 0 ;; esac\n'
        f'echo "start $$" >> "{trace.as_posix()}"\n'
        "sleep 0.3\n"
        f'echo "end $$" >> "{trace.as_posix()}"\n'
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
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
