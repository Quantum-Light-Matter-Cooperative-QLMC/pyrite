"""cxr_mc.remote (``cxr remote``): material-name validation and the
detached-queue runner generation.

These are pure-string/logic checks (no ssh), so they run anywhere. The one
exception is the clear-listing regression test, which executes the box-side
shell snippet under a local bash (skipped when bash is unavailable)."""

import io
import json
import os
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from cxr_mc import remote
from cxr_mc._remote import (  # noqa: F401
    cli,
    config,
    lifecycle,
    presentation,
    scripts,
    state,
    transport,
    viewer,
)


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


@pytest.mark.parametrize(
    "value",
    [
        "-oProxyCommand=touch-/tmp/pwn",
        "user@host",
        "host name",
        "host;touch",
        "host\nother",
        "",
    ],
)
def test_remote_host_rejects_ssh_option_and_shell_syntax_before_subprocess(monkeypatch, value):
    monkeypatch.setattr(config, "HOST", value)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid host must not reach ssh"),
    )

    with pytest.raises(SystemExit, match="CXR_REMOTE_HOST"):
        transport._ssh_capture(":")


def test_non_nvidia_remote_vendor_fails_before_script_generation(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_GPU_VENDOR", "amd")

    with pytest.raises(ValueError, match="do not yet support amd"):
        scripts._slurm_batch_script("job1", "echo ok", job_name="cxr-test")


def test_sync_rejects_hostile_scp_host_before_transport(monkeypatch):
    monkeypatch.setattr(config, "HOST", "-oProxyCommand=touch-/tmp/pwn")
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    monkeypatch.setattr(
        transport,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid host must not reach scp"),
    )

    with pytest.raises(SystemExit, match="CXR_REMOTE_HOST"):
        transport.sync_code()


def test_sync_excludes_generated_caches(monkeypatch, tmp_path):
    source = tmp_path / "src"
    (source / "pkg" / "__pycache__").mkdir(parents=True)
    (source / "pkg" / ".pytest_cache").mkdir()
    (source / "pkg" / "module.py").write_text("VALUE = 1\n")
    (source / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"bytecode")
    (source / "pkg" / ".pytest_cache" / "state").write_text("generated")
    archived = []

    def inspect_transfer(command, **_kwargs):
        if command[0] == "scp":
            with tarfile.open(command[1], "r:gz") as archive:
                archived.extend(archive.getnames())

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(transport, "_run", inspect_transfer)

    transport.sync_code()

    assert archived == ["src/pkg/module.py"]


def test_ssh_download_streams_bytes_and_removes_partial_failure(monkeypatch, tmp_path):
    destination = tmp_path / "incoming.pkl"

    def succeed(command, **kwargs):
        assert command[:3] == ["ssh", "-n", remote.HOST]
        kwargs["stdout"].write(b"payload")

    monkeypatch.setattr(transport, "_run", succeed)
    transport._ssh_download("cat artifact", destination)
    assert destination.read_bytes() == b"payload"

    def fail(_command, **kwargs):
        kwargs["stdout"].write(b"partial")
        raise subprocess.CalledProcessError(7, "ssh")

    monkeypatch.setattr(transport, "_run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        transport._ssh_download("cat artifact", destination)
    assert not destination.exists()


@pytest.mark.parametrize("value", ["relative/path", "/safe\ninjected", "/safe\0injected", ""])
def test_remote_dir_rejects_nonabsolute_and_control_values(monkeypatch, value):
    monkeypatch.setattr(config, "REMOTE_DIR", value)

    with pytest.raises(SystemExit, match="CXR_REMOTE_DIR"):
        remote._queue_script("j", ["hopg"], quick=False, workers=None)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: remote._queue_script("j", ["hopg"], False, None), id="scan-monolithic"
        ),
        pytest.param(
            lambda: remote._chunked_queue_script("j", ["hopg"], False, None, 10),
            id="scan-chunked",
        ),
        pytest.param(lambda: remote._zhai_queue_script("j", 1, 1, 1, 0, False), id="zhai"),
        pytest.param(
            lambda: remote._rebrem_queue_script("j", ["hopg"], None, None, False),
            id="rebrem-monolithic",
        ),
        pytest.param(
            lambda: remote._rebrem_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="rebrem-chunked",
        ),
        pytest.param(
            lambda: scripts._reline_queue_script("j", ["hopg"], None, None, False),
            id="reline-monolithic",
        ),
        pytest.param(
            lambda: scripts._reline_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="reline-chunked",
        ),
    ],
)
def test_every_generated_bash_payload_quotes_hostile_remote_dir(monkeypatch, build):
    hostile = '/safe"; SENTINEL_REMOTE_DIR; #'
    monkeypatch.setattr(config, "REMOTE_DIR", hostile)

    script = build()

    expected_jobdir = config.shell_word(config.remote_path(config.JOBS_SUBDIR, "j"))
    assert f"JOBDIR={expected_jobdir}" in script
    assert "timing: uv sync %d.%03d s" in script
    syntax = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: scripts._release_checkpoint_stems_command("j", ["hopg"]),
            id="release-stems",
        ),
        pytest.param(lambda: scripts._release_job_reservations_command("j"), id="release-job"),
        pytest.param(
            lambda: scripts._reserve_checkpoint_stems_command("j", ["hopg"]),
            id="reserve",
        ),
        pytest.param(lambda: scripts._submit_slurm_command("j", ["hopg"]), id="submit"),
        pytest.param(
            lambda: scripts._write_job_script_command(
                config.remote_path(config.JOBS_SUBDIR, "j"), "job: j\n"
            ),
            id="upload",
        ),
        pytest.param(
            lambda: scripts._clear_checkpoint_stems_command("j", ["hopg"]),
            id="clear",
        ),
        pytest.param(lambda: scripts._reap_job_command("j"), id="reap"),
        pytest.param(lambda: viewer._status_remote_command('JOB="j"', 0), id="status"),
    ],
)
def test_every_remote_bash_command_family_quotes_hostile_remote_dir(monkeypatch, build):
    hostile = '/safe"; SENTINEL_REMOTE_DIR; #'
    monkeypatch.setattr(config, "REMOTE_DIR", hostile)

    command = build()

    assert '\\"; SENTINEL_REMOTE_DIR' in command or "'/safe\"; SENTINEL_REMOTE_DIR" in command
    syntax = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: remote._queue_script("j", ["hopg"], False, None), id="scan-monolithic"
        ),
        pytest.param(
            lambda: remote._chunked_queue_script("j", ["hopg"], False, None, 10),
            id="scan-chunked",
        ),
        pytest.param(lambda: remote._zhai_queue_script("j", 1, 1, 1, 0, False), id="zhai"),
        pytest.param(
            lambda: remote._rebrem_queue_script("j", ["hopg"], None, None, False),
            id="rebrem-monolithic",
        ),
        pytest.param(
            lambda: remote._rebrem_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="rebrem-chunked",
        ),
        pytest.param(
            lambda: scripts._reline_queue_script("j", ["hopg"], None, None, False),
            id="reline-monolithic",
        ),
        pytest.param(
            lambda: scripts._reline_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="reline-chunked",
        ),
    ],
)
def test_every_generated_bash_payload_rejects_remote_uv_program_text(monkeypatch, build):
    monkeypatch.setattr(config, "REMOTE_UV", "uv; SENTINEL_REMOTE_UV #")

    with pytest.raises(SystemExit, match="CXR_REMOTE_UV"):
        build()


@pytest.mark.parametrize("value", ["/safe path", "/safe#comment"])
def test_sbatch_rejects_remote_dir_unsupported_by_directives(monkeypatch, value):
    monkeypatch.setattr(config, "REMOTE_DIR", value)

    with pytest.raises(SystemExit, match="SBATCH"):
        remote._slurm_batch_script("j", ":", job_name="cxr-j")


def test_scp_remote_path_quotes_hostile_but_valid_posix_path(monkeypatch):
    monkeypatch.setattr(config, "HOST", "qlmc")
    path = "/srv/cxr data/$(touch SENTINEL)"

    rendered = config.scp_remote_path(path)

    assert rendered == "qlmc:'/srv/cxr data/$(touch SENTINEL)'"


def test_queue_script_has_per_material_scan_calls():
    s = remote._queue_script("20260101-000000", ["mose2", "wse2"], quick=True, workers=8)
    assert 'scan_launcher=("/home/aamador/.local/bin/uv" run --no-sync python)' in s
    assert '"${scan_launcher[@]}" -m cxr_mc._entry.scan' in s
    assert "--quick" in s and "--workers 8" in s
    assert "mose2" in s and "wse2" in s
    assert "20260101-000000" in s  # job id is embedded
    assert "mats=(mose2 wse2)" in s  # bash array drives the loop
    assert "parallel_materials=1" in s
    assert 'mkdir -p "$JOBDIR/progress"' in s
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    assert "--no-progress" in s
    assert "/pid" not in s


def test_queue_script_accepts_three_parallel_materials():
    script = remote._queue_script(
        "j", ["hopg", "hbn", "hfse2"], quick=False, workers=4, parallel_materials=3
    )

    assert "parallel_materials=3" in script
    assert "export CXR_MC_GPU_SHARE=3" in script  # co-tenants split the VRAM pool cap
    assert "wait -n" in script


def test_queue_script_records_uv_sync_timing_and_preserves_failure_state(monkeypatch, tmp_path):
    bash = _bash_or_skip(tmp_path)
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\nexit 7\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    script = remote._queue_script("j", ["hopg"], quick=False, workers=None)

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode != 0
    assert (jobdir / "state").read_text().startswith("FAILED (uv sync)")
    assert "timing: uv sync " in (jobdir / "log").read_text()


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s
    assert "--profile" not in s


def test_queue_script_profiles_uncached_repetitions_with_fixed_runtime_knobs():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        performance_profile="compute_test_300keV",
        performance_repetitions=3,
        performance_interval=1.0,
        spec_chunk=20_000,
        brem_chunk=10_000,
    )

    assert "export CXR_MC_SPEC_CHUNK=20000" in script
    assert "export CXR_MC_BREM_CHUNK=10000" in script
    assert "performance_repetitions=3" in script
    assert "repetition<=performance_repetitions" in script
    assert '--checkpoint-dir "$JOBDIR/performance-checkpoints/$m/$repetition"' in script
    assert "--perf-interval 1" in script


def test_queue_script_wraps_single_profile_session_with_nsys():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        performance_profile="compute_test_300keV",
        nsys=True,
    )
    metadata = scripts._queue_metadata(
        "j",
        ["mos2"],
        False,
        6,
        performance_profile="compute_test_300keV",
        nsys=True,
    )

    assert "command -v nsys" in script
    assert "export CXR_MC_NSYS=1" in script
    assert "nsys profile" in script
    assert "--trace=cuda,nvtx,osrt" in script
    assert "--wait=all" in script
    # Python stack-walkers stay opt-in (SIGSEGV on CPython 3.14 under nsys
    # 2025.6.x); gated behind CXR_MC_NSYS_PYSTACK, not on by default.
    assert 'if [ -n "${CXR_MC_NSYS_PYSTACK:-}" ]; then' in script
    assert "--python-backtrace=cuda" in script
    assert '--output="$trace_base"' in script
    assert 'scan_launcher=("/home/aamador/dev/cxr-mc/.venv/bin/python")' in script
    assert '--checkpoint-dir "$JOBDIR/performance-checkpoints/$m/$repetition"' in script
    assert "nsys stats" in script
    assert "--report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_sum" in script
    assert "nsys: True" in metadata
    bash = shutil.which("bash")
    if bash is not None:
        assert subprocess.run([bash, "-n"], input=script, text=True).returncode == 0


def test_queue_script_emits_positional_profile_when_not_standard():
    script = scripts._queue_script("j", ["mos2"], False, None, catalog_profile="sub_100keV")
    assert '_entry.scan "sub_100keV" -m "$m"' in script

    chunked = scripts._chunked_queue_script(
        "j", ["mos2"], False, None, 10, catalog_profile="sub_100keV"
    )
    assert '_entry.scan "sub_100keV" -m "$m"' in chunked


def test_queue_metadata_records_catalog_profile():
    default = scripts._queue_metadata("j", ["mos2"], False, None)
    assert "catalog_profile: standard" in default

    custom = scripts._queue_metadata("j", ["mos2"], False, None, catalog_profile="sub_100keV")
    assert "catalog_profile: sub_100keV" in custom


def test_queue_scripts_and_metadata_record_performance_profile():
    monolithic = scripts._queue_script(
        "j",
        ["mos2"],
        False,
        None,
        performance_profile="baseline",
    )
    chunked = scripts._chunked_queue_script(
        "j",
        ["mos2"],
        False,
        None,
        10,
        performance_profile="baseline",
    )
    metadata = scripts._queue_metadata(
        "j",
        ["mos2"],
        False,
        None,
        performance_profile="baseline",
    )

    for script in (monolithic, chunked):
        assert "--performance-profile baseline" in script
        assert '--performance-dir "$JOBDIR/performance"' in script
    assert "performance_profile: baseline" in metadata


def test_pull_performance_profile_fetches_each_matching_job(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: "job-1\thopg.ndjson\njob-2\tmos2.ndjson\n",
    )
    remote_commands = []

    def fake_download(command, destination):
        remote_commands.append(command)
        destination.write_text('{"schema":"cxr.performance.v1"}\n')

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    pulled = lifecycle.pull_performance_profile("baseline")

    assert pulled == [
        tmp_path / "performance-profiles" / "baseline" / "job-1" / "hopg.ndjson",
        tmp_path / "performance-profiles" / "baseline" / "job-2" / "mos2.ndjson",
    ]
    assert all(path.is_file() for path in pulled)
    assert all("/performance/baseline/" in command for command in remote_commands)


def test_pull_performance_profile_fetches_nsys_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: (
            "job-3\tmos2.ndjson\n"
            "job-3\tmos2.nsys-rep\n"
            "job-3\tmos2.sqlite\n"
            "job-3\tmos2.nsys-stats.txt\n"
            "job-3\tunsafe.report\n"
        ),
    )

    def fake_download(_command, destination):
        destination.write_bytes(b"artifact")

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    pulled = lifecycle.pull_performance_profile("baseline")

    assert pulled == [
        tmp_path / "performance-profiles" / "baseline" / "job-3" / "mos2.ndjson",
        tmp_path / "performance-profiles" / "baseline" / "job-3" / "mos2.nsys-rep",
        tmp_path / "performance-profiles" / "baseline" / "job-3" / "mos2.sqlite",
        tmp_path / "performance-profiles" / "baseline" / "job-3" / "mos2.nsys-stats.txt",
    ]


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

    output = presentation._format_job_status(sections, 0)

    assert "PERFORMANCE PROFILE" in output
    assert "CPU  80%" in output
    assert "GPU  90%" in output
    assert "workers 4" in output
    assert "chunks 2000/4000" in output


def test_stems_predicts_qualified_stem_for_non_standard_catalog_profile(monkeypatch):
    import cxr_mc.profiles as profiles_module

    assert scripts._stems(["mos2"], False) == ["mos2"]

    calls = []

    def fake_named_profile_stem(material, fidelity, *, catalog_profile="standard"):
        calls.append((material, fidelity, catalog_profile))
        return f"{material}--{fidelity}-qualified"

    monkeypatch.setattr(profiles_module, "named_profile_stem", fake_named_profile_stem)

    assert scripts._stems(["mos2"], False, catalog_profile="sub_100keV") == ["mos2--full-qualified"]
    assert calls == [("mos2", "full", "sub_100keV")]


def test_queue_script_and_stem_resolve_survey_profile():
    from cxr_mc.profiles import named_profile_stem

    script = remote._queue_script("j", ["mos2"], quick=False, workers=None, fidelity="survey")
    assert "--fidelity survey" in script
    assert remote._stems(["mos2"], False, "survey") == [named_profile_stem("mos2", "survey")]


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
    assert "#SBATCH --cpus-per-task=8" in script
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

    real_datetime = scripts.datetime.datetime

    class FixedDatetime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 7, 15, 12, 0, 0)

    suffixes = iter(["a" * 32, "b" * 32])
    uploads = []
    monkeypatch.setattr(scripts.datetime, "datetime", FixedDatetime)
    monkeypatch.setattr(
        scripts,
        "uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(suffixes))),
        raising=False,
    )
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda command, **_kwargs: uploads.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    first = remote.start_queue(["hopg"], no_sync=True)
    second = remote.start_queue(["hbn"], no_sync=True)

    assert first == "20260715-120000-aaaaaaaa"
    assert second == "20260715-120000-bbbbbbbb"
    assert first != second
    assert all("mkdir -p" not in upload[-1] for upload in uploads)
    assert all("mkdir '" in upload[-1] for upload in uploads)


def test_checkpoint_reservation_rejects_a_second_stager_for_the_same_stem(monkeypatch, tmp_path):
    """The remote ``mkdir`` lock closes the gap between a busy check and sbatch."""
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))

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
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
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
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(state, "_slurm_state", lambda _scheduler_id: "RUNNING")
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))

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

    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(state, "_slurm_job_id", lambda _jobid: "48291")
    monkeypatch.setattr(state, "_slurm_state", lambda _scheduler_id: "RUNNING")

    def run_remote(command, **_kwargs):
        subprocess.run(
            ["bash", "-c", command[-1]],
            check=True,
            env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        )

    monkeypatch.setattr(transport, "_run", run_remote)

    with pytest.raises(subprocess.CalledProcessError):
        remote._stop_jobid("j")

    assert reservation.exists()
    assert (jobdir / "state").read_text().startswith("cancelling")


def test_remote_dry_run_describes_sbatch_submission(monkeypatch, capsys):
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

    remote.start_queue(["hopg"], dry_run=True)

    out = capsys.readouterr().out
    assert "#SBATCH --partition=gpu" in out
    assert "sbatch --parsable" in out
    assert "nohup" not in out and "setsid" not in out


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


def test_chunk_minutes_zero_emits_monolithic_script(monkeypatch, capsys):
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

    remote.start_queue(["hopg"], dry_run=True, chunk_minutes=0)

    out = capsys.readouterr().out
    assert "#SBATCH --time=UNLIMITED" in out
    assert "parallel_materials=1" in out
    assert "--max-minutes" not in out


def test_parallel_materials_rejected_in_chunked_mode(monkeypatch):
    with pytest.raises(SystemExit, match="chunk-minutes 0"):
        remote.start_queue(["hopg"], dry_run=True, parallel_materials=2)


def test_cli_start_chunk_flags(monkeypatch, capsys):
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

    remote.main(
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--dry-run",
            "--chunk-minutes",
            "0",
            "--parallel-materials",
            "3",
        ]
    )  # legal: monolithic
    assert (
        remote.main(["run", "standard", "-m", "hopg", "--dry-run", "--parallel-materials", "3"])
        == 2
    )  # illegal: chunked default


def test_start_writes_static_metadata_before_sbatch(monkeypatch):
    uploads = []
    submissions = []
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: submissions.append(command))
    monkeypatch.setattr(
        transport,
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
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *args, **kwargs: uploads.append((args, kwargs))
    )
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: submissions.append(command) or "48291\n"
    )

    remote.start_queue(["hopg"], no_sync=True)  # chunked is the default

    assert "chunk_minutes: 10.0" in uploads[0][0][0][-1]
    sbatch_commands = [command for command in submissions if "sbatch" in command]
    assert sbatch_commands, "the live path must reach sbatch"
    assert all("sbatch --parsable --nice=10000" in command for command in sbatch_commands)


def test_start_reports_the_submitted_slurm_job_id(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg"], no_sync=True)

    output = capsys.readouterr().out
    assert "· SUBMITTED" in output
    assert "SLURM" in output
    assert "48291" in output
    assert "cxr remote status" in output
    assert "--attach" in output
    assert "cxr remote attach" not in output


def test_start_profile_submit_suggests_profile_pull_not_a_stem_wall(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg", "hbn", "mose2", "wse2"], no_sync=True, catalog_profile="sub_100keV")

    output = capsys.readouterr().out
    assert "cxr remote pull --profile sub_100keV" in output
    assert "cxr remote pull hopg--" not in output  # no hash-stem wall


def test_start_standard_submit_suggests_stem_pull(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg"], no_sync=True)

    output = capsys.readouterr().out
    assert "cxr remote pull hopg " in output


def test_run_profile_uses_shipped_membership_without_material_option(capsys):
    """A positional profile selects its explicit material membership."""
    remote.main(["run", "sub_100keV", "--dry-run"])

    out = capsys.readouterr().out
    assert "hopg" in out and "zrte3" in out


def test_run_perf_uses_positional_catalog_profile(capsys):
    remote.main(["run", "sub_100keV", "--perf", "--dry-run"])

    out = capsys.readouterr().out
    assert "hopg" in out and "zrte3" in out
    assert '_entry.scan "sub_100keV" -m' in out
    assert "--performance-profile sub_100keV" in out


def test_run_rejects_retired_profile_option(capsys):
    result = remote.main(
        [
            "run",
            "sub_100keV",
            "--profile",
            "standard",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "No such option '--profile'" in capsys.readouterr().err


def test_run_rejects_perf_reps_without_perf(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "--perf-reps",
            "3",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--perf-reps requires --perf" in capsys.readouterr().err


def test_run_rejects_perf_reps_in_chunked_mode(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "--perf",
            "--perf-reps",
            "3",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--perf-reps requires --chunk-minutes 0" in capsys.readouterr().err


def test_run_rejects_nsys_without_performance_profile(capsys):
    result = remote.main(
        ["run", "compute_test_300keV", "--nsys", "--chunk-minutes", "0", "--dry-run"]
    )

    assert result == 2
    assert "--nsys requires --perf" in capsys.readouterr().err


def test_run_rejects_nsys_in_chunked_mode(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "--perf",
            "--nsys",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires --chunk-minutes 0" in capsys.readouterr().err


def test_run_rejects_nsys_with_multiple_repetitions(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "--perf",
            "--perf-reps",
            "2",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires --perf-reps 1" in capsys.readouterr().err


def test_run_rejects_nsys_with_multiple_materials(capsys):
    result = remote.main(
        [
            "run",
            "sub_100keV",
            "--perf",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires exactly one material" in capsys.readouterr().err


def test_run_performance_runtime_knobs_reach_monolithic_dry_run(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "--material",
            "mos2",
            "--perf",
            "--perf-reps",
            "3",
            "--perf-interval",
            "1",
            "--workers",
            "6",
            "--spec-chunk",
            "20000",
            "--brem-chunk",
            "10000",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result is None
    output = capsys.readouterr().out
    assert "#SBATCH --cpus-per-task=6" in output
    assert "--perf-interval 1" in output
    assert "performance_repetitions=3" in output
    assert "export CXR_MC_SPEC_CHUNK=20000" in output
    assert "export CXR_MC_BREM_CHUNK=10000" in output


def test_run_nsys_reaches_monolithic_dry_run(capsys):
    result = remote.main(
        [
            "run",
            "compute_test_300keV",
            "-m",
            "mos2",
            "-p",
            "--workers",
            "6",
            "--spec-chunk",
            "20000",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result is None
    output = capsys.readouterr().out
    assert "nsys profile" in output
    assert "nsys: True" in output
    assert "export CXR_MC_NSYS=1" in output


def test_run_defaults_to_attach_and_pull(monkeypatch):
    events = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda mats, **kwargs: events.append(("run", mats, kwargs)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: list(mats))
    monkeypatch.setattr(scripts, "_stems", lambda mats, *args, **kwargs: list(mats))
    monkeypatch.setattr(
        lifecycle, "pull", lambda stems, **kwargs: events.append(("pull", stems, kwargs))
    )

    remote.main(["run", "standard", "-m", "hopg", "--no-sync"])

    assert [event[0] for event in events] == ["run", "attach", "pull"]
    assert events[-1][2]["no_sync"] is True


def test_run_headless_skips_attach_and_pull(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: pytest.fail("must not attach"))
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: pytest.fail("must not pull"))

    remote.main(["run", "standard", "-m", "hopg", "--headless", "--no-sync"])


def test_run_no_pull_still_attaches(monkeypatch):
    attached = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda jobid: attached.append(jobid) or True)
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: pytest.fail("must not pull"))

    remote.main(["run", "standard", "-m", "hopg", "--no-pull", "--no-sync"])

    assert attached == ["j"]


def test_run_perf_reps_attach_but_skip_checkpoint_pull(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(
        state,
        "_completed_materials",
        lambda *_args: pytest.fail("must not resolve profiling checkpoints"),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda *_args, **_kwargs: pytest.fail("must not pull profiling checkpoints"),
    )

    remote.main(
        [
            "run",
            "compute_test_300keV",
            "-m",
            "mos2",
            "--perf",
            "--perf-reps",
            "3",
            "--chunk-minutes",
            "0",
            "--no-sync",
        ]
    )

    assert "isolated job-local checkpoints" in capsys.readouterr().err


def test_run_rejects_headless_with_no_pull(capsys):
    assert remote.main(["run", "standard", "-m", "hopg", "--headless", "--no-pull"]) == 2
    assert "--headless cannot be combined with --no-pull" in capsys.readouterr().err


@pytest.mark.parametrize("retired", ["scan", "submit", "start"])
def test_retired_remote_run_commands_are_removed(retired, capsys):
    assert remote.main([retired, "--help"]) == 2
    assert f"No such command '{retired}'" in capsys.readouterr().err


def _quiet_submission(monkeypatch):
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")


def test_profile_submit_names_job_after_profile(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_profile_jobdirs", lambda _profile: set())

    jobid = remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")

    assert jobid == "sub_100keV"


def test_profile_submit_suffixes_when_finished_run_holds_the_name(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(
        state,
        "_profile_jobdirs",
        lambda _profile: {"sub_100keV", "sub_100keV-2"},
    )

    jobid = remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")

    assert jobid == "sub_100keV-3"


def test_profile_submit_refuses_while_same_profile_is_live(monkeypatch):
    _quiet_submission(monkeypatch)
    monkeypatch.setattr(state, "_live_jobs", lambda: [("sub_100keV", False, ["hopg"])])
    monkeypatch.setattr(state, "_job_profiles", lambda jobids: {"sub_100keV": "sub_100keV"})

    with pytest.raises(SystemExit, match="already has a live job"):
        remote.start_queue(["hopg"], no_sync=True, catalog_profile="sub_100keV")


def test_profile_jobdirs_parses_existing_dirs(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "sub_100keV\nsub_100keV-2\n")

    assert state._profile_jobdirs("sub_100keV") == {"sub_100keV", "sub_100keV-2"}


def test_interrupted_job_upload_releases_its_checkpoint_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: pytest.fail("must not submit"))

    with pytest.raises(KeyboardInterrupt):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(commands) == 2
    assert 'J="' in commands[-1][-1]
    assert 'if [ "$(cat "$R/hopg/jobid"' in commands[-1][-1]


def test_ambiguous_submit_failure_inspects_queued_state_and_keeps_reservations(monkeypatch):
    commands = []
    captures = []
    monkeypatch.setattr(lifecycle, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)

    def capture(command):
        captures.append(command)
        if "sbatch --parsable" in command:
            raise SystemExit("ssh command failed (exit 255)")
        return "queued\n"

    monkeypatch.setattr(transport, "_ssh_capture", capture)

    with pytest.raises(SystemExit, match="ssh command failed"):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(captures) == 2
    assert "slurm_job_id" in captures[-1]
    assert 'cat "$D/state"' in captures[-1]
    assert len(commands) == 1  # reservation remains while queued submission is ambiguous


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


def test_job_status_reports_scheduler_state_not_process_liveness(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or presentation._encode_sections(
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


def _queue_payload(*rows, partition="gpu"):
    return "\n".join(
        [f"cohort_partition={partition}|order=priority_desc_job_id_asc", *rows]
    )


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

    context = presentation._pending_queue_context(payload, target)

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

    assert presentation._pending_queue_context(before, "20")["rank"] == 2
    assert presentation._pending_queue_context(after, "20")["rank"] == 1
    assert presentation._pending_queue_context(before, "999") is None
    assert presentation._pending_queue_context(
        before + "\njob_id=bad|state=PENDING|partition=gpu|priority=nan", "bad"
    ) is None
    injected = _queue_payload(
        "job_id=20|state=PENDING|name=x|priority=999|partition=gpu|"
        "reason=Priority|priority=10|user=u"
    )
    assert presentation._pending_queue_context(injected, "20") is None


def test_pending_queue_render_sanitizes_identity_reason_and_explains_backfill(capsys, monkeypatch):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: presentation._encode_sections(
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


def test_status_snapshot_uses_one_partition_bounded_squeue_query():
    command = viewer._status_remote_command('JOB="j"', 2)

    assert command.count("squeue ") == 1
    assert "--partition=" in command
    assert "gpu" in command
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


def test_status_framing_keeps_marker_like_payload_inside_original_section():
    payload = "running\n@@META\njob: forged\nCXR_REMOTE_V1\tJOB\tZm9yZ2Vk"
    wire = presentation._encode_sections({"STATE": payload, "JOB": "real"})

    sections = remote._marked_sections(wire)

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

    sections = remote._marked_sections(result.stdout)
    assert sections["JOB"] == "j"
    assert sections["STATE"] == hostile
    assert sections["LOG"] == "log\n@@STATE\nforged"


def test_job_status_sanitizes_hostile_remote_fields(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: presentation._encode_sections(
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


def test_job_status_verbose_adds_scheduler_allocation_fields(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or presentation._encode_sections(
                {
                    "JOB": "j",
                    "META": "job: j\nmaterials: hopg\nworkers: None\nslurm_job_id: 48291",
                    "STATE": "running hopg",
                    "SQUEUE": "job_id=48291|state=RUNNING|name=cxr-j|partition=gpu|"
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
            or presentation._encode_sections(
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
    assert "COUPLING PROVENANCE" in output
    assert "1 stored spectra; 2 cases recomputed χ_g/U_g" in output
    assert '{"material"' not in output
    assert "last log line" in output
    assert '"$D"/progress/*.json' in commands[0]
    assert 'tail -c 32768 "$D/log"' in commands[0]
    assert "nvidia-smi --query-gpu" in commands[0]


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
        lambda _command: presentation._encode_sections(
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
        transport,
        "_ssh_capture",
        lambda _command: presentation._encode_sections(
            {
                "JOB": "j",
                "META": "job: j\nkind: line-grid-bounds\nslice_minutes: 10\n"
                "energies: 200,250,300\nslurm_job_id: 227",
                "STATE": "FAILED (signal TERM) now",
                "SQUEUE": "job_id=227|state=NOT_QUEUED",
                "PROGRESS": "",
                "LOG": "cases: 40%|████      | 2/5 [17:31<26:18, 526.18s/it]\r",
            }
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
    monkeypatch.setattr(viewer, "job_status", lambda jobid, detail: calls.append((jobid, detail)))

    remote.main(["status", "j", "-vv"])

    assert calls == [("j", 2)]


def _status_output(state, *, squeue_state="RUNNING", sid="48291", progress=""):
    """Fake one round-trip of the shared status command (the marked sections
    _format_job_status / the attach loop read)."""
    return presentation._encode_sections(
        {
            "JOB": "j",
            "META": f"job: j\nmaterials: hopg\nslurm_job_id: {sid}",
            "STATE": state,
            "SQUEUE": f"job_id={sid}|state={squeue_state}",
            "PROGRESS": progress,
        }
    )


def test_attach_redraws_the_status_report_until_terminal(monkeypatch, capsys):
    running = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":2,"state":"running"}'
    )
    done = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":3,"state":"done"}'
    )
    outputs = iter(
        [
            _status_output("running hopg [1/1] since now", progress=running),
            _status_output("done [1/1] now", squeue_state="NOT_QUEUED", progress=done),
        ]
    )
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)

    assert remote.attach("20260101-000000") is True

    out = capsys.readouterr().out
    assert "ATTACHED" in out  # live banner with refresh counter
    assert "Progress" in out  # overall aggregate bar in the header
    assert "CASE PROGRESS" in out  # per-material bars at base verbosity
    assert "3/4" in out
    assert "JOB 20260101-000000 · FINISHED" in out
    assert "done [1/1] now" in out


def test_attach_sanitizes_hostile_state(monkeypatch, capsys):
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _command: iter(
            [_status_output("done\x1b[2J hopg [1/1] now", squeue_state="NOT_QUEUED")]
        ),
    )

    assert remote.attach("20260101-000000") is True

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "done?[2J hopg [1/1] now" in output


def test_attach_omits_log_tail_at_base_verbosity_but_still_fetches_progress(monkeypatch):
    commands = []
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda cmd: (
            commands.append(cmd) or iter([_status_output("done now", squeue_state="NOT_QUEUED")])
        ),
    )

    remote.attach("20260101-000000")

    assert '"$D"/progress/*.json' in commands[0]  # bars at every level need it
    assert "tail -c 32768" not in commands[0]  # log tail is still -vv only


def test_attach_forwards_double_verbose_to_the_status_command(monkeypatch):
    commands = []
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda cmd: (
            commands.append(cmd) or iter([_status_output("done now", squeue_state="NOT_QUEUED")])
        ),
    )

    remote.attach("20260101-000000", detail=2)

    assert 'tail -c 32768 "$D/log"' in commands[0]
    assert "squeue" in commands[0]
    assert "kill -0" not in commands[0]
    assert "/pid" not in commands[0]


def test_attach_watchdog_exits_on_a_stalled_chain(monkeypatch, capsys):
    output = _status_output("queued slice 2 now", squeue_state="NOT_QUEUED")
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _cmd: iter([output] * viewer._POLL_GRACE_POLLS),
    )

    assert remote.attach("20260101-000000") is False
    assert "CHAIN STALLED" in capsys.readouterr().out


@pytest.mark.parametrize("attach_flag", ["-a", "--attach"])
def test_status_attach_cli_repeats_verbose_for_the_live_report(monkeypatch, attach_flag):
    calls = []
    monkeypatch.setattr(viewer, "attach", lambda jobid, detail: calls.append((jobid, detail)))

    remote.main(["status", "j", "-vv", attach_flag])

    assert calls == [("j", 2)]


def test_remote_attach_command_is_absent(capsys):
    assert remote.main(["attach"]) == 2
    assert "No such command 'attach'" in capsys.readouterr().err


def test_status_attach_rejects_json(monkeypatch):
    monkeypatch.setattr(viewer, "attach", lambda *_args: pytest.fail("must reject before attach"))

    assert remote.main(["status", "--attach", "--json"]) == 2


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


def test_logs_identify_resolved_job_and_host(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "LOG j · qlmc\n\nline\n",
    )

    remote.tail_logs("j")

    assert capsys.readouterr().out == "LOG j · qlmc\n\nline\n"
    assert 'printf "LOG %s · qlmc' in commands[0]


def test_attach_returns_false_when_the_viewer_is_interrupted(monkeypatch, capsys):
    monkeypatch.setattr(
        viewer,
        "_status_stream",
        lambda _cmd: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert remote.attach("20260101-000000") is False
    assert "VIEWER DISCONNECTED" in capsys.readouterr().out


class _FakeKeyListener:
    """Scripted stand-in for viewer._KeyListener: one poll() result per call."""

    def __init__(self, polls, *, active=True):
        self._polls = list(polls)
        self.active = active
        self.stopped = False

    def poll(self):
        return self._polls.pop(0) if self._polls else []

    def stop(self):
        self.stopped = True


def test_key_listener_inactive_when_stdin_is_not_a_tty(monkeypatch):
    monkeypatch.setattr(viewer.sys.stdin, "isatty", lambda: False)

    listener = viewer._KeyListener()

    assert listener.active is False
    assert listener.poll() == []
    listener.stop()  # must be a harmless no-op


def test_attach_header_shows_cancel_hint_only_when_keys_are_active():
    assert f"{viewer._CANCEL_ARM_KEY} cancels job" in viewer._attach_header(1, cancel_hint=True)
    assert f"{viewer._PULL_ARM_KEY} pulls progress" in viewer._attach_header(1, cancel_hint=True)
    assert "cancels job" not in viewer._attach_header(1, cancel_hint=False)


def test_attach_header_shows_armed_confirm_banner():
    header = viewer._attach_header(1, armed=True)
    assert "CANCEL ARMED" in header
    assert f"press {viewer._CANCEL_CONFIRM_KEY} to confirm" in header


def test_attach_pull_keybinding_confirms_without_stopping_job(monkeypatch, capsys):
    running = _status_output(
        "running hopg [1/1] since now",
        squeue_state="RUNNING",
        progress=(
            '{"material":"hopg","total_cases":4,"cached_cases":1,'
            '"completed_new_cases":1,"state":"running"}'
        ),
    )
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: iter([running, running, done]))
    fake_keys = _FakeKeyListener([["p"], ["y"], []])
    monkeypatch.setattr(viewer, "_KeyListener", lambda: fake_keys)
    pulled = []
    monkeypatch.setattr(
        viewer,
        "_pull_attached_progress",
        lambda jobid, sections: pulled.append((jobid, sections)),
    )
    monkeypatch.setattr(
        lifecycle, "_stop_jobid", lambda _jobid: pytest.fail("pull must not cancel job")
    )

    assert remote.attach("20260101-000000") is True

    assert len(pulled) == 1
    assert pulled[0][0] == "20260101-000000"
    assert "PULL ARMED" in capsys.readouterr().out


def test_pull_attached_progress_uses_reporting_profile_stems(monkeypatch):
    sections = {
        "META": ("materials: hopg hbn\nquick: False\nfidelity: full\ncatalog_profile: sub_100keV"),
        "PROGRESS": (
            '{"material":"hopg","total_cases":4,"cached_cases":1,'
            '"completed_new_cases":1,"state":"running"}'
        ),
    }
    stem_calls = []
    pulls = []
    monkeypatch.setattr(
        scripts,
        "_stems",
        lambda materials, quick, fidelity, **kwargs: (
            stem_calls.append((materials, quick, fidelity, kwargs)) or ["hopg-profile"]
        ),
    )
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kwargs: pulls.append((stems, kwargs)))

    viewer._pull_attached_progress("j", sections)

    assert stem_calls == [
        (
            ["hopg"],
            False,
            "full",
            {"high_energy_min_kev": None, "catalog_profile": "sub_100keV"},
        )
    ]
    assert pulls == [(["hopg-profile"], {"no_sync": True})]


def test_attach_cancel_keybinding_confirms_and_scancels_the_job(monkeypatch, capsys):
    """item 5: 'x' arms, 'y' on a LATER poll confirms -- two different keys, not
    a bare single keystroke -- and cancels via lifecycle._stop_jobid."""
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    outputs = iter([running, running])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["y"]])
    monkeypatch.setattr(viewer, "_KeyListener", lambda: fake_keys)
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is False

    assert stopped == ["20260101-000000"]
    assert fake_keys.stopped is True
    out = capsys.readouterr().out
    assert "CANCEL ARMED" in out
    assert "CANCELLED BY USER" in out


def test_attach_cancel_keybinding_disarms_on_any_other_key(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    outputs = iter([running, done])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["z"]])
    monkeypatch.setattr(viewer, "_KeyListener", lambda: fake_keys)
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is True

    assert stopped == []  # wrong confirm key disarms instead of cancelling
    assert "FINISHED" in capsys.readouterr().out


def test_attach_cancel_arm_expires_without_a_confirm_key(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    done = _status_output("done now", squeue_state="NOT_QUEUED")
    outputs = iter([running, done])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], []])
    monkeypatch.setattr(viewer, "_KeyListener", lambda: fake_keys)
    # First frame arms at t=0 (armed_until = 0 + _CANCEL_ARM_SECONDS); every
    # later monotonic() call reads t=100, well past the window, so the second
    # frame's expiry check must disarm before it ever looks at (the empty)
    # keys.poll() for that frame.
    clock = iter([0.0])
    monkeypatch.setattr(viewer.time, "monotonic", lambda: next(clock, 100.0))
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", lambda jobid: stopped.append(jobid))

    assert remote.attach("20260101-000000") is True

    assert stopped == []
    out = capsys.readouterr().out
    assert out.count("CANCEL ARMED") == 1  # armed on frame 1, disarmed by frame 2
    assert "FINISHED" in out


def test_attach_cancel_request_failure_is_reported_without_a_traceback(monkeypatch, capsys):
    running = _status_output("running hopg [1/1] since now", squeue_state="RUNNING")
    outputs = iter([running, running])
    monkeypatch.setattr(viewer, "_status_stream", lambda _cmd: outputs)
    fake_keys = _FakeKeyListener([["x"], ["y"]])
    monkeypatch.setattr(viewer, "_KeyListener", lambda: fake_keys)

    def _raise(jobid):
        raise SystemExit(f"job {jobid} is not an active SLURM job")

    monkeypatch.setattr(lifecycle, "_stop_jobid", _raise)

    assert remote.attach("20260101-000000") is False

    assert "CANCEL REQUEST FAILED" in capsys.readouterr().out


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


def test_overall_progress_line_sums_cases_and_counts_done_materials():
    records = remote._parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":4,"cached_cases":2,'
                '"completed_new_cases":2,"state":"done"}',
                '{"material":"hbn","total_cases":6,"cached_cases":0,'
                '"completed_new_cases":3,"state":"running"}',
            ]
        )
    )

    line = remote._overall_progress_line(records)

    assert line is not None
    assert "7/10 cases" in line  # 4 (hopg) + 3 (hbn)
    # material-weighted over the two started materials: (4/4 + 3/6) / 2 = 75%.
    assert " 75%" in line
    assert "materials" not in line
    assert "●" in line  # a still-running material keeps the aggregate active


def test_overall_progress_line_counts_unstarted_materials_in_the_denominator():
    # 2 of 4 rostered materials have started (one done, one half); the other two
    # have no progress record yet. The headline must weight over ALL four, not
    # read ~75% off the two that reported.
    records = remote._parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":4,"cached_cases":4,'
                '"completed_new_cases":0,"state":"done"}',
                '{"material":"hbn","total_cases":4,"cached_cases":0,'
                '"completed_new_cases":2,"state":"running"}',
            ]
        )
    )

    line = remote._overall_progress_line(records, ["hopg", "hbn", "mos2", "wse2"])

    assert line is not None
    # (4/4 + 2/4 + 0 + 0) / 4 = 37.5% -> 38%, not ~75%.
    assert " 38%" in line
    assert "6/16 cases" in line
    assert "materials" not in line


def test_compute_progress_reports_percentage_without_case_counts():
    records = remote._parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"running",'
        '"done_cost":25.0,"total_cost":100.0}'
    )

    line = remote._overall_progress_line(records, ["hopg"], use_cost=True)

    assert " 25%" in line
    assert "cases" not in line


def test_paused_overall_progress_uses_warning_state():
    records = remote._parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"running"}'
    )

    line = remote._overall_progress_line(records, ["hopg"], state_override="paused")

    assert "Ⅱ" in line


def test_overall_progress_line_is_none_without_records():
    assert remote._overall_progress_line({}) is None


def test_parse_progress_records_keeps_valid_cost_fields():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":25.0,"total_cost":100.0}'
    )
    records = remote._parse_progress_records(payload)
    assert records["hopg"]["done_cost"] == 25.0
    assert records["hopg"]["total_cost"] == 100.0


def test_parse_progress_records_drops_cost_fields_when_done_exceeds_total():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":150.0,"total_cost":100.0}'
    )
    records = remote._parse_progress_records(payload)
    assert "done_cost" not in records["hopg"]
    assert "total_cost" not in records["hopg"]


def test_parse_progress_records_drops_non_numeric_cost_fields():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":"lots","total_cost":100.0}'
    )
    records = remote._parse_progress_records(payload)
    assert "done_cost" not in records["hopg"]
    assert "total_cost" not in records["hopg"]


def test_overall_progress_line_use_cost_weights_by_compute_not_case_count():
    # hopg: 1 cheap case done of 1 total by count (100%), but that one case is
    # only 10% of the material's compute; hbn hasn't started. A case-count bar
    # would read 50% (1 material fully "done" of 2); the compute bar must read
    # far lower since hopg's done work is cheap relative to the whole sweep.
    records = remote._parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","total_cases":1,"cached_cases":0,'
                '"completed_new_cases":1,"state":"done",'
                '"done_cost":10.0,"total_cost":100.0}',
                '{"material":"hbn","total_cases":4,"cached_cases":0,'
                '"completed_new_cases":0,"state":"running",'
                '"done_cost":0.0,"total_cost":50.0}',
            ]
        )
    )

    cases_line = remote._overall_progress_line(records, ["hopg", "hbn"], use_cost=False)
    compute_line = remote._overall_progress_line(records, ["hopg", "hbn"], use_cost=True)

    assert " 50%" in cases_line
    assert " 5%" in compute_line  # (10/100 + 0/50) / 2 = 5%


def test_overall_progress_line_use_cost_falls_back_to_cases_without_cost_data():
    records = remote._parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running"}'
    )
    line = remote._overall_progress_line(records, ["hopg"], use_cost=True)
    assert " 50%" in line
    assert "cases" not in line


def _timed_record(
    *,
    material="hopg",
    total=10,
    cached=0,
    completed=0,
    state="running",
    seconds=None,
    measured=None,
    done_cost=None,
    total_cost=None,
    measured_cost=None,
):
    record = {
        "material": material,
        "total_cases": total,
        "cached_cases": cached,
        "completed_new_cases": completed,
        "state": state,
    }
    optional = {
        "active_compute_seconds": seconds,
        "measured_new_cases": measured,
        "done_cost": done_cost,
        "total_cost": total_cost,
        "measured_new_cost": measured_cost,
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    return record


def test_compute_eta_is_unavailable_before_a_measured_tick_and_excludes_cached_work():
    records = {
        "hopg": _timed_record(cached=5, seconds=20.0, measured=0),
    }

    estimate = presentation._compute_time_estimate(records, ["hopg"])

    assert estimate == {"elapsed": 20.0, "remaining": None, "total": None}
    assert "ETA —" in presentation._compute_time_suffix(estimate)


def test_compute_eta_uses_measured_work_rate_across_pause_and_chunk_resume():
    # Five done: three cached and two measured over 20 active seconds. Remaining
    # five therefore estimate to 50 seconds, not 20 seconds from cached-inflated
    # 50% overall progress.
    records = {
        "hopg": _timed_record(
            cached=3,
            completed=2,
            state="paused",
            seconds=20.0,
            measured=2,
        ),
    }

    estimate = presentation._compute_time_estimate(records, ["hopg"])

    assert estimate == {"elapsed": 20.0, "remaining": 50.0, "total": 70.0}


def test_compute_eta_prefers_cost_weighting():
    records = {
        "hopg": _timed_record(
            total=4,
            completed=1,
            seconds=10.0,
            measured=1,
            done_cost=10.0,
            total_cost=100.0,
            measured_cost=10.0,
        )
    }

    estimate = presentation._compute_time_estimate(records, ["hopg"], use_cost=True)

    assert estimate == {"elapsed": 10.0, "remaining": 90.0, "total": 100.0}


def test_compute_eta_accounts_for_parallel_material_processes_and_unstarted_work():
    records = {
        "hopg": _timed_record(material="hopg", completed=5, seconds=20.0, measured=5),
        "hbn": _timed_record(material="hbn", completed=5, seconds=20.0, measured=5),
    }

    estimate = presentation._compute_time_estimate(
        records,
        ["hopg", "hbn", "mos2"],
        parallel_materials=2,
    )

    assert estimate["elapsed"] == 20.0
    assert estimate["remaining"] == 40.0
    assert estimate["total"] == 60.0


@pytest.mark.parametrize(
    ("record", "remaining", "total"),
    [
        (_timed_record(total=0, state="done", seconds=3.0), 0.0, 3.0),
        (_timed_record(total=10, completed=10, state="done", seconds=30.0), 0.0, 30.0),
        (_timed_record(total=10, completed=2, state="failed", seconds=8.0, measured=2), None, None),
        (_timed_record(total=10, completed=2), None, None),
    ],
)
def test_compute_eta_defines_zero_done_failed_and_legacy_records(record, remaining, total):
    estimate = presentation._compute_time_estimate({"hopg": record}, ["hopg"])

    assert estimate["remaining"] == remaining
    assert estimate["total"] == total


def test_parse_progress_records_drops_non_finite_or_hostile_timing_values():
    payload = json.dumps(
        _timed_record(
            completed=2,
            seconds=float("nan"),
            measured=-3,
            measured_cost=float("inf"),
        )
    )

    record = remote._parse_progress_records(payload)["hopg"]

    assert "active_compute_seconds" not in record
    assert "measured_new_cases" not in record
    assert "measured_new_cost" not in record


def test_status_shows_one_compute_bar_at_base_verbosity_when_cost_data_present(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _cmd: presentation._encode_sections(
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
        lambda _cmd: presentation._encode_sections(
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
        lambda _cmd: presentation._encode_sections(
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
        lambda _cmd: presentation._encode_sections(
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
        lambda _cmd: presentation._encode_sections(
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
    monkeypatch.setattr(viewer.sys.stdout, "isatty", lambda: True)

    colored = remote._format_case_progress(record, ["hopg"])
    monkeypatch.setenv("NO_COLOR", "1")
    plain = remote._format_case_progress(record, ["hopg"])

    assert "\033[38;2;92;207;230m" in colored
    assert "\033[" not in plain


def test_case_progress_shows_current_crystal_parameters_for_running_material():
    records = remote._parse_progress_records(
        '{"material":"hopg","total_cases":5,"cached_cases":1,'
        '"completed_new_cases":2,"state":"running",'
        '"current":{"energy_keV":30,"tilt_deg":25,"azimuth_deg":110,"thickness_um":0.5}}'
    )

    output = remote._format_case_progress(records, ["hopg"])

    assert "NOW TESTING" in output
    assert "30 keV" in output
    assert "tilt 25°" in output
    assert "azim 110°" in output
    assert "0.5 µm" in output


def test_case_progress_omits_current_params_for_non_running_material():
    records = remote._parse_progress_records(
        '{"material":"hopg","total_cases":5,"cached_cases":5,'
        '"completed_new_cases":0,"state":"done",'
        '"current":{"energy_keV":30,"tilt_deg":25,"azimuth_deg":110,"thickness_um":0.5}}'
    )

    output = remote._format_case_progress(records, ["hopg"])

    assert "keV" not in output  # done row falls back to the cached/new tally


def test_material_roster_summarizes_and_wraps_a_long_list():
    short = remote._format_material_roster(["hopg", "hbn"])
    assert short == "hopg, hbn"

    long_roster = [f"mat{i:02d}" for i in range(20)]
    rendered = remote._format_material_roster(long_roster)
    assert rendered.splitlines()[0] == "20 total"
    assert "mat00" in rendered
    assert "\n" in rendered  # wrapped, not one ragged line


def test_stop_jobid_uses_scancel_not_kill(monkeypatch):
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))

    remote._stop_jobid("j")

    assert "scancel 48291" in commands[0][-1]
    assert "kill -TERM" not in commands[0][-1]
    assert "scancel 48291 || exit" in commands[0][-1]


def test_stop_jobid_rejects_legacy_job_without_scheduler_id(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("must not scancel")
    )

    with pytest.raises(SystemExit, match="not an active SLURM job"):
        remote._stop_jobid("j")


def test_stop_help_describes_slurm_cancellation(capsys):
    assert remote.main(["stop", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "cancel active SLURM job" in help_text
    assert "SIGTERM" not in help_text


def test_follow_logs_use_stdin_closed_ssh(monkeypatch):
    runs = []
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda cmd: runs.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    assert remote.tail_logs("20260101-000000", follow=True) == 0

    assert len(runs) == 1
    assert runs[0][:3] == ["ssh", "-n", remote.HOST]
    assert len(runs[0]) == 4


def test_follow_logs_propagates_ssh_failure(monkeypatch):
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda cmd: subprocess.CompletedProcess(cmd, 255),
    )

    assert remote.tail_logs("20260101-000000", follow=True) == 1


def test_follow_logs_maps_interrupt_to_130_and_stderr(monkeypatch, capsys):
    def interrupt(_cmd):
        raise KeyboardInterrupt

    monkeypatch.setattr(transport.subprocess, "run", interrupt)

    assert remote.tail_logs("20260101-000000", follow=True) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "job is unaffected" in captured.err


@pytest.mark.parametrize("status", [1, 130])
def test_follow_logs_status_propagates_through_remote_cli(monkeypatch, status):
    monkeypatch.setattr(viewer, "tail_logs", lambda _jobid, _follow: status)

    assert remote.main(["logs", "--follow"]) == status


def test_stems_quick_suffix():
    assert remote._stems(["mose2", "wse2"], True) == ["mose2_quick", "wse2_quick"]
    assert remote._stems(["mose2"], False) == ["mose2"]


def test_stop_materials_resolve_unique_live_jobs(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(lifecycle, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(["wse2", "mose2"])

    assert stopped == ["job2"]


def test_stop_all_stops_every_live_job(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", True, ["mose2"])],
    )
    monkeypatch.setattr(lifecycle, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(all_jobs=True)

    assert stopped == ["job1", "job2"]


def test_stop_rejects_bad_material_before_live_job_lookup(monkeypatch):
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: pytest.fail("must validate before checking live jobs"),
    )

    with pytest.raises(SystemExit):
        remote.stop_jobs(["bad;material"])


def test_stop_cli_accepts_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "stop_jobs",
        lambda materials, all_jobs, *, yes, profile: calls.append(
            (materials, all_jobs, yes, profile)
        ),
    )

    remote.main(["stop", "hopg", "mose2", "--yes"])

    assert calls == [(["hopg", "mose2"], False, True, None)]


def test_stop_cli_accepts_all(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "stop_jobs",
        lambda materials, all_jobs, *, yes, profile: calls.append(
            (materials, all_jobs, yes, profile)
        ),
    )

    remote.main(["stop", "--all", "--yes"])

    assert calls == [([], True, True, None)]


def test_stop_profile_matches_live_job_metadata(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: [("job1", False, ["hopg"]), ("job2", False, ["mose2", "wse2"])],
    )
    monkeypatch.setattr(
        state,
        "_job_profiles",
        lambda jobids: {"job1": "sub_100keV", "job2": "standard"},
    )
    monkeypatch.setattr(lifecycle, "_stop_jobids", lambda jobids: stopped.extend(jobids))

    remote.stop_jobs(profile="sub_100keV")

    assert stopped == ["job1"]


def test_stop_profile_without_live_match_errors(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(state, "_job_profiles", lambda jobids: {"job1": "standard"})

    with pytest.raises(SystemExit, match="no live job found for profile"):
        remote.stop_jobs(profile="sub_100keV")


def test_stop_profile_rejects_materials_and_all(monkeypatch):
    monkeypatch.setattr(
        state,
        "_live_jobs",
        lambda: pytest.fail("must reject the combination before checking live jobs"),
    )
    with pytest.raises(SystemExit):
        remote.stop_jobs(["hopg"], profile="sub_100keV")
    with pytest.raises(SystemExit):
        remote.stop_jobs(all_jobs=True, profile="sub_100keV")


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


def test_scancel_jobs_command_batches_sentinel_scancel_and_release():
    cmd = scripts._scancel_jobs_command(["job1", "job2"])

    # every STOP sentinel lands before the single scancel
    assert cmd.index(': > "$D/STOP"') < cmd.index("scancel $SIDS")
    assert cmd.count("scancel ") == 1
    # one shared squeue poll loop, then per-job release + terminal state
    assert cmd.count("squeue -h") == 1
    assert 'echo "cancelled [$SID]' in cmd
    assert '"$R"/*' in cmd  # reservation release


def test_scancel_jobs_command_functional(monkeypatch, tmp_path):
    """Run the batched cancel: STOP sentinels, one scancel, terminal states,
    reservation release -- and a job without a scheduler id is skipped, not
    fatal."""
    bash = _bash_or_skip(tmp_path)
    jobs = tmp_path / "jobs"
    for jobid, sid in [("j1", "101"), ("j2", "102")]:
        jobdir = jobs / jobid
        jobdir.mkdir(parents=True)
        (jobdir / "meta").write_text(f"slurm_job_id: {sid}\n")
    (jobs / "j3").mkdir()  # no meta -> skipped
    reservation = jobs / "reservations" / "hopg"
    reservation.mkdir(parents=True)
    (reservation / "jobid").write_text("j1")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scancel_log = tmp_path / "scancel.args"
    fake_scancel = bin_dir / "scancel"
    fake_scancel.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{scancel_log.as_posix()}"\n')
    fake_squeue = bin_dir / "squeue"
    fake_squeue.write_text("#!/bin/sh\nexit 0\n")  # nothing live -> poll loop exits at once
    for fake in (fake_scancel, fake_squeue):
        fake.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setenv("PATH", f"{bin_dir.as_posix()}:{os.environ['PATH']}")

    command = scripts._scancel_jobs_command(["j1", "j2", "j3"])
    result = subprocess.run(
        [bash, "-c", command], capture_output=True, text=True, env=os.environ.copy()
    )

    assert result.returncode == 0, result.stderr
    assert "skipping j3" in result.stderr
    assert scancel_log.read_text().split() == ["101", "102"]  # one batched scancel
    assert (jobs / "j1" / "STOP").exists() and (jobs / "j2" / "STOP").exists()
    assert (jobs / "j1" / "state").read_text().startswith("cancelled [101]")
    assert not reservation.exists()  # released
    assert "cancelled SLURM job 102 for job j2" in result.stdout


# ---- clear <material> (checkpoint lifecycle, component 3) ----------------------
def _no_live_jobs(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [])


def test_clear_refuses_when_a_live_job_produces_the_stem(monkeypatch):
    # a live job producing hopg -> clearing hopg must refuse before any ssh
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg", yes=True)


def test_clear_refuses_for_quick_stem_collision(monkeypatch):
    # a live --quick job producing hopg_quick still blocks a clear of hopg
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", True, ["hopg"])])
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg")


def test_clear_refuses_when_an_ambiguous_submission_holds_a_reservation(monkeypatch):
    """A retained pre-sbatch lock protects a possibly queued job without an ID."""
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *_args: "RESERVED\thopg\n")

    with pytest.raises(SystemExit, match="reservation"):
        remote.clear_remote("hopg", yes=True)


def test_clear_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "hopg.pkl\nhopg_quick.pkl\n")
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "hopg_quick.pkl" in out
    assert runs == []  # nothing deleted in a dry preview


def test_clear_yes_deletes_existing_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "CLEARED\thopg.pkl\nCLEARED\thopg_quick.pkl\n",
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert runs == []
    assert 'mkdir "$R/$stem"' in commands[0]
    assert 'rm -f "$f"' in commands[0]
    assert "cleared on the box" in capsys.readouterr().out


def test_clear_reports_nothing_when_no_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "\n")
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert "nothing to clear" in capsys.readouterr().out
    assert runs == []  # nothing to delete


def test_clear_multiple_materials_yes_deletes_all_stems(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or "CLEARED\thopg.pkl\nCLEARED\thbn.pkl\nCLEARED\tdiamond.pkl\n"
        ),
    )
    remote.clear_remote(["hopg", "hbn", "diamond"], yes=True)
    # one reserving delete command covering every stem (material + _quick)
    assert len(commands) == 1
    for stem in ("hopg", "hopg_quick", "hbn", "hbn_quick", "diamond", "diamond_quick"):
        assert stem in commands[0]
    out = capsys.readouterr().out
    assert "cleared on the box" in out and "diamond.pkl" in out


def test_clear_multiple_materials_dry_preview_lists_each(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "hopg.pkl\nhbn.pkl\n"
    )
    remote.clear_remote(["hopg", "hbn"], yes=False)
    listing = commands[-1]
    for name in ("hopg.pkl", "hopg_quick.pkl", "hbn.pkl", "hbn_quick.pkl"):
        assert name in listing
    assert "would delete" in capsys.readouterr().out


def test_clear_profile_targets_only_profile_stems(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or "hopg--profile-sub_100keV.pkl\nhbn--profile-sub_100keV.pkl\n"
        ),
    )
    monkeypatch.setattr(
        scripts,
        "_stems",
        lambda materials, quick, fidelity, **kwargs: [
            f"{material}--{fidelity}-profile-{kwargs['catalog_profile']}" for material in materials
        ],
    )

    remote.clear_remote(
        ["hopg", "hbn"],
        catalog_profile="sub_100keV",
    )

    listing = commands[-1]
    assert "hopg--full-profile-sub_100keV" in listing
    assert "hopg--survey-profile-sub_100keV" in listing
    assert "hbn--full-profile-sub_100keV" in listing
    assert "hbn--survey-profile-sub_100keV" in listing
    assert "hopg_quick" not in listing
    assert "would delete" in capsys.readouterr().out


def test_clear_profile_cli_uses_profile_membership(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_profile_default_materials", lambda profile: ("hopg", "hbn"))
    monkeypatch.setattr(
        lifecycle,
        "clear_remote",
        lambda materials, yes, **kwargs: calls.append((materials, yes, kwargs)),
    )

    remote.main(["clear", "--profile", "sub_100keV", "--yes"])

    assert calls == [(["hopg", "hbn"], True, {"catalog_profile": "sub_100keV"})]


def test_clear_profile_rejects_materials_and_all(capsys):
    assert remote.main(["clear", "hopg", "--profile", "sub_100keV"]) == 2
    assert "--profile takes no material" in capsys.readouterr().err
    assert remote.main(["clear", "--all", "--profile", "sub_100keV"]) == 2
    assert "--profile takes no material" in capsys.readouterr().err


def test_clear_rejects_bad_material(monkeypatch):
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must validate before touching jobs")
    )
    with pytest.raises(SystemExit):
        remote.clear_remote("rm -rf /")


def test_clear_all_refuses_when_a_live_job_is_running(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear --all"):
        remote.clear_all_remote(yes=True)


def test_clear_all_refuses_when_a_reservation_is_held(monkeypatch):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, [("hopg", "job1", 900)]))
    with pytest.raises(SystemExit, match="reservation"):
        remote.clear_all_remote(yes=True)


def test_clear_all_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "hopg.pkl\nzhai_reproduction/a.pkl\n",
    )
    remote.clear_all_remote(yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "zhai_reproduction/a.pkl" in out
    assert len(commands) == 1  # listing only; no delete
    assert "-delete" not in commands[0]


def test_clear_all_yes_deletes_every_pkl(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    commands = []

    def capture(command):
        commands.append(command)
        return "" if "-delete" in command else "hopg.pkl\nhopg_quick.pkl\n"

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    remote.clear_all_remote(yes=True)
    out = capsys.readouterr().out
    assert "cleared on the box: 2 checkpoint file(s)" in out
    assert any("-delete" in c for c in commands)


def test_clear_all_reports_nothing_when_empty(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "\n")
    remote.clear_all_remote(yes=True)
    assert "nothing to clear" in capsys.readouterr().out


def test_prune_remote_refuses_while_any_job_is_live(monkeypatch):
    target = type("Target", (), {"stem": "hopg"})()
    monkeypatch.setattr("cxr_mc.prune._targets", lambda *_args: [target])
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: pytest.fail("must not mutate remote checkpoints"),
    )

    with pytest.raises(SystemExit, match="live job"):
        lifecycle.prune_remote()


def test_prune_checkpoint_command_reserves_runs_and_releases():
    command = scripts._prune_checkpoint_stems_command(
        "prune-job",
        ["hopg", "hopg--survey-deadbeef0000"],
        catalog_profile="standard",
        yes=True,
    )

    assert "for stem in hopg hopg--survey-deadbeef0000" in command
    assert "trap release_prune EXIT" in command
    assert "run --no-sync cxr prune --profile standard --yes" in command
    assert command.index("for stem in hopg") < command.index("run --no-sync cxr prune")


def test_prune_remote_dispatches_exact_reserved_stems(monkeypatch, capsys):
    target = type("Target", (), {"stem": "hopg"})()
    commands = []
    monkeypatch.setattr("cxr_mc.prune._targets", lambda *_args: [target])
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_reservation_holders", lambda stems: [])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "would prune remote",
    )

    lifecycle.prune_remote(catalog_profile="standard")

    assert len(commands) == 1
    assert "for stem in hopg" in commands[0]
    assert "cxr prune --profile standard" in commands[0]
    assert "would prune remote" in capsys.readouterr().out


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
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
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
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
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
    fake_uv.write_text('#!/bin/sh\nif [ "$1" = sync ]; then exit 0; fi\nexit 7\n')
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
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        env = {**os.environ, "PATH": f"{commands}:{os.environ.get('PATH', '')}"}
        result = subprocess.run([bash, "-c", remote_cmd], capture_output=True, env=env)
        if result.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {result.returncode})")
        return result.stdout.decode()

    monkeypatch.setattr(transport, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kw: pytest.fail("must not delete"))

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
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        # same semantics as _ssh_capture, but run the snippet locally
        r = subprocess.run([bash, "-c", remote_cmd], capture_output=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {r.returncode})")
        return r.stdout

    monkeypatch.setattr(transport, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(
        transport, "_run", lambda cmd, **kw: pytest.fail("dry preview must not delete")
    )
    remote.clear_remote("hopg", yes=False)  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "hopg.pkl" in out and "hopg_quick.pkl" not in out


# ---- run --quick --grid must be rejected at parse time --------------------------
def test_run_rejects_quick_plus_grid_before_any_work(monkeypatch):
    """--quick checkpoints aren't grid-filterable (cxr slim rejects _quick stems),
    so run --quick --grid must fail up front -- not run the whole sweep and then
    traceback on the trailing pull."""
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must reject before the busy check")
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must reject before syncing"))
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda *a, **kw: pytest.fail("must reject before submitting")
    )
    assert remote.main(["run", "standard", "-m", "hopg", "--quick", "--grid"]) == 2


# ---- pull defaults to --grid; -f/--full opts into the plain whole-file pull -----
def test_pull_defaults_to_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg"])
    assert calls[0]["grid"] is True


def test_pull_full_flag_disables_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "--full"])
    assert calls[0]["grid"] is False


def test_pull_short_full_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg", "-f"])
    assert calls[0]["grid"] is False


def test_pull_level9_flag_defaults_off_and_wires_through(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda *a, **kw: calls.append(kw))
    remote.main(["pull", "hopg"])
    assert calls[0]["level9"] is False
    remote.main(["pull", "hopg", "--level9"])
    assert calls[1]["level9"] is True


def test_component_pull_projects_transfer_pickle_and_installs_split_store(monkeypatch, tmp_path):
    import numpy as np

    from cxr_mc import _checkpoint_io, _checkpoint_store

    payload = {
        "cfg": {
            30.0: {
                "case": {"crystal": "hopg", "Ne_brem": 10},
                "E_grid": np.array([1.0, 2.0]),
                "spec": np.array([3.0, 4.0]),
                "E_grid_brem": np.array([1.0, 2.0]),
                "brem_wide": np.array([5.0, 6.0]),
                "brem": np.array([5.0, 6.0]),
            }
        }
    }
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))
    transfers = []

    def fake_download(command, destination):
        transfers.append((command, destination))
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "_ssh_download", fake_download)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")  # no survey siblings

    remote.pull(["hopg"], grid=True, level9=True, no_sync=True)

    assert (tmp_path / "checkpoints" / "hopg" / "line.pkl").is_file()
    assert (tmp_path / "checkpoints" / "hopg" / "brem.pkl").is_file()
    loaded = _checkpoint_store.load("hopg", tmp_path / "checkpoints")
    assert np.array_equal(loaded["cfg"][30.0]["spec"], np.array([3.0, 4.0]))
    assert len(transfers) == 1
    transfer_command, destination = transfers[0]
    assert "/checkpoints/hopg" in transfer_command
    assert "--grid" in transfer_command and "--compresslevel 9" in transfer_command
    assert "trap cleanup EXIT" in transfer_command
    assert "trap 'exit 143' TERM" in transfer_command
    assert '1>&2 && cat "$T"' in transfer_command
    assert destination == tmp_path / "checkpoints" / ".hopg.incoming.pkl"
    syntax = subprocess.run(["bash", "-n", "-c", transfer_command], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


# ---- pull resolves identity-qualified survey checkpoints for a bare material ----
def test_resolve_survey_stems_discovers_matching_survey_dir(monkeypatch, capsys):
    from cxr_mc.profiles import named_profile_stem

    survey_stem = named_profile_stem("hopg", "survey")
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: f"{survey_stem}\nhopg\nunrelated--full-{'b' * 12}\n",
    )

    resolved = lifecycle._resolve_survey_stems(["hopg"])

    assert resolved == ["hopg", survey_stem]
    assert "survey checkpoint" in capsys.readouterr().out


def test_resolve_survey_stems_ignores_other_materials_and_profiles(monkeypatch):
    other_material_survey = f"wse2--survey-{'c' * 12}"
    same_material_full_variant = f"hopg--full-{'d' * 12}"
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: f"{other_material_survey}\n{same_material_full_variant}\n",
    )

    assert lifecycle._resolve_survey_stems(["hopg"]) == ["hopg"]


def test_resolve_survey_stems_does_not_duplicate_an_explicitly_requested_stem(monkeypatch):
    from cxr_mc.profiles import named_profile_stem

    survey_stem = named_profile_stem("hopg", "survey")
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: f"{survey_stem}\n")

    assert lifecycle._resolve_survey_stems(["hopg", survey_stem]) == ["hopg", survey_stem]


def test_resolve_survey_stems_skips_quick_and_already_qualified_stems(monkeypatch):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *_a: pytest.fail("must not list checkpoints/")
    )

    assert lifecycle._resolve_survey_stems(["hopg_quick"]) == ["hopg_quick"]
    qualified = f"hopg--survey-{'a' * 12}"
    assert lifecycle._resolve_survey_stems([qualified]) == [qualified]


# ---- pull MATERIAL@PROFILE selector (Phase 3 decision 3/design) -----------
def test_split_profile_selector_parses_and_rejects_empty_halves():
    assert lifecycle._split_profile_selector("hopg") is None
    assert lifecycle._split_profile_selector("hopg@sub_100keV") == ("hopg", "sub_100keV")
    with pytest.raises(SystemExit, match="invalid MATERIAL@PROFILE"):
        lifecycle._split_profile_selector("hopg@")
    with pytest.raises(SystemExit, match="invalid MATERIAL@PROFILE"):
        lifecycle._split_profile_selector("@sub_100keV")


def test_split_profile_selector_treats_full_at_stem_as_literal():
    """After the @-stem migration a resolved on-disk stem carries an ``@`` too
    (``<material>@<label>-<digest>``). It is an exact checkpoint, not a
    MATERIAL@PROFILE query, so pull must pass it through literally rather than
    try to re-resolve ``<label>-<digest>`` as a catalog profile."""
    from cxr_mc.profiles import named_profile_stem

    at_stem = named_profile_stem("hopg", "full", catalog_profile="sub_100keV")
    assert "@" in at_stem and at_stem != "hopg@sub_100keV"  # full stem, has digest tail
    assert lifecycle._split_profile_selector(at_stem) is None
    # a bare selector (no digest tail) still resolves as a profile query
    assert lifecycle._split_profile_selector("hopg@sub_100keV") == ("hopg", "sub_100keV")


def test_check_shell_tokens_accepts_at_stems():
    """@-stems must survive the remote shell-token gate so pull/prune/slim can
    name them on the box (regression for the 2026-07-29 stem migration; ``@``
    was previously rejected by ``_SHELL_TOKEN_RE``)."""
    from cxr_mc.profiles import named_profile_stem

    at_stem = named_profile_stem("hopg", "full", catalog_profile="sub_100keV")
    remote._check_shell_tokens([at_stem])  # must not raise


def _fake_remote_catalog(listing, meta_by_stem):
    """Route ``transport._ssh_capture`` calls: the checkpoint-dir listing
    command returns ``listing`` verbatim; any other command is treated as a
    ``[ -f ... ] ... stat ... cat ...`` meta.json fetch and matched by which
    stem's path it names."""

    def fake(command):
        if command.startswith("[ -d "):
            return listing
        for stem, (mtime, meta) in meta_by_stem.items():
            if f"checkpoints/{stem}/meta.json" in command:
                return f"{mtime}\n{json.dumps(meta)}"
        return ""

    return fake


def test_resolve_profile_stem_picks_newest_and_reports_alternates(monkeypatch, capsys):
    a = f"hopg--full-{'a' * 12}"
    b = f"hopg--full-{'b' * 12}"
    meta = {
        "hopg": (
            50,
            {"dataset_identity": {"catalog_profile": "standard", "parameter_sha256": "c" * 64}},
        ),
        a: (
            100,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "a" * 64}},
        ),
        b: (
            200,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "b" * 64}},
        ),
    }
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog(f"hopg\n{a}\n{b}\n", meta))

    stem = lifecycle.resolve_profile_stem("hopg", "sub_100keV")

    assert stem == b  # newest mtime (200) wins
    out = capsys.readouterr().out
    assert "2 hashes found" in out
    assert ("a" * 12) in out


def test_resolve_profile_stem_hash_prefix_pins_one(monkeypatch):
    a = f"hopg--full-{'a' * 12}"
    b = f"hopg--full-{'b' * 12}"
    meta = {
        a: (
            100,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "a" * 64}},
        ),
        b: (
            200,
            {"dataset_identity": {"catalog_profile": "sub_100keV", "parameter_sha256": "b" * 64}},
        ),
    }
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog(f"{a}\n{b}\n", meta))

    assert lifecycle.resolve_profile_stem("hopg", "sub_100keV", hash_prefix="a" * 12) == a


def test_resolve_profile_stem_raises_with_no_match(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", _fake_remote_catalog("hopg\n", {}))

    with pytest.raises(SystemExit, match=r"no remote checkpoint matches hopg@sub_100keV"):
        lifecycle.resolve_profile_stem("hopg", "sub_100keV")


def test_pull_hash_requires_exactly_one_qualified_selector(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

    with pytest.raises(SystemExit, match="--hash requires exactly one MATERIAL@PROFILE"):
        remote.pull(["hopg@standard", "wse2@standard"], grid=True, hash_prefix="a" * 12)


def test_pull_resolves_profile_selector_to_the_predicted_stem(monkeypatch, tmp_path):
    import numpy as np

    from cxr_mc import _checkpoint_io, _checkpoint_store

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    resolved = []

    def fake_resolve(material, profile, **kwargs):
        resolved.append((material, profile, kwargs.get("hash_prefix")))
        return "hopg"

    monkeypatch.setattr(lifecycle, "resolve_profile_stem", fake_resolve)
    monkeypatch.setattr(transport, "sync_code", lambda: None)
    # resolves to bare "hopg" -- also triggers _resolve_survey_stems' listing;
    # empty means no sibling survey checkpoint to also pull.
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")
    payload = {"cfg": {30.0: {"case": {}, "E_grid": np.array([1.0]), "spec": np.array([1.0])}}}
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    def fake_download(_command, destination):
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg@sub_100keV"], grid=True, no_sync=True)

    assert resolved == [("hopg", "sub_100keV", None)]
    assert _checkpoint_store.checkpoint_exists("hopg", tmp_path / "checkpoints")


def test_pull_bare_material_also_pulls_matching_survey_checkpoint(monkeypatch, tmp_path, capsys):
    import numpy as np

    from cxr_mc import _checkpoint_io, _checkpoint_store
    from cxr_mc.profiles import named_profile_stem

    survey_stem = named_profile_stem("hopg", "survey")
    payload = {
        "cfg": {
            30.0: {
                "case": {"crystal": "hopg", "Ne_brem": 10},
                "E_grid": np.array([1.0, 2.0]),
                "spec": np.array([3.0, 4.0]),
                "E_grid_brem": np.array([1.0, 2.0]),
                "brem_wide": np.array([5.0, 6.0]),
                "brem": np.array([5.0, 6.0]),
            }
        }
    }
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: f"{survey_stem}\n")
    transfers = []

    def fake_download(command, destination):
        transfers.append((command, destination))
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg"], grid=True, no_sync=True)

    assert len(transfers) == 2  # canonical stem + discovered survey variant
    assert _checkpoint_store.checkpoint_exists("hopg", tmp_path / "checkpoints")
    assert _checkpoint_store.checkpoint_exists(survey_stem, tmp_path / "checkpoints")
    assert "also pulling identity-qualified survey checkpoint" in capsys.readouterr().out


def test_pull_quick_stem_does_not_resolve_survey_siblings(monkeypatch, tmp_path):
    import numpy as np

    from cxr_mc import _checkpoint_io

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *_a: pytest.fail("--quick pull must not list checkpoints/"),
    )
    payload = {"cfg": {30.0: {"case": {}, "E_grid": np.array([1.0]), "spec": np.array([1.0])}}}
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))

    def fake_download(_command, destination):
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(transport, "_ssh_download", fake_download)

    remote.pull(["hopg_quick"], grid=False, no_sync=True)  # must not raise


def test_pull_dataset_merge_skips_survey_discovery(monkeypatch, tmp_path):
    def _boom(*_a, **_kw):
        raise RuntimeError("stub -- dataset merge test does not need a real download")

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *_a: pytest.fail("--brem-only pull must not list checkpoints/"),
    )
    monkeypatch.setattr(transport, "_ssh_download", _boom)

    with pytest.raises(SystemExit, match="remote pull failed"):
        lifecycle.pull(["hopg"], dataset="brem", no_sync=True)


def test_remote_run_uses_profile_membership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda materials, *args, **kwargs: calls.append(materials)
    )

    remote.main(["run", "sub_100keV", "--dry-run"])

    assert calls and "hopg" in calls[0] and "zrte3" in calls[0]


def test_remote_run_defers_parallel_materials_default_to_start_queue(monkeypatch):
    """The CLI no longer bakes in a default: --parallel-materials is None
    unless given explicitly, so start_queue (mode-aware) resolves it. The
    monolithic default of 2 is exercised by
    test_chunk_minutes_zero_emits_monolithic_script."""
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    remote.main(["run", "standard", "-m", "hopg", "--dry-run", "--chunk-minutes", "0"])

    assert calls[0][1]["parallel_materials"] is None
    assert calls[0][1]["chunk_minutes"] == 0.0


def test_remote_run_accepts_parallel_materials_three_and_four(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    remote.main(
        ["run", "sub_100keV", "--parallel-materials", "3", "--chunk-minutes", "0", "--dry-run"]
    )
    remote.main(
        ["run", "sub_100keV", "--parallel-materials", "4", "--chunk-minutes", "0", "--dry-run"]
    )

    assert [kwargs["parallel_materials"] for _materials, kwargs in calls] == [3, 4]


def test_remote_run_rejects_parallel_materials_above_four(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda *_args, **_kwargs: pytest.fail("must reject before start")
    )

    assert (
        remote.main(["run", "standard", "-m", "hopg", "--parallel-materials", "5", "--dry-run"])
        == 2
    )


def test_start_queue_rejects_parallel_materials_above_four():
    with pytest.raises(SystemExit, match="between 1 and 4"):
        remote.start_queue(["hopg"], parallel_materials=5, chunk_minutes=0, dry_run=True)


def test_remote_run_forwards_parallel_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)

    remote.main(
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--parallel-materials",
            "3",
            "--chunk-minutes",
            "0",
            "--no-sync",
        ]
    )

    assert calls[0][1]["parallel_materials"] == 3


def test_remote_run_preserves_hyphenated_catalog_material(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda materials, **_kw: calls.append(materials) or "j"
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: None)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, materials: materials)
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: None)

    remote.main(["run", "standard", "-m", "mos2-on-sio2-si", "--no-sync"])

    assert calls == [["mos2-on-sio2-si"]]


def test_completed_materials_reads_success_markers_from_the_queue_log(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "hopg\nhbn\n",
    )

    assert remote._completed_materials("j", ["hopg", "wse2", "hbn"]) == ["hopg", "hbn"]
    assert "s/^completed: //p" in commands[0]


def test_remote_run_rejects_unknown_profile_before_busy_or_sync(monkeypatch):
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must validate before checking busy jobs")
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

    assert remote.main(["run", "not_in_catalog"]) == 2


def test_pull_rejects_unsafe_stem_before_sync_or_local_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

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
    assert "python -m cxr_mc._entry.reproduce_zhai" in s
    assert "--ne 11" in s and "--ne-brem 3" in s and "--ne-supp 5" in s and "--refresh" in s
    assert "--tmd-azimuth 35.0" in s
    assert "20260101-000000" in s
    assert "materials: zhai" not in s and "quick: False" not in s
    assert "started: $(date -Is)" in s
    assert "/pid" not in s


def test_zhai_queue_metadata_records_cache_and_detector_provenance():
    metadata = remote._zhai_queue_metadata("j", ne=11, ne_brem=3, ne_supp=5)

    assert "zhai_cache_schema: 4" in metadata
    assert "detector_observation_angle_deg: 119" in metadata
    assert "detector_polar_acceptance_deg: 16.6" in metadata
    assert "detector_solid_angle_sr: 0.066" in metadata


def test_zhai_queue_script_no_refresh_flag_when_unset():
    s = remote._zhai_queue_script("j", ne=1, ne_brem=1, ne_supp=1, tmd_azimuth=0.0, refresh=False)
    assert "--refresh" not in s


def test_zhai_start_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_zhai_queue()


def test_zhai_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(state, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_zhai_queue(dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "python -m cxr_mc._entry.reproduce_zhai" in out


def test_remote_check_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.remote_check()


def test_foreground_check_submits_then_attaches_and_pulls(monkeypatch):
    events = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: events.append("start") or "j")
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True, raising=False)
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: events.append("pull"))

    remote.remote_check(no_sync=True)

    assert events == ["start", ("attach", "j"), "pull"]


def test_interrupted_foreground_check_does_not_pull(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("must not pull after interruption")
    )

    remote.remote_check(no_sync=True)


@pytest.mark.parametrize("job_state", ["FAILED (exit 1)", "cancelled [48291]"])
def test_failed_foreground_check_does_not_pull_stale_cache(monkeypatch, job_state):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: False, raising=False)
    monkeypatch.setattr(state, "_job_state", lambda _jobid: job_state, raising=False)
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("failed job must not pull stale cache")
    )

    with pytest.raises(SystemExit, match="did not complete successfully"):
        remote.remote_check(no_sync=True)


def test_remote_check_no_sync_skips_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: calls.append(kw) or "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: calls.append("attach") or True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: calls.append("pull"))

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
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda *a: (
            "/r/checkpoints/zhai_reproduction/zhai-a.pkl\n"
            "/r/checkpoints/zhai_reproduction/zhai-b.pkl\n"
        ),
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.pull_zhai_cache()

    assert len(runs) == 2
    assert runs[0][0] == "scp" and runs[0][1].endswith("zhai-a.pkl")
    assert (tmp_path / "checkpoints" / "zhai_reproduction").is_dir()


def test_pull_zhai_cache_reports_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: pytest.fail("nothing to pull"))

    remote.pull_zhai_cache()

    assert "no zhai cache files" in capsys.readouterr().out
    assert "find" in commands[0]


def test_check_cli_pull_flag_skips_run(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: calls.append("pull"))
    monkeypatch.setattr(
        cli, "remote_check", lambda **kw: pytest.fail("--pull must not run the reproduction")
    )

    remote.main(["check", "--pull"])

    assert calls == ["pull"]


def test_check_cli_detached_starts_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: calls.append(kw) or "jid")
    monkeypatch.setattr(viewer, "attach", lambda jobid: pytest.fail("no --follow: must not attach"))

    remote.main(["check", "--detached", "--ne", "11"])

    assert calls[0]["ne"] == 11


def test_check_cli_detached_follow_attaches(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: "jid")
    attached = []
    monkeypatch.setattr(viewer, "attach", attached.append)

    remote.main(["check", "--detached", "--follow"])

    assert attached == ["jid"]


def test_check_cli_foreground_calls_remote_check(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "remote_check", lambda **kw: calls.append(kw))

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
    monkeypatch.setattr(cli, "remote_check", lambda **kw: pytest.fail("must reject before running"))

    assert remote.main(["check", "--follow"]) == 2

    assert "--follow requires --detached" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--pull", "--detached"], ["--pull", "--detached", "--follow"]])
def test_check_cli_rejects_pull_with_detached(monkeypatch, capsys, args):
    monkeypatch.setattr(
        lifecycle, "pull_zhai_cache", lambda: pytest.fail("must reject before pulling")
    )

    assert remote.main(["check", *args]) == 2

    assert "--pull and --detached are mutually exclusive" in capsys.readouterr().err


def test_sync_paths_ship_checks_and_entry_shims():
    # The entry shims live under src/cxr_mc/_entry/ now, so they travel via "src";
    # checks/ still ships the real reproduce_all logic.
    assert "checks" in remote.SYNC_PATHS
    assert "src" in remote.SYNC_PATHS
    assert "scan.py" not in remote.SYNC_PATHS
    assert "reproduce_zhai.py" not in remote.SYNC_PATHS


def test_stop_writes_stop_sentinel_before_scancel(monkeypatch):
    commands = []
    monkeypatch.setattr(state, "_slurm_job_id", lambda jobid: "123")
    monkeypatch.setattr(state, "_slurm_state", lambda sid: "RUNNING")
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: commands.append(cmd[-1]))
    remote._stop_jobid("20260717-abc")
    (cmd,) = commands
    assert cmd.index("STOP") < cmd.index("scancel")


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
        state,
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
        state, "_slurm_job_id", lambda jobid: "48291" if jobid in {"live", "fresh"} else None
    )
    monkeypatch.setattr(state, "_slurm_state", lambda sid: "RUNNING" if sid == "48291" else None)

    orphans, protected = remote._orphaned_reservation_jobs(300.0)

    assert orphans == {"dead": ["hbn", "hopg"]}
    assert protected == {"live": ["mose2"], "fresh": ["wse2"]}


def test_reap_reservations_dry_run_previews_without_releasing(monkeypatch, capsys):
    monkeypatch.setattr(
        state,
        "_orphaned_reservation_jobs",
        lambda _min_age: ({"dead": ["hopg", "hbn"]}, {"live": ["mose2"]}),
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.reap_reservations(yes=False)

    out = capsys.readouterr().out
    assert "would release dead" in out
    assert "keeping live" in out
    assert "re-run with --yes" in out
    assert runs == []


def test_reap_reservations_yes_releases_orphans_and_stamps_terminal_state(monkeypatch, capsys):
    monkeypatch.setattr(
        state, "_orphaned_reservation_jobs", lambda _min_age: ({"dead": ["hopg"]}, {})
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd[-1]))

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

    old = config.REMOTE_DIR
    config.REMOTE_DIR = str(monkeypatch_dir)
    try:
        result = subprocess.run(
            [bash, "-c", scripts._reap_job_command("dead")], capture_output=True, text=True
        )
    finally:
        config.REMOTE_DIR = old

    assert result.returncode == 0, result.stderr
    assert not (reservations / "hopg").exists()
    assert not (reservations / "mose2").exists()
    assert (reservations / "wse2").exists()  # another owner's lock untouched
    assert "reaped" in (monkeypatch_dir / "jobs" / "dead" / "state").read_text()


# ---- cxr remote rebrem (brem-only checkpoint recompute) ---------------------
def test_rebrem_queue_script_flags_progress_and_markers():
    s = remote._rebrem_queue_script(
        "20260101-000000", ["hopg", "hbn"], ne_brem=1000, brem_step_eV=25.0, redo_all=True
    )
    assert "cxr rebrem" in s
    assert "--ne-brem 1000" in s and "--step 25" in s and "--redo-all" in s
# per-material progress record feeds the shared plain/attached status dashboard
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    # completion markers match the scan queue's so _completed_materials works
    assert 'echo "completed: $m"' in s and 'echo "failed: $m"' in s
    assert "mats=(hopg hbn)" in s


def test_rebrem_queue_script_omits_unset_flags():
    s = remote._rebrem_queue_script("j", ["hopg"], ne_brem=None, brem_step_eV=None, redo_all=False)
    assert "--ne-brem" not in s and "--step" not in s and "--redo-all" not in s


def test_rebrem_metadata_keys_mode_line_and_live_job_plumbing():
    meta = remote._rebrem_queue_metadata("j", ["hopg"], 1000, 25.0, False)
    assert "kind: rebrem" in meta
    assert "materials: hopg" in meta and "quick: False" in meta  # _live_jobs fields
    summary = remote._mode_summary(meta)
    assert "brem-only recompute" in summary
    assert "Ne_brem=1000" in summary and "step 25.0 eV" in summary
    # zhai detection must not shadow rebrem despite the ne_brem field
    assert "Zhai" not in summary


def test_rebrem_start_refuses_when_material_checkpoint_is_busy(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_rebrem_queue(["hopg"], ne_brem=1000)


def test_rebrem_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(state, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_rebrem_queue(["hopg"], ne_brem=1000, dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "cxr rebrem" in out


def test_rebrem_cli_submits_attaches_and_pulls_completed(monkeypatch):
    events = []
    monkeypatch.setattr(
        lifecycle,
        "start_rebrem_queue",
        lambda mats, **kw: events.append(("start", mats, kw)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: ["hopg"])
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kw: events.append(("pull", stems)))

    remote.main(["rebrem", "hopg", "hbn", "--ne-brem", "1000", "--step", "25"])

    assert events[0][0] == "start" and events[0][1] == ["hopg", "hbn"]
    assert events[0][2]["ne_brem"] == 1000 and events[0][2]["brem_step_eV"] == 25.0
    assert events[1] == ("attach", "j")
    assert events[2] == ("pull", ["hopg"])


def test_rebrem_cli_skips_pull_when_viewer_disconnects(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_rebrem_queue", lambda mats, **kw: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)
    monkeypatch.setattr(
        lifecycle, "pull", lambda *a, **kw: pytest.fail("must not pull after disconnect")
    )

    remote.main(["rebrem", "hopg"])


def test_cli_rebrem_pulls_brem_dataset(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_rebrem_queue", lambda *a, **k: "J1")
    monkeypatch.setattr(viewer, "attach", lambda jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: ["mos2"])
    captured = {}
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kw: captured.update(stems=stems, kw=kw))
    args = type(
        "A",
        (),
        dict(
            material=["mos2"],
            all=False,
            ne_brem=None,
            step=None,
            redo_all=False,
            no_sync=False,
            dry_run=False,
            chunk_minutes=10.0,
            remote_command="rebrem",
        ),
    )()
    cli._cli_rebrem(args)
    assert captured["stems"] == ["mos2"]
    assert captured["kw"].get("dataset") == "brem"


def test_rebrem_chunked_queue_script_self_resubmits():
    s = scripts._rebrem_chunked_queue_script(
        "J1", ["mos2", "w"], ne_brem=1000, brem_step_eV=None, redo_all=False, chunk_minutes=10.0
    )
    assert "cxr rebrem" in s
    assert "--ne-brem 1000" in s
    assert "--max-minutes" in s
    assert "--nice=10000" in s
    assert '--progress-file "$JOBDIR/progress/$m.json"' in s
    # metadata unchanged by chunking
    meta = scripts._rebrem_queue_metadata("J1", ["mos2", "w"], 1000, None, False)
    assert "kind: rebrem" in meta


def test_reline_queue_script_and_metadata():
    from cxr_mc._remote import scripts

    # Default remote reline is chunked like `cxr remote run`.
    s = scripts._reline_chunked_queue_script(
        "J1", ["mos2", "w"], line_ne=40000, line_step_eV=None, redo_all=True, chunk_minutes=10.0
    )
    assert "cxr reline" in s
    assert "--line-ne 40000" in s
    assert "--redo-all" in s
    assert "--max-minutes" in s
    assert "--nice=10000" in s
    assert "--progress-file" in s
    # The monolithic (chunk_minutes==0) path still runs `cxr reline` per material.
    mono = scripts._reline_queue_script("J1", ["mos2", "w"], 40000, None, True)
    assert "cxr reline" in mono and "--line-ne 40000" in mono and "--redo-all" in mono
    assert "--progress-file" in mono
    meta = scripts._reline_queue_metadata("J1", ["mos2", "w"], 40000, None, True)
    assert "kind: reline" in meta
    assert "line_ne: 40000" in meta


def test_pull_dataset_merges_and_archives(monkeypatch, tmp_path):
    import numpy as np

    from cxr_mc import _checkpoint_io
    from cxr_mc._remote import lifecycle

    # local checkpoint with a line record
    ckpt = tmp_path / "checkpoints" / "mos2.pkl"
    ckpt.parent.mkdir(parents=True)
    local = {
        "n": {
            30.0: {
                "case": {},
                "spec": np.zeros(3),
                "E_grid": np.array([1.0, 2.0, 3.0]),
                "brem_wide": np.array([1.0, 1.0, 1.0]),
                "E_grid_brem": np.array([1.0, 2.0, 3.0]),
                "brem": np.zeros(3),
            }
        }
    }
    _checkpoint_io.dump(local, str(ckpt))
    # the "remote" sub-pickle (what scp would land): new spec only
    remote_tmp = tmp_path / "remote.pkl"
    _checkpoint_io.dump(
        {"n": {30.0: {"case": {}, "spec": np.full(3, 7.0), "E_grid": np.array([1.0, 2.0, 3.0])}}},
        str(remote_tmp),
    )

    monkeypatch.setattr(lifecycle.config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle.transport, "sync_code", lambda: None)
    archived = []
    monkeypatch.setattr(
        lifecycle.archive, "archive_checkpoint", lambda stem, **kw: archived.append(stem)
    )

    # Stub the one-session box round-trip into the incoming transfer file.
    def fake_download(_command, destination):
        shutil.copy(remote_tmp, destination)

    monkeypatch.setattr(lifecycle.transport, "_ssh_download", fake_download)

    lifecycle.pull(["mos2"], dataset="line")

    from cxr_mc import _checkpoint_store

    merged = _checkpoint_store.load("mos2", ckpt.parent)["n"][30.0]
    assert np.all(merged["spec"] == 7.0)  # line overwritten
    assert archived == ["mos2"]  # archived before merge


@pytest.mark.parametrize("dataset", ["spec", "", "BREM", 1])
def test_pull_rejects_invalid_public_dataset_before_side_effects(monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(lifecycle.config, "LOCAL_ROOT", tmp_path / "must-not-exist")
    monkeypatch.setattr(
        lifecycle.transport,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid dataset must not reach transport"),
    )

    with pytest.raises(ValueError, match="dataset must be"):
        lifecycle.pull(["mos2"], dataset=dataset)

    assert not lifecycle.config.LOCAL_ROOT.exists()


# --- attach: warn when defaulting to a terminal-state job ----------------------


def test_attach_warns_when_defaulting_to_terminal_job(monkeypatch, capsys):
    """No job id + resolved job already terminal -> stderr warning, still attaches."""
    monkeypatch.setattr(state, "_latest_jobid", lambda: "sub_100keV")
    monkeypatch.setattr(
        state, "_job_state", lambda jobid: "cancelled [932] 2026-07-28T18:15:56-07:00"
    )
    seen = []
    monkeypatch.setattr(
        viewer, "_live_status", lambda jobid, detail: seen.append((jobid, detail)) or True
    )

    viewer.attach()

    captured = capsys.readouterr()
    assert "defaulting to sub_100keV" in captured.err
    assert "not running" in captured.err
    assert "defaulting" not in captured.out  # warning belongs on stderr
    assert seen == [("sub_100keV", 0)]


def test_attach_does_not_warn_for_explicit_jobid(monkeypatch, capsys):
    """An explicit job id is an informed choice: never read state, never warn."""

    def forbidden(jobid):
        raise AssertionError("state must not be read for an explicit job id")

    monkeypatch.setattr(state, "_job_state", forbidden)
    monkeypatch.setattr(viewer, "_live_status", lambda jobid, detail: True)

    viewer.attach("sub_100keV-3")

    assert "defaulting" not in capsys.readouterr().err


def test_attach_does_not_warn_when_default_job_is_live(monkeypatch, capsys):
    monkeypatch.setattr(state, "_latest_jobid", lambda: "sub_100keV-7")
    monkeypatch.setattr(state, "_job_state", lambda jobid: "running mos2 [4/21] since now")
    monkeypatch.setattr(viewer, "_live_status", lambda jobid, detail: True)

    viewer.attach()

    assert "defaulting" not in capsys.readouterr().err


# --- prune-jobs: delete terminal, non-live job directories ---------------------


def _run_prune_jobs_command(command, jobs_dir, live_ids):
    """Run a _prune_job_dirs_command against a real tmp jobs tree with a squeue shim."""
    bindir = jobs_dir.parent / "bin"
    bindir.mkdir(exist_ok=True)
    squeue = bindir / "squeue"
    squeue.write_text('#!/bin/sh\nprintf "%s\\n" ' + " ".join(f'"{i}"' for i in live_ids) + "\n")
    squeue.chmod(0o755)
    command = command.replace(
        f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs_dir}"'
    )
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env)


def _make_job(jobs_dir, name, state_line, slurm_ids):
    job = jobs_dir / name
    job.mkdir(parents=True)
    (job / "meta").write_text("".join(f"slurm_job_id: {i}\n" for i in slurm_ids))
    (job / "state").write_text(state_line + "\n")
    return job


def test_prune_jobs_command_keeps_live_and_non_terminal(monkeypatch, tmp_path):
    """Only done/FAILED/cancelled dirs whose latest SLURM id is not live are removed."""
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932] 2026-07-28", [915, 932])
    live = _make_job(jobs, "sub_100keV-7", "running mos2 [4/21] since now", [942, 947])
    crashed = _make_job(jobs, "sub_100keV-6", "running hbn [2/21] since then", [941])
    done = _make_job(jobs, "20260715-113910-abcd", "done [21/21] 2026-07-15", [800])

    command = scripts._prune_job_dirs_command(all_jobs=True, yes=True)
    result = _run_prune_jobs_command(command, jobs, live_ids=["947"])

    assert result.returncode == 0, result.stderr
    assert "PRUNED\tsub_100keV\t" in result.stdout
    assert "PRUNED\t20260715-113910-abcd\t" in result.stdout
    assert "KEPT\tsub_100keV-7\tlive" in result.stdout
    assert not cancelled.exists()
    assert not done.exists()
    assert live.exists()  # live chain never removed
    assert crashed.exists()  # non-terminal state kept


def test_prune_jobs_command_preview_deletes_nothing(tmp_path):
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932] 2026-07-28", [932])

    command = scripts._prune_job_dirs_command(all_jobs=True, yes=False)
    result = _run_prune_jobs_command(command, jobs, live_ids=[])

    assert result.returncode == 0, result.stderr
    assert "WOULD-PRUNE\tsub_100keV\t" in result.stdout
    assert cancelled.exists()  # preview never mutates


def test_prune_jobs_command_profile_scopes_to_family(tmp_path):
    jobs = tmp_path / "jobs"
    _make_job(jobs, "sub_100keV", "cancelled [932]", [932])
    _make_job(jobs, "sub_100keV-2", "cancelled [933]", [933])
    other = _make_job(jobs, "20260715-113910-abcd", "cancelled [800]", [800])

    command = scripts._prune_job_dirs_command(profile="sub_100keV", yes=True)
    result = _run_prune_jobs_command(command, jobs, live_ids=[])

    assert result.returncode == 0, result.stderr
    assert "PRUNED\tsub_100keV\t" in result.stdout
    assert "PRUNED\tsub_100keV-2\t" in result.stdout
    assert other.exists()  # a standard timestamp job is outside the profile family


def test_prune_jobs_command_fail_closed_when_squeue_errors(tmp_path):
    """squeue failure must not delete anything (liveness unverifiable)."""
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932]", [932])
    command = scripts._prune_job_dirs_command(all_jobs=True, yes=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # A squeue that errors: liveness is unverifiable, so nothing may be pruned.
    (bindir / "squeue").write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
    (bindir / "squeue").chmod(0o755)
    command = command.replace(f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs}"')
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env=dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}"),
    )

    assert result.returncode != 0
    assert cancelled.exists()


def test_prune_job_dirs_reports_preview(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: "WOULD-PRUNE\tsub_100keV\tcancelled [932]\nKEPT\tsub_100keV-7\tlive\n",
    )

    lifecycle.prune_job_dirs(profile="sub_100keV")

    out = capsys.readouterr().out
    assert "jobs/sub_100keV" in out
    assert "cancelled" in out
    assert "kept 1 live job" in out


def test_prune_job_dirs_reports_deletions(monkeypatch, capsys):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: "PRUNED\tsub_100keV\tcancelled [932]\n"
    )

    lifecycle.prune_job_dirs(all_jobs=True, yes=True)

    out = capsys.readouterr().out
    assert "pruned job directories" in out
    assert "jobs/sub_100keV" in out


def test_prune_job_dirs_requires_exactly_one_selector(monkeypatch):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: pytest.fail("must not reach ssh")
    )

    with pytest.raises(SystemExit, match="exactly one"):
        lifecycle.prune_job_dirs()
    with pytest.raises(SystemExit, match="exactly one"):
        lifecycle.prune_job_dirs(profile="sub_100keV", all_jobs=True)


def test_prune_jobs_cli_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "prune_job_dirs", lambda **kwargs: calls.append(kwargs))

    remote.main(["prune-jobs", "--profile", "sub_100keV", "--yes"])

    assert calls == [{"profile": "sub_100keV", "all_jobs": False, "yes": True}]


def test_prune_jobs_cli_rejects_bad_selectors(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "prune_job_dirs", lambda **kwargs: pytest.fail("must not dispatch")
    )

    assert remote.main(["prune-jobs"]) == 2
    assert remote.main(["prune-jobs", "--all", "--profile", "sub_100keV"]) == 2
