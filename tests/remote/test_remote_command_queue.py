"""Remote queue tests."""

import shutil
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


def test_queue_script_has_per_material_scan_calls():
    s = remote._queue_script("20260101-000000", ["mose2", "wse2"], quick=True, workers=8)
    assert 'scan_launcher=("uv" run --no-sync python)' in s
    assert '"${scan_launcher[@]}" -u -m pyrite._entry.scan' in s
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
    assert "export PYRITE_MC_GPU_SHARE=3" in script  # co-tenants split the VRAM pool cap
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


def test_queue_script_syncs_runtime_member_without_test_tools():
    script = remote._queue_script("j", ["hopg"], quick=False, workers=None)

    assert "sync --package pyrite-mc --no-default-groups --extra" in script


def test_queue_script_no_flags_when_unset():
    s = remote._queue_script("j", ["mos2"], quick=False, workers=None)
    assert "--quick" not in s and "--workers" not in s
    assert "--profile" not in s


@pytest.mark.parametrize(
    "build",
    [
        lambda: remote._queue_script("j", ["hopg"], quick=False, workers=None),
        lambda: remote._chunked_queue_script("j", ["hopg"], False, None, 10),
    ],
)
@pytest.mark.parametrize(
    ("name", "value", "invalid"),
    [
        ("PYRITE_MC_TRANSPORT_CORE", "cuda", "gpu"),
        ("PYRITE_MC_TRANSPORT_SCHEDULE", "auto", "streaming"),
        ("PYRITE_MC_RESOURCE_POLICY", "throughput", "max"),
        ("PYRITE_MC_MIN_CHUNK", "100", "0"),
    ],
)
def test_queue_scripts_forward_local_runtime_pins(monkeypatch, build, name, value, invalid):
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_SCHEDULE", raising=False)
    monkeypatch.delenv("PYRITE_MC_RESOURCE_POLICY", raising=False)
    monkeypatch.delenv("PYRITE_MC_MIN_CHUNK", raising=False)
    assert name not in build()

    monkeypatch.setenv(name, value.upper())
    assert f"export {name}={value}" in build()

    monkeypatch.setenv(name, invalid)
    with pytest.raises(ValueError, match=name):
        build()


def test_queue_script_profiles_uncached_repetitions_with_fixed_runtime_knobs():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        performance_profile="sub_100keV",
        performance_repetitions=3,
        performance_interval=1.0,
        spec_chunk=20_000,
        brem_chunk=10_000,
    )

    assert "export PYRITE_MC_CXR_CHUNK=20000" in script
    assert "export PYRITE_MC_BREM_CHUNK=10000" in script
    assert "performance_repetitions=3" in script
    assert "repetition<=performance_repetitions" in script
    assert '--checkpoint-dir "$PERFCKPT/$m/$repetition"' in script
    assert "--perf-interval 1" in script


def test_queue_script_wraps_single_profile_session_with_nsys():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        performance_profile="sub_100keV",
        nsys=True,
    )
    metadata = scripts._queue_metadata(
        "j",
        ["mos2"],
        False,
        6,
        performance_profile="sub_100keV",
        nsys=True,
    )

    assert "command -v nsys" in script
    assert "export PYRITE_MC_NSYS=1" in script
    assert "nsys profile" in script
    assert "--trace=cuda,nvtx,osrt" in script
    assert "--wait=all" in script
    # Python stack-walkers stay opt-in (SIGSEGV on CPython 3.14 under nsys
    # 2025.6.x); gated behind PYRITE_MC_NSYS_PYSTACK, not on by default.
    assert 'if [ -n "${PYRITE_MC_NSYS_PYSTACK:-}" ]; then' in script
    assert "--python-backtrace=cuda" in script
    assert '--output="$trace_base"' in script
    assert 'scan_launcher=("/path/to/pyrite/.venv/bin/python")' in script
    assert '--checkpoint-dir "$PERFCKPT/$m/$repetition"' in script
    assert "nsys stats" in script
    assert "--report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_sum" in script
    assert "nsys: True" in metadata
    bash = shutil.which("bash")
    if bash is not None:
        assert subprocess.run([bash, "-n"], input=script, text=True).returncode == 0


def test_queue_script_cpu_flag_adds_cprofile_after_primary():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        catalog_profile="sub_100keV",
        performance_profile="sub_100keV",
        cpu=True,
    )
    # Serial NumPy pass under cProfile, after the primary performance run.
    assert "-m pyrite._entry.profile_module " in script
    assert "env -u PYRITE_MC_NSYS PYRITE_MC_BACKEND=cpu" in script
    # Always the tiny --quick grid (representative call distribution in minutes,
    # not a >1h full-fidelity serial pass) with a hard --max-minutes backstop.
    assert '_entry.scan "sub_100keV" -m "$m" --quick --workers 0' in script
    assert "--max-minutes 10" in script
    # Fresh checkpoint dir (a cache hit would profile zero compute) and no
    # --performance-profile sampler that would clobber the GPU tick file.
    assert '--checkpoint-dir "$cpu_ckpt"' in script
    assert "cpu-profile-checkpoints" in script
    assert "--performance-profile" not in script.split("printf '%s\\n' \"cProfile")[1]
    assert "profiling CPU $m" in script
    assert "completed primary: $m" in script
    assert "FAILED CPU profile" in script
    assert "exit 1" in script
    bash = shutil.which("bash")
    if bash is not None:
        assert subprocess.run([bash, "-n"], input=script, text=True).returncode == 0


@pytest.mark.parametrize("cpu_only", [False, True])
def test_queue_script_cpu_profile_passes_only_known_scan_options(cpu_only):
    from pyrite.cli.commands.scan import _command

    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        catalog_profile="sub_100keV",
        performance_profile="sub_100keV",
        cpu=not cpu_only,
        cpu_only=cpu_only,
    )
    cpu_call = script.split("pyrite._entry.profile_module", 1)[1].split(">>", 1)[0]
    scan_args = cpu_call.split("pyrite._entry.scan", 1)[1].replace("\\\n", " ").split()
    known = {opt for param in _command.params for opt in (*param.opts, *param.secondary_opts)}

    assert {arg for arg in scan_args if arg.startswith("-")} <= known


def test_queue_script_nsys_alone_does_not_add_cpu_cprofile():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        catalog_profile="sub_100keV",
        performance_profile="sub_100keV",
        nsys=True,
    )
    assert "cProfile" not in script
    assert "PYRITE_MC_BACKEND=cpu" not in script


def test_queue_script_cpu_only_has_no_primary_gpu_or_sampler_phase():
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=6,
        catalog_profile="sub_100keV",
        performance_profile="sub_100keV",
        cpu_only=True,
    )

    assert "-m pyrite._entry.profile_module " in script
    assert "PYRITE_MC_BACKEND=cpu" in script
    assert "--progress-phase cpu" in script
    assert "cpu_only_enabled=1" in script
    assert 'if [ "$cpu_only_enabled" -eq 0 ]; then' in script
    assert "--performance-profile" not in script
    assert "nsys_enabled=0" in script


def test_queue_metadata_distinguishes_cpu_modes_from_nsys():
    metadata = scripts._queue_metadata(
        "j",
        ["mos2"],
        False,
        None,
        performance_profile="baseline",
        cpu=True,
    )
    cpu_only = scripts._queue_metadata(
        "j",
        ["mos2"],
        False,
        None,
        performance_profile="baseline",
        cpu_only=True,
    )

    assert "nsys: False" in metadata
    assert "cpu: True" in metadata
    assert "cpu_only: False" in metadata
    assert "cpu: False" in cpu_only
    assert "cpu_only: True" in cpu_only


@pytest.mark.parametrize("cpu_only", [False, True])
def test_queue_script_cpu_failure_is_terminal_and_keeps_primary_artifacts(
    monkeypatch, tmp_path, cpu_only
):
    fake_uv = tmp_path / "fake-uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\ncase " $* " in *" pyrite._entry.profile_module "*) exit 9 ;; esac\nexit 0\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "REMOTE_UV", str(fake_uv))
    artifact = tmp_path / "pyrite-output" / "performance" / "j" / "baseline" / "mos2.ndjson"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("primary artifact\n", encoding="utf-8")
    script = remote._queue_script(
        "j",
        ["mos2"],
        quick=False,
        workers=0,
        performance_profile="baseline",
        cpu=not cpu_only,
        cpu_only=cpu_only,
    )

    result = subprocess.run(["bash"], input=script, capture_output=True, text=True)

    assert result.returncode != 0
    assert (tmp_path / "jobs" / "j" / "state").read_text().startswith("FAILED CPU profile")
    assert artifact.read_text() == "primary artifact\n"


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
        assert '--performance-dir "$PERFDIR"' in script
    assert "performance_profile: baseline" in metadata


def test_queue_script_continues_then_fails_the_job_when_a_material_fails():
    script = remote._queue_script("j", ["hopg", "hbn"], quick=False, workers=None)

    assert "WARNING: scan failed for $m; continuing" in script
    assert "wait -n" in script
    assert "FAILED ($failures of $total material(s) failed)" in script
    assert "done with $failures warning(s)" not in script


def test_queue_script_wraps_sessions_with_py_spy_and_isolates_checkpoints():
    script = remote._queue_script(
        "j",
        ["hopg"],
        quick=False,
        workers=None,
        performance_profile="sub_100keV",
        py_spy=True,
    )

    assert "py_spy_enabled=1" in script
    assert 'tool run --from "py-spy>=0.4" py-spy' in script
    assert '"--output" "$trace_base.py-spy.json" "--"' in script
    assert '"--subprocesses"' in script and '"--nonblocking"' in script
    assert '-m pyrite.perf.py_spy \\\n        "$trace_base.py-spy.status" --' in script
    assert 'scan_rc=$(cat "$trace_base.py-spy.status" 2>/dev/null || echo 1)' in script
    assert '[ "$py_spy_enabled" -eq 1 ]; then\n      checkpoint_flags=' in script
    metadata = remote._queue_metadata("j", ["hopg"], False, None, py_spy=True)
    assert "py_spy: True" in metadata


def test_queue_script_refuses_py_spy_with_nsys():
    with pytest.raises(ValueError, match="py_spy"):
        remote._queue_script(
            "j",
            ["hopg"],
            quick=False,
            workers=None,
            performance_profile="p",
            nsys=True,
            py_spy=True,
        )
