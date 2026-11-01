"""Allocation fault injection through the public runner topologies."""

import numpy as np
import pytest

from pyrite._backend import BackendResourceError
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import scheduling


class AllocationError(Exception):
    pass


@pytest.fixture
def execution(monkeypatch):
    tp = {"E_grid": np.arange(2), "E_brem": np.arange(2)}
    calls = []
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (AllocationError,))
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom_retries", 2)
    monkeypatch.setattr(runner, "_admit_chunk", lambda chunk, *_args, **_kw: chunk)
    monkeypatch.setattr("pyrite.montecarlo.runner.stage_chunks._admit_chunk", runner._admit_chunk)
    monkeypatch.setattr(runner, "_transport_case", lambda *_args, **_kwargs: tp)
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: None)
    monkeypatch.setattr(scheduling, "case_runtime_plan", lambda case: {})
    monkeypatch.setattr(scheduling, "_cuda_transport_run", lambda cases: True)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")
    return tp, calls


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("phase", ["line", "characteristic", "brem"])
def test_public_runner_retries_phase_without_retransport(execution, monkeypatch, serial, phase):
    tp, calls = execution
    case = {"spec_chunk": 8, "brem_chunk": 8, "seed": 42}

    def spectrum(work, transport, record_timing=False):
        assert transport is tp
        calls.append((work["spec_chunk"], work["brem_chunk"]))
        if len(calls) == 1:
            raise runner._SpectrumPhaseOOM(phase, AllocationError("allocation failed"))
        return {"spec": np.array([1.0, 2.0])}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    if serial:
        timings = []
        out = runner.run_cases([case], progress=False, on_timing=timings.append)[0]
        assert timings[0][f"{phase}_gpu_oom_retries"] == 1
    else:
        out = runner.run_case(case)
    np.testing.assert_array_equal(out["spec"], [1.0, 2.0])
    assert calls == [(8, 8), (4, 8) if phase == "line" else (8, 4)]
    assert case == {"spec_chunk": 8, "brem_chunk": 8, "seed": 42}


@pytest.mark.parametrize("error", [AllocationError("oom"), BackendResourceError("budget")])
def test_direct_auto_fallback_preserves_transport_and_reports_path(execution, monkeypatch, error):
    tp, calls = execution
    monkeypatch.setattr(runner, "_admit_cpu_fallback", lambda: calls.append("admit"))

    def spectrum(case, transport, record_timing=False):
        assert transport is tp
        calls.append(runner._RESOURCE_POLICY.gpu)
        if runner._RESOURCE_POLICY.gpu:
            raise error
        return {"spec": np.array([1.0])}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    with pytest.warns(RuntimeWarning, match="CPU NumPy"):
        out = runner.run_case({"spec_chunk": 4, "brem_chunk": 4})
    assert out["_recovery_path"] == "cpu_spectrum"
    assert calls[-2:] == ["admit", False]
    assert calls.count(True) == (1 if isinstance(error, BackendResourceError) else 3)
    assert out.get("_gpu_oom_retries", 0) == (0 if isinstance(error, BackendResourceError) else 2)
    assert runner._RESOURCE_POLICY.gpu is True


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_direct_forced_backend_never_falls_back(execution, monkeypatch, backend):
    monkeypatch.setenv("PYRITE_MC_BACKEND", backend)
    monkeypatch.setattr(
        runner, "_spectrum_case", lambda *_a, **_kw: (_ for _ in ()).throw(AllocationError("oom"))
    )
    with pytest.raises(AllocationError):
        runner.run_case({"spec_chunk": 4, "brem_chunk": 4})


def test_direct_runtime_bug_is_not_retried(execution, monkeypatch):
    _, calls = execution

    def spectrum(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("kernel bug")

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    with pytest.raises(RuntimeError, match="kernel bug"):
        runner.run_case({"spec_chunk": 4, "brem_chunk": 4})
    assert calls == [1]


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("admission", [False, True])
def test_exhaustion_and_host_admission(execution, monkeypatch, serial, admission):
    _, calls = execution

    def spectrum(case, tp, record_timing=False):
        calls.append(runner._RESOURCE_POLICY.gpu)
        raise BackendResourceError("device budget") if admission else AllocationError("device oom")

    def refuse_host():
        raise BackendResourceError("cannot admit CPU fallback")

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    monkeypatch.setattr(runner, "_admit_cpu_fallback", refuse_host)
    case = {"spec_chunk": 8, "brem_chunk": 8}
    with pytest.raises(BackendResourceError, match="cannot admit CPU fallback") as caught:
        if serial:
            runner.run_cases([case], progress=False)
        else:
            runner.run_case(case)
    assert calls == [True] * (1 if admission else 3)
    assert isinstance(
        caught.value.__context__, BackendResourceError if admission else AllocationError
    )
    assert runner._RESOURCE_POLICY.gpu is True


def test_explicit_serial_gpu_engine_never_falls_back(execution, monkeypatch):
    _, calls = execution

    def spectrum(case, tp, record_timing=False):
        calls.append(1)
        raise runner._SpectrumPhaseOOM("line", AllocationError("device oom"))

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    with pytest.raises(AllocationError) as caught:
        runner.run_cases([{"spec_chunk": 8, "brem_chunk": 8}], engine="gpu", progress=False)
    assert calls == [1, 1, 1]
    assert any("phase=line, retries=2" in note for note in caught.value.__notes__)


def test_resident_fallback_downloads_arrays_before_cpu_scoring(execution, monkeypatch):
    tp, calls = execution

    class DeviceArray:
        pass

    device = DeviceArray()
    host = np.array([1.0, 2.0])
    tp["segs"] = {"L_ang": device, "nested": [{"energy": device}], "Ne": 2}
    monkeypatch.setattr(runner.BACKEND, "name", "cuda")
    monkeypatch.setattr(runner.BACKEND, "is_array", lambda a: isinstance(a, DeviceArray))
    monkeypatch.setattr(runner.BACKEND, "to_cpu", lambda a: calls.append("download") or host)
    monkeypatch.setattr(runner, "_admit_cpu_fallback", lambda: calls.append("admit"))

    def spectrum(case, transport, record_timing=False):
        if runner._RESOURCE_POLICY.gpu:
            raise BackendResourceError("device budget")
        assert transport["segs"]["L_ang"] is host
        assert transport["segs"]["nested"][0]["energy"] is host
        assert transport["segs"]["Ne"] == 2
        return {"spec": host}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    with pytest.warns(RuntimeWarning, match="CPU NumPy"):
        runner.run_case({"spec_chunk": 4, "brem_chunk": 4})
    assert calls == ["admit", "download", "download"]


@pytest.mark.parametrize("failure", ["pool", "delayed", "admission", "bug", "exhausted"])
def test_cuda_resident_transport_recovery(monkeypatch, failure):
    from pyrite.montecarlo.runner import pool
    from tests.helpers.segments import fake_segments

    class DelayedOOM(Exception):
        pass

    calls = []
    segments = fake_segments(2)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (AllocationError,))
    monkeypatch.setattr(runner.BACKEND, "is_oom_error", lambda error: isinstance(error, DelayedOOM))
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: calls.append("release"))

    def admit():
        calls.append("admit")
        if failure == "admission":
            raise BackendResourceError("cannot admit host segments")

    monkeypatch.setattr(pool, "_admit_host_segments", admit)
    monkeypatch.setattr(
        runner.case_driver,
        "_case_geometry",
        lambda case: (
            np.arange(2),
            np.arange(2),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            None,
        ),
    )
    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(runner.case_driver, "_case_stopping_tables", lambda case: None)
    monkeypatch.setattr(runner.case_driver, "resolve_cxr_grid", lambda *args: (np.arange(2), None))

    def simulate(*args, **kwargs):
        resident = kwargs["keep_segments_on_device"]
        calls.append((resident, kwargs["seed"], kwargs["transport_core"]))
        if failure == "bug":
            raise RuntimeError("illegal memory access")
        if resident or failure == "exhausted":
            raise (
                DelayedOOM("delayed allocation")
                if failure == "delayed"
                else AllocationError("allocation")
            )
        return segments

    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", simulate)
    case = {
        "Ne": 2,
        "Ne_brem": 2,
        "seed": 42,
        "E0_keV": 30.0,
        "thickness_ang": 10.0,
        "composition": [],
    }
    if failure in ("admission", "bug", "exhausted"):
        error_type = {
            "admission": BackendResourceError,
            "bug": RuntimeError,
            "exhausted": AllocationError,
        }[failure]
        with pytest.raises(error_type):
            runner._transport_case(case, transport_core="cuda", keep_segments_on_device=True)
    else:
        result = runner._transport_case(case, transport_core="cuda", keep_segments_on_device=True)
        assert result["segs"] is segments
        assert result["_transport_gpu_oom_retries"] == 1
        assert result["_recovery_path"] == "host_segments"
    expected = [(True, 42, "cuda")]
    if failure != "bug":
        expected += ["release", "admit"]
    if failure not in ("bug", "admission"):
        expected += [(False, 42, "cuda")]
    assert calls == expected


def test_chunk_retry_shrinks_admitted_chunk(monkeypatch):
    from pyrite.montecarlo.runner import stage_chunks

    monkeypatch.setattr(
        stage_chunks, "_admit_chunk", lambda requested, *_a, **_kw: min(16, requested)
    )
    work = {"spec_chunk": 1024, "brem_chunk": 1024}
    runner._halve_case_spec_chunk(work, {"E_grid": np.arange(2)})
    runner._halve_case_brem_chunk(work, {"E_brem": np.arange(2)})
    assert work == {"spec_chunk": 8, "brem_chunk": 8}


def test_cpu_fallback_failure_restores_backend(execution, monkeypatch):
    monkeypatch.setattr(runner, "_admit_cpu_fallback", lambda: None)

    def spectrum(*_args, **_kwargs):
        if runner._RESOURCE_POLICY.gpu:
            raise BackendResourceError("device budget")
        raise MemoryError("host allocation failed")

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    with pytest.warns(RuntimeWarning, match="CPU NumPy"):
        with pytest.raises(MemoryError, match="host allocation failed") as caught:
            runner.run_case({"spec_chunk": 4, "brem_chunk": 4})
    assert runner._RESOURCE_POLICY.gpu is True
    assert any("device_budget_infeasible" in note for note in caught.value.__notes__)


def test_serial_timing_includes_transport_and_spectrum_mitigations(execution, monkeypatch):
    tp, calls = execution
    tp.update(
        _transport_gpu_oom_retries=1,
        _recovery_path="host_segments",
        _recovery_events=[{"phase": "transport", "mitigation": "host_segments"}],
    )

    def spectrum(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise runner._SpectrumPhaseOOM("line", AllocationError("device allocation"))
        return {"spec": np.zeros(2)}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    timing = []
    out = runner.run_cases(
        [{"spec_chunk": 4, "brem_chunk": 4}], progress=False, on_timing=timing.append
    )[0]
    assert timing[0]["gpu_oom_retry_count"] == 2
    assert timing[0]["recovery_events"] == [
        {"phase": "transport", "mitigation": "host_segments"},
        {"phase": "line", "mitigation": "halve_chunks"},
    ]
    assert timing[0]["recovery_path"] == "host_segments"
    assert not any(key.startswith("_") for key in out)


def test_host_segment_admission_uses_transport_budget(monkeypatch):
    from pyrite.montecarlo.runner import pool

    policy = runner._RESOURCE_POLICY
    monkeypatch.setattr(policy, "total_mem_mb", 10000)
    monkeypatch.setattr(policy, "available_mem_mb", lambda: 2000)
    monkeypatch.setattr(policy, "host_fraction", 1.0)
    monkeypatch.setattr(policy, "host_reserve_bytes", 0)
    monkeypatch.setattr(policy, "pipeline_worker_mem_mb", 1500)
    monkeypatch.setattr(policy, "worker_mem_mb", 6000)
    pool._admit_host_segments()
    with pytest.raises(BackendResourceError, match="CPU fallback"):
        pool._admit_cpu_fallback()
    monkeypatch.setattr(policy, "available_mem_mb", lambda: 1000)
    with pytest.raises(BackendResourceError, match="host segments"):
        pool._admit_host_segments()
