from contextlib import contextmanager

import numpy as np
import pytest

import pyrite.montecarlo.runner as runner
from pyrite.montecarlo.runner import scheduling


class DummyOOM(Exception):
    pass


@pytest.fixture(autouse=True)
def _chunk_admission_off_device(monkeypatch):
    """Report chunks without real device-memory admission.

    ``_effective_spec_chunk``/``_effective_brem_chunk`` push the requested chunk
    through ``_admit_chunk``, which clamps against free VRAM whenever the policy
    says a GPU is present. On a CUDA box the eight dense EEDL intermediates then
    drag a requested 100_000 brem chunk down to whatever the device budget
    allows, so the fixed expectations below would be hardware-dependent. These
    tests are about retry bookkeeping, not admission, so pin the policy off.
    """
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)


def _tp():
    return {"E_grid": np.zeros(2000), "E_brem": np.zeros(2000)}


def test_retry_succeeds_after_halving(monkeypatch):
    calls = []

    def fake_spectrum(case, tp, record_timing=False):
        calls.append((case.get("spec_chunk"), case.get("brem_chunk")))
        if len(calls) <= 3:
            raise runner._SpectrumPhaseOOM("line", DummyOOM())
        return {"ok": True}

    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    case = {"spec_chunk": 30_000, "brem_chunk": 100_000}
    out = runner._spectrum_case_retry(case, _tp(), max_retries=3)

    assert out["ok"]
    assert case == {"spec_chunk": 30_000, "brem_chunk": 100_000}
    assert calls == [
        (30_000, 100_000),
        (15_000, 100_000),
        (7_500, 100_000),
        (3_750, 100_000),
    ]
    assert out["_attempted_spec_chunk"] == 30_000
    assert out["_effective_spec_chunk"] == 3_750
    assert out["_gpu_oom_retries"] == 3
    assert out["_line_gpu_oom_retries"] == 3
    assert out["_brem_gpu_oom_retries"] == 0
    assert out["_generic_gpu_oom_retries"] == 0


def test_retry_reraises_on_exhaustion(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_spectrum_case",
        lambda c, tp, record_timing=False: (_ for _ in ()).throw(
            runner._SpectrumPhaseOOM("line", DummyOOM())
        ),
    )

    with pytest.raises(DummyOOM):
        runner._spectrum_case_retry({}, _tp(), max_retries=2)


def test_generic_oom_preserves_legacy_dual_chunk_fallback(monkeypatch):
    calls = []

    def fake_spectrum(case, tp, record_timing=False):
        calls.append((case.get("spec_chunk"), case.get("brem_chunk")))
        if len(calls) == 1:
            raise DummyOOM
        return {"ok": True}

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (DummyOOM,))
    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    case = {"spec_chunk": 30_000, "brem_chunk": 100_000}

    out = runner._spectrum_case_retry(case, _tp(), max_retries=3)

    assert out["ok"]
    assert calls == [(30_000, 100_000), (15_000, 50_000)]
    assert case == {"spec_chunk": 30_000, "brem_chunk": 100_000}
    assert out["_generic_gpu_oom_retries"] == 1
    assert out["_line_gpu_oom_retries"] == 0
    assert out["_brem_gpu_oom_retries"] == 0


def test_brem_oom_halves_only_brem_and_preserves_original_case(monkeypatch):
    calls = []

    def fake_spectrum(case, tp, record_timing=False):
        calls.append((case.get("spec_chunk"), case.get("brem_chunk")))
        if len(calls) == 1:
            raise runner._SpectrumPhaseOOM("brem", DummyOOM())
        return {"ok": True}

    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    case = {"spec_chunk": 30_000, "brem_chunk": 100_000}

    out = runner._spectrum_case_retry(case, _tp(), max_retries=3)

    assert out["ok"]
    assert calls == [(30_000, 100_000), (30_000, 50_000)]
    assert case == {"spec_chunk": 30_000, "brem_chunk": 100_000}
    assert out["_attempted_brem_chunk"] == 100_000
    assert out["_effective_brem_chunk"] == 50_000
    assert out["_brem_gpu_oom_retries"] == 1
    assert out["_line_gpu_oom_retries"] == 0


def test_delayed_runtime_oom_inside_brem_is_phase_tagged(monkeypatch):
    from tests.helpers.segments import runner_transport_payload

    class CUDARuntimeError(Exception):
        pass

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", ())
    monkeypatch.setattr(
        runner.BACKEND,
        "is_oom_error",
        lambda error: isinstance(error, CUDARuntimeError),
    )
    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda *_args, **_kwargs: np.zeros(2),
    )
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *_args, **_kwargs: np.zeros(2),
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CUDARuntimeError("cudaErrorMemoryAllocation: out of memory")
        ),
    )
    segments = {
        "n_backscattered": 0,
        "n_missed": 0,
        "Ne": 1,
        "L_ang": np.array([1.0]),
    }
    tp = runner_transport_payload(
        segs=segments,
        E_grid=np.array([100.0, 200.0]),
        E_brem=np.array([100.0, 200.0]),
        n_hat=np.array([0.0, 0.0, 1.0]),
        groove=None,
    )

    with pytest.raises(runner._SpectrumPhaseOOM) as caught:
        runner._spectrum_case(
            {"name": "oom", "crystal": "hopg", "E0_keV": 30.0},
            tp,
        )

    assert caught.value.phase == "brem"
    assert isinstance(caught.value.error, CUDARuntimeError)


class _SyncProcessPoolExecutor:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_result(fn(*args))
        return future


def test_gpu_pipeline_reuses_successful_line_fallback(monkeypatch):
    calls = []

    def fake_transport(case, record_timing=False, **_kwargs):
        assert record_timing
        return {
            "E_grid": np.zeros(2000),
            "E_brem": np.zeros(2000),
            "_t_transport": 0.1,
            "name": case["name"],
        }

    def fake_spectrum(case, tp, record_timing=False):
        assert record_timing
        calls.append((tp["name"], case.get("spec_chunk"), case.get("brem_chunk")))
        if len(calls) == 1:
            raise runner._SpectrumPhaseOOM("line", DummyOOM())
        return {"name": tp["name"], "_t_spectrum": 0.2}

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", fake_transport)
    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)
    cases = [
        {
            "name": f"case-{index}",
            "spec_chunk": 30_000,
            "brem_chunk": 100_000,
        }
        for index in range(3)
    ]
    timings = []

    out = runner.run_cases(
        cases,
        max_workers=2,
        progress=False,
        on_timing=timings.append,
    )

    assert [item["name"] for item in out] == ["case-0", "case-1", "case-2"]
    assert calls == [
        ("case-0", 30_000, 100_000),
        ("case-0", 15_000, 100_000),
        ("case-1", 15_000, 100_000),
        ("case-2", 15_000, 100_000),
    ]
    assert all(case["spec_chunk"] == 30_000 for case in cases)
    assert all(case["brem_chunk"] == 100_000 for case in cases)
    assert timings[0]["gpu_oom_retry_count"] == 1
    assert timings[0]["line_gpu_oom_retries"] == 1
    assert timings[0]["brem_gpu_oom_retries"] == 0
    assert timings[0]["attempted_spec_chunk"] == 30_000
    assert timings[0]["effective_spec_chunk"] == 15_000
    assert [timing["learned_spec_chunk"] for timing in timings] == [15_000] * 3


def test_gpu_pipeline_auto_fallback_recognizes_delayed_runtime_oom(monkeypatch):
    class CUDARuntimeError(Exception):
        pass

    def fake_transport(case, record_timing=False):
        return {
            "E_grid": np.zeros(2000),
            "E_brem": np.zeros(2000),
            "name": case["name"],
        }

    @contextmanager
    def cpu_spectrum_backend():
        yield

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", ())
    monkeypatch.setattr(
        runner.BACKEND,
        "is_oom_error",
        lambda error: isinstance(error, CUDARuntimeError),
    )
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_admit_cpu_fallback", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", fake_transport)
    monkeypatch.setattr(
        scheduling,
        "_spectrum_case_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CUDARuntimeError("cudaErrorMemoryAllocation: out of memory")
        ),
    )
    monkeypatch.setattr(scheduling, "_cpu_spectrum_backend", cpu_spectrum_backend)
    monkeypatch.setattr(
        scheduling,
        "_spectrum_case",
        lambda case, *_args, **_kwargs: {"name": case["name"]},
    )
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)
    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")

    timings = []
    with pytest.warns(RuntimeWarning, match="rerunning spectrum phase on CPU NumPy"):
        out = runner.run_cases(
            [{"name": "case-0", "spec_chunk": 30_000, "brem_chunk": 100_000}],
            max_workers=2,
            progress=False,
            engine="auto",
            on_timing=timings.append,
        )

    assert out[0]["name"] == "case-0"
    assert "cudaErrorMemoryAllocation" in timings[0]["backend_fallback_reason"]


def test_spectrum_case_emits_nsys_phase_ranges(monkeypatch):
    from tests.helpers.segments import runner_transport_payload

    entered = []

    @contextmanager
    def record_range(message):
        entered.append(message)
        yield

    monkeypatch.setattr(runner, "_nsys_range", record_range)
    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda *_args, **_kwargs: np.array([1.0, 2.0]),
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda *_args, **_kwargs: np.array([3.0, 4.0]),
    )
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *_args, **_kwargs: np.zeros(2),
    )
    segments = {
        "n_backscattered": 0,
        "n_missed": 0,
        "Ne": 1,
        "L_ang": np.array([1.0]),
    }
    case = {"name": "heavy case", "crystal": "mos2", "E0_keV": 300.0}
    tp = runner_transport_payload(
        segs=segments,
        E_grid=np.array([100.0, 200.0]),
        E_brem=np.array([100.0, 200.0]),
        n_hat=np.array([0.0, 0.0, 1.0]),
        groove=None,
    )

    runner._spectrum_case(case, tp)

    assert entered == [
        "cxr.spectrum_case:heavy case",
        "cxr.lines",
        "cxr.characteristic",
        "cxr.brem",
        "cxr.interpolate",
    ]


def test_gpu_pipeline_evaluates_directions_on_the_pipelined_transport(monkeypatch):
    transports = []
    directional_calls = []

    def fake_transport(case, record_timing=False, **_kwargs):
        transports.append(case["name"])
        return {"E_grid": np.zeros(2000), "E_brem": np.zeros(2000), "name": case["name"]}

    def fake_directional(case, tp, directions, spectrum=None):
        assert tp["name"] == case["name"]
        assert spectrum.func is scheduling._spectrum_case_retry
        directional_calls.append((case["name"], len(directions)))
        return {"spec_by_direction": np.zeros((len(directions), 2))}

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", fake_transport)
    monkeypatch.setattr(
        scheduling, "_spectrum_case_retry", lambda case, tp, **_kwargs: {"name": case["name"]}
    )
    monkeypatch.setattr(scheduling, "_directional_outputs", fake_directional)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    out = runner.run_cases(
        [{"name": "observed"}, {"name": "scalar"}],
        max_workers=2,
        progress=False,
        observation_directions=[np.zeros((3, 3)), None],
    )

    assert transports == ["observed", "scalar"]
    assert directional_calls == [("observed", 3)]
    assert out[0]["directional"]["spec_by_direction"].shape == (3, 2)
    assert "directional" not in out[1]


def test_device_transport_serial_path_reports_timing(monkeypatch):
    seen = []

    def fake_run_case(case, record_timing=False, **kwargs):
        seen.append((record_timing, kwargs.get("keep_segments_on_device")))
        return {"name": case["name"], "_t_spectrum": 0.3, "_line_axis_nodes": 1234}

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "_cuda_transport_run", lambda cases: True)
    monkeypatch.setattr(scheduling, "case_runtime_plan", lambda case: {})
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "run_case", fake_run_case)
    timings = []

    out = runner.run_cases(
        [{"name": "case-0"}, {"name": "case-1"}],
        progress=False,
        on_timing=timings.append,
    )

    assert [item["name"] for item in out] == ["case-0", "case-1"]
    assert seen == [(True, True), (True, True)]
    assert [item["spectrum_seconds"] for item in timings] == [0.3, 0.3]
    assert all(item["line_axis_nodes"] == 1234 for item in timings)


def test_gpu_pipeline_falls_back_per_case_when_a_later_case_is_infeasible(monkeypatch):
    # #266: only the later wide case exceeds the device budget; earlier cases
    # stay on the device and the run completes.
    from pyrite._backend import BackendResourceError

    def fake_transport(case, record_timing=False):
        return {"E_grid": np.zeros(2), "E_brem": np.zeros(2), "name": case["name"]}

    def fake_retry(case, tp, **_kwargs):
        if case["name"] == "wide":
            raise BackendResourceError("device budget cannot admit 1 segment")
        return {"name": case["name"], "backend": "device"}

    @contextmanager
    def cpu_spectrum_backend():
        yield

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "case_runtime_plan", lambda case: {})
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_admit_cpu_fallback", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", fake_transport)
    monkeypatch.setattr(scheduling, "_spectrum_case_retry", fake_retry)
    monkeypatch.setattr(scheduling, "_cpu_spectrum_backend", cpu_spectrum_backend)
    monkeypatch.setattr(
        scheduling,
        "_spectrum_case",
        lambda case, *_args, **_kwargs: {"name": case["name"], "backend": "cpu"},
    )
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)
    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")

    timings = []
    with pytest.warns(RuntimeWarning, match="device_budget_infeasible"):
        out = runner.run_cases(
            [{"name": "narrow-0"}, {"name": "narrow-1"}, {"name": "wide"}],
            max_workers=2,
            progress=False,
            engine="auto",
            on_timing=timings.append,
        )

    assert [(item["name"], item["backend"]) for item in out] == [
        ("narrow-0", "device"),
        ("narrow-1", "device"),
        ("wide", "cpu"),
    ]
    reasons = [timing.get("backend_fallback_reason") for timing in timings]
    assert reasons[:2] == [None, None]
    assert reasons[2].startswith("device_budget_infeasible: ")


def test_gpu_pipeline_reraises_infeasible_case_when_engine_is_forced(monkeypatch):
    from pyrite._backend import BackendResourceError

    def fake_transport(case, record_timing=False):
        return {"E_grid": np.zeros(2), "E_brem": np.zeros(2), "name": case["name"]}

    def fake_retry(case, tp, **_kwargs):
        raise BackendResourceError("device budget cannot admit 1 segment")

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "case_runtime_plan", lambda case: {})
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", fake_transport)
    monkeypatch.setattr(scheduling, "_spectrum_case_retry", fake_retry)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    with pytest.raises(BackendResourceError, match="cannot admit"):
        runner.run_cases([{"name": "wide"}], max_workers=2, progress=False, engine="gpu")


def test_startup_check_sends_run_to_cpu_when_a_later_case_is_infeasible(monkeypatch):
    # #266: a fresh run used to plan only cases[0] (narrow), stay on the
    # device, and die on the later wide case.
    from pyrite._backend import BackendResourceError

    narrow = {"name": "narrow", "E_grid": [1.0, 2.0], "E_grid_brem": [1.0, 2.0]}
    wide = {"name": "wide", "E_grid": [1.0, 2.0, 3.0], "E_grid_brem": [1.0, 2.0, 3.0]}
    planned = []

    def fake_plan(case):
        planned.append(case["name"])
        if case is wide:
            raise BackendResourceError("device budget cannot admit 1 segment")
        return {}

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "case_runtime_plan", fake_plan)
    monkeypatch.setattr(scheduling, "_admit_cpu_fallback", lambda: None)
    monkeypatch.setattr(
        scheduling,
        "run_case",
        lambda case, *_args, **_kwargs: {"name": case["name"], "gpu": runner._RESOURCE_POLICY.gpu},
    )
    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")

    timings = []
    with pytest.warns(RuntimeWarning, match="device_budget_infeasible"):
        out = runner.run_cases(
            [narrow, wide],
            max_workers=0,
            progress=False,
            engine="auto",
            on_timing=timings.append,
        )

    assert planned == ["wide"]
    assert [(item["name"], item["gpu"]) for item in out] == [("narrow", False), ("wide", False)]
    assert all(
        timing["backend_fallback_reason"].startswith("device_budget_infeasible: ")
        for timing in timings
    )
