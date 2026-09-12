from contextlib import contextmanager

import numpy as np

import pyrite.montecarlo.runner as runner
from pyrite.montecarlo.runner import scheduling


class DummyOOM(Exception):
    pass


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
    import pytest

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

    import pytest

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

    import pytest

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
