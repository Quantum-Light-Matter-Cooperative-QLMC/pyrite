"""``run_cases(on_case_error=...)``: skip a failed case and keep running the rest."""

from concurrent.futures import Future

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import scheduling

CASES = [{"name": "case-0"}, {"name": "bad"}, {"name": "case-2"}]


class _SyncPool:
    """In-process pool whose futures carry a raised exception like real ones."""

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def _compute(case, *_args, **_kwargs):
    if case["name"] == "bad":
        raise ValueError("bandwidth refused")
    return {"name": case["name"]}


def _transport(case, *_args, **_kwargs):
    return {"E_grid": np.zeros(2), "E_brem": np.zeros(2), "name": case["name"]}


def _branch(monkeypatch, branch):
    """Route ``run_cases`` onto ``branch`` with ``bad`` failing in the named phase."""
    monkeypatch.setattr(scheduling, "case_runtime_plan", lambda case: {})
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncPool)
    gpu = branch != "cpu-pool"
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", gpu)
    monkeypatch.setattr(scheduling, "_cuda_transport_run", lambda cases: branch == "serial")
    if branch == "serial":
        monkeypatch.setattr(scheduling, "run_case", _compute)
    elif branch.startswith("pipeline"):
        monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
        failing_transport = branch == "pipeline-transport"
        worker = _compute if failing_transport else _transport

        def spectrum(case, tp, **_kwargs):
            return {"name": case["name"]} if failing_transport else _compute(case)

        monkeypatch.setattr(scheduling, "_pool_worker", lambda _case, *, transport: worker)
        monkeypatch.setattr(scheduling, "_spectrum_with_recovery", spectrum)
    else:
        monkeypatch.setattr(scheduling, "_pool_worker", lambda _case, *, transport: _compute)
        monkeypatch.setattr(scheduling, "_cpu_pool_workers", lambda *_args: 2)


BRANCHES = ["serial", "pipeline-transport", "pipeline-spectrum", "cpu-pool"]


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("bounded", [False, True])
def test_absorbed_failure_skips_only_that_case(monkeypatch, branch, bounded):
    _branch(monkeypatch, branch)
    errors, delivered = [], []

    out = runner.run_cases(
        CASES,
        progress=False,
        callback=lambda _i, case, _out: delivered.append(case["name"]),
        should_stop=(lambda: False) if bounded else None,
        on_case_error=lambda i, case, exc: errors.append((i, case["name"], str(exc))) or True,
    )

    assert errors == [(1, "bad", "bandwidth refused")]
    assert sorted(delivered) == ["case-0", "case-2"]
    assert [None if item is None else item["name"] for item in out] == ["case-0", None, "case-2"]


@pytest.mark.parametrize("branch", BRANCHES)
def test_refused_failure_aborts(monkeypatch, branch):
    _branch(monkeypatch, branch)

    with pytest.raises(ValueError, match="bandwidth refused"):
        runner.run_cases(CASES, progress=False, on_case_error=lambda *_args: False)


def test_without_hook_first_failure_aborts(monkeypatch):
    _branch(monkeypatch, "serial")

    with pytest.raises(ValueError, match="bandwidth refused"):
        runner.run_cases(CASES, progress=False)


def test_callback_errors_are_never_absorbed(monkeypatch):
    _branch(monkeypatch, "serial")

    def callback(_i, _case, _out):
        raise RuntimeError("checkpoint write failed")

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        runner.run_cases(CASES, progress=False, callback=callback, on_case_error=lambda *_a: True)


def test_broken_pool_is_never_absorbed(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    _branch(monkeypatch, "cpu-pool")

    def broken(_case, *_args, **_kwargs):
        raise BrokenProcessPool("worker died")

    monkeypatch.setattr(scheduling, "_pool_worker", lambda _case, *, transport: broken)

    with pytest.raises(BrokenProcessPool):
        runner.run_cases(CASES, progress=False, on_case_error=lambda *_a: True)
