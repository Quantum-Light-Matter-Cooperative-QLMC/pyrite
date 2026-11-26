"""Bounded ``cpu_pool``: no case is submitted once ``should_stop`` is true (#1478).

CPython's ``ProcessPoolExecutor`` marks up to ``max_workers + 1`` queued items
running when it moves them to its call queue, so ``Future.cancel()`` cannot stop
them. The fake pool below models that: every submitted future is running (not
cancellable) and only executes when the driver blocks on a completion.
"""

import concurrent.futures as cf

import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import scheduling


class _LazyPool:
    """Records submissions; a submitted job runs only when the driver waits."""

    instances: list["_LazyPool"] = []

    def __init__(self, max_workers=None, **_kwargs):
        self.max_workers = max_workers
        self.submitted: list[str] = []
        self.unfinished: list[tuple[cf.Future, object, tuple, dict]] = []
        self.max_unfinished = 0
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        future = cf.Future()
        future.set_running_or_notify_cancel()  # call-queue items cannot be cancelled
        self.submitted.append(args[0]["name"])
        self.unfinished.append((future, fn, args, kwargs))
        self.max_unfinished = max(self.max_unfinished, len(self.unfinished))
        return future

    def run_next(self):
        if not self.unfinished:
            return
        future, fn, args, kwargs = self.unfinished.pop(0)
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)

    def run_all(self):
        while self.unfinished:
            self.run_next()


@pytest.fixture
def pool(monkeypatch):
    _LazyPool.instances = []
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _LazyPool)
    real_wait, real_as_completed = cf.wait, cf.as_completed

    def wait(fs, *args, **kwargs):
        _LazyPool.instances[-1].run_next()
        return real_wait(fs, *args, **kwargs)

    def as_completed(fs, *args, **kwargs):
        _LazyPool.instances[-1].run_all()
        return real_as_completed(fs, *args, **kwargs)

    monkeypatch.setattr("concurrent.futures.wait", wait)
    monkeypatch.setattr("concurrent.futures.as_completed", as_completed)
    monkeypatch.setattr(scheduling, "run_case", _compute)
    return _LazyPool


def _compute(case, *_args, **_kwargs):
    if case["name"] == "bad":
        raise ValueError("boom")
    return {"name": case["name"]}


def _cases(n):
    return [{"name": f"c{i}"} for i in range(n)]


def test_bounded_stop_submits_nothing_after_deadline(pool):
    delivered = []
    results = runner.run_cases(
        _cases(6),
        max_workers=2,
        progress=False,
        callback=lambda i, case, out: delivered.append(case["name"]),
        should_stop=lambda: len(delivered) >= 2,
    )
    submitted = pool.instances[0].submitted
    # c0, c1 up front; c2 refills after c0 (stop false); nothing after c1 trips it.
    assert submitted == ["c0", "c1", "c2"]
    assert delivered == ["c0", "c1", "c2"]  # in-flight c2 still drains
    assert [r["name"] if r else None for r in results] == ["c0", "c1", "c2", None, None, None]
    assert pool.instances[0].max_unfinished <= 2


def test_bounded_stop_already_true_submits_nothing(pool):
    results = runner.run_cases(_cases(4), max_workers=2, progress=False, should_stop=lambda: True)
    assert not pool.instances or pool.instances[0].submitted == []
    assert results == [None] * 4


def test_bounded_never_stopping_runs_every_case_within_window(pool):
    results = runner.run_cases(_cases(5), max_workers=2, progress=False, should_stop=lambda: False)
    assert [r["name"] for r in results] == [f"c{i}" for i in range(5)]
    assert pool.instances[0].submitted == [f"c{i}" for i in range(5)]
    assert pool.instances[0].max_unfinished == 2


def test_unbounded_submits_every_case_up_front(pool):
    results = runner.run_cases(_cases(5), max_workers=2, progress=False)
    assert [r["name"] for r in results] == [f"c{i}" for i in range(5)]
    assert pool.instances[0].max_unfinished == 5


def test_bounded_skipped_failure_refills_window(pool):
    cases = [{"name": "c0"}, {"name": "bad"}, {"name": "c2"}, {"name": "c3"}]
    errors = []
    results = runner.run_cases(
        cases,
        max_workers=1,
        progress=False,
        should_stop=lambda: False,
        on_case_error=lambda i, case, exc: errors.append(i) or True,
    )
    assert errors == [1]
    assert pool.instances[0].submitted == ["c0", "bad", "c2", "c3"]
    assert [r["name"] if r else None for r in results] == ["c0", None, "c2", "c3"]
    assert pool.instances[0].max_unfinished == 1


def test_bounded_skipped_failure_does_not_submit_after_stop(pool):
    cases = [{"name": "c0"}, {"name": "bad"}, {"name": "c2"}]
    stop = {"now": False}

    def on_error(i, case, exc):
        stop["now"] = True
        return True

    results = runner.run_cases(
        cases,
        max_workers=1,
        progress=False,
        should_stop=lambda: stop["now"],
        on_case_error=on_error,
    )
    assert pool.instances[0].submitted == ["c0", "bad"]
    assert [r["name"] if r else None for r in results] == ["c0", None, None]
