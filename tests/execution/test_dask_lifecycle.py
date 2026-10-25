"""Backpressure and cancellation at the Dask submission boundary."""

import sys
import types
from collections import deque
from concurrent.futures import Future

import numpy as np
import pytest

from pyrite.execution import CaseArtifactStore, DaskConfig, DaskExecutor, RetryPolicy


class TrackedFuture(Future):
    def release(self):
        self.released = True


class TrackingClient:
    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.peak_outstanding = 0

    def submit(self, fn, *args, **kwargs):
        future = TrackedFuture()
        future.released = False
        self.submitted.append(future)
        self.peak_outstanding = max(
            self.peak_outstanding, sum(not f.released for f in self.submitted)
        )
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def cancel(self, futures):
        self.cancelled.extend(futures)


class ReadyQueue:
    def __init__(self, futures=()):
        self.ready = deque(futures)

    def add(self, future):
        self.ready.append(future)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.ready:
            raise StopIteration
        return self.ready.popleft()


@pytest.fixture
def submitted_campaign(tmp_path, monkeypatch):
    import pyrite.execution.dask as backend

    monkeypatch.setitem(sys.modules, "distributed", types.SimpleNamespace(as_completed=ReadyQueue))
    cases = [{"crystal": "hopg", "key": f"{i:064x}"} for i in range(1, 8)]
    store = CaseArtifactStore(tmp_path)
    calls = []

    def publish(_run_dir, unit):
        calls.append(unit)
        store.publish("hopg", cases[unit]["key"], {"spec": np.array([unit])})

    monkeypatch.setattr(backend, "_worker_run_unit", publish)
    client = TrackingClient()
    executor = DaskExecutor(
        store, lambda case: case["key"], config=DaskConfig(max_in_flight=2), client=client
    )
    return executor, cases, client, calls


def test_submission_keeps_only_bounded_unreleased_futures(submitted_campaign):
    executor, cases, client, calls = submitted_campaign
    collected = []
    executor.execute(cases, callback=lambda i, *_: collected.append(i))
    assert client.peak_outstanding == 2
    assert calls == collected == list(range(len(cases)))
    assert all(f.released for f in client.submitted)
    assert not client.cancelled


def test_callback_cancellation_stops_submission_and_recovers_survivors(submitted_campaign):
    executor, cases, client, calls = submitted_campaign
    collected = []

    def stop(i, *_):
        collected.append(i)
        executor.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        executor.execute(cases, callback=stop)
    assert calls == [0, 1]
    assert collected == [0, 1]
    assert client.cancelled
    assert all(f.released for f in client.submitted)


def test_callback_failure_cancels_outstanding_work(submitted_campaign):
    executor, cases, client, calls = submitted_campaign

    def fail(*_):
        raise ValueError("coordinator failed")

    with pytest.raises(ValueError, match="coordinator failed"):
        executor.execute(cases, callback=fail)
    assert calls == [0, 1]
    assert client.cancelled
    assert all(f.released for f in client.submitted)


def test_retry_only_submits_missing_work_units(submitted_campaign, monkeypatch):
    import pyrite.execution.dask as backend

    executor, cases, client, calls = submitted_campaign
    executor.retry_policy = RetryPolicy(max_attempts=2)
    publish = backend._worker_run_unit
    attempts = []

    def fail_once(run_dir, unit):
        attempts.append(unit)
        if unit == 1 and attempts.count(1) == 1:
            raise RuntimeError("worker failed before publication")
        return publish(run_dir, unit)

    monkeypatch.setattr(backend, "_worker_run_unit", fail_once)
    collected = []
    executor.execute(cases, callback=lambda i, *_: collected.append(i))
    assert attempts == [*range(len(cases)), 1]
    assert sorted(collected) == list(range(len(cases)))
    assert len(calls) == len(cases)
    assert client.peak_outstanding == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_in_flight_limit_refuses(limit):
    with pytest.raises(ValueError, match="max_in_flight"):
        DaskConfig(max_in_flight=limit)
