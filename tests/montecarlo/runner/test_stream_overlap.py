"""Overlapped live block streaming (#862) on a fake device backend."""

import threading

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import block_transport, stream_overlap


class FakeOOM(RuntimeError):
    pass


class _Stream:
    def __init__(self, non_blocking=False):
        self.waited = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait_event(self, event):
        assert event.recorded
        self.waited += 1

    def synchronize(self):
        pass


class _Event:
    def __init__(self, block=False, disable_timing=False):
        self.recorded = False

    def record(self):
        self.recorded = True


class _Cuda:
    Stream = _Stream
    Event = _Event


class _Cp:
    cuda = _Cuda


class FakeBackend:
    name = "cuda"
    xp = _Cp

    def __init__(self):
        self.releases = 0

    def release_memory(self):
        self.releases += 1


@pytest.fixture
def fake_cuda(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(stream_overlap, "BACKEND", backend)
    monkeypatch.setattr(block_transport, "BACKEND", backend)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda error: isinstance(error, FakeOOM))
    monkeypatch.setenv("PYRITE_STREAM_OVERLAP", "1")
    return backend


def _simulate(start, stop):
    n = stop - start
    return {"Ne": n, "n_missed": 1, "electron_id": np.arange(n)}


def _run(consume, simulate=_simulate, replay=None):
    return block_transport.transport_electron_blocks(
        simulate, 10, 3, seed=1, bunch={}, consume=consume, replay_block=replay
    )


def test_overlap_scores_blocks_in_order_off_the_main_thread(fake_cuda):
    seen, threads = [], set()

    def consume(block):
        seen.append(int(block["electron_id"][0]))
        threads.add(threading.get_ident())

    counts = _run(consume)
    assert seen == [0, 3, 6, 9]
    assert threading.get_ident() not in threads
    assert counts["Ne"] == 10 and counts["n_missed"] == 4


def test_overlap_matches_serial_counts(fake_cuda, monkeypatch):
    overlapped = _run(lambda block: None)
    monkeypatch.setenv("PYRITE_STREAM_OVERLAP", "0")
    assert _run(lambda block: None) == overlapped


def test_scoring_oom_resumes_serially_from_the_failed_block(fake_cuda):
    seen, failed = [], []

    def consume(block):
        first = int(block["electron_id"][0])
        if first == 3 and not failed:
            failed.append(threading.get_ident())
            raise FakeOOM
        seen.append((first, threading.get_ident()))

    counts = _run(consume, replay=_simulate)
    assert [first for first, _ in seen] == [0, 3, 6, 9]
    # Block 3 and every later block rerun on the serial (main-thread) loop.
    assert all(ident == threading.get_ident() for first, ident in seen if first >= 3)
    assert counts["Ne"] == 10 and counts["n_missed"] == 4


def test_transport_oom_resumes_serially_after_committing_the_previous_block(fake_cuda):
    seen, calls = [], []

    def simulate(start, stop):
        calls.append(start)
        if start == 6 and calls.count(6) == 1:
            raise FakeOOM
        return _simulate(start, stop)

    counts = _run(lambda block: seen.append(int(block["electron_id"][0])), simulate)
    assert seen == [0, 3, 6, 9]
    assert counts["Ne"] == 10


def test_non_oom_scoring_error_propagates(fake_cuda):
    def consume(block):
        raise ValueError("bad block")

    with pytest.raises(ValueError, match="bad block"):
        _run(consume)


def test_overlap_needs_cuda_fixed_blocks_and_no_stop(fake_cuda, monkeypatch):
    assert stream_overlap.overlap_requested(None, None)
    assert not stream_overlap.overlap_requested(object(), None)
    assert not stream_overlap.overlap_requested(None, lambda stop: False)
    monkeypatch.setenv("PYRITE_STREAM_OVERLAP", "0")
    assert not stream_overlap.overlap_requested(None, None, 1 << 22, 1 << 20)
    monkeypatch.setattr(stream_overlap, "BACKEND", type("CpuBackend", (), {"name": "cpu"})())
    monkeypatch.setenv("PYRITE_STREAM_OVERLAP", "1")
    assert not stream_overlap.overlap_requested(None, None)


def test_overlap_defaults_on_for_two_large_blocks(fake_cuda, monkeypatch):
    """Unforced, overlap engages with >= 2 full blocks of >= 2**18 electrons (#862)."""
    monkeypatch.delenv("PYRITE_STREAM_OVERLAP")
    block = stream_overlap.AUTO_MIN_BLOCK
    assert stream_overlap.overlap_requested(None, None, 2 * block, block)
    assert stream_overlap.overlap_requested(None, None, 16 * block, 4 * block)
    assert not stream_overlap.overlap_requested(None, None, 2 * block - 1, block)
    assert not stream_overlap.overlap_requested(None, None, 64 * block, block // 2)
    assert not stream_overlap.overlap_requested(object(), None, 64 * block, block)
    assert not stream_overlap.overlap_requested(None, lambda stop: False, 64 * block, block)


def test_small_blocks_stream_serially_by_default(fake_cuda, monkeypatch):
    monkeypatch.delenv("PYRITE_STREAM_OVERLAP")
    threads = []

    def consume(block):
        threads.append(threading.current_thread())

    _run(consume)
    assert threads and all(t is threading.main_thread() for t in threads)
