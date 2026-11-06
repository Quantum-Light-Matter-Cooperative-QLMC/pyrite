"""Memory-target block sizing of live streamed scoring (#868)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.config import TransportKernelTuning
from pyrite.montecarlo.runner import stream_sizing, transport_options
from pyrite.montecarlo.runner.stream_sizing import PROBE_ELECTRONS, MemoryBlockSizer


def _rows(electrons, bytes_per_electron):
    return {"L_ang": np.zeros(electrons * bytes_per_electron // 8), "Ne": electrons}


def _walk(sizer, n, density):
    start = 0
    for start, stop in sizer.blocks(n):
        sizer.observe(start, stop, _rows(stop - start, density(start)))
    return sizer.spans


def test_blocks_cover_every_electron_once_and_follow_the_target():
    sizer = MemoryBlockSizer(1024, TransportKernelTuning())  # 256 MiB retained
    spans = _walk(sizer, 1 << 20, lambda _start: 1 << 10)
    assert spans[0] == (0, PROBE_ELECTRONS)
    assert spans[0][0] == 0 and spans[-1][1] == 1 << 20
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert spans[1][1] - spans[1][0] == (256 << 20) // (1 << 10)


def test_a_denser_stretch_shrinks_later_blocks():
    sizer = MemoryBlockSizer(1024, TransportKernelTuning())
    spans = _walk(sizer, 1 << 20, lambda start: 1 << 10 if start == 0 else 1 << 14)
    assert spans[2][1] - spans[2][0] == (256 << 20) // (1 << 14)


def test_block_is_at_least_one_electron_and_rowless_runs_finish():
    dense = MemoryBlockSizer(1, TransportKernelTuning())
    assert _walk(dense, 2048, lambda _start: 1 << 22)[1] == (PROBE_ELECTRONS, PROBE_ELECTRONS + 1)
    assert MemoryBlockSizer(1, TransportKernelTuning()).next_stop(0, 10) == 10
    assert _walk(MemoryBlockSizer(64, TransportKernelTuning()), 4096, lambda _: 0)[-1][1] == 4096


@pytest.mark.parametrize(
    ("explicit", "scratch", "batch"),
    [({}, 512, 1024), ({"scratch_mib": 7, "electron_batch": 9}, 7, 9)],
)
def test_block_tuning_scales_only_unset_controls(monkeypatch, explicit, scratch, batch):
    seen = {}
    monkeypatch.setattr(
        transport_options,
        "per_electron_config",
        lambda tuning: seen.update(s=tuning.scratch_mib, b=tuning.electron_batch),
    )
    with MemoryBlockSizer(1024, TransportKernelTuning(**explicit)).block_tuning(1024):
        pass
    assert seen == {"s": scratch, "b": batch}


def test_scratch_is_bounded_by_the_target(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        transport_options, "per_electron_config", lambda tuning: seen.update(s=tuning.scratch_mib)
    )
    with MemoryBlockSizer(32_768, TransportKernelTuning()).block_tuning(1 << 20):
        pass
    assert seen["s"] == 4096


@pytest.mark.parametrize(
    ("previous", "applied"),
    [(0, 1024 << 19), (1 << 40, 1024 << 19), (1 << 20, 1 << 20)],
)
def test_pool_limit_never_raises_a_cap_and_restores_it(monkeypatch, previous, applied):
    calls = []
    backend = SimpleNamespace(
        memory_limit_bytes=lambda: previous, set_memory_limit_bytes=calls.append
    )
    monkeypatch.setattr(stream_sizing, "BACKEND", backend)
    with MemoryBlockSizer(1024, TransportKernelTuning()).pool_limit():
        assert calls == [applied]
    assert calls == [applied, previous]


@pytest.mark.parametrize(
    ("target_mib", "pool_bytes"),
    [
        (1024, 1024 << 19),  # small target: half, best effort
        (16_384, (16_384 << 20) - (1280 << 20)),  # fixed reserve binds
        (65_536, int(0.95 * (65_536 << 20))),  # fraction binds
    ],
)
def test_pool_cap_leaves_room_for_memory_outside_the_pool(monkeypatch, target_mib, pool_bytes):
    calls = []
    backend = SimpleNamespace(memory_limit_bytes=lambda: 0, set_memory_limit_bytes=calls.append)
    monkeypatch.setattr(stream_sizing, "BACKEND", backend)
    with MemoryBlockSizer(target_mib, TransportKernelTuning()).pool_limit():
        assert calls == [pool_bytes]
