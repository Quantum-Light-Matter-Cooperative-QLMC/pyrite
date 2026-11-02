"""Allocator-only row recovery without replaying committed scoring work (#915)."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.emission_adapters import EmissionBatch, EmissionOutput
from pyrite.montecarlo.runner.streamed_recovery import RowScoringRecovery


class AllocatorOOM(RuntimeError):
    pass


@pytest.fixture
def batch():
    return EmissionBatch(
        segments={
            "L_ang": np.arange(1.0, 10.0),
            "elec_id": np.repeat([4, 5, 6], 3),
            "Ne": 7,
            "thickness_ang": 100.0,
            # Non-row arrays must survive even when their length equals rows.
            "initial_E_keV": np.full(9, 30.0),
        },
        energy_eV=np.array([100.0]),
        case={},
        n_hat=np.array([0.0, 0.0, -1.0]),
        electron_limit=6,
    )


def test_retry_keeps_successful_slices_and_reuses_learned_limit(batch, monkeypatch):
    calls, scratch = [], []
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom_retries", 3)

    def evaluate(part, slot):
        assert not scratch or scratch[-1]() is None
        rows = part.segments["L_ang"]
        calls.append(tuple(rows))
        assert part.segments["initial_E_keV"] is batch.segments["initial_E_keV"]
        assert part.segments["Ne"] == 7 and part.electron_limit == 6
        # Force a second failure after one slice was already scored.
        if rows.size > 4 or (rows[0] == 5 and rows.size > 2):
            temporary = np.ones(20)
            scratch.append(weakref.ref(temporary))
            raise AllocatorOOM()
        selected = rows[part.segments["elec_id"] < part.electron_limit]
        return EmissionOutput(slot, np.array([selected.sum() / part.electron_limit]))

    adapter = SimpleNamespace(evaluate=evaluate)
    recovery = RowScoringRecovery()
    np.testing.assert_allclose(recovery.evaluate(adapter, batch, "spec_characteristic"), [3.5])
    assert calls == [tuple(range(1, 10)), (1, 2, 3, 4), (5, 6, 7, 8), (5, 6), (7, 8), (9,)]
    calls.clear()
    np.testing.assert_allclose(recovery.evaluate(adapter, batch, "spec_characteristic"), [3.5])
    assert max(map(len, calls)) == 2


@pytest.mark.parametrize("error", [ValueError("bad input"), MemoryError("host OOM")])
def test_non_allocator_errors_propagate_without_retry(batch, monkeypatch, error):
    calls = []
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))

    def evaluate(*args):
        calls.append(True)
        raise error

    with pytest.raises(type(error), match=str(error)):
        RowScoringRecovery().evaluate(SimpleNamespace(evaluate=evaluate), batch, "brem_wide")
    assert calls == [True]


@pytest.mark.parametrize("budget, attempts", [(0, 1), (1, 2), (20, 4)])
def test_terminal_oom_is_bounded_by_retry_budget_or_one_row(batch, monkeypatch, budget, attempts):
    calls = []
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom_retries", budget)

    def evaluate(*args):
        calls.append(True)
        raise AllocatorOOM("no device space")

    with pytest.raises(AllocatorOOM, match="no device space") as caught:
        RowScoringRecovery().evaluate(SimpleNamespace(evaluate=evaluate), batch, "brem_wide")
    assert len(calls) == attempts
    assert "recovery exhausted" in caught.value.__notes__[0]


@pytest.mark.parametrize("layered", [False, True])
def test_escape_chunks_preserve_geometry_and_selected_row_order(monkeypatch, layered):
    from pyrite.montecarlo.runner.streamed_recovery import EscapeScoringRecovery
    from pyrite.montecarlo.spectrum import segment_escape

    segments = {
        "r_mid": np.column_stack((np.linspace(-40, 40, 9), np.zeros(9), np.linspace(10, 90, 9))),
        "v_hat": np.tile([1.0, 0.0, 0.0], (9, 1)),
        "L_ang": np.full(9, 15.0),
        "thickness_ang": 100.0,
        "crystal_width_ang": 100.0,
        "crystal_height_ang": 100.0,
    }
    index = np.array([8, 4, 5, 2, 0])
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    layers = [(0.0, 50.0, []), (50.0, 100.0, [])] if layered else None
    expected = segment_escape.segment_escape_paths(segments, index, direction, layers=layers)
    core = segment_escape._segment_escape_paths
    calls = []

    def bounded(rows, selected, *args, **kwargs):
        calls.append(int(selected.size))
        if selected.size > 2:
            raise AllocatorOOM()
        return core(rows, selected, *args, **kwargs)

    monkeypatch.setattr(segment_escape, "_segment_escape_paths", bounded)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    recovery = EscapeScoringRecovery()
    for _ in range(2):
        actual = segment_escape.segment_escape_paths(
            {**segments, "_stream_escape_recovery": recovery}, index, direction, layers=layers
        )
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, reference)
    assert calls == [5, 2, 2, 1, 2, 2, 1]
