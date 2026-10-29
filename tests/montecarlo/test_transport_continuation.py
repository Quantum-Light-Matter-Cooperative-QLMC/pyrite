"""Bounded transport continuation scheduling (#789), without a device.

A NumPy stand-in follows the CUDA kernel's continuation contract: thread
``i`` owns history ``slots[i]``, emits at most ``cap`` rows from its saved row
total, and suspends with ``EXIT_SUSPENDED`` while rows remain. Each row is a
pure function of (history, ordinal), so any lost, duplicated or misordered
row is visible. The device kernel itself is exercised by
``test_transport_continuation_cuda.py``.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.batching import (
    PerElectronTransportConfig,
    _alloc_hard_scratch,
    _alloc_scratch,
)
from pyrite.montecarlo.transport.continuation import (
    EXIT_SUSPENDED,
    TransportContinuationConfig,
    _drive_continuation_rounds,
    active_population,
    continuation_row_bytes,
)
from pyrite.montecarlo.transport.cores import (
    EXIT_BACKSCATTERED,
    EXIT_NOT_ENTERED,
    EXIT_TRANSMITTED,
)

# Mixed track lengths, including zero-row (not entered) and one-row histories.
LENGTHS = np.array([0, 1, 40, 3, 17, 0, 2, 64, 5, 1, 33, 9], dtype=np.int64)
NE = LENGTHS.size
WRITTEN = (0, 2, 5, 7, 12)


def _value(history, ordinal):
    return 1000.0 * history + ordinal


def _fake_core(lengths, launches):
    """NumPy core with the continuation kernel's slot/state/row contract."""

    def core(run, *args, continuation):
        _start, count, cap, _keys = run
        scratch, (seg_count, exit_code) = args[0], args[1]
        slots, state_i32, _state_u64, _state_f64 = continuation
        launches.append(np.asarray(slots[:count]).copy())
        for lane in range(count):
            history = int(slots[lane])
            done = int(state_i32[history, 0])  # the stand-in's step is its row count
            emit = min(cap, int(lengths[history]) - done)
            for j in range(emit):
                slot = lane * cap + j
                scratch[0][slot] = _value(history, done + j)
                scratch[2][slot] = done + j
                scratch[5][slot] = history
                scratch[7][slot] = -_value(history, done + j)
                scratch[12][slot] = 0.5 * (done + j)
            seg_count[lane] = emit
            if lengths[history] == 0:
                exit_code[lane] = EXIT_NOT_ENTERED
            elif done + emit < lengths[history]:
                exit_code[lane] = EXIT_SUSPENDED
                state_i32[history, 0] = done + emit
            else:
                exit_code[lane] = EXIT_TRANSMITTED if history % 2 else EXIT_BACKSCATTERED

    return core


def _alloc(rows):
    return _alloc_scratch(np, rows, 1, midpoint=True) + _alloc_hard_scratch(np, rows, 1)


def _shell_alloc(secondaries):
    """Shell soft/hard schema (#797): hard pair, secondary rows, photon pair."""

    def alloc(rows):
        return (
            _alloc_scratch(np, rows, 1, midpoint=True)
            + _alloc_hard_scratch(np, rows, 1, secondaries)
            + _alloc_hard_scratch(np, rows, 1)
        )

    return alloc


def _expected():
    history = np.repeat(np.arange(NE), LENGTHS)
    ordinal = np.concatenate([np.arange(n) for n in LENGTHS])
    return history, ordinal


def _drive(
    cont, *, budget=1 << 30, keep_on_device=False, order=None, lengths=LENGTHS, alloc=_alloc
):
    launches = []
    total = int(lengths.sum())
    out = tuple(
        np.full((total,) + a.shape[1:], -7, dtype=a.dtype) if a.shape[0] else a for a in alloc(1)
    )
    config = PerElectronTransportConfig(scratch_budget_bytes=budget, continuation=cont)
    progress = []
    counts = _drive_continuation_rounds(
        _fake_core(lengths, launches),
        (),
        np,
        lengths.size,
        total,
        np.zeros(lengths.size, dtype=np.uint64),
        np.zeros(lengths.size),
        (False, None),
        (),
        alloc,
        out,
        lambda a: a,
        config,
        keep_on_device,
        on_batch=lambda done, n: progress.append((done, n)),
        history_order=order,
    )
    rows = counts[6] if keep_on_device else out
    return counts, rows, launches, progress


@pytest.mark.parametrize("quantum", [1, 4, 64, 1000])
@pytest.mark.parametrize("active", [1, 3, NE])
@pytest.mark.parametrize("order", ["forward", "reversed", "shuffled"])
def test_rows_reassemble_in_canonical_order(quantum, active, order):
    permutation = {
        "forward": None,
        "reversed": np.arange(NE)[::-1],
        "shuffled": np.random.default_rng(3).permutation(NE),
    }[order]
    cont = TransportContinuationConfig(rows_per_round=quantum, active_histories=active)
    counts, rows, launches, progress = _drive(cont, order=permutation)

    history, ordinal = _expected()
    assert counts[0] == LENGTHS.sum()
    np.testing.assert_array_equal(rows[5], history)
    np.testing.assert_array_equal(rows[2], ordinal)
    np.testing.assert_array_equal(rows[0][:, 0], _value(history, ordinal))
    np.testing.assert_array_equal(rows[7], -_value(history, ordinal))
    np.testing.assert_array_equal(rows[12], 0.5 * ordinal)
    # Not-entered histories are finished, never tallied as an exit.
    entered = np.flatnonzero(LENGTHS > 0)
    assert counts[1] == np.count_nonzero(entered % 2 == 0)
    assert counts[2] == np.count_nonzero(entered % 2 == 1)
    assert counts[3:6] == (0, 0, 0)
    assert max(len(slots) for slots in launches) <= active
    launched = np.concatenate(launches)
    assert set(launched.tolist()) == set(range(NE))
    assert progress[-1] == (NE, NE)
    assert [done for done, _ in progress] == sorted({done for done, _ in progress})


def test_finished_lanes_are_refilled_and_long_histories_resume():
    records = []
    cont = TransportContinuationConfig(
        rows_per_round=4, active_histories=4, on_round=records.append
    )
    _counts, _rows, launches, _progress = _drive(cont)
    assembly = records.pop()
    assert assembly["phase"] == "assembly" and assembly["rows"] == LENGTHS.sum()
    # Short histories finish first; pending ones join while long ones resume.
    assert launches[0].tolist() == [0, 1, 2, 3]
    assert launches[1].tolist() == [2, 4, 5, 6]
    # Every lane stays occupied while primaries are pending.
    for record, following in zip(records, records[1:], strict=False):
        if record["pending"]:
            assert following["launched"] == 4
    assert records[-1]["pending"] == 0
    resumed = [slots for slots in launches if 7 in slots]
    assert len(resumed) == int(np.ceil(LENGTHS[7] / 4))


def test_device_resident_join_matches_host_rows():
    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    _counts, host_rows, _, _ = _drive(cont)
    _counts, device_rows, _, _ = _drive(cont, keep_on_device=True)
    assert len(host_rows) == len(device_rows)
    for column in WRITTEN:  # the stand-in core leaves other scratch unset
        np.testing.assert_array_equal(host_rows[column], device_rows[column])


def test_device_staging_matches_host_rows():
    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    _counts, host_rows, _, _ = _drive(cont)
    counts, staged_rows, _, _ = _drive(replace(cont, device_staging=True))
    assert counts[6] is None
    for column in WRITTEN:
        np.testing.assert_array_equal(host_rows[column], staged_rows[column])


def test_scratch_respects_budget_with_staging_and_event_columns():
    row_bytes = continuation_row_bytes(_alloc(1))
    # Midpoint schema plus the photon pair: 7 base + 5 end fields + 2 photon.
    assert row_bytes == 82 + 33 + 10
    cont = TransportContinuationConfig(rows_per_round=8)
    budget = 3 * (8 * (2 * row_bytes + 24) + 13) + 5
    records = []
    _drive(replace(cont, on_round=records.append), budget=budget)
    records = [r for r in records if r["phase"] == "round"]
    assert {r["launched"] for r in records} <= {1, 2, 3}
    assert max(r["launched"] for r in records) == 3
    assert all(r["scratch_bytes"] <= budget for r in records)
    assert all(r["scratch_bytes"] == 3 * 8 * row_bytes for r in records)


@pytest.mark.parametrize("secondaries", [False, True])
@pytest.mark.parametrize("keep_on_device", [False, True])
def test_shell_schema_rows_and_budget(secondaries, keep_on_device):
    alloc = _shell_alloc(secondaries)
    row_bytes = continuation_row_bytes(alloc(1))
    # Midpoint 115 + hard W/channel 10 + photon pair 10, plus (3,) float64 directions.
    assert row_bytes == 135 + (24 if secondaries else 0)
    budget = 2 * (4 * (2 * row_bytes + 24) + 13)
    records = []
    cont = TransportContinuationConfig(rows_per_round=4, on_round=records.append)
    counts, rows, _, _ = _drive(cont, budget=budget, keep_on_device=keep_on_device, alloc=alloc)
    history, ordinal = _expected()
    np.testing.assert_array_equal(rows[5], history)
    np.testing.assert_array_equal(rows[2], ordinal)
    secondary = rows[14]
    assert secondary.shape == ((LENGTHS.sum(), 3) if secondaries else (0, 3))
    records = [r for r in records if r["phase"] == "round"]
    assert max(r["launched"] for r in records) == 2
    assert all(r["scratch_bytes"] <= budget for r in records)
    assert all(r["scratch_bytes"] == 2 * 4 * row_bytes for r in records)


@pytest.mark.parametrize("secondaries", [False, True])
def test_shell_schema_zero_row_launch(secondaries):
    lengths = np.zeros(4, dtype=np.int64)
    cont = TransportContinuationConfig(rows_per_round=3)
    counts, rows, _, _ = _drive(cont, lengths=lengths, alloc=_shell_alloc(secondaries))
    assert counts[0] == 0
    assert all(column.shape[0] == 0 for column in rows)


def test_one_lane_above_budget_is_rejected():
    cont = TransportContinuationConfig(rows_per_round=64)
    with pytest.raises(ValueError, match="lower rows_per_round"):
        active_population(10, 125, PerElectronTransportConfig(scratch_budget_bytes=100), cont)


def test_population_honours_max_batch_and_explicit_cap():
    config = PerElectronTransportConfig(max_batch=7, continuation=None)
    cont = TransportContinuationConfig(rows_per_round=2)
    assert active_population(100, 125, config, cont)[0] == 7
    assert active_population(100, 125, config, replace(cont, active_histories=3))[0] == 3
    assert active_population(2, 125, config, cont)[0] == 2


@pytest.mark.parametrize(
    "kwargs", [{"rows_per_round": 0}, {"active_histories": 0}, {"rows_per_round": -1}]
)
def test_invalid_schedule_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TransportContinuationConfig(**kwargs)


def test_history_order_must_be_a_permutation():
    cont = TransportContinuationConfig(rows_per_round=2)
    with pytest.raises(ValueError, match="permutation"):
        _drive(cont, order=np.zeros(NE, dtype=np.int32))


def test_segment_overflow_raises():
    cont = TransportContinuationConfig(rows_per_round=2)
    with pytest.raises(RuntimeError, match="segment buffer exhausted"):
        launches = []
        out = tuple(np.empty((1,) + a.shape[1:], a.dtype) if a.shape[0] else a for a in _alloc(1))
        _drive_continuation_rounds(
            _fake_core(LENGTHS, launches),
            (),
            np,
            NE,
            3,
            np.zeros(NE, dtype=np.uint64),
            np.zeros(NE),
            (False, None),
            (),
            _alloc,
            out,
            lambda a: a,
            PerElectronTransportConfig(continuation=cont),
            False,
        )


@pytest.mark.parametrize("core", ["per-electron", "lockstep"])
def test_host_cores_reject_continuation(core):
    config = PerElectronTransportConfig(continuation=TransportContinuationConfig())
    with pytest.raises(ValueError, match="requires transport_core='cuda'"):
        simulate_trajectories(
            E0_keV=60.0,
            Ne=4,
            thickness_ang=1_000.0,
            element="C",
            n_atoms_per_ang3=0.1,
            seed=1,
            transport_core=core,
            per_electron_config=config,
        )
