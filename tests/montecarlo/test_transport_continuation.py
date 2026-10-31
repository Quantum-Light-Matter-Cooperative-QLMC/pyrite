"""Bounded transport continuation scheduling (#789), without a device.

A NumPy stand-in follows the CUDA kernel's continuation contract: thread
``i`` owns history ``slots[i]``, emits at most ``cap`` rows from its saved row
total, and suspends with ``EXIT_SUSPENDED`` while rows remain. Each row is a
pure function of (history, ordinal), so any lost, duplicated or misordered
row is visible. The device kernel itself is exercised by
``test_transport_continuation_cuda.py``.
"""

import os
import subprocess
import sys
import textwrap
from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.groove import GrooveSpec
from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.batching import (
    PerElectronTransportConfig,
    _alloc_hard_scratch,
    _alloc_scratch,
)
from pyrite.montecarlo.transport.continuation import (
    CONTINUATION_AUTO_MIN_HISTORIES,
    EXIT_SUSPENDED,
    TransportContinuationConfig,
    _drive_continuation_rounds,
    active_population,
    continuation_row_bytes,
    effective_schedule,
    resolve_schedule,
    schedule_from_env,
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
    assert not cont.device_staging  # host reassembly is the default (#825)
    _counts, host_rows, _, _ = _drive(cont)
    counts, staged_rows, _, _ = _drive(replace(cont, device_staging=True))
    assert counts[6] is None
    for column in WRITTEN:
        np.testing.assert_array_equal(host_rows[column], staged_rows[column])


@pytest.mark.parametrize(("budget", "spills"), [(0, True), (4000, True), (None, False)])
def test_staging_spill_matches_host_rows(budget, spills):
    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    _counts, host_rows, _, _ = _drive(replace(cont, device_staging=False))
    records = []
    staged = replace(
        cont, device_staging=True, staging_budget_bytes=budget, on_round=records.append
    )
    counts, rows, _, _ = _drive(staged)
    assert counts[6] is None
    for column in WRITTEN:
        np.testing.assert_array_equal(host_rows[column], rows[column])
    assembly = records[-1]
    assert assembly["phase"] == "assembly"
    assert assembly["staged"] is not spills
    if budget == 4000:
        assert assembly["spilled_round"] > 0  # spills mid-run, not at once


def test_spill_releases_the_device_pool_once():
    """Spilled rounds' cached pool blocks go back to the driver (#825)."""
    from types import SimpleNamespace

    from pyrite.montecarlo.transport.continuation import _Retained

    released = []
    pool = SimpleNamespace(free_all_blocks=lambda: released.append(True))
    xp = SimpleNamespace(get_default_memory_pool=lambda: pool)
    cont = TransportContinuationConfig(device_staging=True, staging_budget_bytes=1000)
    retained = _Retained(xp, cont, keep_on_device=False, to_host=np.array)
    retained.append((np.zeros(50),))  # 400 B staged; 2x fits
    assert retained.on_device and not released
    retained.append((np.zeros(50),))  # 800 B staged; 2x exceeds: spill
    retained.append((np.zeros(50),))  # already host-side
    assert not retained.on_device and retained.spilled_round == 1
    assert released == [True]


class _DeviceLike:
    """Minimal device array: ``get(out=...)`` copies into a host array."""

    def __init__(self, values):
        self.values = values
        self.shape, self.dtype, self.nbytes = values.shape, values.dtype, values.nbytes

    def get(self, out):
        out[...] = self.values
        return out


def test_round_downloads_pack_into_column_blocks():
    """Device rounds download into per-column arena blocks, values unchanged (#825)."""
    from pyrite.montecarlo.transport.continuation import _Retained

    cont = TransportContinuationConfig(device_staging=False)
    retained = _Retained(np, cont, keep_on_device=False, to_host=np.asarray)
    rounds = [
        (
            np.arange(12.0).reshape(4, 3) + 100 * r,
            np.arange(4, dtype=np.int32) + r,
            np.empty((0, 3)),
        )
        for r in range(3)
    ]
    for chunk in rounds:
        retained.append(tuple(_DeviceLike(a) for a in chunk))
    for chunk, got in zip(rounds, retained.chunks, strict=True):
        for want, have in zip(chunk, got, strict=True):
            np.testing.assert_array_equal(have, want)
            assert have.dtype == want.dtype and have.flags.c_contiguous
    # One block per column: rounds of a column share it, columns never do.
    first, second = retained.chunks[0], retained.chunks[1]
    assert first[0].base is second[0].base  # NumPy collapses views to the block
    assert first[0].base is not first[1].base
    retained.close()
    assert retained.arena is None


def test_device_resident_rows_never_spill():
    cont = TransportContinuationConfig(rows_per_round=5, active_histories=3)
    records = []
    _drive(replace(cont, staging_budget_bytes=0, on_round=records.append), keep_on_device=True)
    assert records[-1]["spilled_round"] is None


def test_negative_staging_budget_is_rejected():
    with pytest.raises(ValueError, match="staging_budget_bytes"):
        TransportContinuationConfig(staging_budget_bytes=-1)


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
@pytest.mark.parametrize(
    "config",
    [
        PerElectronTransportConfig(continuation=TransportContinuationConfig()),
        PerElectronTransportConfig(schedule="continuation"),
    ],
)
def test_host_cores_reject_continuation(core, config):
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


# Every ``auto`` support flag off: a supported CUDA pass.
SUPPORTED = dict(cuda=True, lut=False, launched=False, gdf=False)
AUTO = PerElectronTransportConfig(schedule="auto")


def test_schedule_defaults_to_continuation_field():
    assert effective_schedule(PerElectronTransportConfig()) == "full-history"
    cont = PerElectronTransportConfig(continuation=TransportContinuationConfig())
    assert effective_schedule(cont) == "continuation"
    forced = replace(cont, schedule="full-history")
    assert resolve_schedule(forced, 10**6, **SUPPORTED).continuation is None


def test_explicit_continuation_keeps_or_creates_its_config():
    created = resolve_schedule(PerElectronTransportConfig(schedule="continuation"), 1, **SUPPORTED)
    assert created.continuation == TransportContinuationConfig()
    cont = TransportContinuationConfig(rows_per_round=9)
    kept = PerElectronTransportConfig(schedule="continuation", continuation=cont)
    # Explicit requests are not second-guessed: unsupported modes raise later.
    assert resolve_schedule(kept, 1, **(SUPPORTED | {"lut": True})).continuation is cont


def test_auto_continues_a_supported_large_pass(caplog):
    caplog.set_level("INFO", logger="pyrite.montecarlo.transport.continuation")
    resolved = resolve_schedule(AUTO, CONTINUATION_AUTO_MIN_HISTORIES, **SUPPORTED)
    assert resolved.schedule == "continuation"
    assert resolved.continuation == TransportContinuationConfig()
    assert "continuation for" in caplog.text


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("cuda", "not 'cuda'"),
        ("lut", "transport LUT"),
        ("launched", "launched passes"),
        ("gdf", "GDF"),
    ],
)
def test_auto_falls_back_with_a_stated_reason(flag, reason, caplog):
    caplog.set_level("INFO", logger="pyrite.montecarlo.transport.continuation")
    support = SUPPORTED | {flag: not SUPPORTED[flag]}
    resolved = resolve_schedule(AUTO, 10 * CONTINUATION_AUTO_MIN_HISTORIES, **support)
    assert (resolved.schedule, resolved.continuation) == ("full-history", None)
    assert reason in caplog.text


def test_auto_keeps_full_history_below_the_history_threshold(caplog):
    caplog.set_level("INFO", logger="pyrite.montecarlo.transport.continuation")
    resolved = resolve_schedule(AUTO, CONTINUATION_AUTO_MIN_HISTORIES - 1, **SUPPORTED)
    assert resolved.continuation is None
    assert "below" in caplog.text


def test_unknown_schedule_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="transport schedule"):
        PerElectronTransportConfig(schedule="streaming")
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_SCHEDULE", "fast")
    with pytest.raises(ValueError, match="PYRITE_MC_TRANSPORT_SCHEDULE"):
        schedule_from_env()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "auto"), ("", "auto"), (" auto ", "auto"), ("full-history", "full-history")],
)
def test_schedule_env_default(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PYRITE_MC_TRANSPORT_SCHEDULE", raising=False)
    else:
        monkeypatch.setenv("PYRITE_MC_TRANSPORT_SCHEDULE", raw)
    assert schedule_from_env() == expected


def test_auto_on_a_host_core_is_the_full_history_run():
    kw = dict(
        E0_keV=60.0,
        Ne=4,
        thickness_ang=1_000.0,
        element="C",
        n_atoms_per_ang3=0.1,
        seed=1,
        transport_core="per-electron",
    )
    reference = simulate_trajectories(**kw)
    auto = simulate_trajectories(**kw, per_electron_config=AUTO)
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(auto[key], value, err_msg=key)


@pytest.mark.slow
def test_auto_groove_logs_fallback_and_preserves_output(caplog):
    kw = dict(
        E0_keV=60.0,
        Ne=4,
        thickness_ang=1_000.0,
        element="C",
        n_atoms_per_ang3=0.1,
        seed=1,
        transport_core="auto",
        groove=GrooveSpec(spacing_ang=1.0e4, depth_ang=1.0e3, tilt_polar_rad=0.2),
    )
    reference = simulate_trajectories(**kw)
    caplog.set_level("INFO", logger="pyrite.montecarlo.transport.continuation")
    auto = simulate_trajectories(**kw, per_electron_config=AUTO)
    assert "full-history because the transport core is not 'cuda'" in caplog.text
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(auto[key], value, err_msg=key)


@pytest.mark.slow
def test_schedule_environment_does_not_change_case_identity():
    script = textwrap.dedent("""
        from pyrite.campaign.config import material_sweep
        from pyrite.campaign.profiles import case_content_key
        from pyrite.campaign.sweep import build_cases
        from pyrite.montecarlo.transport.batching import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
        case = build_cases(material_sweep("silicon"), 4, 4)[0]
        print(DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG.schedule, case_content_key(case))
    """)
    keys = []
    for schedule in (None, "full-history", "continuation", "auto"):
        env = os.environ.copy()
        env.pop("PYRITE_MC_TRANSPORT_SCHEDULE", None)
        if schedule is not None:
            env["PYRITE_MC_TRANSPORT_SCHEDULE"] = schedule
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        selected, key = result.stdout.strip().split()
        assert selected == (schedule or "auto")
        keys.append(key)
    assert len(set(keys)) == 1
