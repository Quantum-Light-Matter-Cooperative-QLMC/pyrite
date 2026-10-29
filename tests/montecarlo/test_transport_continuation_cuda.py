"""Bounded transport continuation on the exact CUDA kernel (#789).

Continuation changes only the schedule: each history is resumed from the
kernel's own loop-carried state, so every returned array must equal the
full-history run of the same kernel *exactly*, across row quanta, active
populations and refill orders. No tolerance is relaxed. Scheduling without a
device is covered by ``test_transport_continuation.py``. These tests need a
CUDA device and run through ``pyrite remote``.

Validation: gpu-transport-core
"""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    batching,
    check_segment_event_contract,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
from pyrite.montecarlo.transport.continuation import (
    TransportContinuationConfig,
    _drive_continuation_rounds,
)
from pyrite.montecarlo.transport.events import EVENT_HARD_RADIATIVE, TransportStepLimitError
from tests.helpers.bremslib import synthetic_bremslib_arrays

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

NO_LUT = TransportLUTConfig(enabled=False)
REFERENCE = PerElectronTransportConfig(seg_capacity=16, probe_electrons=0)


def _continued(rows_per_round, active_histories=None, **kwargs):
    cont = TransportContinuationConfig(
        rows_per_round=rows_per_round, active_histories=active_histories
    )
    return replace(REFERENCE, continuation=cont, **kwargs)


def _table(scale):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * scale,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * scale,
    )


def _radiative(config, *, Ne=300, seed=42, **kw):
    return simulate_trajectories(
        E0_keV=60.0,
        Ne=Ne,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=seed,
        energy_model="midpoint",
        max_dE_frac=0.05,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": _table(1e5)},
        transport_core="cuda",
        transport_lut_config=NO_LUT,
        per_electron_config=config,
        **kw,
    )


def _layered(config, *, energy_model="midpoint", **kw):
    """Uncoupled two-layer prism: layer boundaries, side, top and bottom exits."""
    return simulate_trajectories(
        E0_keV=40.0,
        Ne=400,
        thickness_ang=5.0e4,
        layers=[(0.0, 1.0e4, [("C", 0.176)]), (1.0e4, 5.0e4, [("W", 0.0632)])],
        elastic_model="sr",
        crystal_width_mm=2.0e-4,
        crystal_height_mm=2.0e-4,
        E_cut_keV=5.0,
        seed=7,
        energy_model=energy_model,
        radiative_model="uncoupled",
        transport_core="cuda",
        transport_lut_config=NO_LUT,
        per_electron_config=config,
        **kw,
    )


def _host(value):
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else value


def _assert_identical(reference, candidate):
    assert reference.keys() == candidate.keys()
    for key, expected in reference.items():
        expected, actual = _host(expected), _host(candidate[key])
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype, key
            assert actual.shape == expected.shape, key
            # Bitwise, NaN-aware: the same kernel arithmetic must be replayed.
            assert np.array_equal(
                expected.view(np.uint8), np.ascontiguousarray(actual).view(np.uint8)
            ), key
        elif isinstance(expected, dict):
            _assert_identical(expected, actual)
        else:
            assert actual == expected, key


@pytest.fixture(scope="module")
def radiative_reference():
    return _radiative(REFERENCE)


@pytest.mark.parametrize(
    ("rows_per_round", "active"), [(1, None), (7, None), (7, 37), (64, 5), (4096, None)]
)
def test_coupled_radiative_rows_match_full_history(radiative_reference, rows_per_round, active):
    result = _radiative(_continued(rows_per_round, active))
    _assert_identical(radiative_reference, result)
    assert np.count_nonzero(result["event_kind"] == EVENT_HARD_RADIATIVE) > 0
    check_segment_event_contract(result)


def test_refill_order_does_not_change_rows(radiative_reference, monkeypatch):
    order = np.random.default_rng(11).permutation(300).astype(np.int32)
    monkeypatch.setattr(
        batching,
        "_drive_continuation_rounds",
        partial(_drive_continuation_rounds, history_order=order),
    )
    _assert_identical(radiative_reference, _radiative(_continued(5, 23)))


def test_budget_bounds_population_and_matches(radiative_reference):
    records = []
    cont = TransportContinuationConfig(rows_per_round=8, on_round=records.append)
    budget = 50 * (8 * (2 * 125 + 24) + 13)
    config = replace(REFERENCE, scratch_budget_bytes=budget, continuation=cont)
    _assert_identical(radiative_reference, _radiative(config))
    assert max(r["launched"] for r in records) == 50
    assert all(r["scratch_bytes"] <= budget for r in records)
    assert sum(r["rows"] for r in records) == radiative_reference["electron_id"].size


@pytest.mark.parametrize("energy_model", ["midpoint", "frozen"])
def test_layered_prism_exits_match_full_history(energy_model):
    reference = _layered(REFERENCE, energy_model=energy_model)
    assert reference["n_side_exited"] > 0
    assert reference["n_transmitted"] + reference["n_backscattered"] > 0
    _assert_identical(reference, _layered(_continued(3, 41), energy_model=energy_model))


def test_device_resident_rows_match_full_history():
    reference = _layered(REFERENCE, keep_segments_on_device=True)
    candidate = _layered(_continued(9, 64), keep_segments_on_device=True)
    _assert_identical(reference, candidate)


def test_step_limit_is_reported_identically():
    with pytest.raises(TransportStepLimitError) as expected:
        _layered(REFERENCE, max_steps=6)
    with pytest.raises(TransportStepLimitError) as actual:
        _layered(_continued(4, 50), max_steps=6)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"transport_lut_config": TransportLUTConfig()}, "transport LUT"),
        ({"straggling": True}, "straggling"),
    ],
)
def test_unsupported_modes_are_rejected(kwargs, message):
    base = {
        "E0_keV": 40.0,
        "Ne": 16,
        "thickness_ang": 1.0e4,
        "element": "C",
        "n_atoms_per_ang3": 0.176,
        "elastic_model": "sr",
        "radiative_model": "uncoupled",
        "transport_core": "cuda",
        "transport_lut_config": NO_LUT,
        "per_electron_config": _continued(8),
    }
    with pytest.raises(NotImplementedError, match=message):
        simulate_trajectories(**(base | kwargs))
