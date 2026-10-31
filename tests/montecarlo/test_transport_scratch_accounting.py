"""Full-history batch sizing counts every allocated scratch column (#611).

``scratch_budget_bytes`` must bound the dense ``(batch, capacity)`` grid the
driver actually allocates, whatever optional columns the schema enables, and a
history too long for the budget must run alone with an explicit warning.
"""

import logging

import numpy as np
import pytest

from pyrite.montecarlo.transport import PerElectronTransportConfig, simulate_trajectories
from pyrite.montecarlo.transport.batching import (
    _ELECTRON_STATUS_BYTES,
    _SEG_SCRATCH_BYTES,
    _alloc_batch_scratch,
    _batch_size,
    _scratch_row_bytes,
)

# (midpoint, inelastic, radiative, secondaries) -> bytes per slot.
SCHEMAS = {
    "frozen": ((False, False, False, None), 82),
    "midpoint": ((True, False, False, None), 115),
    "midpoint+photons": ((True, False, True, None), 125),
    "shell": ((True, True, False, False), 125),
    "shell+secondaries": ((True, True, False, True), 149),
    "shell+photons": ((True, True, True, False), 135),
}


def _allocated_bytes(schema, m, cap):
    scratch = _alloc_batch_scratch(np, m, cap, *schema)
    return sum(int(a.nbytes) for a in scratch) + m * _ELECTRON_STATUS_BYTES


@pytest.mark.parametrize("name", SCHEMAS)
def test_row_bytes_match_allocated_columns(name):
    schema, expected = SCHEMAS[name]
    m, cap = 3, 7
    assert _scratch_row_bytes(*schema) == expected
    assert _allocated_bytes(schema, m, cap) == m * (cap * expected + _ELECTRON_STATUS_BYTES)


def test_frozen_schema_keeps_legacy_constant():
    assert _scratch_row_bytes() == _SEG_SCRATCH_BYTES


@pytest.mark.parametrize("name", SCHEMAS)
@pytest.mark.parametrize("cap", [1, 37, 512, 12348])
@pytest.mark.parametrize("budget", [1 << 16, 1 << 20, 1 << 24])
def test_batch_scratch_fits_budget_and_is_tight(name, cap, budget):
    schema, _ = SCHEMAS[name]
    row = _scratch_row_bytes(*schema)
    config = PerElectronTransportConfig(scratch_budget_bytes=budget)
    m = _batch_size(cap, config, row)
    if _allocated_bytes(schema, 1, cap) > budget:
        assert m == 1  # one history must still run; see the warning test
        return
    assert _allocated_bytes(schema, m, cap) <= budget
    assert m == config.max_batch or _allocated_bytes(schema, m + 1, cap) > budget


def test_coupled_midpoint_no_longer_overshoots_the_786_budget():
    # #611: a 4096 MiB budget at 12348 slots admitted 4241 histories whose
    # coupled midpoint scratch was 6.10 GiB, because rows were costed at 82 B.
    schema, _ = SCHEMAS["midpoint+photons"]
    budget, cap = 4096 << 20, 12348
    config = PerElectronTransportConfig(scratch_budget_bytes=budget)
    assert _batch_size(cap, config) == 4241
    m = _batch_size(cap, config, _scratch_row_bytes(*schema))
    assert m == 2782
    nbytes = sum(int(a.nbytes) for a in _alloc_batch_scratch(np, m, cap, *schema))
    assert nbytes + m * _ELECTRON_STATUS_BYTES <= budget


def test_one_history_over_budget_runs_alone_and_warns_once(caplog):
    base = simulate_trajectories(
        E0_keV=30.0, Ne=40, thickness_ang=5.0e4, element="Si",
        n_atoms_per_ang3=0.04996, seed=11, transport_core="per-electron",
    )  # fmt: skip
    tiny = PerElectronTransportConfig(scratch_budget_bytes=64, seg_capacity=4, probe_electrons=0)
    with caplog.at_level(logging.WARNING, logger="pyrite.montecarlo.transport.batching"):
        small = simulate_trajectories(
            E0_keV=30.0, Ne=40, thickness_ang=5.0e4, element="Si",
            n_atoms_per_ang3=0.04996, seed=11, transport_core="per-electron",
            per_electron_config=tiny,
        )  # fmt: skip
    warnings = [r for r in caplog.records if "above scratch_budget_bytes=64" in r.getMessage()]
    assert len(warnings) == 1
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "elec_id", "layer"):
        np.testing.assert_array_equal(small[key], base[key])


def test_midpoint_results_do_not_depend_on_budget():
    case = dict(
        E0_keV=30.0, Ne=60, thickness_ang=5.0e4, element="Si",
        n_atoms_per_ang3=0.04996, seed=5, transport_core="per-electron",
        energy_model="midpoint",
    )  # fmt: skip
    base = simulate_trajectories(**case)
    small = simulate_trajectories(
        **case, per_electron_config=PerElectronTransportConfig(scratch_budget_bytes=1 << 14)
    )
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "elec_id", "layer", "E_end_keV"):
        if key in base:
            np.testing.assert_array_equal(small[key], base[key])
