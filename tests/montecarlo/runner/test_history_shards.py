"""History shards join to the single fixed-count run bit for bit (#1387).

Each shard transports its electron range in isolation and crosses a process
boundary as a pickle; the join must equal ``run_case`` on the per-electron
core exactly, for several shard counts including a non-divisor of N. Seeds
are fixed by the sweep; every comparison is exact.
"""

import pickle

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.realization import TransportRealization, bind_realization
from pyrite.montecarlo.runner.block_transport import ShardJoin
from pyrite.montecarlo.runner.history_shards import (
    refuse_unshardable,
    run_case_from_shards,
    shard_ranges,
    transport_history_shard,
)
from pyrite.montecarlo.runner.scheduling import _UNTIMED_OUTPUT_KEYS


def _case():
    # Default shell + coupled radiative modes, spot, spread and bunch: both
    # population-wide host steps run; brem is a prefix of the transport.
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0, energy_spread_frac=1e-3, bunch_length_fs=5.0),
        tilt_deg=30.0,
        n_electrons=13,
        n_electrons_brem=7,
    )
    return build_cases(sweep)[0]


def _assert_identical(got, ref, path=""):
    assert set(got) == set(ref), path
    for key, value in ref.items():
        where = f"{path}.{key}"
        if where in {".line_grid_diagnostic.cache", ".line_grid_resolved.cache"}:
            # Cache warmth is operational telemetry, not a scientific result.
            assert got[key] in {"hit", "miss"}
            continue
        if isinstance(value, dict):
            _assert_identical(got[key], value, where)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value, err_msg=where)
        elif isinstance(value, int | float | str | bool):
            assert got[key] == value, where


@pytest.fixture(scope="module")
def case_and_reference():
    case = bind_realization(_case(), TransportRealization())
    ref = runner.run_case(case, transport_core="per-electron")
    for key in _UNTIMED_OUTPUT_KEYS:
        ref.pop(key, None)
    segs = runner._transport_case(case, transport_core="per-electron")["segs"]
    return case, ref, segs


@pytest.mark.parametrize(("shard_electrons", "block_electrons"), [(5, None), (4, 3), (13, 6)])
def test_sharded_run_equals_one_fixed_n_run(case_and_reference, shard_electrons, block_electrons):
    """K = 3, 4 and 1 shards over N = 13 (5 and 4 do not divide it)."""
    case, ref, segs = case_and_reference
    ranges = shard_ranges(case, shard_electrons)
    assert ranges[-1][1] == 13
    # Reverse order: no shard depends on another having run first.
    published = {
        span: pickle.dumps(
            transport_history_shard(case, *span, block_electrons=block_electrons),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for span in reversed(ranges)
    }
    shards = [pickle.loads(published[span]) for span in ranges]
    # Rows, bunch timing and completed hard-photon vectors, then the spectra.
    joined = runner._transport_case(
        case, transport_core="per-electron", block_electrons=ShardJoin(tuple(shards))
    )
    _assert_identical(joined["segs"], segs)
    _assert_identical(run_case_from_shards(case, shards), ref)


def test_shard_rows_are_the_matching_slice(case_and_reference):
    case, _, _ = case_and_reference
    whole = transport_history_shard(case, 0, 13)
    part = transport_history_shard(case, 5, 9)
    rows = (whole["electron_id"] >= 5) & (whole["electron_id"] < 9)
    np.testing.assert_array_equal(part["L_ang"], whole["L_ang"][rows])
    np.testing.assert_array_equal(part["electron_id"], whole["electron_id"][rows])
    assert part["Ne"] == 4
    assert part["hard_radiative_direction"] is None  # completed once, at the join


def test_join_refuses_an_incomplete_shard_set(case_and_reference):
    case, _, _ = case_and_reference
    shard = transport_history_shard(case, 0, 5)
    with pytest.raises(ValueError, match="5 electrons; the case has 13"):
        run_case_from_shards(case, [shard])


def test_unsupported_cores_and_cases_fail_early(monkeypatch):
    case = _case()
    with pytest.raises(ValueError, match="lockstep"):
        refuse_unshardable(case, "lockstep")
    with pytest.raises(ValueError, match="'auto' resolved to 'lockstep'"):
        refuse_unshardable(case, "auto")
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    with pytest.raises(ValueError, match="lockstep"):
        transport_history_shard(case, 0, 2)
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE")
    with pytest.raises(ValueError, match="adaptive"):
        refuse_unshardable({**case, "adaptive_precision": {}})
    with pytest.raises(ValueError, match="cascades"):
        refuse_unshardable({**case, "secondary_threshold_eV": 1e3})
    with pytest.raises(ValueError, match="within N"):
        transport_history_shard(case, 10, 14)


def test_marker_isolates_grid_cache_without_changing_per_electron_numbers(case_and_reference):
    case, reference, _ = case_and_reference
    ordinary = dict(case)
    ordinary.pop("transport_realization")
    plain = runner.run_case(ordinary, transport_core="per-electron")
    assert (
        plain["line_grid_diagnostic"]["cache_key"] != reference["line_grid_diagnostic"]["cache_key"]
    )
    for key in ("spec", "brem", "E_grid", "E_grid_brem", "brem_wide"):
        np.testing.assert_array_equal(plain[key], reference[key], err_msg=key)


def test_marked_case_honors_auto_env_and_pool_default(case_and_reference, monkeypatch):
    from pyrite.montecarlo.runner.scheduling import _pool_worker

    case, reference, _ = case_and_reference
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "auto")
    np.testing.assert_array_equal(runner.run_case(case)["spec"], reference["spec"])
    # A pool's injected lockstep default is not an inherited user pin.
    monkeypatch.setattr(runner, "_WORKER_INHERITED_CORE", "")
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    worker = _pool_worker(case, transport=False)
    np.testing.assert_array_equal(worker(case)["spec"], reference["spec"])
    assert runner.env_value("PYRITE_MC_TRANSPORT_CORE") == "lockstep"
