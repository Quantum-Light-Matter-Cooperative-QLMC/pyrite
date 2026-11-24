"""Shard-summed spectra (#1395): each history shard scores its own electrons.

The shards of one case score against one frozen plan (line axis, reflection
selection) and their spectra add in shard order. The result must

- equal live streamed scoring with ``block_electrons = shard_electrons`` bit
  for bit (same blocks, same order of FP64 additions);
- be bitwise repeatable for a fixed shard size, whatever order the shards
  were scored in and after a pickle round trip;
- match the #1387 join (``run_case`` on the per-electron core) to FP64
  summation order, for shard counts that do and do not divide N.

Seeds are fixed by the sweep.
"""

import pickle

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.realization import (
    TransportRealization,
    bind_realization,
    normalize_realization,
)
from pyrite.montecarlo.runner import reflection_cutoff
from pyrite.montecarlo.runner.history_shards import (
    run_case_from_shards,
    shard_ranges,
    transport_history_shard,
)
from pyrite.montecarlo.runner.shard_scoring import (
    SHARD_SCORING_KEY,
    merge_shard_spectra,
    resolve_shard_plan,
    run_case_shard_scored,
    score_history_shard,
    shard_scoring_refusal,
)
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from tests.montecarlo.transport.test_streamed_line_grid import _measured_policy

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
SCALARS = ("eta", "hit_frac", "n_segments")


def _case():
    # Spot, spread and bunch: the join redoes both population-wide host steps,
    # which shard scoring skips. Brem is a prefix of the transport.
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0, energy_spread_frac=1e-3, bunch_length_fs=5.0),
        tilt_deg=30.0,
        n_electrons=13,
        n_electrons_brem=7,
    )
    return build_cases(sweep)[0]


def _shards(case, size):
    return [transport_history_shard(case, start, stop) for start, stop in shard_ranges(case, size)]


def _assert_bitwise(got, ref):
    np.testing.assert_array_equal(got["E_grid"], ref["E_grid"])
    for key in SPECTRA:
        np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
    for key in SCALARS:
        assert got[key] == ref[key], key


def _assert_within_reordering(got, ref):
    """The stated contract against the joined reduction (see the compute note)."""
    np.testing.assert_array_equal(got["E_grid"], ref["E_grid"])
    for key in SPECTRA:
        scale = np.abs(ref[key]).max()
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-11, atol=1e-14 * scale, err_msg=key)
    for key in SCALARS:
        assert got[key] == ref[key], key


@pytest.fixture(scope="module")
def plain():
    case = _case()
    assert shard_scoring_refusal(case) is None
    return case, run_case_from_shards(case, _shards(case, 13))


@pytest.mark.parametrize("size", [5, 4])
def test_shard_sum_equals_streaming_in_shard_blocks(plain, size):
    """K = 3 and 4 shards over N = 13; neither size divides it."""
    case, _ = plain
    got = run_case_shard_scored(case, size)
    _assert_bitwise(got, run_case_streamed(case, size, transport_core="per-electron"))
    assert got[SHARD_SCORING_KEY]["shards"] == len(shard_ranges(case, size))


@pytest.mark.parametrize("size", [5, 4])
def test_shard_sum_is_within_reordering_of_the_join(plain, size):
    case, joined = plain
    _assert_within_reordering(run_case_shard_scored(case, size), joined)


def test_shard_sum_is_repeatable_in_any_scoring_order(plain):
    """Shards scored in reverse, pickled, then merged in order: same bits."""
    case, _ = plain
    shards = _shards(case, 4)
    plan = resolve_shard_plan(case, lambda: iter(shards))
    partials = {
        i: pickle.loads(pickle.dumps(score_history_shard(case, rows, plan)))
        for i, rows in reversed(list(enumerate(shards)))
    }
    got = merge_shard_spectra(case, plan, [partials[i] for i in range(len(shards))])
    _assert_bitwise(got, run_case_shard_scored(case, 4))


def test_merge_refuses_a_stale_plan(plain):
    case, _ = plain
    shards = _shards(case, 5)
    plan = resolve_shard_plan(case, lambda: iter(shards))
    partials = [score_history_shard(case, rows, plan) for rows in shards]
    partials[1] = {**partials[1], "plan": "0" * 64}
    with pytest.raises(ValueError, match=r"\[1\] were scored against another plan"):
        merge_shard_spectra(case, plan, partials)


def test_measured_policy_grid_is_frozen_from_the_shards():
    """resonance-population bandwidth, resonance-local bin-mean axis (the #1320 shape)."""
    case = _case()
    case = {**case, "line_grid_policy": _measured_policy(case)}
    joined = run_case_from_shards(case, _shards(case, 13))
    got = run_case_shard_scored(case, 4)
    _assert_within_reordering(got, joined)
    # The audit sums add across shards to the joined audit's summation order.
    audit, expected = got["line_grid_resolved"], joined["line_grid_resolved"]
    assert audit["start_eV"] == expected["start_eV"] and audit["stop_eV"] == expected["stop_eV"]
    np.testing.assert_allclose(
        audit["truncation_audit"]["upper_fraction_bound"],
        expected["truncation_audit"]["upper_fraction_bound"],
        rtol=1e-12,
    )
    _assert_bitwise(got, run_case_streamed(case, 4, transport_core="per-electron"))


@pytest.mark.slow
def test_automatic_selection_reads_only_the_prefix_shards(monkeypatch):
    from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION, _transported
    from tests.montecarlo.transport.test_streamed_selection import assert_same_selection

    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    case, _ = _transported()
    case = {k: v for k, v in case.items() if k != "line_grid_policy"}
    case = {**case, "reflection_selection": {**SELECTION, "automatic": True}}
    shards = _shards(case, 1)
    read = []

    def tracked():
        for index, rows in enumerate(shards):
            read.append(index)
            yield rows

    plan = resolve_shard_plan(case, tracked)
    # The streamed scorer holds blocks until one reaches electron 2.
    assert read == [0, 1, 2]
    got = merge_shard_spectra(
        case, plan, [score_history_shard(case, rows, plan) for rows in shards]
    )
    streamed = run_case_streamed(case, 1, transport_core="per-electron")
    _assert_bitwise(got, streamed)
    assert_same_selection(got["reflection_selection"], streamed["reflection_selection"])


def test_shard_sum_realization_identity():
    joined = TransportRealization()
    summed = TransportRealization(reduction="shard-sum", shard_electrons=5)
    assert dict(joined) == {"schema": joined.schema, "core": "per-electron"}
    assert dict(summed) == {
        "schema": summed.schema,
        "core": "per-electron",
        "scoring": "cpu",
        "reduction": "shard-sum",
        "shard_electrons": 5,
    }
    assert pickle.loads(pickle.dumps(summed)) == summed
    # HDF5 round trips return NumPy integers.
    assert normalize_realization({**dict(summed), "shard_electrons": np.int64(5)}) == summed
    for bad in ({"shard_electrons": 0}, {"shard_electrons": None}, {"reduction": "tree"}):
        with pytest.raises(ValueError):
            normalize_realization({**dict(summed), **bad})
    with pytest.raises(ValueError, match="shard-sum"):
        TransportRealization(shard_electrons=5)


def test_marked_case_routes_run_case_and_refuses_joined_scoring(plain):
    case, _ = plain
    marked = bind_realization(case, TransportRealization(reduction="shard-sum", shard_electrons=5))
    got = runner.run_case(marked)
    _assert_bitwise(got, run_case_shard_scored(case, 5))
    with pytest.raises(ValueError, match="conflicts"):
        run_case_shard_scored(marked, 4)
    with pytest.raises(ValueError, match="shard-sum realization scores"):
        run_case_from_shards(marked, _shards(case, 13))
