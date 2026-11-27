"""Shard-summed spectra (#1395): each history shard scores its own electrons.

The shards of one case score against one frozen plan (line axis, reflection
selection) and their spectra add in shard order. The result must

- equal live streamed scoring with ``block_electrons = shard_electrons`` bit
  for bit (same blocks, same order of FP64 additions);
- be bitwise repeatable for a fixed shard size, whatever order the shards
  were scored in and after a pickle round trip;
- match the #1387 join (``run_case`` on the per-electron core) to FP64
  summation order, for shard counts that do and do not divide N; ``bin-mean``
  Fourier lines to that reduction's per-batch approximation instead.

Seeds are fixed by the sweep.
"""

import functools
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
from pyrite.montecarlo.runner import lattice_share, reflection_cutoff
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


@pytest.mark.slow
def test_bin_mean_fourier_lines_differ_from_the_join_only_by_their_batches():
    """Fourier bin-mean lines: per-flush approximation, not summation order (#1395).

    The host Fourier route picks its period and broad split per flushed line
    batch, and shards flush other batches than the join. With the dense
    evaluator the reordering contract holds; on the default route the line
    spectrum stays within the restated bound (two reductions, each within the
    ``cxr-line-fourier-reduction`` error of the exact bin means).
    """
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_reduction_override

    case = {**_case(), "line_quadrature": "bin-mean"}
    with bin_reduction_override("auto"):
        dense_joined = run_case_from_shards(case, _shards(case, 13))
        dense = run_case_shard_scored(case, 4)
    _assert_within_reordering(dense, dense_joined)
    joined = run_case_from_shards(case, _shards(case, 13))
    got = run_case_shard_scored(case, 4)
    peak = np.abs(joined["spec"]).max()
    excess = np.abs(got["spec"] - joined["spec"])
    assert excess.max() > 1e-14 * peak  # batching shows: not the reordering contract
    np.testing.assert_allclose(got["spec"], joined["spec"], rtol=0, atol=1e-5 * peak)


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
@pytest.mark.parametrize("rows", [None, 1])
def test_automatic_selection_reads_only_the_prefix_shards(monkeypatch, rows):
    from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION, _transported
    from tests.montecarlo.transport.test_streamed_selection import assert_same_selection

    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    if rows is not None:  # #1368 row budget: met by the first electron's rows
        monkeypatch.setattr(reflection_cutoff, "SELECTION_ROWS", rows)
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
    # The streamed scorer holds blocks until one reaches electron 2, or until
    # the held rows reach the row budget.
    assert read == ([0, 1, 2] if rows is None else [0])
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
        with pytest.raises(ValueError, match=r"shard_electrons|reduction"):
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


@pytest.mark.slow
def test_distributed_selection_pilots_give_the_same_plan(monkeypatch):
    """Pilots shipped part by part (cloudpickle round trip, as Dask does) and summed
    in shard order select the same set and freeze the same plan bit for bit."""
    import cloudpickle

    from pyrite.montecarlo.runner.part_map import (
        StagedParts,
        ordered_values,
        part_mapper,
        staged_part_result,
    )
    from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION, _transported

    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 3)
    case, _ = _transported()
    case = {k: v for k, v in case.items() if k != "line_grid_policy"}
    case = {**case, "reflection_selection": {**SELECTION, "automatic": True}}
    shards = _shards(case, 2)
    calls, continuum = [], []

    def distribute(held):
        def mapper(parts, fn):
            calls.append(len(held))
            if isinstance(fn, functools.partial) and fn.func is lattice_share._block_yield:
                # The continuum share ships without the held rows (#1395).
                carried = [*fn.args, *fn.keywords.values()]
                continuum.append(
                    not any(
                        isinstance(a, StagedParts) or any(a is h for h in held) for a in carried
                    )
                )
            shipped = cloudpickle.loads(cloudpickle.dumps((fn, held, parts.header, parts.prefix)))
            remote_fn, blocks, header, prefix = shipped
            results = [staged_part_result(remote_fn, block, header, prefix) for block in blocks]
            return ordered_values(parts, fn, results)

        return part_mapper(mapper)

    local = resolve_shard_plan(case, lambda: iter(shards))
    remote = resolve_shard_plan(case, lambda: iter(shards), distribute=distribute)
    assert calls and set(calls) == {2}  # prefix 3 is held in shards [0, 2) and [2, 4)
    assert continuum and all(continuum)
    assert remote.digest == local.digest
    assert remote.selection[1] == local.selection[1]


def test_cone_split_tally_absorbs_shard_snapshots():
    """Shard tallies absorbed in order equal one tally over both ranges (#379, #1395)."""
    from pyrite.montecarlo.runner.cone_split_strata import SplitTally

    def block(parents, weights, killed):
        return dict(
            cone_split=dict(
                n_hat=[0.0, 0.0, 1.0],
                half_angle_rad=0.1,
                target_weight=0.25,
                parent=np.asarray(parents),
                weight=np.asarray(weights),
                killed=np.asarray(killed),
            )
        )

    case = {"cone_split": {"policy": "detector-cone"}}
    first = block([0, 0, 1], [0.5, 0.25, 0.25], [False, True])
    second = block([0, 0, 0], [0.25, 0.25, 0.25], [False])
    whole, shards = SplitTally(case), [SplitTally(case), SplitTally(case)]
    for tally, segments in zip(shards, (first, second), strict=True):
        whole.add(segments)
        tally.add(segments)
    merged = SplitTally(case)
    for tally in shards:
        merged.absorb(tally.snapshot())
    assert merged.record() == whole.record()
