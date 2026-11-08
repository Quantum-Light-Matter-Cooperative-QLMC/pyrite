"""Lineage-addressed cascade RNG (#923): keys and block invariance.

Validation: shell-secondary-transport, photon-pair-first-interaction
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import lineage
from pyrite.montecarlo.transport.kinematics import stream_keys
from tests.helpers.positron_tables import require_positron_tables

from .test_pair_production import _cascade
from .test_shell_secondary_transport import _require_pdatconf, _run  # noqa: F401

_PROCESSES = (
    lineage.PROCESS_SHELL,
    lineage.PROCESS_PAIR_PHOTON,
    lineage.PROCESS_PAIR_ELECTRON,
    lineage.PROCESS_PAIR_POSITRON,
    lineage.PROCESS_ANNIHILATION_IN_FLIGHT,
    lineage.PROCESS_ANNIHILATION_EMISSION,
    lineage.PROCESS_HARD_PHOTON_DIRECTION,
)


def test_daughter_keys_separate_processes_parents_and_ordinals():
    parents = stream_keys(7, 50)
    ordinals = np.arange(8)
    keys = np.concatenate(
        [
            lineage.daughter_keys(parents[:, None], process, ordinals[None, :]).ravel()
            for process in _PROCESSES
        ]
    )
    assert keys.dtype == np.uint64
    assert np.unique(keys).size == keys.size
    assert not np.intersect1d(keys, stream_keys(7, 10_000)).size
    # Pure function of (parent key, process, ordinal): no population context.
    one = lineage.daughter_keys(parents[13], lineage.PROCESS_SHELL, 3)
    assert one == lineage.daughter_keys(parents, lineage.PROCESS_SHELL, 3)[13]
    assert one == lineage.daughter_keys(stream_keys(7, 1, start=13), lineage.PROCESS_SHELL, 3)[0]


def test_keyed_uniforms_depend_only_on_key_and_counter():
    keys = stream_keys(3, 4000)
    u = lineage.keyed_uniforms(keys, 2)
    assert u.shape == (4000, 2) and np.all((u >= 0.0) & (u < 1.0))
    permutation = np.random.default_rng(0).permutation(keys.size)
    np.testing.assert_array_equal(lineage.keyed_uniforms(keys[permutation], 2), u[permutation])
    np.testing.assert_array_equal(lineage.keyed_uniforms(keys, 3)[:, :2], u)
    assert abs(u.mean() - 0.5) < 0.01
    assert abs(np.corrcoef(u[:, 0], u[:, 1])[0, 1]) < 0.05


def test_parent_ordinals_count_in_flight_order_whatever_the_row_order():
    parent = np.array([2, 0, 2, 0, 2, 1])
    flight = np.array([5, 1, 3, 0, 4, 9])
    order, ordinal = lineage.parent_ordinals(parent, flight)
    ranked = np.empty_like(ordinal)
    ranked[order] = ordinal
    np.testing.assert_array_equal(ranked, [2, 1, 0, 0, 1, 0])
    permutation = np.array([4, 2, 0, 5, 3, 1])
    order2, ordinal2 = lineage.parent_ordinals(parent[permutation], flight[permutation])
    ranked2 = np.empty_like(ordinal2)
    ranked2[order2] = ordinal2
    np.testing.assert_array_equal(ranked2, ranked[permutation])


def _canonical(result):
    """Rows keyed by (track lineage, flight, substep), with the track's launch data."""
    tracks = result["secondary_tracks"]
    track = np.asarray(result["track_id"], dtype=np.int64)
    lineage_key = tracks["lineage_key"][track]
    order = np.lexsort((result["substep_id"], result["flight_id"], lineage_key))
    rows = {
        "lineage_key": lineage_key[order],
        "flight_id": np.asarray(result["flight_id"])[order],
        "substep_id": np.asarray(result["substep_id"])[order],
        "generation": np.asarray(result["generation"])[order],
    }
    for name in ("r_mid", "v_hat", "L_ang", "E_start_keV", "E_end_keV", "t_end_ang", "hard_W_keV"):
        rows[name] = np.asarray(result[name])[order]
    by_key = np.argsort(tracks["lineage_key"])
    launches = {
        name: np.asarray(tracks[name])[by_key]
        for name in (
            "lineage_key",
            "generation",
            "parent_hard_ordinal",
            "launch_kind",
            "launch_E_keV",
        )
    }
    return rows, launches


@pytest.mark.parametrize("split", ([12], [5, 7], [1, 4, 7], [3, 3, 3, 3]))
def test_shell_cascade_is_invariant_to_primary_blocks(split):
    """Blocks of primaries reproduce the joined cascade history by history (#923)."""
    whole = _run(Ne=12, E0=30.0)
    assert np.any(np.asarray(whole["generation"]) >= 2), "need descendants beyond gen 1"
    parts, start = [], 0
    for n in split:
        parts.append(_run(Ne=n, E0=30.0, _electron_start=start))
        start += n
    _assert_blocks_match(whole, parts)


def _assert_blocks_match(whole, parts):
    rows, launches = _canonical(whole)
    block = [_canonical(part) for part in parts]
    joined_rows = {k: np.concatenate([b[0][k] for b in block]) for k in rows}
    joined_launches = {k: np.concatenate([b[1][k] for b in block]) for k in launches}
    order = np.lexsort(
        (joined_rows["substep_id"], joined_rows["flight_id"], joined_rows["lineage_key"])
    )
    for name, value in rows.items():
        np.testing.assert_array_equal(joined_rows[name][order], value, err_msg=name)
    by_key = np.argsort(joined_launches["lineage_key"])
    for name, value in launches.items():
        np.testing.assert_array_equal(joined_launches[name][by_key], value, err_msg=name)


def _pair_events(result, fields):
    events = result["pair_production"]["events"]
    order = np.lexsort((events["photon_ordinal"], events["parent_key"]))
    return {name: np.asarray(events[name])[order] for name in fields}


_PAIR_FIELDS = ("parent_key", "photon_ordinal", "k_eV", "electron_keV", "positron_keV", "r_ang")


@pytest.mark.parametrize("positrons", (False, True))
@pytest.mark.parametrize("split", ([2, 4], [1, 1, 4], [5, 1]))
def test_pair_cascade_is_invariant_to_primary_blocks(monkeypatch, positrons, split):
    """Pair conversion, both products and annihilation follow lineage, not blocks."""
    fields = _PAIR_FIELDS
    extra = {}
    if positrons:
        require_positron_tables()
        fields += ("positron_fate", "annihilation_keV", "annihilation_t_ang")
        extra = {"positron_transport": True, "thickness_ang": 1.0e6}
    whole = _cascade(monkeypatch, pair_scale=1e9, **extra)
    kinds = whole["secondary_tracks"]["launch_kind"]
    assert np.any(kinds == 2), "need launched pair electrons"
    if positrons:
        assert np.any(kinds == 3), "need launched positrons"
    parts, start = [], 0
    for n in split:
        parts.append(_cascade(monkeypatch, pair_scale=1e9, Ne=n, _electron_start=start, **extra))
        start += n
    _assert_blocks_match(whole, parts)
    expected = _pair_events(whole, fields)
    got = [_pair_events(part, fields) for part in parts]
    joined = {k: np.concatenate([g[k] for g in got]) for k in fields}
    order = np.lexsort((joined["photon_ordinal"], joined["parent_key"]))
    for name in fields:
        np.testing.assert_array_equal(joined[name][order], expected[name], err_msg=name)
    if positrons:
        photons = whole["pair_production"]["annihilation_photons"]["k_keV"]
        block_photons = np.concatenate(
            [p["pair_production"]["annihilation_photons"]["k_keV"] for p in parts]
        )
        np.testing.assert_array_equal(np.sort(block_photons), np.sort(photons))


@pytest.mark.parametrize("split", ([4, 7, 1], [12]))
def test_spread_cascade_blocks_match_with_population_range(split):
    """Energy spread, per-electron cutoffs and T_s below the cutoff (#923).

    Blocks get the population table range, as the block driver passes it.
    """
    from pyrite.montecarlo.transport.beam_entry import initial_energies_keV

    n, spread, seed = 12, 0.08, 11
    cuts = np.where(np.arange(n) % 3 == 0, 3.0, 2.0)
    common = dict(E0=30.0, threshold=1000.0, energy_spread_frac=spread, seed=seed)
    whole = _run(Ne=n, E_cut_by_electrons=cuts, **common)
    assert np.any(np.asarray(whole["generation"]) >= 2)
    energies = initial_energies_keV(30.0, n, seed, spread)
    population = (float(cuts.min()), float(energies.max()))
    parts, start = [], 0
    for size in split:
        parts.append(
            _run(
                Ne=size,
                E_cut_by_electrons=cuts[start : start + size],
                _electron_start=start,
                _energy_range_keV=population,
                **common,
            )
        )
        start += size
    _assert_blocks_match(whole, parts)
