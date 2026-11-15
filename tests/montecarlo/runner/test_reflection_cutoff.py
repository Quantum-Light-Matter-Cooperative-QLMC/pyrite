"""Measured reciprocal-cutoff selection for incoherent CXR lines (#1173)."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.crystal import dominant_reflections
from pyrite.montecarlo.runner.reflection_cutoff import (
    F0_RANGE_INVANG,
    band_bin_grid,
    geometric_tail_fraction,
    iter_reflection_shells,
    prune_case_reflections,
    reflection_shells,
    select_reflection_cutoff,
)
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths


@pytest.mark.parametrize("index", [0, 1, 2], ids=["si-slab", "mos2-film", "si-substrate"])
def test_production_gate_shells_pin_the_selected_set_and_tail(index):
    """Freeze the measured #1173 production sequences, not a synthetic decay.

    Jobs 12672/12673, seed 642501, frozen 12083/12115 transports; source and
    report hashes are in the fixture. This replays selection, not kernels.
    It anchors these scenes without asserting a universal omitted-yield bound.
    """
    evidence = json.loads(
        (Path(__file__).parents[2] / "data" / "reflection_selection_1173.json").read_text()
    )
    sample = evidence["samples"][index]
    measured = iter(sample["shells"])

    def shell_yield(members):
        shell = next(measured)
        assert len(members) == shell["n"]
        return shell["yield_"]

    result = select_reflection_cutoff(
        shell_yield,
        reflection_shells(sample["crystal"], sample["B_ang2"], 32.0),
        evidence["tolerance"],
    )
    assert result.converged and result.cutoff_invang == sample["cutoff_invang"]
    assert len(result.hkl_list) == sample["reflections"]
    digest = hashlib.sha256(
        json.dumps(sorted(result.hkl_list), separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == sample["hkl_sha256"]
    assert result.tail_fraction == pytest.approx(sample["tail_fraction"], rel=1e-12)
    assert result.tail_fraction <= evidence["tolerance"]
    assert abs(sample["yield_gap_vs_g40"]) < evidence["tolerance"]


def test_band_bin_grid_tiles_the_band_exactly():
    grid = band_bin_grid(500.0, 10_000.0, 500.0)
    edges, widths = _energy_bin_edges_and_widths(grid)
    assert edges[0] == pytest.approx(500.0)
    assert edges[-1] == pytest.approx(10_000.0)
    np.testing.assert_allclose(widths, 500.0)
    assert band_bin_grid(2000.0, 2600.0, 500.0).size == 4
    with pytest.raises(ValueError):
        band_bin_grid(3.0, 2.0)


def test_geometric_tail_is_exact_for_a_geometric_sequence():
    ratio = 0.5
    shells = ratio ** np.arange(6)
    remaining = shells[-1] * ratio / (1 - ratio)
    assert geometric_tail_fraction(shells) == pytest.approx(remaining / shells.sum())


@pytest.mark.parametrize(
    "shells",
    [[1.0, 0.5], [1.0, 1.0, 1.0], [1.0, 0.5, 0.6], [1.0, 0.5, 0.0], [1.0, 0.5, math.nan]],
)
def test_geometric_tail_refuses_until_the_sequence_decays(shells):
    assert geometric_tail_fraction(shells) == math.inf


def test_shells_partition_the_dominant_reflection_set():
    shells = reflection_shells("silicon", 0.46, 12.0)
    union = sorted(hkl for _, members in shells for hkl in members)
    assert union == sorted(
        dominant_reflections("silicon", n_families=10**9, B_ang2=0.46, g_max_invang=12.0)
    )
    uppers = [upper for upper, _ in shells]
    assert uppers == sorted(uppers) and uppers[-1] == pytest.approx(12.0)
    first_eight = sorted(hkl for upper, members in shells if upper <= 8.0 for hkl in members)
    assert first_eight == sorted(
        dominant_reflections("silicon", n_families=10**9, B_ang2=0.46, g_max_invang=8.0)
    )
    with pytest.raises(ValueError):
        reflection_shells("silicon", 0.46, F0_RANGE_INVANG + 1.0)


def _synthetic(yields):
    shells = [(2.0 * (i + 1), [(i, 0, 0)]) for i in range(len(yields))]
    by_member = {members[0]: value for (_, members), value in zip(shells, yields, strict=True)}
    return shells, lambda hkls: sum(by_member[h] for h in hkls)


def test_selection_stops_at_the_first_converged_shell_and_skips_low_order_dips():
    # A dip at shell 2 (0.3 < 0.5) recovers at shell 3; only the geometric tail stops.
    yields = [1.0, 0.3, 0.6, 0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375]
    shells, score = _synthetic(yields)
    result = select_reflection_cutoff(score, shells, 0.02)
    assert result.converged
    kept = yields[: len(result.hkl_list)]
    assert result.tail_fraction <= 0.02
    assert result.tail_fraction == pytest.approx(geometric_tail_fraction(kept))
    assert len(result.hkl_list) == 7
    assert result.cutoff_invang == pytest.approx(14.0)
    assert [row[0] for row in result.shells] == [2.0 * (i + 1) for i in range(7)]


def test_selection_reports_an_unconverged_cap():
    shells, score = _synthetic([1.0, 0.9, 0.8])
    result = select_reflection_cutoff(score, shells, 1e-3)
    assert not result.converged
    assert result.cutoff_invang == pytest.approx(6.0)
    assert len(result.hkl_list) == 3
    with pytest.raises(ValueError):
        select_reflection_cutoff(score, shells, 0.0)


def test_tail_takes_the_larger_of_the_last_two_ratios():
    # After a dip (0.1 then 0.08), one ratio (0.8) would claim a smaller tail
    # than the ratio across the dip allows; max picks 0.8 only when it is larger.
    assert geometric_tail_fraction([1.0, 0.1, 0.08]) == pytest.approx(0.08 * 0.8 / 0.2 / 1.18)
    assert geometric_tail_fraction([1.0, 0.5, 0.05]) == pytest.approx(0.05 * 0.5 / 0.5 / 1.55)


def test_a_lone_deep_dip_does_not_stop_the_search():
    shells, score = _synthetic([1.0, 1.2, 0.001, 1.0, 0.9])
    result = select_reflection_cutoff(score, shells, 1e-2)
    assert not result.converged
    assert len(result.hkl_list) == 5


def test_a_zero_shell_does_not_stop_the_search():
    shells, score = _synthetic([1.0, 0.5, 0.0, 0.4, 0.3])
    result = select_reflection_cutoff(score, shells, 1e-3)
    assert not result.converged
    assert len(result.hkl_list) == 5


def test_lazy_shells_match_the_eager_partition_and_stop_early():
    eager = reflection_shells("silicon", 0.46, 13.0)
    lazy = iter_reflection_shells("silicon", 0.46, 13.0)
    assert [(u, sorted(m)) for u, m in lazy] == [(u, sorted(m)) for u, m in eager]
    first = next(iter_reflection_shells("silicon", 0.46))
    assert first == eager[0]
    with pytest.raises(ValueError):
        next(iter_reflection_shells("silicon", 0.46, F0_RANGE_INVANG + 1.0))


def _g_decay_density(calls):
    """Line density whose band yield per reflection falls as exp(-|g|^2 / 50)."""
    from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector

    def density(pilot, grid):
        calls.append(pilot)
        radiators = pilot.get("layer_radiators") or [pilot]
        (rad,) = [r for r in radiators if r is not None]
        lattice = CRYSTALS[rad["crystal"]]["lattice"]
        g2 = [np.sum(reciprocal_g_vector(h, lattice)[0] ** 2) for h in rad["hkl_list"]]
        return np.full(grid.size, np.exp(-np.asarray(g2) / 50.0).sum() / grid.size)

    return density


def test_prune_cuts_each_radiator_on_its_own_shells_in_bounded_calls():
    calls = []
    slab = dict(crystal="silicon", B_ang2=0.46, hkl_list=[(1, 1, 1)])
    pruned, (outcome,) = prune_case_reflections(
        slab, _g_decay_density(calls), 1e-3, (500.0, 10_000.0), max_reflections=64
    )
    assert outcome.converged and outcome.tail_fraction <= 1e-3
    assert pruned["hkl_list"] == outcome.hkl_list and slab["hkl_list"] == [(1, 1, 1)]
    assert all(c["line_quadrature"] == "bin-mean" and len(c["hkl_list"]) <= 64 for c in calls)
    shell_sum = sum(row[2] for row in outcome.shells)
    whole = _g_decay_density([])(dict(slab, hkl_list=outcome.hkl_list), np.zeros(19))
    assert shell_sum == pytest.approx(np.sum(whole) * 9500.0 / 19)

    calls.clear()
    film = dict(crystal="mos2", B_ang2=0.5, hkl_list=[(0, 0, 2)])
    layered = dict(film, layer_radiators=[film, None, dict(slab)])
    pruned, outcomes = prune_case_reflections(
        layered, _g_decay_density(calls), 1e-2, (500.0, 10_000.0)
    )
    assert len(outcomes) == 2 and all(o.converged for o in outcomes)
    radiators = pruned["layer_radiators"]
    assert radiators[1] is None
    assert radiators[0]["hkl_list"] == outcomes[0].hkl_list == pruned["hkl_list"]
    assert radiators[2]["hkl_list"] == outcomes[1].hkl_list
    assert layered["layer_radiators"][2]["hkl_list"] == [(1, 1, 1)]
    assert {sum(r is not None for r in c["layer_radiators"]) for c in calls} == {1}


def test_selection_keeps_the_spectrum_of_the_selected_set_on_the_axis():
    from pyrite.montecarlo.runner.reflection_cutoff import (
        axis_yield_weights,
        select_case_reflections,
    )

    calls = []
    grid = np.linspace(500.0, 10_000.0, 41)
    weights = axis_yield_weights(grid, "node")
    np.testing.assert_allclose(weights @ np.ones(grid.size), 9500.0)
    slab = dict(crystal="silicon", B_ang2=0.46, hkl_list=[(1, 1, 1)], line_quadrature="node")
    pruned, (outcome,), spectrum = select_case_reflections(
        slab, _g_decay_density(calls), 1e-3, grid, weights, max_reflections=64
    )
    assert {c["line_quadrature"] for c in calls} == {"node"}
    whole = _g_decay_density([])(dict(slab, hkl_list=outcome.hkl_list), grid)
    np.testing.assert_allclose(spectrum, whole, rtol=1e-12)
    assert sum(row[2] for row in outcome.shells) == pytest.approx(whole @ weights)
    assert pruned["hkl_list"] == outcome.hkl_list
    bins = axis_yield_weights(grid, "bin-mean")
    assert bins.sum() == pytest.approx(9500.0 + (grid[1] - grid[0]))
    with pytest.raises(ValueError):
        axis_yield_weights(grid, "simpson")
