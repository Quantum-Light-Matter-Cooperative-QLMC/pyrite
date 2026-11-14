"""Measured reciprocal-cutoff selection for incoherent CXR lines (#1173)."""

import math

import numpy as np
import pytest

from pyrite.materials.crystal import dominant_reflections
from pyrite.montecarlo.runner.reflection_cutoff import (
    F0_RANGE_INVANG,
    band_bin_grid,
    geometric_tail_fraction,
    reflection_shells,
    select_reflection_cutoff,
)
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths


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


@pytest.mark.parametrize("shells", [[1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, math.nan]])
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
