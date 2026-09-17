"""Window-refinement ladder on a tiny real transport (issue #101).

Ne=3 through 1 um of HOPG at 30 keV is nowhere near a converged spectrum; these
tests pin plumbing that does not depend on Monte Carlo statistics: windows
refine while the backbone stays put, rungs are labelled by the refined
quantity, one transport serves every rung, and the dense uniform reference is
never finer than the budget allows.
"""

import numpy as np
import pytest

from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import (
    SpectrumSample,
    evaluate_ladder,
    segment_fingerprint,
)

SAMPLES = (2, 4, 8)
PROVIDERS = ("pxr-kinematic",)
BACKBONE_EV = 12.0


@pytest.fixture(scope="module")
def ladder():
    case = cc.build_ladder_case("hopg", 30.0, 5.0, 95.0, thickness_ang=1.0e4, n_electrons=3, seed=0)
    return cc.CaseLadder(case, transport_core="lockstep")


def _rungs(ladder):
    return cc.window_ladder(ladder, SAMPLES, backbone_eV=BACKBONE_EV, providers=PROVIDERS)


def test_windows_refine_while_the_backbone_stays_fixed(ladder):
    width, rungs = _rungs(ladder)

    assert [rung["spacing_eV"] for rung in rungs] == [width / count for count in SAMPLES]
    grids = [rung["plan"].coordinates() for rung in rungs]
    # The backbone is the largest spacing of every rung, so it cannot order them.
    assert {round(float(np.diff(grid).max()), 9) for grid in grids} == {
        round(float(np.diff(grids[0]).max()), 9)
    }
    assert [grid.size for grid in grids] == sorted(grid.size for grid in grids)
    assert [float(np.diff(grid).min()) for grid in grids] == sorted(
        (float(np.diff(grid).min()) for grid in grids), reverse=True
    )
    assert all(rung["seeds"]["pxr-kinematic"]["seeds"] >= 1 for rung in rungs)


def test_samples_must_be_positive_and_strictly_increasing(ladder):
    with pytest.raises(ValueError, match="strictly increasing"):
        cc.window_ladder(ladder, (4, 2), backbone_eV=BACKBONE_EV, providers=PROVIDERS)


def test_rungs_are_labelled_by_the_refined_window_spacing(ladder):
    _width, rungs = _rungs(ladder)

    evaluated = evaluate_ladder(
        [rung["plan"].coordinates() for rung in rungs],
        ladder.evaluate,
        spacings=[rung["spacing_eV"] for rung in rungs],
        segments=ladder.segments,
    )

    assert [rung.spacing_eV for rung in evaluated] == [rung["spacing_eV"] for rung in rungs]
    assert [rung.n_points for rung in evaluated] == [rung["plan"].num for rung in rungs]
    assert segment_fingerprint(ladder.segments) == ladder.fingerprint


def test_explicit_labels_must_still_refine_strictly_and_cover_every_grid():
    grid = np.linspace(100.0, 200.0, 11)

    def evaluate(E):
        return SpectrumSample(line=np.zeros(E.size), background=np.zeros(E.size))

    with pytest.raises(ValueError, match="refine strictly"):
        evaluate_ladder([grid, grid], evaluate, spacings=[1.0, 1.0])
    with pytest.raises(ValueError, match="label every grid"):
        evaluate_ladder([grid, grid], evaluate, spacings=[1.0])


def test_reference_grid_is_capped_by_its_point_budget():
    fine, spacing = cc.reference_grid(10.0, 2010.0, 0.05, max_points=1_000_000)
    # Credited 1e-9 relative for float64 linspace rounding at an exact divisor.
    assert spacing <= 0.05 * (1.0 + 1.0e-9) and fine.size < 1_000_000
    coarse, capped = cc.reference_grid(10.0, 2010.0, 0.05, max_points=2000)
    assert coarse.size <= 2001
    assert capped == pytest.approx(1.0, rel=1e-6)


def test_point_budget_truncates_the_ladder_and_records_what_it_dropped(ladder):
    _width, rungs = _rungs(ladder)
    grids = [rung["plan"].coordinates() for rung in rungs]
    budget = grids[1].size

    kept, coordinates, skipped = cc._affordable_rungs(rungs, grids, budget)

    assert [rung["samples_per_feature"] for rung in kept] == list(SAMPLES[:2])
    assert [grid.size for grid in coordinates] == [grids[0].size, grids[1].size]
    assert skipped == [
        {
            "samples_per_feature": SAMPLES[2],
            "n_points": grids[2].size,
            "reason": "point budget",
        }
    ]


def test_point_budget_keeps_every_rung_that_fits(ladder):
    _width, rungs = _rungs(ladder)
    grids = [rung["plan"].coordinates() for rung in rungs]

    kept, coordinates, skipped = cc._affordable_rungs(rungs, grids, grids[-1].size)

    assert len(kept) == len(coordinates) == len(SAMPLES)
    assert skipped == []


def test_a_budget_under_the_coarsest_rung_raises_instead_of_reporting_nothing(ladder):
    _width, rungs = _rungs(ladder)
    grids = [rung["plan"].coordinates() for rung in rungs]

    with pytest.raises(ValueError, match="coarsest window rung"):
        cc._affordable_rungs(rungs, grids, grids[0].size - 1)
