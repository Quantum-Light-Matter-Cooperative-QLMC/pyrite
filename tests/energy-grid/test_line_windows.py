"""Piecewise line-axis window plans (issue #101).

The planner is pure policy: these tests pin determinism, coverage, exact
anchors, the uniform no-seed limit, payload round trips, and the interval-wise
backend-precision floor. Convergence of the windowed spectrum is a ladder
measurement, not a unit test.
"""

import itertools
import json

import numpy as np
import pytest

from pyrite._grid_semantics import resolution_num, validate_backend_coordinates
from pyrite._line_windows import (
    LINE_WINDOW_PLAN_SCHEMA,
    FeatureSeed,
    build_window_plan,
    window_plan_from_payload,
)


def _seed(centre, below, above, spacing, *, source="test", label=None, anchor=False):
    return FeatureSeed(
        source=source,
        label=label or f"{centre:g}",
        centre_eV=centre,
        below_eV=below,
        above_eV=above,
        spacing_eV=spacing,
        anchor=anchor,
    )


def _spacing_at(coordinates, lo, hi):
    """Largest node interval intersecting the open interval (lo, hi).

    Credited 1e-9 relative for float64 ``linspace`` rounding, so a window whose
    length is an exact multiple of its spacing compares equal to that spacing.
    """
    left = np.searchsorted(coordinates, lo, side="right") - 1
    right = np.searchsorted(coordinates, hi, side="left")
    return float(np.diff(coordinates[max(left, 0) : right + 1]).max()) * (1.0 - 1.0e-9)


def test_no_seeds_is_the_uniform_backbone_bit_for_bit():
    plan = build_window_plan(50.0, 16400.0, 0.75)
    expected = np.linspace(50.0, 16400.0, resolution_num(50.0, 16400.0, 0.75))
    assert not plan.windowed
    assert plan.num == expected.size
    np.testing.assert_array_equal(plan.coordinates(), expected)


def test_a_window_is_fine_inside_and_backbone_outside():
    plan = build_window_plan(10.0, 5000.0, 3.0, [_seed(2000.0, 20.0, 30.0, 0.1)])
    grid = plan.coordinates()
    assert plan.windowed
    assert np.all(np.diff(grid) > 0.0)
    assert grid[0] == 10.0 and grid[-1] == 5000.0
    assert 1980.0 in grid and 2030.0 in grid
    assert _spacing_at(grid, 1980.0, 2030.0) <= 0.1
    assert _spacing_at(grid, 10.0, 1980.0) <= 3.0
    assert _spacing_at(grid, 2030.0, 5000.0) <= 3.0
    assert plan.num == grid.size
    assert grid.size < resolution_num(10.0, 5000.0, 0.1) // 10


def test_plan_depends_on_the_seed_set_not_its_order():
    seeds = [
        _seed(900.0, 50.0, 50.0, 0.5, label="a"),
        _seed(930.0, 10.0, 80.0, 0.2, label="b"),
        _seed(284.2, 0.0, 15.0, 0.25, label="C K", anchor=True),
        _seed(930.0, 10.0, 80.0, 0.2, label="b"),
    ]
    reference = build_window_plan(100.0, 2000.0, 2.0, seeds)
    for order in itertools.permutations(seeds):
        plan = build_window_plan(100.0, 2000.0, 2.0, order)
        assert plan.pieces == reference.pieces
        np.testing.assert_array_equal(plan.coordinates(), reference.coordinates())
    assert len(reference.seeds) == 3


def test_overlapping_windows_take_the_finest_spacing_without_duplicate_nodes():
    plan = build_window_plan(
        100.0, 2000.0, 2.0, [_seed(900.0, 50.0, 50.0, 0.5), _seed(940.0, 20.0, 60.0, 0.1)]
    )
    grid = plan.coordinates()
    assert np.unique(grid).size == grid.size
    assert _spacing_at(grid, 850.0, 920.0) <= 0.5
    assert _spacing_at(grid, 920.0, 1000.0) <= 0.1
    assert _spacing_at(grid, 1000.0, 2000.0) <= 2.0


def test_anchor_is_an_exact_single_node_with_a_one_sided_window():
    edge = 284.2
    seeds = [
        _seed(edge, 0.0, 15.0, 0.25, source="absorption-edge", label="C K", anchor=True),
        _seed(edge, 5.0, 0.0, 0.5, source="other", label="C K below", anchor=True),
    ]
    grid = build_window_plan(10.0, 1000.0, 3.0, seeds).coordinates()
    assert np.count_nonzero(grid == edge) == 1
    assert _spacing_at(grid, edge, edge + 15.0) <= 0.25
    assert _spacing_at(grid, edge - 5.0, edge) <= 0.5


def test_windows_are_clipped_and_seeds_outside_the_bandwidth_are_reported():
    inside = _seed(15.0, 10.0, 5.0, 0.2, label="clipped")
    outside = _seed(9000.0, 5.0, 5.0, 0.2, label="outside")
    outside_anchor = _seed(-1.0, 0.0, 0.0, 0.2, label="edge below", anchor=True)
    plan = build_window_plan(10.0, 1000.0, 3.0, [inside, outside, outside_anchor])
    grid = plan.coordinates()
    assert grid[0] == 10.0
    assert _spacing_at(grid, 10.0, 20.0) <= 0.2
    assert {seed.label for seed in plan.dropped} == {"outside", "edge below"}
    assert plan.payload()["dropped"] == ["test:edge below", "test:outside"]


def test_nearly_coincident_window_ends_do_not_leave_a_sliver():
    # Window ends 1e-3 eV apart would otherwise create a 1e-3 eV piece.
    seeds = [_seed(500.0, 0.0, 50.0, 0.5), _seed(550.001, 0.0, 50.0, 0.5)]
    plan = build_window_plan(100.0, 1000.0, 2.0, seeds)
    grid = plan.coordinates()
    assert np.diff(grid).min() > 0.25
    assert _spacing_at(grid, 500.0, 600.0) <= 0.5


def test_a_window_coarser_than_the_backbone_refines_nothing():
    plan = build_window_plan(10.0, 1000.0, 1.0, [_seed(500.0, 50.0, 50.0, 5.0)])
    assert not plan.windowed
    np.testing.assert_array_equal(
        plan.coordinates(), np.linspace(10.0, 1000.0, resolution_num(10.0, 1000.0, 1.0))
    )


def test_payload_round_trips_through_json_bit_for_bit():
    seeds = [
        _seed(284.2, 0.0, 15.0, 0.25, source="absorption-edge", label="C K", anchor=True),
        _seed(1740.0, 3.0, 3.0, 0.05, source="characteristic", label="Si Ka1"),
        _seed(4123.456789, 40.0, 60.0, 0.33, source="pxr-kinematic", label="(002)"),
    ]
    plan = build_window_plan(10.0, 16400.0, 1.5, seeds)
    payload = json.loads(json.dumps(plan.payload(), sort_keys=True))
    assert payload["schema"] == LINE_WINDOW_PLAN_SCHEMA
    rebuilt = window_plan_from_payload(payload)
    assert rebuilt == plan
    np.testing.assert_array_equal(rebuilt.coordinates(), plan.coordinates())


def test_payload_whose_pieces_disagree_with_its_seeds_is_refused():
    payload = build_window_plan(10.0, 1000.0, 3.0, [_seed(500.0, 5.0, 5.0, 0.1)]).payload()
    payload["pieces"][1][2] = 0.2
    with pytest.raises(ValueError, match="schema bump"):
        window_plan_from_payload(payload)
    payload["schema"] = LINE_WINDOW_PLAN_SCHEMA + 1
    with pytest.raises(ValueError, match="not supported"):
        window_plan_from_payload(payload)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"below": -1.0}, "negative"),
        ({"spacing": 0.0}, "positive"),
        ({"below": 0.0, "above": 0.0}, "empty window"),
        ({"centre": float("nan")}, "finite"),
    ],
)
def test_malformed_seeds_are_refused(kwargs, match):
    arguments = {"centre": 100.0, "below": 1.0, "above": 1.0, "spacing": 0.1} | kwargs
    with pytest.raises(ValueError, match=match):
        _seed(arguments["centre"], arguments["below"], arguments["above"], arguments["spacing"])


def test_backend_floor_is_judged_at_each_interval_not_at_the_axis_top():
    # 5e-4 eV at 200 eV is ~33 float32 ulp there, but only ~0.25 ulp at 16.4 keV.
    plan = build_window_plan(50.0, 16400.0, 3.0, [_seed(200.0, 0.05, 0.05, 5.0e-4)])
    grid = plan.coordinates()
    assert validate_backend_coordinates(grid, dtype=np.float32, safety_ulps=8.0) > 0.0


def test_backend_floor_refuses_a_fine_window_high_on_the_axis_in_float32_only():
    plan = build_window_plan(50.0, 16400.0, 3.0, [_seed(15000.0, 0.05, 0.05, 5.0e-3)])
    grid = plan.coordinates()
    with pytest.raises(ValueError, match="ULP safety floor"):
        validate_backend_coordinates(grid, dtype=np.float32, safety_ulps=8.0)
    assert validate_backend_coordinates(grid, dtype=np.float64, safety_ulps=8.0) > 0.0


def test_backend_coordinates_refuse_nonmonotonic_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_backend_coordinates(np.array([1.0, 2.0, 2.0]), dtype=np.float64)
