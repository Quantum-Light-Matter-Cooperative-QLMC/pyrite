"""Photon-continuum node refinement at absorption edges and kinematic endpoints.

Issue #100. The geometric baseline equidistributes relative quadrature error for
a locally power-law integrand; these tests pin the two places the modelled
continuum is not one, that refinement is *absent* everywhere else, and that a
refined source mesh does not move the Timepix input channels.

Everything here is deterministic: fixed seed, fixed trajectories, and grids
built from the medium's own catalog data.
"""

import numpy as np
import pytest

from pyrite._grid_semantics import node_bin_edges_and_widths, validate_backend_coordinates
from pyrite.detectors.spec import EagleXO, Timepix3
from pyrite.detectors.timepix_response import TimepixResponse
from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import spectrum_observables
from pyrite.energy_grid.floor import geometric_continuum_grid, photon_continuum_floor_eV
from pyrite.energy_grid.refine import (
    EDGE_KIND,
    ENDPOINT_KIND,
    case_endpoint_eV,
    continuum_refinement_marks,
    refined_continuum_grid,
)

HOPG_FLOOR = photon_continuum_floor_eV("hopg")
#: Above the carbon K edge search interval but below the C K jump itself.
NO_FEATURE_STOP_EV = 250.0
#: 30 keV case, band deliberately taken above it so the tip is interior.
ENDPOINT_CASE_STOP_EV = 40_000.0
TINY = dict(thickness_ang=1.0e4, n_electrons=3, seed=0)


# --- edges ------------------------------------------------------------------


def test_edge_refinement_anchors_the_jump_where_an_edge_exists():
    """The located bracket nodes appear in the grid exactly."""
    marks, summary = continuum_refinement_marks("hopg", HOPG_FLOOR, 29_000.0)
    labels = {mark.label for mark in marks}
    # The medium's own carbon in the table its escape mu reads (EPDL), and the
    # detector path's silicon sensor in the table its response reads (Chantler).
    assert labels == {"C K (EPDL)", "Si K"}
    assert all(mark.kind == EDGE_KIND for mark in marks)
    assert summary["edges_outside_band"] == []

    grid = refined_continuum_grid("hopg", 29_000.0, 2049)
    for mark in marks:
        for anchor in mark.anchors_eV:
            assert np.any(grid == anchor), f"{mark.label} anchor {anchor} missing from the grid"


def test_edge_anchors_bracket_the_jump_in_one_interval():
    """The two anchors are adjacent nodes, so the jump lies in a single interval."""
    marks, _ = continuum_refinement_marks("hopg", HOPG_FLOOR, 29_000.0)
    grid = refined_continuum_grid("hopg", 29_000.0, 2049)
    for mark in marks:
        below, above = mark.anchors_eV
        index = int(np.searchsorted(grid, below))
        assert grid[index] == below
        assert grid[index + 1] == above


def test_edges_come_from_the_medium_not_a_hardcoded_list():
    """A different medium yields its own elements' edges."""
    tungsten_selenide, _ = continuum_refinement_marks(
        "wse2", photon_continuum_floor_eV("wse2"), 29_000.0
    )
    labels = {mark.label for mark in tungsten_selenide}
    assert {"W L3 (EPDL)", "Se K (EPDL)"} <= labels
    assert not any(label.startswith("C ") for label in labels)

    graphite, _ = continuum_refinement_marks("hopg", HOPG_FLOOR, 29_000.0)
    assert not any(mark.label.startswith("W ") for mark in graphite)


def test_no_edge_and_no_endpoint_returns_the_geometric_baseline_identically():
    """The limiting case: nothing to refine leaves every node where it was."""
    marks, summary = continuum_refinement_marks("hopg", HOPG_FLOOR, NO_FEATURE_STOP_EV)
    assert marks == []
    assert summary["edges_refined"] == 0

    baseline = geometric_continuum_grid("hopg", NO_FEATURE_STOP_EV, 513)
    refined, cost = refined_continuum_grid("hopg", NO_FEATURE_STOP_EV, 513, return_summary=True)
    assert np.array_equal(refined, baseline)
    assert cost["added_nodes"] == 0


def test_an_edge_just_outside_the_band_degrades_gracefully():
    """A jump above the ceiling is reported and dropped, never refused."""
    stop = 288.0  # EPDL C K jumps at exactly 288 eV: its anchor pair straddles the ceiling
    marks, summary = continuum_refinement_marks("hopg", HOPG_FLOOR, stop)
    assert marks == []
    assert "C K (EPDL)" in summary["edges_outside_band"]

    refined = refined_continuum_grid("hopg", stop, 513)
    assert np.array_equal(refined, geometric_continuum_grid("hopg", stop, 513))


# --- kinematic endpoint -----------------------------------------------------


def test_endpoint_refinement_puts_a_bin_edge_exactly_at_the_brem_tip():
    """The straddling pair makes the cutoff a bin EDGE, killing the O(h) term."""
    endpoint = 30_000.0
    refined, summary = refined_continuum_grid(
        "hopg", ENDPOINT_CASE_STOP_EV, 2049, endpoint_eV=endpoint, return_summary=True
    )
    assert summary["endpoint_eV"] == endpoint

    edges, _widths = node_bin_edges_and_widths(refined)
    assert np.min(np.abs(edges - endpoint)) == pytest.approx(0.0, abs=1e-9)

    # The plain baseline puts no edge there, and its local bin is far wider than
    # the miss, so the cutoff falls strictly inside one bin.
    baseline = geometric_continuum_grid("hopg", ENDPOINT_CASE_STOP_EV, 2049)
    base_edges, base_widths = node_bin_edges_and_widths(baseline)
    index = int(np.argmin(np.abs(base_edges - endpoint)))
    assert abs(base_edges[index] - endpoint) > 1.0
    assert base_widths[min(index, base_widths.size - 1)] > 10.0


def test_endpoint_marks_carry_the_endpoint_kind():
    marks, _ = continuum_refinement_marks(
        "hopg",
        HOPG_FLOOR,
        ENDPOINT_CASE_STOP_EV,
        endpoint_eV=30_000.0,
        local_spacing_eV=100.0,
    )
    endpoints = [mark for mark in marks if mark.kind == ENDPOINT_KIND]
    assert len(endpoints) == 1
    assert endpoints[0].anchors_eV == (29_950.0, 30_050.0)


def test_endpoint_above_the_ceiling_is_dropped():
    """A cutoff outside the modelled band places no requirement on the grid."""
    _refined, summary = refined_continuum_grid(
        "hopg", 29_000.0, 2049, endpoint_eV=30_000.0, return_summary=True
    )
    assert summary["endpoint_eV"] is None


def test_endpoint_needs_the_local_spacing():
    with pytest.raises(ValueError, match="local_spacing_eV"):
        continuum_refinement_marks("hopg", HOPG_FLOOR, 29_000.0, endpoint_eV=1000.0)


# --- grid invariants --------------------------------------------------------


@pytest.mark.parametrize("num", [257, 513, 2049, 8193])
def test_refinement_keeps_the_band_and_strict_monotonicity(num):
    baseline = geometric_continuum_grid("hopg", ENDPOINT_CASE_STOP_EV, num)
    refined = refined_continuum_grid("hopg", ENDPOINT_CASE_STOP_EV, num, endpoint_eV=30_000.0)
    assert refined[0] == baseline[0] == photon_continuum_floor_eV("hopg")
    assert refined[-1] == baseline[-1]
    assert np.all(np.diff(refined) > 0.0)
    assert refined.size >= baseline.size


@pytest.mark.parametrize("material", ["hopg", "silicon", "wse2", "diamond"])
def test_refined_grids_survive_backend_coordinate_precision(material):
    """No merge leaves a near-degenerate interval, on any catalog medium.

    ``wse2`` seeds nine marks, several of them overlapping shells, so this is
    where a missing merge rule would show up first.
    """
    refined = refined_continuum_grid(material, 29_000.0, 4097, endpoint_eV=20_000.0)
    validate_backend_coordinates(refined, dtype=np.float32)


def test_refinement_cost_stays_small_and_shrinks_as_the_baseline_refines():
    """Node cost is bounded and falls with node count; pinned, not assumed."""
    added = []
    for num in (2049, 4097, 8193):
        _grid, cost = refined_continuum_grid(
            "hopg", 29_000.0, num, endpoint_eV=30_000.0, return_summary=True
        )
        added.append(cost["added_nodes"])
        assert cost["added_nodes"] / num < 0.02
    # More baseline nodes are displaced by mark nodes, so fewer are net added.
    assert added == sorted(added, reverse=True)


# --- Timepix channels under a refined mesh ----------------------------------


@pytest.mark.parametrize("num", [257, 513, 2049])
def test_refining_a_mesh_does_not_move_the_timepix_input_channels(num):
    """The #100 padding fix holds under refinement.

    Refinement only adds interior nodes and never moves the band endpoints, so
    the outermost midpoint half-widths can only narrow. A mesh already covered
    by the padded band therefore stays covered, and keeps exactly the channels
    -- and so the seeded MC response matrix -- the plain baseline had.
    """
    baseline = geometric_continuum_grid("hopg", 20_000.0, num)
    refined = refined_continuum_grid("hopg", 20_000.0, num, endpoint_eV=15_000.0)

    plain_response = TimepixResponse(baseline, n_mc=8, seed=7)
    refined_response = TimepixResponse(refined, n_mc=8, seed=7)

    assert np.array_equal(plain_response.in_edges, refined_response.in_edges)
    assert plain_response.n_in == refined_response.n_in
    assert plain_response.zero_channel == refined_response.zero_channel


# --- observable budget on identical trajectories ----------------------------


@pytest.fixture(scope="module")
def ladder():
    case = cc.build_ladder_case("hopg", 30.0, 5.0, 95.0, **TINY)
    return cc.CaseLadder(case, transport_core="lockstep")


def test_continuum_ladder_starts_at_the_derived_floor(ladder):
    """The production selector reads the floor from the case's own medium."""
    grids = ladder.continuum_grids((513,), 29_000.0, refine=False)
    assert grids[0][0] == photon_continuum_floor_eV("hopg")
    assert grids[0][0] != 100.0  # the literal these ladders used to start at


def test_refined_continuum_observables_stay_within_budget(ladder):
    """Acceptance: refinement moves every continuum observable under budget.

    Compared against the plain geometric baseline on identical trajectories --
    one transport, two meshes -- so any difference is the grid alone.
    """
    budget = {
        "continuum_yield": 1.0e-3,
        "continuum_centroid_eV": 1.0e-3,
        "timepix3_continuum_counts": 1.0e-2,
        "eaglexo_continuum_counts": 1.0e-2,
    }
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    endpoint = case_endpoint_eV(ladder.case)

    def observe(grid):
        continuum = ladder.continuum(grid)
        return spectrum_observables(grid, np.zeros_like(continuum), continuum, detectors=detectors)

    for count in (2049, 4097):
        (plain,) = ladder.continuum_grids((count,), ENDPOINT_CASE_STOP_EV, refine=False)
        (refined,) = ladder.continuum_grids((count,), ENDPOINT_CASE_STOP_EV, refine=True)
        assert refined.size > plain.size
        assert endpoint == 30_000.0

        before, after = observe(plain), observe(refined)
        for name, allowed in budget.items():
            relative = abs(after[name] - before[name]) / abs(before[name])
            assert relative < allowed, f"{name} moved {relative:.3e} at num={count}"
