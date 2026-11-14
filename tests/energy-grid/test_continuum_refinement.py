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


def test_automatic_policy_matches_a_resolved_reference_on_fixed_tracks(ladder):
    """Anchor (#278): the adopted ``dlnE=0.0025`` policy against a 4x finer reference.

    Same trajectories, two automatic meshes, so any difference is the grid
    alone. The ledger claims every continuum observable within ``1e-3`` -- ten
    times inside the ``detected_counts`` budget, at the ``intrinsic_source`` one.
    """
    from pyrite._continuum_grid import CONTINUUM_LOG_STEP, automatic_continuum_grid

    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    endpoint = case_endpoint_eV(ladder.case)

    def observe(grid):
        continuum = ladder.continuum(grid)
        return spectrum_observables(grid, np.zeros_like(continuum), continuum, detectors=detectors)

    adopted = automatic_continuum_grid("hopg", endpoint)
    resolved = automatic_continuum_grid("hopg", endpoint, log_step=CONTINUUM_LOG_STEP / 4)
    assert resolved.size > 3 * adopted.size
    production, reference = observe(adopted), observe(resolved)
    for name in (
        "continuum_yield",
        "continuum_centroid_eV",
        "timepix3_continuum_counts",
        "eaglexo_continuum_counts",
    ):
        assert reference[name] > 0.0
        relative = abs(production[name] - reference[name]) / reference[name]
        assert relative < 1.0e-3, f"{name} differs by {relative:.3e} from the resolved mesh"


@pytest.mark.parametrize(
    ("medium", "endpoint_eV", "size", "digest"),
    [
        (
            "hopg",
            30.0e3,
            2772,
            "e1691b83137a135ae70d351746ef071be880eaa6a097af9350a2856a324b3f84",
        ),
        (
            "tungsten",
            30.0e3,
            2406,
            "c8ee638b465c2ff506aa189ca93a0b1b3f59176ad464b2603e13a964ec061871",
        ),
    ],
)
def test_automatic_continuum_grid_bits_are_machine_independent(medium, endpoint_eV, size, digest):
    """Case identity hashes these node bits, so they may use only correctly rounded ops.

    The digest was verified bit-identical on two CPUs (x86 laptop and a Ryzen 9
    9950X3D). A change that routes the nodes through ``log``/``exp``/``geomspace``
    makes them CPU-dependent and fails this pin on some machines.
    """
    import hashlib

    from pyrite._continuum_grid import automatic_continuum_grid

    nodes = np.ascontiguousarray(automatic_continuum_grid(medium, endpoint_eV), dtype="<f8")
    assert nodes.size == size
    assert np.all(np.diff(nodes) > 0.0)
    assert hashlib.sha256(nodes.tobytes()).hexdigest() == digest


def test_stack_media_anchor_every_layer_edge_and_take_the_highest_floor():
    """#1164: a layered target's grid anchors each layer's EPDL edges, not the film's only.

    MoS2 on Si: the substrate's own Si K edge carries no anchor pair in the
    film-only grid (only the detector-path Chantler Si K does), which left a
    5-9 % edge-window error. With the substrate as a stack medium both Si K
    anchors are grid nodes. The floor is the higher of the two media floors.
    """
    from pyrite._continuum_grid import automatic_continuum_grid

    endpoint = 30.0e3
    film = automatic_continuum_grid("mos2", endpoint)
    np.testing.assert_array_equal(film, automatic_continuum_grid("mos2", endpoint, stack_media=()))
    stacked = automatic_continuum_grid("mos2", endpoint, stack_media=("silicon",))

    band = (float(stacked[0]), float(stacked[-1]))
    film_marks, _ = continuum_refinement_marks("mos2", *band)
    marks, summary = continuum_refinement_marks("mos2", *band, stack_media=("silicon",))
    film_labels = {mark.label for mark in film_marks}
    added = [mark for mark in marks if mark.label not in film_labels]
    assert "Si K (EPDL)" in {mark.label for mark in added}
    assert "Si" in summary["elements"]
    for mark in added:
        assert np.isin(mark.anchors_eV, stacked).all(), mark.label
    assert stacked[0] == max(photon_continuum_floor_eV(m) for m in ("mos2", "silicon"))
    # A substrate with the higher floor raises the stack floor.
    assert automatic_continuum_grid("hopg", endpoint, stack_media=("sapphire",))[0] == (
        photon_continuum_floor_eV("sapphire")
    )


# --- overlapping marks (#1171) ----------------------------------------------


def _mark(label, below, above, spacing, *, first=None, last=None, centred=True):
    from pyrite.energy_grid.refine import RefinementMark

    return RefinementMark(
        kind=EDGE_KIND,
        label=label,
        anchors_eV=(below, above),
        first_eV=below if first is None else first,
        last_eV=above if last is None else last,
        spacing_eV=spacing,
        centred=centred,
    )


def _anchor_failures(grid, marks):
    """Marks whose feature does not sit on a bin boundary of ``grid``.

    A centred pair (EPDL edge, kinematic endpoint) must put a bin edge -- a
    midpoint of two adjacent nodes -- exactly on its feature; a Chantler pair
    must be two adjacent grid nodes.
    """
    lo, hi = float(grid[0]), float(grid[-1])
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    failures = []
    for mark in marks:
        below, above = mark.anchors_eV
        if not (lo < below and above < hi):
            continue
        if mark.centred:
            feature = 0.5 * (below + above)
            if np.min(np.abs(midpoints - feature)) > 1.0e-12 * feature:
                failures.append(mark.label)
            continue
        index = int(np.searchsorted(grid, below))
        if not (index + 1 < grid.size and grid[index] == below and grid[index + 1] == above):
            failures.append(mark.label)
    return failures


def test_another_marks_samples_never_split_an_anchor_pair():
    """A neighbourhood sample inside another mark's pair is rejected, not kept.

    Before #1171 the sorted cross-mark merge kept the first coordinate of any
    close pair, so a coarse mark's sample landing inside a fine mark's pair
    either split it or dropped its anchor.
    """
    from pyrite.energy_grid.refine import refine_continuum_nodes

    baseline = np.linspace(100.0, 200.0, 21)
    coarse = _mark("coarse", 140.0, 146.0, 6.0, first=128.0, last=158.0)  # samples 128,134,...,158
    fine = _mark("fine", 151.6, 152.4, 0.8)  # 152 sits inside, 151.6/152.4 ~0.4 off a sample
    nodes, summary = refine_continuum_nodes(baseline, [coarse, fine])
    assert _anchor_failures(nodes, [coarse, fine]) == []
    assert summary["split_anchor_pairs"] == []
    assert not np.any((nodes > 151.6) & (nodes < 152.4))
    assert {151.6, 152.4, 140.0, 146.0} <= set(nodes.tolist())


def test_interleaved_centred_pairs_both_keep_their_bin_boundary():
    """Two edges closer than their pairs are wide share one node between them.

    The PtTe2-on-gold case: Pt M4 at 2206.0 eV and Au M5 at 2210.0 eV, pairs
    5.6 and 4.7 eV wide. No grid holds both pairs adjacent; the joined chain
    2204, 2208, 2212 puts a bin edge exactly on each edge.
    """
    from pyrite.energy_grid.refine import refine_continuum_nodes

    baseline = np.linspace(2100.0, 2300.0, 41)
    platinum = _mark("Pt M4", 2203.194, 2208.806, 5.612)
    gold = _mark("Au M5", 2207.633, 2212.367, 4.734)
    nodes, summary = refine_continuum_nodes(baseline, [gold, platinum])
    assert _anchor_failures(nodes, [platinum, gold]) == []
    assert summary["joined_anchor_pairs"] == 2
    window = nodes[(nodes > 2200.0) & (nodes < 2215.0)]
    np.testing.assert_allclose(window, [2204.0, 2208.0, 2212.0], rtol=0.0, atol=1.0e-9)


def test_a_baseline_node_at_a_pair_midpoint_is_displaced():
    """The strict half-spacing test alone kept a node exactly on the feature."""
    from pyrite.energy_grid.refine import refine_continuum_nodes

    baseline = np.array([100.0, 150.0, 200.0])
    nodes, _ = refine_continuum_nodes(baseline, [_mark("edge", 149.0, 151.0, 2.0)])
    np.testing.assert_array_equal(nodes, [100.0, 149.0, 151.0, 200.0])


#: Films and stacks whose marks overlap: the S L2 doublet in every sulfide,
#: Ta M2 beside S K, and gold M edges under W, Pt and S films (#1171 evidence).
OVERLAP_CASES = [
    ("mos2", ()),
    ("ws2", ()),
    ("2h_tas2", ()),
    ("tungsten", ("gold",)),
    ("ptte2", ("gold",)),
    ("ws2", ("gold",)),
    ("mos2", ("sio2", "silicon")),
]


def _automatic_grid_anchor_failures(film, stack_media, endpoint_eV=30.0e3):
    from pyrite._continuum_grid import CONTINUUM_LOG_STEP, automatic_continuum_grid

    grid = automatic_continuum_grid(film, endpoint_eV, stack_media=stack_media)
    marks, _ = continuum_refinement_marks(
        film,
        float(grid[0]),
        float(grid[-1]),
        endpoint_eV=endpoint_eV,
        local_spacing_eV=endpoint_eV * CONTINUUM_LOG_STEP,
        stack_media=stack_media,
    )
    assert marks
    return grid, _anchor_failures(grid, marks)


@pytest.mark.parametrize(("film", "stack_media"), OVERLAP_CASES)
def test_overlapping_marks_keep_every_feature_on_a_bin_boundary(film, stack_media):
    grid, failures = _automatic_grid_anchor_failures(film, stack_media)
    assert failures == []
    validate_backend_coordinates(grid, dtype=np.float32)


def test_every_catalog_material_keeps_every_feature_on_a_bin_boundary():
    """Anchor integrity over every catalog material, as a bare film and as shipped."""
    from pyrite.materials.catalog import _get_default_catalog

    catalog = _get_default_catalog()
    cases = {}
    for key in catalog.material_keys:
        spec = catalog.materials[key]
        below = [layer.material for layer in spec.stack]
        below += [spec.substrate] if spec.substrate else []
        stack = tuple(dict.fromkeys(m for m in below if m and m != spec.crystal_key))
        cases[(spec.crystal_key, ())] = key
        cases[(spec.crystal_key, stack)] = key
    failures = {}
    for (film, stack), key in cases.items():
        _grid, failed = _automatic_grid_anchor_failures(film, stack)
        if failed:
            failures[f"{key}: {film}/{','.join(stack)}"] = failed
    assert failures == {}


def test_close_centred_pairs_chain_down_to_the_merge_tolerance():
    """Features 0.3 spacing apart (tolerance 0.25) are chained, not left split.

    Fresh-context verification of #1171 found the chain test demanded intervals
    above twice the tolerance, leaving such pairs split with both edges off a
    bin boundary.
    """
    from pyrite.energy_grid.refine import refine_continuum_nodes

    baseline = np.linspace(80.0, 120.0, 9)
    first = _mark("E1", 99.5, 100.5, 1.0)
    second = _mark("E2", 99.8, 100.8, 1.0)
    nodes, summary = refine_continuum_nodes(baseline, [first, second])
    assert _anchor_failures(nodes, [first, second]) == []
    assert summary["split_anchor_pairs"] == []
    assert np.min(np.diff(nodes)) > 0.25
