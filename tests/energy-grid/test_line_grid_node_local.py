"""Node-local refinement of an over-budget ``sinc-oversampled`` axis (#1257, #1271).

Pins the per-line join-mass bound of ``node_join_halo_eV`` under node
(trapezoid) quadrature, its graded-backbone extension ``node_graded_halos_eV``,
and the resolver route: the uniform axis wherever it fits, the local axis above
the budget (graded only where that saves nodes), a refusal only when that also
exceeds it.

Validation: line-grid-node-local-refinement
"""

import numpy as np
import pytest

from pyrite._cxr_grid_policy import (
    KINEMATIC_BANDWIDTH_POLICY,
    LINE_NYQUIST_OVERSAMPLING,
    OVERSAMPLED_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    CxrGridToleranceError,
    resolve_cxr_grid_policy,
    resolved_coordinates,
)
from pyrite._cxr_windows import build_window_plan
from pyrite.montecarlo.runner import cxr_grid
from pyrite.montecarlo.runner.node_local_grid import (
    DEFAULT_NODE_JOIN_LIMIT,
    NODE_LOCAL_FAR_SHARE,
    NODE_LOCAL_GRADED_CAP_EV,
    NODE_LOCAL_GROWTH,
    NODE_LOCAL_ROUTE,
    _graded_gap,
    graded_coordinates,
    node_graded_halos_eV,
    node_join_halo_eV,
    node_local_seeds,
)
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import sincsq_bin_masses
from pyrite.montecarlo.spectrum.cxr_seeds import ResonancePopulation

_BACKBONE = 3.0
_FINE = _BACKBONE / LINE_NYQUIST_OVERSAMPLING


def _line(centre, width):
    return [ResonancePopulation("line", np.array([centre]), np.array([1.0]), np.array([width]))]


def _join_error(centre, width, join_limit, span=60_000.0):
    """Relative node-quadrature mass error of one line on its local axis."""
    start, stop = centre - span, centre + span
    seeds, _ = node_local_seeds(
        [_line(centre, width)],
        start_eV=start,
        stop_eV=stop,
        fine_eV=_FINE,
        backbone_eV=_BACKBONE,
        oversampling=LINE_NYQUIST_OVERSAMPLING,
        join_limit=join_limit,
    )
    grid = build_window_plan(start, stop, _BACKBONE, seeds).coordinates()
    density = np.sinc((grid - centre) / width) ** 2
    exact = sincsq_bin_masses(np.pi / width, centre, grid[[0, -1]]).sum()
    return abs(np.trapezoid(density, grid) / exact - 1.0)


@pytest.mark.parametrize("width", [0.4, 1.0, 3.0, 3.5, 6.0, 11.6, 23.0, 40.0, 130.0, 2000.0])
def test_join_halo_bounds_single_line_node_mass(width):
    """Both sides together misplace at most twice the per-side join limit."""
    rng = np.random.default_rng(1257)
    worst = max(
        _join_error(20_000.0 + rng.uniform(0.0, 100.0), width, DEFAULT_NODE_JOIN_LIMIT)
        for _ in range(12)
    )
    assert worst <= 2.0 * DEFAULT_NODE_JOIN_LIMIT


def test_join_error_grows_when_the_halo_shrinks():
    """The bound discriminates: a 100x looser limit breaks the 2e-4 target near ``w ~ h_b``."""
    rng = np.random.default_rng(7)
    loose = max(
        _join_error(20_000.0 + rng.uniform(0.0, 100.0), 1.0, 100 * DEFAULT_NODE_JOIN_LIMIT)
        for _ in range(12)
    )
    assert loose > 2.0 * DEFAULT_NODE_JOIN_LIMIT


def test_halo_limiting_cases():
    eta = DEFAULT_NODE_JOIN_LIMIT
    narrow = node_join_halo_eV(0.1, _FINE, _BACKBONE, eta)
    assert narrow == pytest.approx(0.1 / (np.pi**2 * eta) + _BACKBONE / 2)
    target = 8 * 40.0**2 * eta / (np.pi * 2 * _BACKBONE**2)
    assert node_join_halo_eV(40.0, _BACKBONE, _BACKBONE, eta) == pytest.approx(
        3 * 40.0 / (np.pi * target)
    )
    # Wider than sqrt(1.5 pi (h_b**2 + h_c**2) / (8 eta)) (about 325 eV here): no halo.
    assert node_join_halo_eV(400.0, _BACKBONE, _BACKBONE, eta) == 0.0
    # The halo peaks near w ~ h_b and falls on both sides.
    assert node_join_halo_eV(4.5, _FINE, _BACKBONE, eta) > max(
        narrow, node_join_halo_eV(40.0, _BACKBONE, _BACKBONE, eta)
    )


def test_seeds_merge_overlapping_halos_and_clip_to_the_band():
    lines = ResonancePopulation(
        "lines",
        np.array([5000.0, 5100.0, 40_000.0, 99_990.0]),
        np.array([1.0, 1.0, 0.0, 1.0]),  # the zero-weight line asks for nothing
        np.array([12.0, 12.0, 12.0, 12.0]),
    )
    seeds, summary = _seeds([[lines]])
    assert summary["lines"] == 3 and summary["windows"] == 2
    halo = float(node_join_halo_eV(12.0, _FINE, _BACKBONE, DEFAULT_NODE_JOIN_LIMIT))
    first, last = seeds
    assert first.centre_eV <= 5000.0 - halo and first.centre_eV + first.above_eV >= 5100.0 + halo
    assert last.centre_eV + last.above_eV == 100_000.0
    assert {seed.spacing_eV for seed in seeds} == {_FINE}


def _seeds(direction_populations):
    return node_local_seeds(
        direction_populations,
        start_eV=50.0,
        stop_eV=100_000.0,
        fine_eV=_FINE,
        backbone_eV=_BACKBONE,
        oversampling=LINE_NYQUIST_OVERSAMPLING,
    )


def test_wide_lines_refine_only_where_their_halo_meets_a_fine_window():
    """A line wider than ``K h_b`` asks for no window, but no join may sit in its halo."""
    eta = DEFAULT_NODE_JOIN_LIMIT
    wide_halo = float(node_join_halo_eV(40.0, _BACKBONE, _BACKBONE, eta))
    narrow = ResonancePopulation("n", np.array([5000.0]), np.ones(1), np.array([12.0]))
    seeds, _ = _seeds([[narrow]])
    (window,) = seeds
    alone = ResonancePopulation(
        "w", np.array([60_000.0]), np.ones(1), np.array([40.0])
    )  # far away: stays on the backbone
    assert _seeds([[narrow, alone]])[0] == seeds
    edge = window.centre_eV + window.above_eV
    touching = ResonancePopulation(
        "w", np.array([edge + 0.5 * wide_halo]), np.ones(1), np.array([40.0])
    )  # its halo straddles the window edge: the halo is refined whole
    (merged,), _ = _seeds([[narrow], [touching]])
    assert merged.centre_eV == window.centre_eV
    assert merged.centre_eV + merged.above_eV == pytest.approx(edge + 1.5 * wide_halo)


def _payload(bandwidth, max_points, stop=966_200.0):
    return resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=stop,
        per_call={"bandwidth": bandwidth, "max_points": max_points},
    ).payload()


class _Evidence:
    def __init__(self, populations):
        self._populations = populations
        self.calls = 0

    def populations(self, n_hat, start, ceiling, profile):
        self.calls += 1
        return self._populations


_LINES = [
    ResonancePopulation(
        "lines",
        np.linspace(5600.0, 6800.0, 32),
        np.ones(32),
        np.full(32, 12.0),
    )
]
_CASE = {"name": "si test", "E0_keV": 1000.0}


def test_uniform_axis_unchanged_wherever_it_fits():
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 3_000_000)
    evidence = _Evidence(_LINES)
    grid, record = cxr_grid._kinematic_cxr_grid(payload, _CASE, (None,), 11.6, evidence)
    reference, expected = resolved_coordinates(payload, 11.6, dtype=cxr_grid.REAL)
    assert np.array_equal(grid, reference) and record == expected
    assert "local_axis" not in record and evidence.calls == 0


def test_kinematic_axis_over_budget_takes_the_node_local_route():
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 600_000)
    grid, record = cxr_grid._kinematic_cxr_grid(payload, _CASE, (None,), 11.6, _Evidence(_LINES))
    local = record["local_axis"]
    assert local["route"] == NODE_LOCAL_ROUTE
    assert local["uniform_points"] == 2_576_401 > 600_000 >= grid.size == record["num"]
    assert "above the 600000-point budget" in local["reason"]
    assert record["resolution_policy"] == OVERSAMPLED_RESOLUTION_POLICY
    steps = np.diff(grid)
    inside = (grid[:-1] >= 5600.0 - 100.0) & (grid[1:] <= 6800.0 + 100.0)
    assert steps[inside].max() <= _FINE * (1 + 1e-12)
    assert grid[0] == 50.0 and grid[-1] == 966_200.0
    # The graded backbone (#1271) beats the ungraded one on this 1 MeV band.
    grading = local["grading"]
    assert grading["cap_eV"] == NODE_LOCAL_GRADED_CAP_EV and grading["growth"] == NODE_LOCAL_GROWTH
    assert grid.size < local["ungraded_points"]
    assert _BACKBONE < steps.max() <= NODE_LOCAL_GRADED_CAP_EV + 1e-9
    assert np.array_equal(cxr_grid._cached_grid(record), grid)


def test_mixed_transport_local_axis_is_the_backbone():
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 600_000)
    evidence = _Evidence(_LINES)
    case = {**_CASE, "transport_mode": "mixed"}
    grid, record = cxr_grid._kinematic_cxr_grid(payload, case, (None,), 11.6, evidence)
    assert evidence.calls == 0 and record["local_axis"]["windows"] == 0
    assert np.diff(grid).min() == pytest.approx(_BACKBONE)


def test_local_axis_over_budget_still_refuses():
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 100_000)
    with pytest.raises(CxrGridToleranceError, match="node-local refinement") as error:
        cxr_grid._kinematic_cxr_grid(payload, _CASE, (None,), 11.6, _Evidence(_LINES))
    assert "sinc-nyquist" in str(error.value)


def test_measured_axis_over_budget_takes_the_node_local_route(monkeypatch):
    from pyrite.montecarlo import runner

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        line = (np.array([20_000.0]), np.array([1.0]), np.array([0.5]))
        kwargs["truncation_audit"]["collect"].append(line)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_cxr_for_segments", collect)
    payload = _payload(RESONANCE_BANDWIDTH_POLICY, 80_000, stop=100_000.0)
    case = {"name": "hbn test", "E0_keV": 5000.0, "composition": [("B", 0.5), ("N", 0.5)]}
    grid, record, bandwidth = cxr_grid._measured_cxr_grid(
        payload, case, {}, (np.array([0.0, 0.0, 1.0]),), 2, 0.5, None, None
    )
    local = record["local_axis"]
    assert local["route"] == NODE_LOCAL_ROUTE and local["windows"] == 1
    assert grid[-1] == bandwidth["stop_eV"] and grid.size <= 80_000 < local["uniform_points"]
    assert record["backbone_spacing_eV"] == 3.0 and record["nyquist_spacing_eV"] == 0.5
    assert np.diff(grid).min() == pytest.approx(0.5 / LINE_NYQUIST_OVERSAMPLING)


def test_bin_mean_axis_over_budget_takes_the_node_local_route_at_the_shape_step():
    """Bin-mean ``sinc-oversampled`` refines locally at its own step (#1265 on #1257).

    Validation: line-grid-node-local-refinement
    """
    from pyrite._cxr_grid_policy import DEFAULT_SHAPE_RTOL, shape_spacing_factor

    payload = resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=966_200.0,
        per_call={
            "bandwidth": KINEMATIC_BANDWIDTH_POLICY,
            "quadrature": "bin-mean",
            "max_points": 600_000,
        },
    ).payload()
    assert "oversampling" not in payload["resolution"]
    grid, record = cxr_grid._kinematic_cxr_grid(payload, _CASE, (None,), 11.6, _Evidence(_LINES))
    local = record["local_axis"]
    fine = shape_spacing_factor(DEFAULT_SHAPE_RTOL) * min(11.6, _BACKBONE)
    assert local["route"] == NODE_LOCAL_ROUTE and grid.size <= 600_000
    assert local["narrow_below_eV"] == pytest.approx(_BACKBONE / shape_spacing_factor(0.01))
    inside = (grid[:-1] >= 5600.0 - 100.0) & (grid[1:] <= 6800.0 + 100.0)
    assert np.diff(grid)[inside].max() <= fine * (1 + 1e-12)
    assert record["target_spacing_eV"] == pytest.approx(fine)


def _graded_join_error(centre, width, *, far=True, span=200_000.0):
    """Relative node mass error of one line on its graded local axis (#1271)."""
    start, stop = centre - span, centre + span
    seeds, summary = node_local_seeds(
        [_line(centre, width)],
        start_eV=start,
        stop_eV=stop,
        fine_eV=_FINE,
        backbone_eV=_BACKBONE,
        oversampling=LINE_NYQUIST_OVERSAMPLING,
        growth=NODE_LOCAL_GROWTH,
        cap_eV=NODE_LOCAL_GRADED_CAP_EV,
    )
    grading = {
        "growth": NODE_LOCAL_GROWTH,
        "cap_eV": NODE_LOCAL_GRADED_CAP_EV,
        "protected_eV": summary["protected_eV"] if far else [],
    }
    grid = graded_coordinates(build_window_plan(start, stop, _BACKBONE, seeds), grading)
    assert np.diff(grid).max() <= NODE_LOCAL_GRADED_CAP_EV + 1e-9
    density = np.sinc((grid - centre) / width) ** 2
    exact = sincsq_bin_masses(np.pi / width, centre, grid[[0, -1]]).sum()
    return abs(np.trapezoid(density, grid) / exact - 1.0)


@pytest.mark.parametrize("width", [0.4, 1.0, 3.0, 6.0, 11.6, 23.0, 40.0, 130.0, 600.0, 2000.0])
def test_graded_halos_bound_single_line_node_mass(width):
    """A graded backbone beyond ``D`` keeps both sides within twice the join limit.

    Validation: line-grid-node-local-refinement
    """
    rng = np.random.default_rng(1271)
    worst = max(_graded_join_error(100_000.0 + rng.uniform(0.0, 100.0), width) for _ in range(8))
    assert worst <= 2.0 * DEFAULT_NODE_JOIN_LIMIT


def test_graded_backbone_needs_the_grading_free_distance():
    """Grading right after the join breaks the bound for a Peano-halo line (D matters)."""
    rng = np.random.default_rng(11)
    loose = max(
        _graded_join_error(100_000.0 + rng.uniform(0.0, 100.0), 40.0, far=False) for _ in range(8)
    )
    assert loose > 2.0 * DEFAULT_NODE_JOIN_LIMIT


def test_graded_halo_limiting_cases():
    eta, rho, cap = DEFAULT_NODE_JOIN_LIMIT, NODE_LOCAL_GROWTH, NODE_LOCAL_GRADED_CAP_EV
    # Narrow line: tail option, d = D, the ungraded tail halo times (1 + rho / 2).
    d, far = node_graded_halos_eV(0.1, _FINE, _BACKBONE, eta, rho, cap)
    assert d == far == pytest.approx((1 + rho / 2) * 0.1 / (np.pi**2 * eta) + _BACKBONE / 2)
    # Peano line: d is the ungraded halo at (1 - share) eta; D bounds the cap's excess.
    d, far = node_graded_halos_eV(40.0, _BACKBONE, _BACKBONE, eta, rho, cap)
    assert d == pytest.approx(node_join_halo_eV(40.0, _BACKBONE, _BACKBONE, 0.5 * eta))
    target = 8 * 40.0**2 * NODE_LOCAL_FAR_SHARE * eta / (np.pi * (cap**2 - _BACKBONE**2))
    assert far == pytest.approx(3 * 40.0 / (np.pi * target)) and far > d
    # No cap above the backbone: no grading-free distance beyond the join.
    d, far = node_graded_halos_eV(40.0, _BACKBONE, _BACKBONE, eta, rho, _BACKBONE)
    assert far == d


def test_graded_gap_steps_grow_from_both_ends_up_to_the_cap():
    lo, hi = 1000.0, 9000.0
    nodes = np.r_[lo, _graded_gap(lo, hi, _BACKBONE, 0.1, 10.0), hi]
    steps = np.diff(nodes)
    assert steps[0] == pytest.approx(_BACKBONE) and steps[-1] == pytest.approx(_BACKBONE)
    near = np.minimum(nodes[:-1] - lo, hi - nodes[1:])
    assert np.all(steps <= np.minimum(10.0, _BACKBONE + 0.1 * near) + 1e-9)
    assert steps.max() == pytest.approx(10.0)
    short = np.diff(np.r_[0.0, _graded_gap(0.0, 5.0, _BACKBONE, 0.1, 10.0), 5.0])
    assert short.max() <= _BACKBONE and short.min() >= 0.5 * _BACKBONE


def test_protected_runs_keep_the_backbone():
    """DBR/Kossel spans and line halos passed as protected runs are never graded."""
    plan = build_window_plan(0.0, 100_000.0, _BACKBONE)
    grading = {"growth": 0.1, "cap_eV": 10.0, "protected_eV": [[40_000.0, 41_500.0]]}
    grid = graded_coordinates(plan, grading)
    inside = (grid[:-1] >= 40_000.0) & (grid[1:] <= 41_500.0)
    assert np.diff(grid)[inside].max() <= _BACKBONE + 1e-9
    assert np.diff(grid).max() == pytest.approx(10.0)
    assert np.array_equal(graded_coordinates(plan, None), plan.coordinates())


def test_graded_axis_kept_only_where_it_saves_nodes():
    """On a narrow band the graded halos cost more than they save: the #1257 axis stays."""
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 25_000, stop=12_000.0)
    grid, record = cxr_grid._kinematic_cxr_grid(payload, _CASE, (None,), 11.6, _Evidence(_LINES))
    local = record["local_axis"]
    assert local["grading"] is None and grid.size == local["ungraded_points"]
    assert np.diff(grid).max() <= _BACKBONE * (1 + 1e-12)


@pytest.mark.parametrize("extra", [{"channeling": {}}, {"transition_radiation": {}}])
def test_unmeasured_spectra_keep_the_ungraded_backbone(extra):
    """Channeling and TR spectra on the line axis lie outside the join bound (#1271)."""
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 600_000)
    case = {**_CASE, **extra}
    grid, record = cxr_grid._kinematic_cxr_grid(payload, case, (None,), 11.6, _Evidence(_LINES))
    assert record["local_axis"]["grading"] is None
    assert np.diff(grid).max() <= _BACKBONE * (1 + 1e-12)


def test_cache_key_separates_graded_and_ungraded_cases():
    """A channeling twin must not reuse a line-only case's graded axis (#1271)."""
    payload = _payload(KINEMATIC_BANDWIDTH_POLICY, 600_000)
    plain, _ = cxr_grid._policy_cache_key(payload, _CASE, (None,), observation=False)
    channeling, _ = cxr_grid._policy_cache_key(
        payload, {**_CASE, "channeling": {}}, (None,), observation=False
    )
    assert plain != channeling
