"""Measured ``resonance-population`` line bandwidth (issue #192).

Pins the tail bounds the edge is chosen from, the opt-in policy payload and its
refusals, case construction where the ceiling alone overflows the budget, and
the production-weight truncation audit.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import _to_cpu, xp
from pyrite._line_grid_policy import (
    AUTOMATIC_BANDWIDTH_POLICY,
    DEFAULT_BANDWIDTH_TRUNCATION,
    LINE_GRID_POLICY_SCHEMA,
    RESONANCE_BANDWIDTH_POLICY,
    RESONANCE_LINE_GRID_POLICY_SCHEMA,
    LineGridToleranceError,
    LineGridTruncationWarning,
    LineYieldStatisticsWarning,
    resolve_line_grid_policy,
    resolved_coordinates,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.energy_grid.bandwidth_check import reference_axis
from pyrite.montecarlo.runner.line_grid import (
    LINE_YIELD_RELATIVE_SE_LIMIT,
    _measured_line_grid,
    check_line_truncation,
    line_truncation_audit,
    line_yield_statistics,
)
from pyrite.montecarlo.spectrum.line_seeds import (
    ResonancePopulation,
    case_line_stop_eV,
    characteristic_stop_eV,
    resonance_population_stop_eV,
    sincsq_upper_tail_bound,
)
from pyrite.montecarlo.spectrum.lines._bin_quadrature import sincsq_window_mass
from pyrite.montecarlo.spectrum.lines._kernels import _accumulate_edge_truncation

_RESONANCE = {"bandwidth": RESONANCE_BANDWIDTH_POLICY}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "PYRITE_ENERGY_GRID_RTOL",
        "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE",
        "PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS",
        "PYRITE_ENERGY_GRID_MAX_SPACING_EV",
        "PYRITE_ENERGY_GRID_ULPS",
        "PYRITE_ENERGY_GRID_MAX_POINTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sincsq_tail_bound_dominates_exact_tail():
    rng = np.random.default_rng(192)
    width = rng.uniform(0.1, 50.0, 400)
    distance = rng.uniform(0.01, 5000.0, 400)
    a_width = np.pi / width
    # Upper edge at E_res + distance: shift the resonance below a fixed edge.
    edges = np.array([-1e12, 0.0])
    _captured, truncated = sincsq_window_mass(a_width, -distance, edges)
    exact = truncated / width
    bound = sincsq_upper_tail_bound(width, distance)
    assert np.all(bound >= exact * (1.0 - 1e-12))
    # sin**2 averages 1/2, so the far-tail bound is twice the exact tail.
    far = distance > 100.0 * width
    assert np.allclose(bound[far] / exact[far], 2.0, rtol=2e-2)


def test_resonance_above_edge_counts_wholly():
    assert sincsq_upper_tail_bound(1.0, -3.0) == 1.0
    assert sincsq_upper_tail_bound(1.0, 0.0) == 1.0


def _population(energy, width, weight=None):
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    width = np.broadcast_to(np.asarray(width, dtype=float), energy.shape).copy()
    weight = np.ones_like(energy) if weight is None else np.asarray(weight, dtype=float)
    return ResonancePopulation("(0 0 2)", energy, weight, width)


def test_single_flight_stop_is_its_closed_form_limit():
    limit = 1e-4
    stop, summary = resonance_population_stop_eV(
        [_population(7000.0, 2.0)], ceiling_eV=1.0e6, truncation_limit=limit, round_to_eV=1.0
    )
    expected = 7000.0 + 2.0 / (np.pi**2 * limit)
    assert expected <= stop <= expected + 1.0
    assert summary["proxy_truncated_fraction"] <= limit
    assert not summary["capped"]


def test_stop_never_exceeds_ceiling_and_empty_population_has_none():
    stop, summary = resonance_population_stop_eV(
        [_population(7000.0, 2.0)], ceiling_eV=8000.0, truncation_limit=1e-4
    )
    assert stop == 8000.0 and summary["capped"]
    stop, summary = resonance_population_stop_eV(
        [_population([], 1.0)], ceiling_eV=8000.0, truncation_limit=1e-4
    )
    assert stop is None and summary["n_lines"] == 0


def test_weighted_stop_lets_light_wide_lines_through():
    heavy = _population(5000.0, 1.0, weight=[1.0])
    light = _population(5000.0, 1000.0, weight=[1e-12])
    alone, _ = resonance_population_stop_eV([heavy], ceiling_eV=1e7, truncation_limit=1e-4)
    mixed, _ = resonance_population_stop_eV([heavy, light], ceiling_eV=1e7, truncation_limit=1e-4)
    assert mixed == pytest.approx(alone, abs=100.0)


def test_characteristic_stop_bounds_each_lorentzian_tail():
    limit = 1e-4
    stop, summary = characteristic_stop_eV([[("Mo", 1.0), ("S", 2.0)]], truncation_limit=limit)
    assert summary["line"].startswith("Mo ")
    # Lorentzian: 1/2 - arctan(2 D / Gamma) / pi <= Gamma / (2 pi D) = limit.
    assert stop > 17_000.0


def test_default_policy_payload_is_unchanged():
    policy = resolve_line_grid_policy(start_eV=50.0, stop_eV=1000.0)
    payload = policy.payload()
    assert payload["schema"] == LINE_GRID_POLICY_SCHEMA
    assert payload["bandwidth"] == {
        "policy": AUTOMATIC_BANDWIDTH_POLICY,
        "start_eV": 50.0,
        "stop_eV": 1000.0,
    }
    assert "bandwidth" not in payload["sources"]


def test_resonance_policy_payload_carries_its_share():
    payload = resolve_line_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call=_RESONANCE).payload()
    assert payload["schema"] == RESONANCE_LINE_GRID_POLICY_SCHEMA
    assert payload["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    assert payload["bandwidth"]["truncation_limit"] == DEFAULT_BANDWIDTH_TRUNCATION
    assert payload["sources"]["bandwidth"] == "per-call"


def test_resonance_policy_refuses_windows_and_unknown_names():
    with pytest.raises(ValueError, match="cannot be combined with windows"):
        resolve_line_grid_policy(
            start_eV=50.0, stop_eV=1000.0, per_call={**_RESONANCE, "windows": True}
        )
    with pytest.raises(ValueError, match="bandwidth must be one of"):
        resolve_line_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call={"bandwidth": "log"})


def test_measured_stop_must_lie_under_the_cap():
    payload = resolve_line_grid_policy(start_eV=50.0, stop_eV=1000.0, per_call=_RESONANCE).payload()
    grid, record = resolved_coordinates(payload, 1.0, dtype=np.float64, stop_eV=500.0)
    assert grid[-1] == 500.0 and record["stop_eV"] == 500.0
    with pytest.raises(ValueError, match="outside"):
        resolved_coordinates(payload, 1.0, dtype=np.float64, stop_eV=2000.0)


def test_high_energy_hbn_builds_under_measured_bandwidth(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_MAX_POINTS", "200000")
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=5000.0), line_grid_policy=_RESONANCE)
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    bandwidth = case["line_grid_policy"]["bandwidth"]
    assert bandwidth["policy"] == RESONANCE_BANDWIDTH_POLICY
    assert bandwidth["stop_eV"] == 1_720_800.0


def _audited_case(limit=1e-4):
    return {
        "name": "hbn test",
        "E0_keV": 5000.0,
        "line_grid_policy": {
            "bandwidth": {
                "policy": RESONANCE_BANDWIDTH_POLICY,
                "start_eV": 50.0,
                "stop_eV": 1e6,
                "truncation_limit": limit,
            }
        },
    }


def test_audit_is_only_built_for_the_measured_bandwidth():
    assert line_truncation_audit({"line_grid_policy": None}, np.arange(10.0)) is None
    default = {"line_grid_policy": {"bandwidth": {"policy": AUTOMATIC_BANDWIDTH_POLICY}}}
    assert line_truncation_audit(default, np.arange(10.0)) is None
    audit = line_truncation_audit(_audited_case(), np.linspace(50.0, 1000.0, 5))
    assert audit == {"start_eV": 50.0, "stop_eV": 1000.0}


def test_audit_accumulates_and_gates_the_upper_edge():
    case = _audited_case()
    audit = {"start_eV": 50.0, "stop_eV": 1000.0}
    # One narrow line well inside, one wide line whose tail crosses the edge.
    E_r = xp.asarray([500.0, 900.0])
    a_width = xp.asarray([np.pi / 1.0, np.pi / 100.0])
    weight = xp.asarray([1.0, 1.0])
    _accumulate_edge_truncation(audit, E_r, a_width, weight)
    record_mass = float(audit["line_mass"])
    assert record_mass == pytest.approx(1.0 + 100.0)
    expected_above = (
        1.0 / (np.pi * np.pi / 1.0 * 500.0) * 1.0
        + min(1.0, 1.0 / (np.pi * np.pi / 100.0 * 100.0)) * 100.0
    )
    assert float(audit["mass_above"]) == pytest.approx(expected_above)
    with pytest.raises(LineGridToleranceError, match="above the 0.0001 bandwidth share"):
        check_line_truncation(case, audit)
    narrow = {"start_eV": 50.0, "stop_eV": 1000.0}
    _accumulate_edge_truncation(narrow, E_r[:1], xp.asarray([np.pi / 0.01]), weight[:1])
    record = check_line_truncation(case, narrow)
    assert record["upper_fraction_bound"] <= 1e-4
    assert record["lower_fraction_bound"] > 0.0


def test_audit_at_the_kinematic_ceiling_warns_instead_of_refusing():
    """#192: at 100 keV the measured stop caps at the closed-form ceiling; tails
    above it are what the automatic bandwidth also drops, so this policy warns."""
    case = _audited_case()
    case["line_grid_policy"]["bandwidth"]["stop_eV"] = 1000.0
    audit = {"start_eV": 50.0, "stop_eV": 1000.0}
    _accumulate_edge_truncation(
        audit, xp.asarray([900.0]), xp.asarray([np.pi / 100.0]), xp.asarray([1.0])
    )
    with pytest.warns(LineGridTruncationWarning, match="kinematic ceiling"):
        record = check_line_truncation(case, audit)
    assert record["capped_at_ceiling"]
    assert record["upper_fraction_bound"] > 1e-4


def test_audit_sums_line_mass_per_electron_when_counted():
    audit = line_truncation_audit(_audited_case(), np.linspace(50.0, 1000.0, 5), n_electrons=3)
    assert audit["n_electrons"] == 3
    E_r = xp.asarray([500.0, 600.0, 700.0])
    a_width = xp.asarray([np.pi, np.pi, np.pi / 2.0])
    _accumulate_edge_truncation(audit, E_r, a_width, xp.ones(3), xp.asarray([0, 2, 0]))
    np.testing.assert_allclose(_to_cpu(audit["electron_mass"]), [3.0, 0.0, 1.0])
    assert float(audit["line_mass"]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="outside the 3 line electrons"):
        _accumulate_edge_truncation(audit, E_r[:1], a_width[:1], xp.ones(1), xp.asarray([3]))


def test_line_yield_statistics_is_the_relative_standard_error_of_the_mean():
    mass = np.array([1.0, 2.0, 3.0, 0.0])
    stats = line_yield_statistics(mass)
    expected = mass.std(ddof=1) / np.sqrt(mass.size) / mass.mean()
    assert stats["relative_se"] == pytest.approx(expected)
    assert stats["max_electron_share"] == pytest.approx(0.5)
    assert line_yield_statistics(np.zeros(4))["relative_se"] is None


def test_audit_warns_when_a_rare_electron_carries_the_line_yield():
    """#201: a statistics-limited yield is flagged and warned about, not refused."""
    case = _audited_case()
    even = {"start_eV": 50.0, "stop_eV": 1000.0, "n_electrons": 400}
    rare = dict(even)
    E_r = xp.full(400, 500.0)
    a_width = xp.full(400, np.pi / 0.01)
    ids = xp.arange(400)
    _accumulate_edge_truncation(even, E_r, a_width, xp.ones(400), ids)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LineYieldStatisticsWarning)
        record = check_line_truncation(case, even)
    stats = record["line_yield_statistics"]
    assert stats["relative_se"] < LINE_YIELD_RELATIVE_SE_LIMIT
    assert not stats["statistics_limited"]
    weights = xp.asarray(np.r_[1.0e3, np.ones(399)])
    _accumulate_edge_truncation(rare, E_r, a_width, weights, ids)
    with pytest.warns(LineYieldStatisticsWarning, match="relative standard error"):
        record = check_line_truncation(case, rare)
    stats = record["line_yield_statistics"]
    assert stats["statistics_limited"]
    assert stats["max_electron_share"] == pytest.approx(1.0e3 / 1399.0)


def test_production_line_collection_retains_resonance_width_and_weight():
    audit = {"start_eV": 50.0, "stop_eV": 2000.0, "collect": []}
    _accumulate_edge_truncation(
        audit,
        xp.asarray([500.0, 900.0]),
        xp.asarray([np.pi, np.pi / 10]),
        xp.asarray([2.0, 0.5]),
    )
    energy, width, weight = audit["collect"][0]
    np.testing.assert_allclose(energy, [500.0, 900.0])
    np.testing.assert_allclose(width, [1.0, 10.0])
    np.testing.assert_allclose(weight, [2.0, 0.5])
    assert "line_mass" not in audit


def test_measured_grid_uses_collected_production_weights(monkeypatch):
    from pyrite.montecarlo import runner

    policy = resolve_line_grid_policy(
        start_eV=50.0,
        stop_eV=100_000.0,
        per_call={"bandwidth": RESONANCE_BANDWIDTH_POLICY},
    ).payload()
    case = {"composition": [("B", 0.5), ("N", 0.5)]}
    collected = (
        np.array([500.0, 900.0]),
        np.array([1.0, 100.0]),
        np.array([100.0, 1.0]),
    )

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        np.testing.assert_array_equal(axis, [50.0, 100_000.0])
        kwargs["truncation_audit"]["collect"].append(collected)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_lines_for_segments", collect)
    grid, _record, bandwidth = _measured_line_grid(
        policy, case, {}, (np.array([0.0, 0.0, 1.0]),), 2, 1.0, None, None
    )
    expected, _ = case_line_stop_eV(
        case,
        [ResonancePopulation("production lines", collected[0], collected[2], collected[1])],
        start_eV=50.0,
        ceiling_eV=100_000.0,
        truncation_limit=1e-4,
        proxy_safety=2.0,
    )
    assert grid[-1] == expected == bandwidth["stop_eV"]


def test_small_hbn_case_collects_production_lines_before_resolution():
    from pyrite.montecarlo.runner import _transport_case

    sweep = material_sweep("hbn")
    sweep = replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=5000.0),
        line_grid_policy=_RESONANCE,
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    result = _transport_case(case, transport_core="lockstep")
    record = result["diagnostic_grid"]
    assert record["measured_bandwidth"]["kinematic"]["n_lines"] > 0
    assert result["E_grid"][-1] < case["line_grid_policy"]["bandwidth"]["stop_eV"]


def test_local_grid_reference_keeps_sinc_spacing_to_ceiling():
    local = np.array([50.0, 50.5, 51.0, 54.0, 57.0])
    reference = reference_axis(local, 100.0, spacing_eV=0.5)
    assert reference[0] == 50.0
    assert reference[-1] == 100.0
    np.testing.assert_allclose(np.diff(reference), 0.5)


def test_light_far_line_spends_the_share_instead_of_widening_the_axis():
    main = _population(5000.0, 1.0, weight=[1.0])
    stray = _population(60_000.0, 1.0, weight=[1e-9])
    alone, _ = resonance_population_stop_eV([main], ceiling_eV=1e6, truncation_limit=1e-5)
    both, summary = resonance_population_stop_eV(
        [main, stray], ceiling_eV=1e6, truncation_limit=1e-5
    )
    assert both < 60_000.0
    assert both == pytest.approx(alone, abs=200.0)
    assert summary["proxy_truncated_fraction"] <= 1e-5
