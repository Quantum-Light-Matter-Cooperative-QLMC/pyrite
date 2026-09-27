"""Measured ``resonance-population`` line bandwidth (issue #192).

Pins the tail bounds the edge is chosen from, the opt-in policy payload and its
refusals, case construction where the ceiling alone overflows the budget, and
the production-weight truncation audit.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import xp
from pyrite._line_grid_policy import (
    AUTOMATIC_BANDWIDTH_POLICY,
    DEFAULT_BANDWIDTH_TRUNCATION,
    LINE_GRID_POLICY_SCHEMA,
    RESONANCE_BANDWIDTH_POLICY,
    RESONANCE_LINE_GRID_POLICY_SCHEMA,
    LineGridToleranceError,
    resolve_line_grid_policy,
    resolved_coordinates,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo.runner.line_grid import check_line_truncation, line_truncation_audit
from pyrite.montecarlo.spectrum.line_seeds import (
    ResonancePopulation,
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
