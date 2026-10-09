"""Tests for config.gate_cases_by_penetration -- the pre-run penetration
watchdog that drops thickness values a beam energy has already died in.
These tests mock pyrite.campaign.config.simulate_trajectories so they exercise ONLY
the gating logic (grouping, cutoff detection, early-exit, reference-case
selection), not real electron transport; test_gate_cases_by_penetration_
drops_with_real_transport at the bottom of this file covers real physics.
"""

import math

import pytest

from pyrite.campaign import config
from pyrite.montecarlo import runner
from pyrite.montecarlo.transport import TransportStepLimitError


def _case(E0_keV, thickness_ang, tilt_deg=0.0, composition="C", abs_layers=None):
    return {
        "E0_keV": E0_keV,
        "thickness_ang": thickness_ang,
        "tilt_deg": tilt_deg,
        "composition": composition,
        "abs_layers": abs_layers,
    }


def test_gate_cases_by_penetration_keeps_dying_case_drops_thicker_ones(monkeypatch):
    calls = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, **kwargs):
        calls.append((E0_keV, thickness_ang))
        fraction = 1.0 if thickness_ang <= 200.0 else (0.03 if thickness_ang <= 400.0 else 0.01)
        return {"n_transmitted": int(round(fraction * Ne))}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [
        _case(30.0, t, tilt_deg=tilt) for t in (100.0, 200.0, 400.0, 800.0) for tilt in (0.0, 30.0)
    ]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=100, seed=0)

    assert sorted({c["thickness_ang"] for c in kept}) == [100.0, 200.0, 400.0]
    assert {c["thickness_ang"] for c in dropped} == {800.0}
    # 800.0 was never simulated -- it was inferred dead once 400.0 crossed the floor
    assert calls == [(30.0, 100.0), (30.0, 200.0), (30.0, 400.0)]


def test_gate_cases_by_penetration_is_independent_per_energy(monkeypatch):
    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, **kwargs):
        fraction = 0.01 if (E0_keV == 30.0 and thickness_ang >= 200.0) else 1.0
        return {"n_transmitted": int(round(fraction * Ne))}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [_case(E0, t) for E0 in (30.0, 300.0) for t in (100.0, 200.0, 400.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=50, seed=0)

    assert {c["thickness_ang"] for c in kept if c["E0_keV"] == 30.0} == {100.0, 200.0}
    assert {c["thickness_ang"] for c in kept if c["E0_keV"] == 300.0} == {100.0, 200.0, 400.0}
    assert {c["thickness_ang"] for c in dropped} == {400.0}


def test_gate_cases_by_penetration_keeps_everything_when_nothing_dies(monkeypatch):
    monkeypatch.setattr(
        config,
        "simulate_trajectories",
        lambda E0_keV, Ne, thickness_ang, **kwargs: {"n_transmitted": Ne},
    )

    cases = [_case(30.0, t) for t in (100.0, 200.0, 400.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert kept == cases
    assert dropped == []


def test_gate_cases_by_penetration_uses_normal_incidence_reference(monkeypatch):
    seen_compositions = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, composition=None, **kwargs):
        seen_compositions.append(composition)
        return {"n_transmitted": Ne}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    cases = [
        _case(30.0, 100.0, tilt_deg=45.0, composition="wrong"),
        _case(30.0, 100.0, tilt_deg=0.0, composition="right"),
    ]
    config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert seen_compositions == ["right"]


def test_gate_cases_by_penetration_forwards_abs_layers_total_thickness(monkeypatch):
    # Film-on-substrate materials (e.g. mos2-on-sapphire) carry an
    # ``abs_layers`` stack; the watchdog must simulate through the FULL
    # stack (abs_layers[-1][1], the substrate's far face) rather than the
    # case's own (film-only) thickness_ang, and forward ``abs_layers``
    # verbatim as the ``layers`` kwarg.
    calls = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, layers=None, **kwargs):
        calls.append((E0_keV, thickness_ang, layers))
        return {"n_transmitted": Ne}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)

    abs_layers = [(0.0, 50.0, []), (50.0, 150.0, [])]
    cases = [_case(30.0, 50.0, abs_layers=abs_layers)]
    config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert calls == [(30.0, 150.0, abs_layers)]


@pytest.mark.slow
def test_gate_cases_by_penetration_drops_with_real_transport():
    # A real (unmocked) pyrite.montecarlo.simulate_trajectories call: a 15 keV
    # beam through a light element (carbon) should fully transmit at 10nm,
    # be fully absorbed well before 100um, and 200um (never checked) is
    # inferred dead too. If this fails, print `fraction` per thickness (see
    # the mocked tests above for the shape) and adjust thickness/energy --
    # do not weaken the assertion to pass regardless of physics.
    composition = [("C", 0.1136)]  # graphite-like number density, atoms/Ang^3
    cases = [_case(15.0, t, composition=composition) for t in (100.0, 1_000_000.0, 2_000_000.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=80, seed=0)

    assert 100.0 in {c["thickness_ang"] for c in kept}
    assert {c["thickness_ang"] for c in dropped} == {2_000_000.0}


def test_gate_cases_by_penetration_follows_the_case_elastic_model(monkeypatch):
    seen = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, **kwargs):
        seen.append(kwargs)
        return {"n_transmitted": Ne}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)
    monkeypatch.setattr(
        config,
        "_case_elastic_kwargs",
        lambda case: {"elastic_model": "elsepa", "elastic_tables": ["sentinel"]},
    )
    config.gate_cases_by_penetration([_case(30.0, 100.0)], Ne=10)

    assert seen[0]["elastic_model"] == "elsepa"
    assert seen[0]["elastic_tables"] == ["sentinel"]


def _step_limited(monkeypatch, fail_below):
    budgets = []

    def fake_simulate_trajectories(E0_keV, Ne, thickness_ang, *, max_steps, **kwargs):
        budgets.append(max_steps)
        if max_steps < fail_below:
            raise TransportStepLimitError(Ne, Ne, max_steps)
        return {"n_transmitted": Ne if thickness_ang <= 100.0 else 0}

    monkeypatch.setattr(config, "simulate_trajectories", fake_simulate_trajectories)
    return budgets


def test_gate_cases_by_penetration_retries_the_step_budget(monkeypatch):
    # Thick MeV cases outlive the default budget (#315); the gate doubles it
    # like the runner does instead of aborting the sweep.
    base = runner.TRANSPORT_MAX_STEPS
    budgets = _step_limited(monkeypatch, fail_below=2 * base)
    cases = [_case(5000.0, t) for t in (100.0, 200.0, 400.0)]
    kept, dropped = config.gate_cases_by_penetration(cases, floor=0.05, Ne=10, seed=0)

    assert {c["thickness_ang"] for c in kept} == {100.0, 200.0}
    assert {c["thickness_ang"] for c in dropped} == {400.0}
    assert budgets == [base, 2 * base, base, 2 * base]


def test_gate_cases_by_penetration_step_budget_stops_at_the_ceiling(monkeypatch):
    budgets = _step_limited(monkeypatch, fail_below=math.inf)
    with pytest.raises(TransportStepLimitError):
        config.gate_cases_by_penetration([_case(5000.0, 100.0)], Ne=10)
    assert budgets[-1] == runner.TRANSPORT_MAX_STEPS_CEILING
