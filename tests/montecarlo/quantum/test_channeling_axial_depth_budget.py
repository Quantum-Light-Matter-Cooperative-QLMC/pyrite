"""Source-normalized, fail-closed research depth-budget contracts (#884).

Validation: cr-axial-depth-budget
"""

import subprocess
import sys
from dataclasses import replace

import pytest

from pyrite.montecarlo.channeling.axial_depth_budget import (
    AxialPhotonBoundInterval,
    AxialPhotonDepthEvidence,
    require_axial_photon_depth_budget,
    sample_budgeted_physical_handoff,
)

CASE = "si-001/5MeV/tds-electronic/2x2-32grid/95ev-cut/solver-v1"
OBS = "photon/energy-70-90ev/polar-20-25deg/s-pol/partition-v1"
MODEL = "research-source-v1"


def interval(start=0.0, stop=1000.0, **changes):
    args = dict(
        start_depth_ang=start,
        stop_depth_ang=stop,
        reference_lower=0.08,
        return_upper=0.0001,
        unresolved_upper=0.0002,
        grid_supercell_upper=0.0001,
        source_model_upper=0.0001,
        evidence_ids=("paired-source-001", "source-envelope-002"),
    )
    args.update(changes)
    return AxialPhotonBoundInterval(**args)


def evidence(*intervals):
    return AxialPhotonDepthEvidence(
        CASE, OBS, MODEL, "joint-source-measurement-2026", 0.99, tuple(intervals)
    )


def require(ev, depth=1000.0, **changes):
    args = dict(
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        requested_depth_ang=depth,
        evidence=ev,
    )
    args.update(changes)
    return require_axial_photon_depth_budget(**args)


def test_source_normalization_uses_lower_reference_yield_and_additive_errors():
    part = interval()
    assert part.absolute_error_upper == pytest.approx(0.0005)
    assert part.relative_error_upper == pytest.approx(0.00625)
    assert require(evidence(part)) is part
    # An electron *population* error of 0.5% is not interchangeable with the
    # source error. With small photon yield, the same absolute photon error
    # produces a much larger relative change.
    weaker_source = replace(part, reference_lower=0.01)
    with pytest.raises(ValueError, match="photon-source error"):
        require(evidence(weaker_source))
    # A source >1 photon/incident does not imply a fraction upper bound.
    strong = interval(reference_lower=3.0, return_upper=0.01)
    assert require(evidence(strong)).relative_error_upper < 0.01


def test_depth_intervals_are_uniform_and_are_not_interpolated_or_extrapolated():
    ev = evidence(interval(200.0, 500.0), interval(700.0, 900.0))
    for depth in (200.0, 501.0, 600.0, 700.0, 901.0, 1000.0, 0.0):
        with pytest.raises(ValueError, match="outside certified|positive"):
            require(ev, depth)
    assert require(ev, 500.0) is ev.intervals[0]
    assert require(ev, 900.0) is ev.intervals[1]
    # Adjacent intervals are legal but not extrapolated past their ends.
    adjacent = evidence(interval(0.0, 500.0), interval(500.0, 1000.0))
    assert require(adjacent, 500.0) is adjacent.intervals[0]
    assert require(adjacent, 500.00001) is adjacent.intervals[1]
    with pytest.raises(ValueError, match="nonoverlapping"):
        evidence(interval(0, 800), interval(700, 1000))
    with pytest.raises(ValueError, match="nonoverlapping"):
        evidence(interval(700, 1000), interval(0, 500))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"evidence": None}, "no matching"),
        ({"case_key": "si-111"}, "no matching"),
        ({"observable_key": "different-energy-band"}, "no matching"),
        ({"model_id": "different-source-model"}, "no matching"),
        ({"min_confidence": 0.999}, "joint confidence"),
        ({"max_relative_error": 0.001}, "photon-source error"),
        ({"requested_depth_ang": float("nan")}, "finite and positive"),
        ({"requested_depth_ang": float("inf")}, "finite and positive"),
        ({"max_relative_error": 1.0}, "max_relative_error"),
        ({"min_confidence": 0.0}, "min_confidence"),
    ],
)
def test_preflight_fails_closed_on_missing_incompatible_or_uncertain_data(change, message):
    with pytest.raises(ValueError, match=message):
        require(evidence(interval()), **change)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"stop_depth_ang": 0.0}, "greater"),
        ({"start_depth_ang": -1.0}, "start_depth_ang"),
        ({"reference_lower": 0.0}, "positive reference_lower"),
        ({"return_upper": float("nan")}, "return_upper"),
        ({"unresolved_upper": -0.01}, "unresolved_upper"),
        ({"grid_supercell_upper": float("inf")}, "grid_supercell_upper"),
        ({"source_model_upper": -1.0}, "source_model_upper"),
        ({"evidence_ids": ()}, "evidence_ids"),
        ({"evidence_ids": ("same", "same")}, "duplicate evidence_ids"),
    ],
)
def test_interval_schema_rejects_nonconservative_measurements(changes, message):
    with pytest.raises(ValueError, match=message):
        interval(**changes)


def test_evidence_schema_rejects_wrong_types_and_bad_confidence():
    good = evidence(interval())
    with pytest.raises(ValueError, match="simultaneous_confidence"):
        replace(good, simultaneous_confidence=1.0)
    with pytest.raises(ValueError, match="at least one"):
        replace(good, intervals=())
    with pytest.raises(TypeError, match="intervals must contain"):
        replace(good, intervals=("not-an-interval",))
    with pytest.raises(ValueError, match="observable_key"):
        replace(good, observable_key=" ")


def test_budgeted_handoff_refuses_before_loading_solver_or_writing(monkeypatch):
    # No real channels/arrays/zone are constructed. If preflight were moved
    # behind solver setup this test would fail on None or an unexpected call.
    from pyrite.montecarlo.channeling import axial_outgoing_launch

    called = []

    def fake_handoff(*args, **kwargs):
        called.append((args, kwargs))
        return ["sampled"]

    monkeypatch.setattr(axial_outgoing_launch, "sample_physical_handoff", fake_handoff)
    args = dict(
        channels=None,
        entry_orders=None,
        evidence=None,
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        hbar_v_eV_ang=1.0,
        window=None,
        max_depth_ang=1500.0,
        incidents=(),
        zone=None,
        matching=None,
    )
    with pytest.raises(ValueError, match="no matching"):
        sample_budgeted_physical_handoff(**args)
    assert called == []
    args["evidence"] = evidence(interval())
    with pytest.raises(ValueError, match="outside certified"):
        sample_budgeted_physical_handoff(**args)
    assert called == []
    args["max_depth_ang"] = 1000.0
    assert sample_budgeted_physical_handoff(**args) == ["sampled"]
    assert len(called) == 1
    assert called[0][1]["max_depth_ang"] == 1000.0


def test_error_exactly_at_tolerance_is_accepted_and_just_above_is_refused():
    part = interval(reference_lower=1.0, return_upper=0.25, unresolved_upper=0.0)
    part = replace(part, grid_supercell_upper=0.0, source_model_upper=0.0)
    exact = evidence(part)
    assert require(exact, max_relative_error=0.25) is part
    with pytest.raises(ValueError, match="photon-source error"):
        require(exact, max_relative_error=0.2499999)


def test_guard_refuses_in_a_fresh_process_before_solver_modules_load():
    program = """
import sys
from pyrite.montecarlo.channeling.axial_depth_budget import sample_budgeted_physical_handoff
try:
    sample_budgeted_physical_handoff(
        None, None, evidence=None, case_key="c", observable_key="o", model_id="m",
        hbar_v_eV_ang=1.0, window=None, max_depth_ang=10.0, incidents=(), zone=None, matching=None,
    )
except ValueError as error:
    assert "no matching" in str(error)
else:
    raise SystemExit("guard did not refuse")
base = "pyrite.montecarlo.channeling."
for name in ("axial_outgoing_launch", "axial_fiber_unraveling"):
    assert base + name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", program], check=True)
