"""Photon-source depth measurement producer; arithmetic, guard and refusals (#884).

These are synthetic numerical-contract tests, NOT physical calibration.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_depth_budget import (
    require_axial_photon_depth_budget,
    sample_budgeted_physical_handoff,
)
from pyrite.montecarlo.channeling.axial_depth_measurements import (
    AxialPhotonPairedProbe,
    AxialPhotonSourceIntervalMeasurement,
    compile_axial_photon_depth_evidence,
    source_interval_bounds,
)

CASE = "si-001/5MeV/tds-electronic/cut95/model1"
OBS = "photons-75-90eV/20-25deg/s-pol/partition1"
MODEL = "axial-photon-source-v1"
REFERENCE = (1.0, 0.5)


def probe(name, difference=(0.0, 0.0), stat=(0.0, 0.0), drift=(0.0, 0.0)):
    return AxialPhotonPairedProbe(
        candidate_yield_bins=tuple(r + d for r, d in zip(REFERENCE, difference, strict=True)),
        difference_stat_error_upper_bins=stat,
        difference_depth_drift_upper_bins=drift,
        evidence_id=f"paired-{name}",
    )


def measurement(start=100.0, stop=200.0, **changes):
    values = dict(
        start_depth_ang=start,
        stop_depth_ang=stop,
        reference_yield_bins=REFERENCE,
        reference_stat_error_upper_bins=(0.01, 0.0),
        reference_depth_drift_upper_bins=(0.01, 0.0),
        return_flux=probe("return", (.001, -.001), (.0001, 0), (.0001, 0)),
        unresolved=probe("unresolved", (.001, 0)),
        matching_surface=probe("matching", (.0005, 0)),
        grid_supercell=probe("grid", (.0007, 0)),
        source_model=probe("source", (.0008, 0)),
        evidence_id=f"interval-{start:g}-{stop:g}",
        simultaneous_confidence=0.99,
    )
    values.update(changes)
    return AxialPhotonSourceIntervalMeasurement(**values)


def compile_rows(*rows):
    return compile_axial_photon_depth_evidence(
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        evidence_id="combined-source-2026",
        measurements=rows,
    )


def require(ev, depth=200.0, **extra):
    return require_axial_photon_depth_budget(
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        requested_depth_ang=depth,
        evidence=ev,
        **extra,
    )


def test_compiles_absolute_source_errors_without_spectral_cancellation():
    bound = source_interval_bounds(measurement())
    assert bound.reference_lower == pytest.approx(1.48)
    # The first paired probe changes +.001 in bin A and -.001 in bin B.
    # Its *signed* integral is zero, but its conservative L1 error is .0022.
    assert bound.return_upper == pytest.approx(.0022)
    assert bound.unresolved_upper == pytest.approx(.001)
    assert bound.grid_supercell_upper == pytest.approx(.0012)  # includes matching
    assert bound.source_model_upper == pytest.approx(.0008)
    assert bound.absolute_error_upper == pytest.approx(.0052)
    assert bound.relative_error_upper == pytest.approx(.0052 / 1.48)
    assert len(bound.evidence_ids) == 6
    assert require(compile_rows(measurement())) == bound


def test_reference_lower_not_central_estimate_or_population_probability():
    row = measurement(
        reference_yield_bins=(.01, .001),
        reference_stat_error_upper_bins=(.001, .0001),
        reference_depth_drift_upper_bins=(.001, .0001),
    )
    # Paired probes must share the new reference. Otherwise a spurious large
    # difference correctly fails the strict 1% source guard.
    ev = compile_rows(row)
    assert ev.intervals[0].reference_lower == pytest.approx(.0088)
    with pytest.raises(ValueError, match="photon-source error"):
        require(ev)
    assert ev.intervals[0].relative_error_upper > .01


def test_uses_union_bound_for_confidence_across_intervals():
    ev = compile_rows(measurement(), measurement(200.0, 300.0))
    assert ev.simultaneous_confidence == pytest.approx(.98)
    assert require(ev, 200.0) is ev.intervals[0]
    assert require(ev, 200.01) is ev.intervals[1]
    with pytest.raises(ValueError, match="joint confidence"):
        require(ev, min_confidence=.985)
    with pytest.raises(ValueError, match="outside certified"):
        require(ev, 350.0)
    with pytest.raises(ValueError, match="outside certified"):
        require(ev, 100.0)
    with pytest.raises(ValueError, match="positive"):
        require(ev, 0.0)


def test_gaps_and_independent_overbudget_intervals_remain_fail_closed():
    weak = measurement(300, 400, source_model=probe("source", (.1, 0)))
    ev = compile_rows(measurement(), weak)
    assert require(ev, 150) is ev.intervals[0]
    with pytest.raises(ValueError, match="outside certified"):
        require(ev, 250)
    with pytest.raises(ValueError, match="photon-source error"):
        require(ev, 350)
    with pytest.raises(ValueError, match="nonoverlapping"):
        compile_rows(measurement(), measurement(180, 210))
    with pytest.raises(ValueError, match="nonoverlapping"):
        compile_rows(measurement(200, 300), measurement())


def test_rejects_invalid_inputs_and_zero_reference_lower():
    with pytest.raises(ValueError, match="no source-depth measurements"):
        compile_rows()
    with pytest.raises(ValueError, match="positive depth"):
        source_interval_bounds(measurement(0, 100))
    with pytest.raises(ValueError, match="positive lower"):
        source_interval_bounds(measurement(reference_stat_error_upper_bins=(2, 0)))
    with pytest.raises(ValueError, match="nonnegative"):
        source_interval_bounds(measurement(reference_yield_bins=(-1, 0)))
    with pytest.raises(ValueError, match="numeric"):
        source_interval_bounds(measurement(reference_yield_bins=(1 + 2j, 0)))
    with pytest.raises(ValueError, match="nonfinite"):
        source_interval_bounds(measurement(reference_yield_bins=(np.nan, .1)))
    with pytest.raises(ValueError, match="shape mismatch"):
        source_interval_bounds(measurement(unresolved=probe("unresolved", stat=(0, 0, 0))))
    with pytest.raises(ValueError, match="nonnegative"):
        source_interval_bounds(measurement(return_flux=probe("return", drift=(-.1, 0))))
    with pytest.raises(ValueError, match="evidence IDs must be distinct"):
        source_interval_bounds(
            measurement(source_model=replace(probe("source"), evidence_id="paired-return"))
        )
    with pytest.raises(TypeError, match="AxialPhotonPairedProbe"):
        source_interval_bounds(measurement(return_flux=None))
    with pytest.raises(ValueError, match="simultaneous_confidence"):
        compile_rows(measurement(simultaneous_confidence=1.0))
    with pytest.raises(ValueError, match="joint confidence"):
        compile_rows(
            measurement(simultaneous_confidence=.1),
            measurement(200, 300, simultaneous_confidence=.1),
        )


def test_preflight_consumes_compiled_evidence_before_any_sampling(monkeypatch):
    from pyrite.montecarlo.channeling import axial_outgoing_launch

    calls = []

    def fake_sampler(*args, **kwargs):
        calls.append((args, kwargs))
        return ["sampled"]

    monkeypatch.setattr(axial_outgoing_launch, "sample_physical_handoff", fake_sampler)
    inputs = dict(
        channels=None,
        entry_orders=None,
        evidence=compile_rows(measurement()),
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        hbar_v_eV_ang=1.0,
        window=None,
        max_depth_ang=300.0,
        incidents=(),
        zone=None,
        matching=None,
    )
    with pytest.raises(ValueError, match="outside certified"):
        sample_budgeted_physical_handoff(**inputs)
    assert calls == []
    inputs["max_depth_ang"] = 200.0
    assert sample_budgeted_physical_handoff(**inputs) == ["sampled"]
    assert len(calls) == 1
    assert calls[0][1]["max_depth_ang"] == 200.0
