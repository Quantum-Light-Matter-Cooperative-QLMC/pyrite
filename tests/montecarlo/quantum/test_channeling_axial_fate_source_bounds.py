"""Synthetic #883-fate-to-#884-photon bounds: numerical contracts only."""

from dataclasses import replace
from math import log, sqrt

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_depth_budget import (
    AxialPhotonBoundInterval,
    AxialPhotonDepthEvidence,
    require_axial_photon_depth_budget,
    sample_budgeted_physical_handoff,
)
from pyrite.montecarlo.channeling.axial_fate_source_bounds import (
    AxialFatePhotonResponseCaps,
    add_physical_fate_source_bounds,
    physical_fate_source_uppers,
)
from pyrite.montecarlo.channeling.axial_outgoing_launch import AxialHandoffRecord

CASE = "si-001/5MeV/tds-electronic/matching95/model-v1"
OBS = "photons/75-90eV/20-25deg/s-pol/native-partition-v1"
MODEL = "research-axial-photon-v1"


def caps(**changes):
    params = dict(
        case_key=CASE,
        observable_key=OBS,
        model_id=MODEL,
        evidence_id="physical-fate-response-proof",
        outgoing_return_error_upper_bins=(0.02, 0.01),
        unresolved_error_upper_bins=(0.10, 0.05),
        failure_probability_upper=0.002,
    )
    params.update(changes)
    return AxialFatePhotonResponseCaps(**params)


def source():
    intervals = tuple(
        AxialPhotonBoundInterval(
            start_depth_ang=a,
            stop_depth_ang=b,
            reference_lower=1.0,
            return_upper=0.0001,
            unresolved_upper=0.0001,
            grid_supercell_upper=0.0001,
            source_model_upper=0.0001,
            evidence_ids=(f"source-interval-{i}", f"paired-comparison-{i}"),
        )
        for i, (a, b) in enumerate(((100.0, 200.0), (200.0, 300.0)))
    )
    return AxialPhotonDepthEvidence(CASE, OBS, MODEL, "joint-source-study", 0.99, intervals)


def records(n=400):
    """One #883 terminal record per independently sampled incident."""
    rows = [AxialHandoffRecord(i, "retained", 300.0) for i in range(n)]
    # Both event fractions grow monotonically with depth.
    rows[0] = AxialHandoffRecord(0, "physical_outgoing", 0.0)
    rows[1] = AxialHandoffRecord(1, "physical_outgoing", 220.0)
    rows[2] = AxialHandoffRecord(2, "unresolved", 150.0)
    rows[3] = AxialHandoffRecord(3, "unresolved", 250.0)
    return rows


def run_bounds(rows=None, **changes):
    params = dict(
        intervals=((100.0, 200.0), (200.0, 300.0)),
        sampled_max_depth_ang=300.0,
        caps=caps(),
        sampling_failure_probability=0.01,
    )
    params.update(changes)
    return physical_fate_source_uppers(records() if rows is None else rows, **params)


def test_joint_hoeffding_cdf_bound_is_uniform_in_depth_and_photon_normalized():
    first, second = run_bounds()
    allowance = sqrt(log(4 / 0.01) / (2 * 400))
    assert first.fate_fraction_allowance == pytest.approx(allowance)
    assert first.observed_outgoing_fraction == pytest.approx(1 / 400)
    assert first.observed_unresolved_fraction == pytest.approx(1 / 400)
    assert second.observed_outgoing_fraction == pytest.approx(2 / 400)
    assert second.observed_unresolved_fraction == pytest.approx(2 / 400)
    assert first.return_source_error_upper == pytest.approx((1 / 400 + allowance) * 0.03)
    assert second.unresolved_source_error_upper == pytest.approx((2 / 400 + allowance) * 0.15)
    assert second.return_source_error_upper >= first.return_source_error_upper
    assert second.unresolved_source_error_upper >= first.unresolved_source_error_upper


def test_zero_observed_exits_is_not_zero_error_at_finite_sample_size():
    empty = [AxialHandoffRecord(i, "retained", 300.0) for i in range(500)]
    bound = run_bounds(empty)[0]
    assert bound.observed_outgoing_fraction == 0
    assert bound.observed_unresolved_fraction == 0
    assert bound.return_source_error_upper > 0
    assert bound.unresolved_source_error_upper > 0


def test_caps_not_clamped_to_one_photon_and_empirical_upper_saturates():
    all_out = [AxialHandoffRecord(i, "physical_outgoing", 0.0) for i in range(40)]
    result = run_bounds(
        all_out,
        caps=caps(outgoing_return_error_upper_bins=(2.0, 3.0)),
    )[0]
    assert result.return_source_error_upper == 5.0


def test_extra_bounds_are_additive_not_a_replacement_and_compose_confidence():
    base = source()
    enriched = add_physical_fate_source_bounds(
        base,
        records(),
        sampled_max_depth_ang=300.0,
        caps=caps(),
        sample_evidence_id="independent-incident-histories",
        sampling_failure_probability=0.01,
    )
    assert enriched is not base
    assert enriched.simultaneous_confidence == pytest.approx(0.978)
    assert enriched.intervals[0].return_upper > base.intervals[0].return_upper
    assert enriched.intervals[0].unresolved_upper > base.intervals[0].unresolved_upper
    assert enriched.intervals[0].source_model_upper == base.intervals[0].source_model_upper
    assert enriched.intervals[1].evidence_ids[-2:] == (
        "physical-fate-response-proof",
        "independent-incident-histories",
    )
    assert (
        require_axial_photon_depth_budget(
            case_key=CASE,
            observable_key=OBS,
            model_id=MODEL,
            requested_depth_ang=200,
            evidence=enriched,
            min_confidence=0.97,
            max_relative_error=0.02,
        )
        is enriched.intervals[0]
    )
    with pytest.raises(ValueError, match="joint confidence"):
        require_axial_photon_depth_budget(
            case_key=CASE,
            observable_key=OBS,
            model_id=MODEL,
            requested_depth_ang=200,
            evidence=enriched,
            min_confidence=0.98,
        )


def test_extra_source_error_can_refuse_before_any_sampling_work(monkeypatch):
    from pyrite.montecarlo.channeling import axial_outgoing_launch

    calls = []
    monkeypatch.setattr(
        axial_outgoing_launch, "sample_physical_handoff", lambda *a, **k: calls.append(1)
    )
    evidence = add_physical_fate_source_bounds(
        source(),
        records(),
        sampled_max_depth_ang=300.0,
        caps=caps(outgoing_return_error_upper_bins=(4.0, 0.0)),
        sample_evidence_id="sample-large-return-cap",
    )
    with pytest.raises(ValueError, match="photon-source error"):
        sample_budgeted_physical_handoff(
            None,
            None,
            evidence=evidence,
            case_key=CASE,
            observable_key=OBS,
            model_id=MODEL,
            hbar_v_eV_ang=1.0,
            window=None,
            max_depth_ang=200.0,
            incidents=(),
            zone=None,
            matching=None,
        )
    assert calls == []


@pytest.mark.parametrize(
    "change,message",
    [
        ({"caps": caps(unresolved_error_upper_bins=(0.1,))}, "matching observable bins"),
        ({"caps": caps(outgoing_return_error_upper_bins=(-1.0, 0.1))}, "nonnegative"),
        ({"caps": caps(unresolved_error_upper_bins=(float("nan"), 0.1))}, "finite"),
        ({"caps": caps(outgoing_return_error_upper_bins=(1j, 0.1))}, "real integrated"),
        ({"caps": caps(failure_probability_upper=1)}, "cap failure_probability"),
        ({"sampling_failure_probability": 0.0}, "sampling_failure_probability"),
        ({"sampling_failure_probability": 1.0}, "sampling_failure_probability"),
        ({"sampled_max_depth_ang": 250.0}, "exceed sampled depth"),
        ({"intervals": ((100.0, 200.0), (150.0, 300.0))}, "overlap"),
        ({"intervals": ((0.0, 200.0),)}, "0 < start"),
        ({"intervals": ()}, "at least one"),
    ],
)
def test_rejects_invalid_caps_and_interval_support(change, message):
    with pytest.raises((TypeError, ValueError), match=message):
        run_bounds(**change)


@pytest.mark.parametrize(
    "bad,message",
    [
        (
            [AxialHandoffRecord(1, "retained", 300), AxialHandoffRecord(1, "retained", 300)],
            "unique",
        ),
        ([AxialHandoffRecord(True, "retained", 300)], "unique"),
        ([AxialHandoffRecord(1, "retained", float("inf"))], "record depth"),
        ([AxialHandoffRecord(1, "retained", 301)], "record depth"),
        ([AxialHandoffRecord(1, "retained", 200)], "retained.*sampled maximum depth"),
        ([AxialHandoffRecord(1, "other", 300)], "invalid physical outgoing fate"),
        ([], "no independently sampled"),
    ],
)
def test_refuses_invalid_terminal_fates(bad, message):
    with pytest.raises(ValueError, match=message):
        run_bounds(bad)


def test_composition_refuses_a_censored_incident_census():
    rows = records()
    # This incident was observed only to 200 Å. Its fate by 300 Å is unknown;
    # counting it as retained would depress both terminal-event estimates.
    rows[-1] = replace(rows[-1], depth_ang=200.0)
    with pytest.raises(ValueError, match="retained.*sampled maximum depth"):
        add_physical_fate_source_bounds(
            source(),
            rows,
            sampled_max_depth_ang=300.0,
            caps=caps(),
            sample_evidence_id="censored-histories",
        )


def test_fails_closed_on_wrong_source_scope_and_confidence_budget():
    with pytest.raises(ValueError, match="do not match"):
        add_physical_fate_source_bounds(
            source(),
            records(),
            sampled_max_depth_ang=300,
            caps=caps(observable_key="wrong-band"),
            sample_evidence_id="histories",
        )
    with pytest.raises(ValueError, match="joint confidence"):
        add_physical_fate_source_bounds(
            replace(source(), simultaneous_confidence=0.9),
            records(),
            sampled_max_depth_ang=300,
            caps=caps(failure_probability_upper=0.8),
            sample_evidence_id="histories",
            sampling_failure_probability=0.2,
        )
    with pytest.raises(ValueError, match="distinct"):
        add_physical_fate_source_bounds(
            source(),
            records(),
            sampled_max_depth_ang=300,
            caps=caps(),
            sample_evidence_id="physical-fate-response-proof",
        )


def test_hoeffding_allowance_covers_true_cdfs_simultaneously_at_declared_rate():
    """Independent coverage check: the true CDFs stay under the bound in >= 1 - delta of samples."""
    rng = np.random.default_rng(884)
    delta, count, trials = 0.2, 200, 2000
    stops = np.array([100.0, 200.0, 300.0])
    # Exponential terminal depth, with independent outgoing or unresolved labels.
    scale, outgoing_share = 400.0, 0.7
    event_cdf = 1 - np.exp(-stops / scale)
    truth = np.stack((outgoing_share * event_cdf, (1 - outgoing_share) * event_cdf))
    allowance = sqrt(log(2 * stops.size / delta) / (2 * count))
    failures = 0
    for _ in range(trials):
        depths = rng.exponential(scale, count)
        labels = rng.random(count) < outgoing_share
        estimate = np.stack(
            (
                [(labels & (depths <= stop)).mean() for stop in stops],
                [(~labels & (depths <= stop)).mean() for stop in stops],
            )
        )
        failures += np.any(truth > estimate + allowance)
    assert failures / trials <= delta


def test_code_allowance_matches_the_coverage_oracle_on_a_sampled_census():
    rng = np.random.default_rng(885)
    count = 300
    depths = rng.exponential(400.0, count)
    rows = [
        AxialHandoffRecord(i, "physical_outgoing", float(z))
        if z <= 300.0
        else AxialHandoffRecord(i, "retained", 300.0)
        for i, z in enumerate(depths)
    ]
    bounds = run_bounds(
        rows,
        intervals=((50.0, 100.0), (100.0, 200.0), (200.0, 300.0)),
        sampling_failure_probability=0.2,
    )
    allowance = sqrt(log(6 / 0.2) / (2 * count))
    for bound, stop in zip(bounds, (100.0, 200.0, 300.0), strict=True):
        assert bound.fate_fraction_allowance == pytest.approx(allowance)
        assert bound.observed_outgoing_fraction == pytest.approx(np.mean(depths <= stop))
