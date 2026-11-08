"""Batch-paired depth calibration (#993) on synthetic arms with known truth."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_depth_calibration import (
    PROBES,
    ArmBatches,
    batch_mean_se,
    calibrate_axial_depth,
    interval_measurement,
)
from pyrite.montecarlo.channeling.axial_depth_measurements import source_interval_bounds
from pyrite.montecarlo.channeling.axial_outgoing_launch import AxialHandoffRecord

THICK = (985.0, 1000.0, 1985.0, 2000.0)
SHIFT = {
    "return_flux": 0.002,
    "unresolved": -0.001,
    "matching_surface": 0.0,
    "grid_supercell": 0.003,
    "source_model": 0.0005,
}


def _arms(rng, batches=200, noise=0.01, interaction=0.0):
    """Yields ~1 per incident at 1000 A, 2 at 2000 A, with shared per-chunk noise."""
    counts = np.full(batches, 100)
    base = np.array([0.985, 1.0, 1.985, 2.0])
    common = rng.normal(0.0, noise, (batches, 1, 1)) * base[None, :, None]
    arms = {}

    def arm(rel):
        own = rng.normal(0.0, noise / 10, (batches, len(THICK), 1))
        values = base[None, :, None] * (1 + rel) + common + own
        return ArmBatches(THICK, tuple(range(0, 100 * batches, 100)), counts, values)

    arms["control"] = arm(0.0)
    for name, rel in SHIFT.items():
        arms[name] = arm(rel)
    arms["all_on"] = arm(sum(SHIFT.values()) + interaction)
    return arms


def _fates(n, outgoing, unresolved, depth=2000.0):
    out = int(outgoing * n)
    lost = int(unresolved * n)
    return [
        AxialHandoffRecord(
            i,
            "physical_outgoing" if i < out else "unresolved" if i < out + lost else "retained",
            0.5 * depth if i < out + lost else depth,
        )
        for i in range(n)
    ]


def _calibrate(arms, **extra):
    kwargs = dict(
        checkpoints_ang=[1000.0, 2000.0],
        span_ang=15.0,
        sampled_max_depth_ang=2000.0,
        case_key="synthetic",
        observable_key="band",
        model_id="fibre",
        evidence_id="test",
    )
    return calibrate_axial_depth(
        arms, _fates(50_000, 0.2, 0.01), _fates(50_000, 0.02, 0.01), **(kwargs | extra)
    )


def test_batch_mean_se_matches_weighted_formula():
    values = np.array([1.0, 2.0, 4.0])
    counts = np.array([1, 1, 2])
    mean, se = batch_mean_se(values, counts)
    assert mean == pytest.approx(11 / 4)
    assert se == pytest.approx(np.sqrt((1.75**2 + 0.75**2 + 2 * 1.25**2) / (2 * 4)))


def test_budget_covers_known_shifts_and_reports_joint_confidence():
    evidence, report = _calibrate(_arms(np.random.default_rng(1)))
    assert evidence.simultaneous_confidence == pytest.approx(0.95)
    assert [(i.start_depth_ang, i.stop_depth_ang) for i in evidence.intervals] == [
        (985.0, 1000.0),
        (1985.0, 2000.0),
    ]
    for interval, scale in zip(evidence.intervals, (1.0, 2.0), strict=True):
        assert interval.reference_lower <= 0.985 * scale
        assert interval.return_upper >= SHIFT["return_flux"] * scale
        assert interval.grid_supercell_upper >= SHIFT["grid_supercell"] * scale
        assert 0 < interval.relative_error_upper < 0.2
    # Caps: returned fraction 0.18 against an all-on shift of 0.0055.
    assert report["caps"]["outgoing"][0] == pytest.approx(0.0055 * 2 / 0.18, rel=0.5)


def test_interaction_enters_the_source_model_bound():
    rng = np.random.default_rng(2)
    plain = interval_measurement(
        _arms(rng), checkpoint_ang=1000.0, span_ang=15.0, alpha=0.01, evidence_prefix="a"
    )[0]
    rng = np.random.default_rng(2)
    coupled = interval_measurement(
        _arms(rng, interaction=0.02),
        checkpoint_ang=1000.0,
        span_ang=15.0,
        alpha=0.01,
        evidence_prefix="a",
    )[0]
    gain = (
        source_interval_bounds(coupled).source_model_upper
        - source_interval_bounds(plain).source_model_upper
    )
    assert gain >= 0.02 * 0.9


def test_bounds_cover_truth_at_nominal_rate():
    """Zero true shifts: the absolute probe bounds must exceed zero error always."""
    misses = 0
    for seed in range(40):
        arms = _arms(np.random.default_rng(100 + seed), batches=20)
        for name in PROBES:
            arms[name] = replace(arms[name], yields=arms["control"].yields + 0.0)
        row, _ = interval_measurement(
            arms, checkpoint_ang=1000.0, span_ang=15.0, alpha=0.05, evidence_prefix="c"
        )
        bounds = source_interval_bounds(row)
        misses += bounds.reference_lower > 1.0
    assert misses <= 6  # nominal miss rate <= 5% per interval; binomial slack


def test_refusals():
    rng = np.random.default_rng(3)
    arms = _arms(rng)
    shifted = dict(arms, return_flux=replace(arms["return_flux"], starts=(1,) * 200))
    with pytest.raises(ValueError, match="chunk-paired"):
        _calibrate(shifted)
    with pytest.raises(ValueError, match="arms must be exactly"):
        _calibrate({k: v for k, v in arms.items() if k != "all_on"})
    with pytest.raises(ValueError, match="no shared recorded thickness"):
        _calibrate(arms, span_ang=5.0)
    with pytest.raises(ValueError, match="response cap undefined"):
        calibrate_axial_depth(
            arms,
            _fates(1000, 0.2, 0.01),
            _fates(1000, 0.2, 0.01),
            checkpoints_ang=[1000.0],
            span_ang=15.0,
            sampled_max_depth_ang=2000.0,
            case_key="s",
            observable_key="b",
            model_id="m",
            evidence_id="e",
        )
