"""Pure NumPy estimator/assignment tests; no tables, transport or GPU needed."""

import math

import numpy as np
import pytest

from checks.fiber_randomized_rolls import (
    SECTOR,
    grouped_plan,
    independent_plan,
    independent_rolls,
    midpoint_rolls,
    row_partitions,
)


def test_counter_addressed_draws():
    ids = np.arange(4096, dtype=np.int64)
    rolls = independent_rolls(ids, 51)
    assert np.all((0 <= rolls) & (rolls < SECTOR))
    np.testing.assert_array_equal(rolls, independent_rolls(ids[::-1], 51)[::-1])
    np.testing.assert_array_equal(rolls[:100], independent_rolls(ids[:100], 51))
    np.testing.assert_array_equal(
        rolls, np.concatenate([independent_rolls(ids[:157], 51), independent_rolls(ids[157:], 51)])
    )
    assert not np.array_equal(rolls, independent_rolls(ids, 52))
    # Deterministic smoke of marginal uniformity, not a RNG qualification.
    assert abs(rolls.mean() / SECTOR - 0.5) < 0.025


@pytest.mark.parametrize("n", [1, 7, 31, 32, 33, 79, 200])
def test_grouped_prefix_and_stratum_occupancy(n):
    k, per_roll = 8, 4
    plan = grouped_plan(n, k, per_roll, 123)
    extended = grouped_plan(n + 100, k, per_roll, 123)
    np.testing.assert_array_equal(plan.labels, extended.labels[:n])
    np.testing.assert_array_equal(plan.rolls, extended.rolls[: plan.rolls.size])
    occ = plan.occupancy().reshape(-1, k)
    assert occ.sum() == n
    assert (occ <= per_roll).all()
    assert (np.ptp(occ, axis=1) <= 1).all()
    assert np.all((0 <= plan.rolls) & (plan.rolls < SECTOR))
    jitter = plan.rolls.reshape(-1, k) / (SECTOR / k) - np.arange(k)
    assert np.all((0 <= jitter) & (jitter < 1))
    assert not np.array_equal(plan.rolls, grouped_plan(n, k, per_roll, 124).rolls)


@pytest.mark.parametrize("grouped", [False, True])
def test_partition_keeps_flights_and_global_denominator(grouped):
    n = 23
    plan = grouped_plan(n, 5, 3, 12) if grouped else independent_plan(n, 12)
    # Step-major, repeated rows of the same primary; 17 primaries are missed.
    ids = np.array([8, 1, 5, 0, 8, 12, 4, 5, 1, 8, 0, 12, 4])
    row_strength = np.linspace(0.3, 2, ids.size)
    actual = 0.0
    seen = []
    for slot, rows in row_partitions(ids, plan):
        assert np.all(np.diff(rows) > 0)
        assert np.all(plan.labels[ids[rows]] == slot)
        actual += row_strength[rows].sum() * np.cos(plan.rolls[slot]) / n
        seen.extend(rows.tolist())
    expected = np.sum(row_strength * np.cos(plan.rolls[plan.labels[ids]])) / n
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert sorted(seen) == list(range(ids.size))
    assert sum(plan.occupancy()) == n
    assert list(row_partitions(np.empty(0, dtype=int), plan)) == []
    with pytest.raises(ValueError, match="primary IDs outside the retained incident population"):
        list(row_partitions(np.array([n]), plan))


def test_narrow_periodic_line_reveals_midpoint_aliasing():
    # Toy spectrum of a sixfold-periodic line center; no PyRITE physics.
    energy = np.linspace(-1.3, 1.3, 401)
    width = 0.009

    def spectrum(rolls):
        centers = np.cos(6 * rolls)
        return np.exp(-0.5 * ((energy[:, None] - centers) / width) ** 2).mean(axis=1)

    ref = spectrum(midpoint_rolls(8192))
    q = spectrum(midpoint_rolls(32))
    shift = spectrum(midpoint_rolls(32, 0.5))
    phase_error = np.sum(abs(q - shift)) / np.sum(abs(ref))
    assert phase_error > 0.25
    samples = [spectrum(independent_rolls(np.arange(4096), seed)) for seed in range(8)]
    mean = np.mean(samples, axis=0)
    assert np.sum(abs(mean - ref)) < np.sum(abs(q - ref)) / 4


def test_grouped_unequal_occupancy_is_unbiased_in_expectation():
    # One partial group, with strengths tied to IDs. Each fixed primary's
    # roll is marginally uniform, including when N is not a multiple of K.
    n = 13
    weights = np.linspace(0.2, 3, n)
    samples = []
    for seed in range(512):
        plan = grouped_plan(n, 8, 3, seed)
        samples.append(np.mean(weights * np.cos(plan.rolls[plan.labels])))
    expected = weights.mean() * math.sin(SECTOR) / SECTOR
    se = np.std(samples, ddof=1) / math.sqrt(len(samples))
    assert abs(np.mean(samples) - expected) < 4 * se
