"""Reported-only precision diagnostics beside the adaptive VOV (#1339).

The tail slope (Hill estimate over the largest scores), MCNP's SHIFT and the
figure of merit are recorded per observable; none enters ``passes``. These
tests pin the bounded top-k merge and third moment against one-pass values,
the slope on synthetic Pareto tails, SHIFT and FOM against closed forms, the
wall-clock interpolation, and that the stop is unchanged.

Seeds are arbitrary fixed integers. Hill tolerances are multiples of the
estimator's standard error ``alpha / sqrt(k)`` for an exact Pareto tail.

Validation: adaptive-sample-size-stopping
"""

import math

import numpy as np
import pytest

from pyrite.montecarlo.runner.adaptive import (
    TAIL_MIN_SAMPLES,
    TAIL_SCORES,
    TAIL_SLOPE_CAP,
    WALL_CLOCK_FIELDS,
    AdaptiveSettings,
    RunningMoments,
    run_stopping_rule,
    without_wall_clock,
)

_DIAGNOSTICS = ("tail_slope", "shift", "shift_over_se", "elapsed_s", "figure_of_merit")


def _blocks(values, sizes):
    moments, start = RunningMoments(), 0
    for size in sizes:
        moments.add_block(values[start : start + size])
        start += size
    moments.add_block(values[start:])
    return moments


@pytest.mark.parametrize("sizes", [(), (1, 7, 64, 300), (150,) * 30, (5000,)])
def test_top_scores_and_third_moment_merge_like_one_pass(sizes):
    values = np.random.default_rng(7).lognormal(0.0, 1.2, 12_001)
    moments = _blocks(values, sizes)
    np.testing.assert_array_equal(np.sort(moments.top), np.sort(values)[-(TAIL_SCORES + 1) :])
    centered = values - values.mean()
    assert moments.m3 == pytest.approx(np.sum(centered**3), rel=1e-10)
    whole = RunningMoments()
    whole.add_block(values)
    assert moments.tail_slope() is not None
    assert moments.tail_slope() == whole.tail_slope()


@pytest.mark.parametrize("alpha", [1.5, 3.0, 5.0])
def test_tail_slope_recovers_pareto_index(alpha):
    # numpy's pareto is Lomax; +1 gives Pareto(alpha) with x_m = 1.
    values = np.random.default_rng(int(alpha * 10)).pareto(alpha, 50_000) + 1.0
    moments = _blocks(values, (4096,) * 12)
    slope = moments.tail_slope()
    assert slope == pytest.approx(1.0 + alpha, abs=4.0 * alpha / math.sqrt(TAIL_SCORES))
    # MCNP's convention: a slope below 3 flags a possibly infinite variance.
    assert (slope < 3.0) is (alpha < 2.0)


def test_tail_slope_limits():
    moments = RunningMoments()
    moments.add_block(np.r_[np.zeros(TAIL_MIN_SAMPLES), np.ones(TAIL_SCORES)])
    assert moments.tail_slope() is None  # threshold score x_(k+1) is zero
    moments.add_block(np.ones(1))
    assert moments.tail_slope() == TAIL_SLOPE_CAP  # flat top: no tail
    few = RunningMoments()
    few.add_block(np.arange(1.0, TAIL_SCORES + 1))
    assert few.tail_slope() is None


@pytest.mark.parametrize("n", [TAIL_MIN_SAMPLES, 200_000])
def test_light_tail_reads_finite_variance_once_reported(n):
    """An exponential's Hill slope is about 2 + ln(n / k), below 3 up to
    n ~ k e, so it is withheld until the top k are at most 2 % of n."""
    values = np.random.default_rng(n).exponential(1.0, n)
    moments = RunningMoments()
    moments.add_block(values[: TAIL_MIN_SAMPLES - 1])
    assert moments.tail_slope() is None
    moments.add_block(values[TAIL_MIN_SAMPLES - 1 :])
    slope = moments.tail_slope()
    assert slope > 3.0
    assert slope == pytest.approx(2.0 + math.log(n / TAIL_SCORES), abs=0.8)


def test_shift_matches_its_closed_form_and_sign():
    values = np.random.default_rng(3).exponential(2.0, 40_000)
    moments = _blocks(values, (999,) * 10)
    n, mean = values.size, values.mean()
    centered = values - mean
    expected = np.sum(centered**3) / (2.0 * np.sum(centered**2) * n) / mean
    assert moments.shift() == pytest.approx(expected, rel=1e-9)
    # Exponential: mu3 = 2 sigma^3 and sigma = mu give SHIFT / mean = 1 / n.
    assert moments.shift() == pytest.approx(1.0 / n, rel=0.1)
    mirrored = RunningMoments()
    mirrored.add_block(values.max() + 1.0 - values)
    assert mirrored.shift() < 0.0
    flat = RunningMoments()
    flat.add_block(np.full(10, 3.0))
    assert flat.shift() == 0.0
    single = RunningMoments()
    single.add_block(np.ones(1))
    assert single.shift() is None


def test_figure_of_merit_is_inverse_rse_squared_time():
    moments = RunningMoments()
    moments.add_block(np.random.default_rng(1).normal(1.0, 0.3, 500))
    rse = moments.relative_se()
    assert moments.figure_of_merit(4.0) == pytest.approx(1.0 / (rse * rse * 4.0), rel=1e-15)
    assert moments.figure_of_merit(8.0) == pytest.approx(moments.figure_of_merit(4.0) / 2.0)
    assert moments.figure_of_merit(None) is None
    assert moments.figure_of_merit(0.0) is None


def _ticking_clock():
    """One second per call: construction reads 0, each chunk end one more."""
    calls = iter(range(10**6))
    return lambda: float(next(calls))


@pytest.mark.parametrize("name", ["cxr", "brem"])
@pytest.mark.parametrize("chunk", [37, 250])
def test_recorded_diagnostics_never_change_the_stop(chunk, name):
    population = np.random.default_rng(11).lognormal(0.0, 1.0, 8000)
    # One monitored observable per run: a shared ticking clock cannot serve the
    # CXR and brem monitors of an independent run at once.
    settings = AdaptiveSettings(0.03, 400, 8000, observables=(name,))

    def measure(start, stop, _):
        return {name: population[start:stop]}

    plain = run_stopping_rule(settings, measure)
    timed = run_stopping_rule(settings, measure, chunk=chunk, clock=_ticking_clock())
    assert without_wall_clock(timed) == without_wall_clock(plain)
    realized = timed["realized_electrons"]
    assert realized is not None and realized < 8000
    status = timed["statistics"][name]
    assert set(_DIAGNOSTICS) <= set(status)
    assert not set(_DIAGNOSTICS) & set(status["passes"])
    assert all(plain["statistics"][name][key] is None for key in WALL_CLOCK_FIELDS)
    # Chunk ends at one second each: T(d) = d / chunk exactly.
    assert status["elapsed_s"] == pytest.approx(realized / chunk, rel=1e-12)
    rse = status["relative_se"]
    assert status["figure_of_merit"] == pytest.approx(chunk / (rse * rse * realized))
    assert status["shift_over_se"] == pytest.approx(status["shift"] / rse)


def test_independent_brem_records_its_own_diagnostics():
    population = np.random.default_rng(5).exponential(1.0, 6000)
    settings = AdaptiveSettings(
        0.05,
        300,
        6000,
        observables=("cxr", "brem"),
        brem_target_rse=0.08,
        brem_min_histories=300,
        brem_max_histories=6000,
    )
    record = run_stopping_rule(
        settings,
        lambda start, stop, _: dict.fromkeys(("cxr", "brem"), population[start:stop]),
        clock=_ticking_clock(),
    )
    brem = record["statistics"]["brem"]
    assert brem["n_electrons"] == record["realized_electrons_brem"]
    assert brem["elapsed_s"] is not None
    assert brem["elapsed_s"] <= record["statistics"]["cxr"]["elapsed_s"]
    assert brem["n_electrons"] < TAIL_MIN_SAMPLES and brem["tail_slope"] is None
    assert brem["shift"] > 0.0
