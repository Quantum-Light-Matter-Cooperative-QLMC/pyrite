"""Adaptive electron counts: the sequential stopping rule and its runner (#361).

The rule watches per-electron, grid-independent scalars block by block and
stops at the first block end where every observable's relative standard error
meets the target and the heavy-tail guards hold. These tests pin

* the running moments against direct NumPy formulas, and the identity that
  ties the effective-sample-size guard to the relative standard error;
* the statistical claim ``adaptive-sample-size-stopping``: near-nominal
  coverage and a stopping bias far below the reported error for a Gaussian
  population (many seeds), and the same on a real light case over 20 seeds;
* that the guards turn a false early stop on a heavy-tailed population into a
  ``statistics_limited`` run at ``max_electrons``;
* the runner contract: the adaptive result is the fixed-N result at the
  realized count bit for bit, replays deterministically, reduces to fixed N
  when ``min == max``, and refuses the coherent route and the lockstep core.

Seeds are arbitrary fixed integers. Statistical tolerances are binomial bounds
on coverage counts and multiples of the ensemble standard error, stated where
used; equalities of realized runs are exact.

Validation: adaptive-sample-size-stopping
"""

import math
import warnings
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from pyrite._backend import REAL
from pyrite._cxr_grid_policy import CxrYieldStatisticsWarning
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import (
    AdaptiveSettings,
    RunningMoments,
    StoppingMonitor,
    band_weights,
    case_measure,
    decision_points,
    run_case_adaptive,
    run_stopping_rule,
    vov_trend_slope,
    without_wall_clock,
)
from pyrite.montecarlo.runner.stream_sizing import AdaptiveChunkPlan
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories

# Device spectra accumulate in float32 by default, so two summation orders of
# one spectrum agree to float32 rounding there, not float64.
_FP32 = np.dtype(REAL) == np.float32

#: Guards disabled: the bare relative-standard-error rule.
_RULE_ONLY = dict(
    max_history_share=1.0,
    min_effective_histories=0.0,
    max_variance_of_variance=1.0,
    stability_checks=0,
)


def _grid(minimum, maximum, growth=0.1):
    return list(decision_points(minimum, maximum, growth))


def _population_measure(population):
    return lambda start, stop, _block: {"cxr": population[start:stop]}


# ---- estimator ----------------------------------------------------------------


def test_running_moments_equal_direct_formulas():
    rng = np.random.default_rng(5)
    values = rng.lognormal(0.0, 1.5, 997)
    moments = RunningMoments()
    for start in range(0, values.size, 64):
        moments.add_block(values[start : start + 64])
    n = values.size
    assert moments.n == n
    assert math.isclose(moments.mean, values.mean(), rel_tol=1e-13)
    expected_rse = values.std(ddof=1) / math.sqrt(n) / values.mean()
    assert math.isclose(moments.relative_se(), expected_rse, rel_tol=1e-12)
    assert math.isclose(moments.max_share(), values.max() / values.sum(), rel_tol=1e-13)
    ess = values.sum() ** 2 / (values**2).sum()
    assert math.isclose(moments.effective_sample_size(), ess, rel_tol=1e-12)
    # The ESS floor and the RSE target constrain the same two sums:
    # RSE^2 = n/(n-1) * (1/ESS - 1/n).
    identity = n / (n - 1) * (1.0 / ess - 1.0 / n)
    assert math.isclose(moments.relative_se() ** 2, identity, rel_tol=1e-9)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("block", [1, 37, 997])
def test_variance_diagnostic_matches_centered_population(scale, block):
    values = scale * np.random.default_rng(75).lognormal(0.0, 1.5, 997)
    moments = RunningMoments()
    for start in range(0, len(values), block):
        moments.add_block(values[start : start + block])
    centered = values - values.mean()
    m2 = np.sum(centered**2)
    m4 = np.sum(centered**4)
    assert moments.m3 == pytest.approx(np.sum(centered**3), rel=1e-12, abs=0)
    assert moments.m4 == pytest.approx(m4, rel=1e-12, abs=0)
    assert moments.variance_of_variance() == pytest.approx(m4 / m2**2 - 1 / len(values), rel=1e-12)


def test_constant_signal_passes_variance_guard_and_zero_signal_does_not_converge():
    settings = AdaptiveSettings(0.001, 3000, 6000, 1000)
    for value, expected in [(1.0, "converged"), (0.0, "max_electrons")]:
        record = run_stopping_rule(settings, _population_measure(np.full(6000, value)))
        assert record["stop_reason"] == expected
        assert record["statistics"]["cxr"]["variance_of_variance"] == 0.0


def test_variance_guard_retains_small_stop_for_nearly_constant_signal():
    population = np.ones(6000)
    population[50] += 0.01
    settings = AdaptiveSettings(0.001, 3000, 6000, 1000)
    record = run_stopping_rule(settings, _population_measure(population))
    assert record["stop_reason"] == "converged"
    # The first point with k = 3 earlier points for the stability window.
    assert record["realized_electrons"] == _grid(3000, 6000)[3]
    assert record["statistics"]["cxr"]["variance_of_variance"] > 0.9


def test_vov_trend_slope_limits():
    """Exact 1/n decay gives -1; only the last half and positive VOV enter."""
    decay = [(n, 7.0 / n) for n in range(100, 1001, 100)]
    assert vov_trend_slope(decay) == pytest.approx(-1.0, abs=1e-12)
    # Points before n_last/2 = 500 are ignored, whatever their trend.
    early = [(n, 1e3 * n) for n in range(100, 500, 100)]
    assert vov_trend_slope(early + decay[4:]) == pytest.approx(-1.0, abs=1e-12)
    assert vov_trend_slope([(n, 3.0 * n**0.5) for n in (600, 800, 1000)]) == pytest.approx(0.5)
    assert vov_trend_slope([(600, 0.1), (800, 0.0), (900, None), (1000, 0.05)]) is None
    assert vov_trend_slope([]) is None


def _prefix_vov(values, n):
    centered = values[:n] - values[:n].mean()
    m2 = np.sum(centered**2)
    return np.sum(centered**4) / m2**2 - 1.0 / n


@pytest.mark.parametrize("late_heavy", [False, True])
def test_vov_trend_slope_is_recorded_and_never_gates(late_heavy):
    """The record's slope is the fit to prefix VOVs over the last half.

    A well-sampled exponential population falls about as 1/n; one heavy score
    in the last half makes the trend positive. Neither changes the stop.
    """
    population = np.random.default_rng(11).exponential(1.0, 4000)
    if late_heavy:
        population[3500] += 500.0
    settings = AdaptiveSettings(1e-6, 2000, 4000, observables=("cxr", "brem"))
    record = run_stopping_rule(
        settings, lambda start, stop, _: dict.fromkeys(("cxr", "brem"), population[start:stop])
    )
    ends = _grid(2000, 4000)  # every decision point lies in the last half
    x = np.log(np.array(ends, dtype=float))
    y = np.log([_prefix_vov(population, n) for n in ends])
    expected = np.polyfit(x, y, 1)[0]
    for name in ("cxr", "brem"):
        status = record["statistics"][name]
        assert status["vov_trend_slope"] == pytest.approx(expected, rel=1e-9)
        assert "vov_trend_slope" not in status["passes"]
    if late_heavy:
        assert expected > 0.0
    else:
        assert -1.5 < expected < -0.5
    assert record["realized_electrons"] == 4000


def test_degenerate_moments_report_no_error():
    moments = RunningMoments()
    moments.add_block(np.zeros(10))
    assert moments.relative_se() is None
    assert moments.max_share() is None
    assert moments.effective_sample_size() == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        dict(target_rse=0.0),
        dict(min_histories=1),
        dict(max_histories=30),
        dict(decision_growth=0.0),
        dict(decision_growth=float("inf")),
        dict(observables=("spec",)),
        dict(max_history_share=0.0),
        dict(block_histories=20.5),
        dict(block_histories=True),
        dict(min_histories=40.5),
        dict(max_histories=200.5),
        dict(pilot_histories=20),
        dict(pilot_histories=220),
        dict(stability_checks=1.5),
        dict(stability_fraction=float("nan")),
        dict(min_effective_histories=float("inf")),
        dict(max_variance_of_variance=0.0),
        dict(max_variance_of_variance=1.5),
        dict(max_variance_of_variance=float("nan")),
        dict(max_variance_of_variance=True),
        dict(observables=("cxr", "cxr")),
        dict(band_eV=(2000.0, 1000.0)),
        dict(band_eV=(0.0, 1000.0)),
        dict(batch_means_band_eV=(1000.0, float("inf"))),
    ],
)
def test_settings_are_validated(bad):
    kw = dict(target_rse=0.1, min_histories=40, max_histories=200, block_histories=20) | bad
    with pytest.raises(ValueError):
        AdaptiveSettings(**kw)


# ---- the statistical claim ------------------------------------------------------


@pytest.mark.parametrize("guards", [_RULE_ONLY, {}], ids=["rule-only", "default-guards"])
def test_gaussian_population_has_near_nominal_coverage(guards):
    """1000 seeds of N(1, 0.5^2) per electron, target 5 % (about 100 electrons).

    Chow-Robbins: the sequential interval's coverage tends to nominal as the
    target shrinks; at N ~ 100 it undercovers slightly. 95 % nominal over 1000
    seeds has binomial SD 0.7 %, so [0.92, 0.98] is a > 4 SD band around the
    measured 0.93-0.94. The stopping bias (mean of mean_N - mu) is required to
    be under a quarter of the target, far below the reported error.
    """
    target = 0.05
    settings = AdaptiveSettings(
        target_rse=target, min_histories=50, max_histories=100_000, block_histories=10, **guards
    )
    errors, rses = [], []
    for seed in range(1000):
        population = np.random.default_rng(seed).normal(1.0, 0.5, 2000)
        record = run_stopping_rule(settings, _population_measure(population))
        assert record["stop_reason"] == "converged"
        line = record["statistics"]["cxr"]
        errors.append(line["mean"] - 1.0)
        rses.append(line["relative_se"])
    errors, rses = np.asarray(errors), np.asarray(rses)
    assert np.all(rses <= target)
    coverage = np.mean(np.abs(errors) <= 1.96 * rses)
    assert 0.92 <= coverage <= 0.98
    assert abs(errors.mean()) < 0.25 * target


_LIGHT_SEEDS = range(1, 21)
_LIGHT_N = 800


@pytest.fixture(scope="module")
def light_line_masses():
    """Per-electron line mass of hopg 30 keV, 1 um, for 20 seeds x 800 electrons.

    The streams are counter-addressed, so a stop at n reads exactly the first
    n values of the seed's population (pinned below against live transport).
    """
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    base = dict(build_cases(sweep)[0])
    settings = AdaptiveSettings(
        target_rse=1.0, min_histories=_LIGHT_N, max_histories=_LIGHT_N, block_histories=_LIGHT_N
    )
    populations = []
    for seed in _LIGHT_SEEDS:
        case = {**base, "seed": seed, "Ne": _LIGHT_N, "Ne_brem": _LIGHT_N}
        tp = runner._transport_case(case, transport_core="per-electron")
        populations.append(np.asarray(case_measure(case, settings)(0, _LIGHT_N, tp["segs"])["cxr"]))
    return base, populations


@pytest.mark.slow
def test_light_case_meets_its_target_over_20_seeds(light_line_masses):
    """Target 10 % on the per-electron line mass (skewness ~ 8).

    Truth is the pooled mean of all 16000 electrons (relative SE ~ 0.8 %).
    The achieved error must lie within 1.96 x target in at least 17 of 20
    seeds (P <= 1.6 % if the true coverage were 95 %). The stopping bias is
    measured paired, against each seed's own 800-electron mean, which cancels
    the shared electrons; it must be below the reported error.
    """
    _, populations = light_line_masses
    target = 0.1
    settings = AdaptiveSettings(
        target_rse=target, min_histories=200, max_histories=_LIGHT_N, block_histories=20
    )
    truth = np.concatenate(populations).mean()
    records = [run_stopping_rule(settings, _population_measure(p)) for p in populations]
    # The variance gate may require more than the captured 800 histories.
    # A capped run remains useful for this accuracy study, but must report
    # that its convergence diagnostics did not pass.
    for record in records:
        if record["stop_reason"] == "max_electrons":
            assert record["statistics_limited"]
            assert record["realized_electrons"] == _LIGHT_N
        else:
            assert record["stop_reason"] == "converged"
            assert record["statistics"]["cxr"]["converged"]
    means = np.array([r["statistics"]["cxr"]["mean"] for r in records])
    reported = np.array([r["statistics"]["cxr"]["relative_se"] for r in records])
    assert np.all(reported <= target)
    within = np.abs(means / truth - 1.0) <= 1.96 * target
    assert within.sum() >= 17
    paired = np.array([(m - p.mean()) / truth for m, p in zip(means, populations, strict=True)])
    assert abs(paired.mean()) < reported.min()


@pytest.mark.slow
def test_stop_inside_transport_equals_the_rule_on_the_population(light_line_masses):
    """The monitor inside block transport reads the per-electron prefix exactly."""
    base, populations = light_line_masses
    settings = AdaptiveSettings(
        target_rse=0.1, min_histories=200, max_histories=_LIGHT_N, block_histories=20
    )
    case = {**base, "seed": _LIGHT_SEEDS[0], "Ne": _LIGHT_N, "Ne_brem": _LIGHT_N}
    monitor = StoppingMonitor(settings, case_measure(case, settings))
    tp = runner._transport_case(
        case, transport_core="per-electron", block_electrons=20, block_monitor=monitor
    )
    live = monitor.record()
    offline = run_stopping_rule(settings, _population_measure(populations[0]))
    assert live["realized_electrons"] == offline["realized_electrons"] == tp["segs"]["Ne"]
    assert live["statistics"] == offline["statistics"]


def test_pilot_projection_skips_checks_until_the_projected_count():
    population = np.random.default_rng(3).normal(1.0, 0.5, 5000)
    base = dict(target_rse=0.02, min_histories=50, max_histories=5000, block_histories=10)
    plain = run_stopping_rule(AdaptiveSettings(**base), _population_measure(population))
    piloted = run_stopping_rule(
        AdaptiveSettings(**base, pilot_histories=100), _population_measure(population)
    )
    pilot = piloted["pilot"]
    rse = pilot["relative_se"]["cxr"]
    # The projection is a decision point itself: no rounding to a transport unit.
    # The pilot count is a decision point of its own.
    assert pilot["n_electrons"] == 100
    assert pilot["projected_electrons"] == math.ceil(pilot["n_electrons"] * (rse / 0.02) ** 2)
    assert piloted["realized_electrons"] >= pilot["projected_electrons"]
    assert piloted["checks"] < plain["checks"]
    assert piloted["stop_reason"] == "converged"


# ---- heavy tails --------------------------------------------------------------


def test_variance_tail_prevents_false_precision_on_nearly_constant_signal():
    """A large baseline hides rare variance carriers from contribution ESS.

    The first 4000 histories have one moderate outlier; the rare class starts
    at 5000. Its eventual RSE exceeds the requested 0.1 percent. No random
    seed or fitted tolerance: the finite population defines the reference.
    """
    population = np.ones(20_000)
    population[500] = 1.5
    population[5000::1000] = 20.0
    settings = AdaptiveSettings(
        target_rse=0.001,
        min_histories=3000,
        max_histories=20_000,
        block_histories=1000,
        pilot_histories=3000,
    )
    record = run_stopping_rule(settings, _population_measure(population))
    assert record["stop_reason"] == "max_electrons"
    assert record["statistics_limited"]
    assert record["realized_electrons"] == len(population)


@pytest.mark.parametrize("observable, vov_limit", [("cxr", 1.0), ("brem", 0.01)])
def test_variance_guard_preserves_disabled_and_brem_stops(observable, vov_limit):
    """Neither explicit opt-out nor brem acquires the CXR-only variance gate."""
    population = np.ones(6000)
    population[500] = 1.5
    settings = AdaptiveSettings(
        0.001,
        3000,
        6000,
        1000,
        observables=(observable,),
        max_variance_of_variance=vov_limit,
    )
    record = run_stopping_rule(
        settings, lambda start, stop, _: {observable: population[start:stop]}
    )
    assert record["stop_reason"] == "converged"
    assert record["realized_electrons"] == _grid(3000, 6000)[3]


@pytest.mark.parametrize(
    "guards, realized",
    [
        ({}, None),  # defaults: one deviation holds all variance, VOV + 1/n ~ 1
        ({"min_effective_histories": 0.0}, None),  # the ESS floor no longer frees VOV
        # VOV off; ESS ~ n passes 100 at the first point with k earlier checks
        ({"max_variance_of_variance": 1.0}, _grid(3000, 6000)[3]),
        # ESS ~ n first reaches 4500 at the grid point above it
        (
            {"max_variance_of_variance": 1.0, "min_effective_histories": 4500.0},
            _grid(3000, 6000)[5],
        ),
    ],
    ids=["defaults", "ess-off", "vov-off", "ess-binds"],
)
def test_vov_threshold_and_ess_floor_gate_independently(guards, realized):
    """#1262: each threshold binds on its own; the record carries both."""
    population = np.ones(6000)
    population[500] = 1.5
    settings = AdaptiveSettings(0.001, 3000, 6000, 1000, **guards)
    record = run_stopping_rule(settings, _population_measure(population))
    status = record["statistics"]["cxr"]
    assert record["realized_electrons"] == (realized or 6000)
    assert record["stop_reason"] == ("converged" if realized else "max_electrons")
    if realized is None:
        assert not status["passes"]["variance_sampling"]
        assert status["passes"]["effective_sample_size"]
    assert record["guards"]["min_effective_electrons"] == settings.min_effective_histories
    assert record["guards"]["max_variance_of_variance"] == settings.max_variance_of_variance


def test_vov_threshold_is_compared_to_m4_over_m2_squared():
    """The gate passes exactly when ``VOV + 1/n = M4/M2**2`` meets the threshold."""
    population = np.random.default_rng(3).lognormal(0.0, 1.0, 4000)
    centered = population - population.mean()
    ratio = float(np.sum(centered**4) / np.sum(centered**2) ** 2)
    base = dict(target_rse=1e-6, min_histories=4000, max_histories=4000, block_histories=4000)
    for limit, expected in ((ratio * (1 + 1e-9), True), (ratio * (1 - 1e-9), False)):
        record = run_stopping_rule(
            AdaptiveSettings(**base, max_variance_of_variance=limit),
            _population_measure(population),
        )
        assert record["statistics"]["cxr"]["passes"]["variance_sampling"] is expected


def test_variance_tail_keeps_cxr_running_after_independent_brem_freezes():
    """The same early scores pass brem but fail CXR's variance gate."""
    population = np.ones(6000)
    population[500] = 1.5
    population[5000] = 20.0
    settings = AdaptiveSettings(
        0.001, 3000, 6000, 1000, observables=("cxr", "brem"), brem_target_rse=0.001
    )
    record = run_stopping_rule(
        settings, lambda start, stop, _: dict.fromkeys(("cxr", "brem"), population[start:stop])
    )
    assert record["statistics_limited"]
    assert record["stop_reason"] == "max_electrons"
    assert record["realized_electrons"] == 6000
    brem_stop = _grid(3000, 6000)[3]
    assert record["realized_electrons_brem"] == brem_stop
    assert record["statistics"]["brem"]["stop_reason"] == "converged"
    assert record["statistics"]["brem"]["mean"] == pytest.approx(1.0 + 0.5 / brem_stop)
    assert not record["statistics"]["cxr"]["passes"]["variance_sampling"]


def _heavy_population(rng, n, *, every=None, rate=None, heavy=400.0):
    """Exp(1) per electron plus rare ``heavy`` contributions (mean 1 + rate*heavy)."""
    values = rng.exponential(1.0, n)
    if every is not None:
        values[every // 2 :: every] += heavy
    else:
        values[rng.random(n) < rate] += heavy
    return values


def test_guards_turn_a_false_early_stop_into_statistics_limited():
    """One heavy electron in 500 carries 44 % of the true mean (1.8).

    Without guards the first 200 electrons hold none: the RSE (~6.5 %) passes
    and the run stops about 37 % low (44 % expected deficit). The ESS floor
    (1000) forces enough electrons to meet the heavy ones, the share cap then refuses to stop on a handful of
    them, and the run ends statistics_limited at max_electrons.
    """
    population = _heavy_population(np.random.default_rng(0), 4000, every=500)
    base = dict(target_rse=0.1, min_histories=200, max_histories=4000, block_histories=100)
    measure = _population_measure(population)
    rule = run_stopping_rule(AdaptiveSettings(**base, **_RULE_ONLY), measure)
    assert rule["stop_reason"] == "converged"
    assert rule["realized_electrons"] == 200
    line = rule["statistics"]["cxr"]
    assert abs(line["mean"] / 1.8 - 1.0) > 5 * line["relative_se"]

    guarded = run_stopping_rule(AdaptiveSettings(**base, min_effective_histories=1000.0), measure)
    assert guarded["stop_reason"] == "max_electrons"
    assert guarded["statistics_limited"]
    assert guarded["realized_electrons"] == 4000
    status = guarded["statistics"]["cxr"]
    assert not status["passes"]["max_electron_share"]


def test_guards_suppress_false_stops_across_seeds():
    """Rate 2e-3: rule-only stops falsely (error > 3 x reported) in most seeds;
    guarded runs do so in at most one of 40 (the tail is then unsampled even
    at the ESS floor: P(no heavy in 2000) = e^-4 ~ 2 %)."""
    base = dict(target_rse=0.1, min_histories=200, max_histories=4000, block_histories=100)
    false_stops = {}
    for label, guards in (("rule", _RULE_ONLY), ("guarded", {"min_effective_histories": 1000.0})):
        settings = AdaptiveSettings(**base, **guards)
        count = 0
        for seed in range(40):
            population = _heavy_population(np.random.default_rng(seed), 4000, rate=2e-3)
            record = run_stopping_rule(settings, _population_measure(population))
            line = record["statistics"]["cxr"]
            if record["stop_reason"] == "converged":
                assert line["max_electron_share"] <= settings.max_history_share
                count += abs(line["mean"] / 1.8 - 1.0) > 3 * line["relative_se"]
        false_stops[label] = count
    assert false_stops["rule"] >= 20
    assert false_stops["guarded"] <= 1


def test_stability_guard_refuses_the_check_a_heavy_electron_arrives_at():
    """A heavy electron moves the running mean by ~20 %: the RSE (~17 %) meets
    the 20 % target at once, but the move exceeds both 0.5 eps and the i.i.d.
    drift allowance 0.9 r sqrt((n / d - 1) / delta) at the arrival check, so the
    stop waits for the next point, where the RSE still covers the jump."""
    population = np.ones(400)
    population[::2] = 0.5  # RSE is tiny from the start
    population[255] = 40.0
    grid = _grid(200, 400)
    arrival = next(point for point in grid if point > 255)
    settings = AdaptiveSettings(
        target_rse=0.2,
        min_histories=200,
        max_histories=400,
        max_history_share=1.0,
        min_effective_histories=0.0,
        max_variance_of_variance=1.0,
        stability_checks=3,
        stability_fraction=0.5,
    )
    record = run_stopping_rule(settings, _population_measure(population))
    assert record["realized_electrons"] == grid[grid.index(arrival) + 1]
    assert record["stop_reason"] == "converged"
    # Without the stability guard the run stops on the arrival check itself.
    bare = run_stopping_rule(replace(settings, stability_checks=0), _population_measure(population))
    assert bare["realized_electrons"] <= arrival


def test_records_do_not_depend_on_chunk_size():
    """Decisions fold values at decision points, so any chunking gives one record."""
    population = _heavy_population(np.random.default_rng(7), 6000, rate=2e-3)
    settings = AdaptiveSettings(target_rse=0.03, min_histories=150, max_histories=6000)
    records = [
        run_stopping_rule(settings, _population_measure(population), chunk=chunk)
        for chunk in (None, 1, 7, 100, 1000, 6000)
    ]
    assert all(record == records[0] for record in records[1:])


def test_decision_points_overshoot_at_most_their_growth():
    for minimum, maximum, growth in [(2, 10**6, 0.1), (200, 20_000, 0.05), (3000, 10**6, 0.3)]:
        grid = _grid(minimum, maximum, growth)
        assert grid[0] == minimum and grid[-1] == maximum
        steps = np.diff(grid)
        assert np.all(steps >= 1)
        assert np.all(np.array(grid[1:-1]) <= np.ceil(np.array(grid[:-2]) * (1 + growth)))
        assert len(grid) <= math.log(maximum / minimum) / math.log1p(growth) + 1 / growth + 2
        # Exact rational ceiling of d (1 + growth), growth read as its decimal.
        ratio = Fraction(repr(growth))
        exact = [min(maximum, d + max(1, math.ceil(d * ratio))) for d in grid[:-1]]
        assert grid[1:] == exact


def test_decision_points_ceiling_is_exact_on_integer_steps():
    """``d delta`` integral: no binary-rounding step past ``d (1 + delta)``."""
    assert _grid(200, 300) == [200, 220, 242, 267, 294, 300]
    assert _grid(3000, 4000) == [3000, 3300, 3630, 3993, 4000]


def test_zero_stability_fraction_admits_only_a_still_mean():
    """``stability_fraction = 0`` is valid: a constant population (zero move,
    zero limit) stops instead of dividing by 0, and a moving mean is judged
    against the drift allowance alone."""
    base = dict(
        max_history_share=1.0,
        min_effective_histories=0.0,
        max_variance_of_variance=1.0,
        stability_fraction=0.0,
    )
    constant = AdaptiveSettings(0.05, 100, 400, **base)
    record = run_stopping_rule(constant, _population_measure(np.ones(400)))
    assert record["stop_reason"] == "converged"
    assert record["realized_electrons"] == _grid(100, 400)[3]
    moving = np.ones(400)
    moving[::2] = 1.001  # nonzero but tiny RSE; drift term well below the move
    settings = AdaptiveSettings(0.05, 100, 400, **base)
    record = run_stopping_rule(settings, _population_measure(moving))
    status = record["statistics"]["cxr"]
    assert status["stability"] > 0.0 and status["stability_limit"] > 0.0
    assert status["passes"]["stability"] == (status["stability"] <= status["stability_limit"])


# ---- per-electron observables ---------------------------------------------------


@pytest.mark.parametrize("band", [(2.2, 2.4), (1.5, 7.0), (-1.0, 20.0), (12.0, 13.0)])
def test_band_weights_integrate_linear_density_at_off_grid_edges(band):
    grid = np.array([1.0, 2.0, 4.0, 10.0])
    weights = band_weights(grid, *band)
    lo, hi = max(band[0], grid[0]), min(band[1], grid[-1])
    width = max(hi - lo, 0.0)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(width)
    # Exact integral of f(E) = 3E + 2, including bands with no interior node.
    expected = 1.5 * (hi**2 - lo**2) + 2.0 * width if hi > lo else 0.0
    assert weights @ (3.0 * grid + 2.0) == pytest.approx(expected)


def test_monitor_requires_one_sample_per_electron():
    settings = AdaptiveSettings(0.1, 40, 200, 20)
    monitor = StoppingMonitor(settings, _population_measure(np.ones(19)))
    with pytest.raises(ValueError, match="one value per electron"):
        monitor.on_block(0, 20, None)


def test_per_electron_brem_band_integrals_average_to_the_band_integral():
    kw = dict(
        E0_keV=30.0,
        thickness_ang=5.0e4,
        element="Si",
        n_atoms_per_ang3=0.04996,
        seed=4,
        transport_core="per-electron",
    )
    segments = simulate_trajectories(Ne=24, **kw)
    E_grid = np.linspace(1000.0, 20000.0, 200)
    q = band_weights(E_grid, 2000.0, 9000.0)
    common = dict(element="Si", n_atoms_per_ang3=0.04996, cross_section_model="eedl")
    spectrum = mc_brem_spectrum(segments, E_grid, **common)
    per_electron = mc_brem_spectrum(segments, E_grid, electron_band_weights=q, **common)
    assert per_electron.shape == (24,)
    assert np.all(per_electron >= 0.0)
    assert math.isclose(per_electron.mean(), q @ spectrum, rel_tol=1e-5 if _FP32 else 1e-10)


# ---- runner ---------------------------------------------------------------------


def _case(n_families=None, **beam):
    sweep = Sweep(
        material="hopg",
        n_families=n_families,
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0, **beam),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    return build_cases(sweep)[0]


def _assert_identical(got, ref):
    assert set(got) == set(ref)
    for key, value in ref.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value, err_msg=key)
        elif isinstance(value, int | float):
            assert got[key] == value, key


def _fixed(case, n):
    fixed = {**case, "Ne": n, "Ne_brem": n}
    return runner._spectrum_case(
        fixed, runner._transport_case(fixed, transport_core="per-electron")
    )


_SETTINGS = AdaptiveSettings(
    target_rse=0.1,
    min_histories=40,
    max_histories=200,
    block_histories=20,
    observables=("cxr", "brem"),
)


def test_adaptive_run_is_the_fixed_n_run_at_its_realized_count():
    """Energy spread and a bunch: the realized population's table range and
    centroid are its own, so the run replays transport when they differ."""
    case = _case(energy_spread_frac=1e-3, bunch_length_fs=5.0)
    # Exercise an early realized-range replay independently of the ESS floors.
    settings = replace(_SETTINGS, min_effective_histories=0.0, max_variance_of_variance=1.0)
    result, stats = run_case_adaptive(case, settings, transport_core="per-electron")
    n = stats["realized_electrons"]
    assert stats["stop_reason"] == "converged" and 40 <= n < 200 and n in _grid(40, 200)
    _assert_identical(result, _fixed(case, n))


@pytest.mark.slow
def test_adaptive_output_does_not_depend_on_chunk_size():
    """#1253 acceptance: chunk sizes (here 1, 7 and 64 electrons, including a
    stop inside a chunk and its re-transported prefix) change neither the
    realized count, the statistics nor the output."""
    case = _case(energy_spread_frac=1e-3)
    settings = replace(
        _SETTINGS, block_histories=None, min_effective_histories=0.0, max_variance_of_variance=1.0
    )
    runs = []
    for floor in (1, 7, 64):
        chunked = replace(settings, block_histories=floor)
        runs.append(run_case_adaptive(case, chunked, transport_core="per-electron"))
    (ref, ref_stats), *others = runs
    assert ref_stats["stop_reason"] == "converged"
    assert ref_stats["realized_electrons"] % 64  # the 64-electron chunk was cut
    for result, stats in others:
        assert stats["realized_electrons"] == ref_stats["realized_electrons"]
        assert (
            without_wall_clock(stats)["statistics"] == without_wall_clock(ref_stats)["statistics"]
        )
        _assert_identical(result, ref)


def test_same_seed_and_target_replay_the_same_count_and_output():
    case = _case()
    first, first_stats = run_case_adaptive(case, _SETTINGS, transport_core="per-electron")
    again, again_stats = run_case_adaptive(case, _SETTINGS, transport_core="per-electron")
    assert without_wall_clock(first_stats) == without_wall_clock(again_stats)
    _assert_identical(again, first)


def test_min_equal_max_is_the_fixed_n_run():
    case = _case()
    settings = AdaptiveSettings(
        target_rse=0.1, min_histories=60, max_histories=60, block_histories=20
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CxrYieldStatisticsWarning)  # ESS < 100 at 60
        result, stats = run_case_adaptive(case, settings, transport_core="per-electron")
    assert stats["realized_electrons"] == 60 and not stats["replayed_transport"]
    _assert_identical(result, _fixed(case, 60))


def test_unconverged_run_completes_statistics_limited():
    settings = AdaptiveSettings(
        target_rse=1e-4, min_histories=20, max_histories=40, block_histories=20
    )
    with pytest.warns(CxrYieldStatisticsWarning, match="max_histories=40") as caught:
        result, stats = run_case_adaptive(_case(), settings, transport_core="per-electron")
    (limit,) = [str(w.message) for w in caught if "adaptive stopping" in str(w.message)]
    assert "CXR at 40/40 electrons: RSE" in limit and "> target 0.0001 (missed)" in limit
    assert stats["statistics_limited"] and stats["stop_reason"] == "max_electrons"
    assert stats["realized_electrons"] == 40
    assert result["spec"].size == result["E_grid"].size


def test_batch_means_reconstruct_the_band_of_the_final_spectra():
    settings = AdaptiveSettings(
        target_rse=1.0,
        min_histories=60,
        max_histories=60,
        block_histories=20,
        batch_means_band_eV=(1000.0, 3000.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CxrYieldStatisticsWarning)
        result, stats = run_case_adaptive(_case(), settings, transport_core="per-electron")
    batch = stats["batch_means"]
    assert batch["n_batches"] == 20
    for key, grid in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = result[grid]
        band = (energies >= 1000.0) & (energies <= 3000.0)
        np.testing.assert_allclose(
            batch[key]["mean"], result[key][band], rtol=1e-5 if _FP32 else 1e-9, atol=0.0
        )
        assert np.all(batch[key]["standard_error"] >= 0.0)


def test_coherent_route_rejects_auto():
    case = {**_case(), "coherent_emission": True}
    with pytest.raises(ValueError, match="coherent"):
        run_case_adaptive(case, _SETTINGS, transport_core="per-electron")


def test_lockstep_core_is_rejected_and_auto_avoids_it():
    with pytest.raises(ValueError, match="lockstep"):
        run_case_adaptive(_case(), _SETTINGS, transport_core="lockstep")
    from pyrite.montecarlo.runner.adaptive import adaptive_transport_core

    assert adaptive_transport_core(_case(), _SETTINGS, "auto") in {"per-electron", "cuda"}


def test_environment_pinned_lockstep_is_rejected(monkeypatch):
    from pyrite.montecarlo.runner.adaptive import adaptive_transport_core

    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    with pytest.raises(ValueError, match="lockstep"):
        adaptive_transport_core(_case(), _SETTINGS)


def test_energy_range_replay_rechecks_statistics_before_accepting_stop(monkeypatch):
    from pyrite.montecarlo.runner import block_transport

    settings = AdaptiveSettings(0.1, 40, 80, **_RULE_ONLY)
    monitor = StoppingMonitor(settings, lambda _start, _stop, block: {"cxr": block})
    calls = []

    def run_chunks(_simulate, plan, monitor, n, *, recheck=False, **_kw):
        calls.append(n)
        start = 0
        while start < n:
            stop = plan.next_stop(start, monitor.next_point(), n)
            values = np.ones(stop - start)
            # The first replay invalidates the apparent convergence at 40.
            if len(calls) == 2 and start == 20:
                values = np.arange(1.0, stop - start + 1.0)
            monitor.on_block(start, stop, values)
            if monitor.should_stop(stop, n if recheck else None):
                return {"Ne": monitor.realized}
            start = stop
        return {"Ne": n}

    monkeypatch.setattr(block_transport, "transport_monitored_blocks", run_chunks)
    monkeypatch.setattr(block_transport, "initial_energies_keV", lambda *_: np.arange(1.0, 81.0))
    monkeypatch.setattr(runner, "_case_radiative_kwargs", lambda _case: {})
    result = block_transport.transport_case_blocks(
        False,
        simulate=None,
        case={"seed": 1, "E0_keV": 30.0},
        E_cut_by_electrons=np.ones(80),
        block_electrons=AdaptiveChunkPlan(20),
        beam_kw={key: None for key in block_transport.BUNCH_KWARGS},
        monitor=monitor,
    )
    grow = _grid(40, 80)[1]
    assert calls == [80, 40, grow]
    assert result["Ne"] == grow
    record = monitor.record()
    assert record["replayed_transport"]
    assert record["realized_electrons"] == grow
    assert record["statistics"]["cxr"]["n_electrons"] == grow
    assert record["statistics"]["cxr"]["mean"] == 1.0
    assert record["stop_reason"] == "converged"


# ---- resident OOM recovery (#1269) ----------------------------------------------


class _AllocationError(Exception):
    pass


def _resident_adaptive(monkeypatch, fail_at=None):
    """Adaptive transport as a resident CUDA run: blocks with ``start >= fail_at``
    raise an allocator OOM while resident; host blocks run the per-electron core."""
    from pyrite.montecarlo.runner import case_driver, pool
    from pyrite.montecarlo.runner.adaptive import transport_case_adaptive

    real = case_driver.simulate_trajectories
    calls, events = [], []

    def simulate(*args, keep_segments_on_device, **kwargs):
        start = kwargs.get("_electron_start", 0)
        calls.append((start, start + args[1], keep_segments_on_device))
        if keep_segments_on_device and fail_at is not None and start >= fail_at:
            raise _AllocationError("out of memory allocating block")
        return real(*args, keep_segments_on_device=False, **kwargs)

    monkeypatch.setattr(case_driver, "simulate_trajectories", simulate)
    monkeypatch.setattr(case_driver, "_resident_transport", lambda *_: True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (_AllocationError,))
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: events.append("release"))
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: events.append("admit"))
    settings = replace(_SETTINGS, min_effective_histories=0.0, max_variance_of_variance=1.0)
    tp = transport_case_adaptive(
        _case(), settings, transport_core="per-electron", keep_segments_on_device=True
    )
    return tp, calls, events


def test_late_resident_oom_reruns_only_the_failed_block(monkeypatch):
    """#1269 acceptance: a late-block OOM in a monitored resident run moves the
    kept blocks to the host and reruns only the failed block; no completed
    block is transported again and the join is the no-OOM transport."""
    ref, ref_calls, _ = _resident_adaptive(monkeypatch)
    spans = [(start, stop) for start, stop, _keep in ref_calls]
    assert len(spans) >= 3 and all(keep for *_, keep in ref_calls)
    fail_at = spans[-1][0]

    tp, calls, events = _resident_adaptive(monkeypatch, fail_at=fail_at)

    # The first resident call at fail_at fails and reruns on the host; every
    # later call (a stop's re-transported prefix included) stays on the host.
    failed = next(i for i, (start, _stop) in enumerate(spans) if start >= fail_at)
    assert calls == [
        *ref_calls[:failed],
        (*spans[failed], True),
        *((start, stop, False) for start, stop in spans[failed:]),
    ]
    assert events[events.index("admit") - 1] == "release"
    assert tp["segs"]["Ne"] == ref["segs"]["Ne"]
    assert without_wall_clock(tp["adaptive_sampling"]) == without_wall_clock(
        ref["adaptive_sampling"]
    )
    _assert_identical(tp["segs"], ref["segs"])
    assert tp["_transport_gpu_oom_retries"] == 1
    assert tp["_recovery_path"] == "host_segments"
    assert tp["_recovery_events"] == [
        {
            "phase": "transport",
            "mitigation": "host_segments",
            "stage": "block",
            "block_start": fail_at,
        }
    ]


def test_resident_join_oom_merges_host_copies(monkeypatch):
    from pyrite.montecarlo.runner import pool
    from pyrite.montecarlo.runner.block_transport import merge_electron_blocks
    from pyrite.montecarlo.runner.recovery import ResidentBlockRecovery

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (_AllocationError,))
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: None)
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: None)
    shared = np.arange(3)
    kept = [
        {"Ne": 3, "electron_id": shared, "elec_id": shared},
        {"Ne": 1, "electron_id": np.arange(3, 4)},
    ]
    kept[1]["elec_id"] = kept[1]["electron_id"]
    recovery = ResidentBlockRecovery(lambda *_a, **_kw: None, resident=True)
    tries = []

    def merge(blocks):
        tries.append(len(blocks))
        if len(tries) == 1:
            raise _AllocationError("join")
        return merge_electron_blocks(blocks)

    merged = recovery.join(merge, kept, 4)
    assert tries == [2, 2] and not recovery.resident
    assert merged["Ne"] == 4 and merged["electron_id"] is merged["elec_id"]
    assert recovery.metrics()["_recovery_events"][0]["stage"] == "join"


def test_adaptive_host_admission_refuses_before_kept_growth(monkeypatch):
    from pyrite.montecarlo.runner.adaptive import transport_case_adaptive
    from pyrite.montecarlo.transport import host_memory

    calls = []
    real = runner.case_driver.simulate_trajectories

    def simulate(*args, **kwargs):
        calls.append(kwargs.get("_electron_start", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", simulate)
    original_admit = host_memory.admit_host_allocation

    def admit(required, phase):
        if phase.startswith("adaptive kept blocks"):
            raise host_memory.HostMemoryResourceError(phase, required, 0)
        original_admit(required, phase)

    monkeypatch.setattr(host_memory, "admit_host_allocation", admit)
    with pytest.raises(host_memory.HostMemoryResourceError, match="lower max_histories.*#1280"):
        transport_case_adaptive(_case(), _SETTINGS, transport_core="per-electron")
    assert calls == [0]


def test_adaptive_host_admission_feasible_transport_is_identical(monkeypatch):
    from pyrite.montecarlo.runner.adaptive import transport_case_adaptive
    from pyrite.montecarlo.transport import host_memory

    settings = replace(_SETTINGS, min_effective_histories=0.0, max_variance_of_variance=1.0)
    reference = transport_case_adaptive(_case(), settings, transport_core="per-electron")
    admitted = []
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 10**12)
    real = host_memory.admit_host_allocation

    def admit(required, phase):
        if phase.startswith("adaptive kept blocks"):
            admitted.append((required, phase))
        real(required, phase)

    monkeypatch.setattr(host_memory, "admit_host_allocation", admit)
    candidate = transport_case_adaptive(_case(), settings, transport_core="per-electron")
    assert any("join" in phase for _, phase in admitted)
    _assert_identical(candidate["segs"], reference["segs"])
    assert without_wall_clock(candidate["adaptive_sampling"]) == without_wall_clock(
        reference["adaptive_sampling"]
    )


def test_measure_skips_the_grid_reduction_with_identical_line_masses(monkeypatch):
    """#1286: the measure reads only ``electron_mass``; skipping the reduction leaves it unchanged.

    Bitwise on NumPy; CuPy's ``bincount`` adds with atomics in unspecified order.
    """
    from pyrite._backend import BACKEND, _to_cpu
    from pyrite.montecarlo.spectrum.cxr import _batched, _per_hkl

    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    case = {**dict(build_cases(sweep)[0]), "seed": 3, "Ne": 40, "Ne_brem": 40}
    tp = runner._transport_case(case, transport_core="per-electron")
    settings = AdaptiveSettings(
        target_rse=1.0, min_histories=40, max_histories=40, block_histories=40
    )
    seen = []
    for module in (_batched, _per_hkl):
        real = module._audit_skips_reduction
        monkeypatch.setattr(
            module, "_audit_skips_reduction", lambda a, real=real: seen.append(a) or real(a)
        )
    skipped = case_measure(case, settings)(0, 40, tp["segs"])["cxr"]
    assert seen and all(a["mass_only"] for a in seen)
    for module in (_batched, _per_hkl):
        monkeypatch.setattr(
            module, "_audit_skips_reduction", lambda a: a is not None and "collect" in a
        )
    reduced = case_measure(case, settings)(0, 40, tp["segs"])["cxr"]
    skipped, reduced = _to_cpu(skipped), _to_cpu(reduced)
    assert skipped.sum() > 0.0
    if BACKEND.name == "cpu":
        np.testing.assert_array_equal(skipped, reduced)
    else:
        np.testing.assert_allclose(skipped, reduced, rtol=1e-12, atol=0.0)
