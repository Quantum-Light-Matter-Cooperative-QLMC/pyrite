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

import numpy as np
import pytest

from pyrite._line_grid_policy import LineYieldStatisticsWarning
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import (
    AdaptiveSettings,
    RunningMoments,
    StoppingMonitor,
    band_weights,
    case_measure,
    run_case_adaptive,
    run_stopping_rule,
)
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.transport import simulate_trajectories

#: Guards disabled: the bare relative-standard-error rule.
_RULE_ONLY = dict(max_electron_share=1.0, min_effective_electrons=0.0, stability_blocks=0)


def _population_measure(population):
    return lambda start, stop, _block: {"line": population[start:stop]}


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
        dict(min_electrons=1),
        dict(max_electrons=30),
        dict(min_electrons=45),
        dict(pilot_electrons=50),
        dict(observables=("spec",)),
        dict(max_electron_share=0.0),
        dict(block_electrons=20.5),
        dict(block_electrons=True),
        dict(min_electrons=40.5),
        dict(max_electrons=200.5),
        dict(pilot_electrons=20),
        dict(pilot_electrons=220),
        dict(stability_blocks=1.5),
        dict(stability_fraction=float("nan")),
        dict(min_effective_electrons=float("inf")),
        dict(observables=("line", "line")),
        dict(band_eV=(2000.0, 1000.0)),
        dict(band_eV=(0.0, 1000.0)),
        dict(batch_means_band_eV=(1000.0, float("inf"))),
    ],
)
def test_settings_are_validated(bad):
    kw = dict(target_rse=0.1, min_electrons=40, max_electrons=200, block_electrons=20) | bad
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
        target_rse=target, min_electrons=50, max_electrons=100_000, block_electrons=10, **guards
    )
    errors, rses = [], []
    for seed in range(1000):
        population = np.random.default_rng(seed).normal(1.0, 0.5, 2000)
        record = run_stopping_rule(settings, _population_measure(population))
        assert record["stop_reason"] == "converged"
        line = record["statistics"]["line"]
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
        target_rse=1.0, min_electrons=_LIGHT_N, max_electrons=_LIGHT_N, block_electrons=_LIGHT_N
    )
    populations = []
    for seed in _LIGHT_SEEDS:
        case = {**base, "seed": seed, "Ne": _LIGHT_N, "Ne_brem": _LIGHT_N}
        tp = runner._transport_case(case, transport_core="per-electron")
        populations.append(
            np.asarray(case_measure(case, settings)(0, _LIGHT_N, tp["segs"])["line"])
        )
    return base, populations


def test_light_case_meets_its_target_over_20_seeds(light_line_masses):
    """Target 10 % on the per-electron line mass (skewness ~ 9).

    Truth is the pooled mean of all 16000 electrons (relative SE ~ 0.8 %).
    The achieved error must lie within 1.96 x target in at least 17 of 20
    seeds (P <= 1.6 % if the true coverage were 95 %). The stopping bias is
    measured paired, against each seed's own 800-electron mean, which cancels
    the shared electrons; it must be below the reported error.
    """
    _, populations = light_line_masses
    target = 0.1
    settings = AdaptiveSettings(
        target_rse=target, min_electrons=200, max_electrons=_LIGHT_N, block_electrons=20
    )
    truth = np.concatenate(populations).mean()
    records = [run_stopping_rule(settings, _population_measure(p)) for p in populations]
    assert all(r["stop_reason"] == "converged" for r in records)
    means = np.array([r["statistics"]["line"]["mean"] for r in records])
    reported = np.array([r["statistics"]["line"]["relative_se"] for r in records])
    assert np.all(reported <= target)
    within = np.abs(means / truth - 1.0) <= 1.96 * target
    assert within.sum() >= 17
    paired = np.array([(m - p.mean()) / truth for m, p in zip(means, populations, strict=True)])
    assert abs(paired.mean()) < reported.min()


def test_stop_inside_transport_equals_the_rule_on_the_population(light_line_masses):
    """The monitor inside block transport reads the per-electron prefix exactly."""
    base, populations = light_line_masses
    settings = AdaptiveSettings(
        target_rse=0.1, min_electrons=200, max_electrons=_LIGHT_N, block_electrons=20
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
    base = dict(target_rse=0.02, min_electrons=50, max_electrons=5000, block_electrons=10)
    plain = run_stopping_rule(AdaptiveSettings(**base), _population_measure(population))
    piloted = run_stopping_rule(
        AdaptiveSettings(**base, pilot_electrons=100), _population_measure(population)
    )
    pilot = piloted["pilot"]
    rse = pilot["relative_se"]["line"]
    assert pilot["projected_electrons"] == 10 * math.ceil(100 * (rse / 0.02) ** 2 / 10)
    assert piloted["realized_electrons"] >= pilot["projected_electrons"]
    assert piloted["checks"] < plain["checks"]
    assert piloted["stop_reason"] == "converged"


# ---- heavy tails --------------------------------------------------------------


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

    Without guards the first 200 electrons hold none: the RSE (~7 %) passes and
    the run stops 44 % low. The ESS floor (1000) forces enough electrons to
    meet the heavy ones, the share cap then refuses to stop on a handful of
    them, and the run ends statistics_limited at max_electrons.
    """
    population = _heavy_population(np.random.default_rng(0), 4000, every=500)
    base = dict(target_rse=0.1, min_electrons=200, max_electrons=4000, block_electrons=100)
    measure = _population_measure(population)
    rule = run_stopping_rule(AdaptiveSettings(**base, **_RULE_ONLY), measure)
    assert rule["stop_reason"] == "converged"
    assert rule["realized_electrons"] == 200
    line = rule["statistics"]["line"]
    assert abs(line["mean"] / 1.8 - 1.0) > 5 * line["relative_se"]

    guarded = run_stopping_rule(AdaptiveSettings(**base, min_effective_electrons=1000.0), measure)
    assert guarded["stop_reason"] == "max_electrons"
    assert guarded["statistics_limited"]
    assert guarded["realized_electrons"] == 4000
    status = guarded["statistics"]["line"]
    assert not status["passes"]["max_electron_share"]


def test_guards_suppress_false_stops_across_seeds():
    """Rate 2e-3: rule-only stops falsely (error > 3 x reported) in most seeds;
    guarded runs do so in at most one of 40 (the tail is then unsampled even
    at the ESS floor: P(no heavy in 2000) = e^-4 ~ 2 %)."""
    base = dict(target_rse=0.1, min_electrons=200, max_electrons=4000, block_electrons=100)
    false_stops = {}
    for label, guards in (("rule", _RULE_ONLY), ("guarded", {"min_effective_electrons": 1000.0})):
        settings = AdaptiveSettings(**base, **guards)
        count = 0
        for seed in range(40):
            population = _heavy_population(np.random.default_rng(seed), 4000, rate=2e-3)
            record = run_stopping_rule(settings, _population_measure(population))
            line = record["statistics"]["line"]
            if record["stop_reason"] == "converged":
                assert line["max_electron_share"] <= settings.max_electron_share
                count += abs(line["mean"] / 1.8 - 1.0) > 3 * line["relative_se"]
        false_stops[label] = count
    assert false_stops["rule"] >= 20
    assert false_stops["guarded"] <= 1


def test_stability_guard_waits_out_a_late_heavy_electron():
    """A heavy electron moves the running mean by ~17 %: the RSE (~17 %) meets
    the 20 % target at once, but the mean must hold within 10 % over three
    block ends, so the stop waits until the jump leaves that window."""
    population = np.ones(400)
    population[::2] = 0.5  # RSE is tiny from the start
    population[255] = 40.0  # block [240, 260)
    settings = AdaptiveSettings(
        target_rse=0.2,
        min_electrons=260,
        max_electrons=400,
        block_electrons=20,
        max_electron_share=1.0,
        min_effective_electrons=0.0,
        stability_blocks=3,
        stability_fraction=0.5,
    )
    record = run_stopping_rule(settings, _population_measure(population))
    assert record["realized_electrons"] == 320
    assert record["stop_reason"] == "converged"


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
    assert math.isclose(per_electron.mean(), q @ spectrum, rel_tol=1e-10)


# ---- runner ---------------------------------------------------------------------


def _case(**beam):
    sweep = Sweep(
        material="hopg",
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
    min_electrons=40,
    max_electrons=200,
    block_electrons=20,
    observables=("line", "brem"),
)


def test_adaptive_run_is_the_fixed_n_run_at_its_realized_count():
    """Energy spread and a bunch: the realized population's table range and
    centroid are its own, so the run replays transport when they differ."""
    case = _case(energy_spread_frac=1e-3, bunch_length_fs=5.0)
    result, stats = run_case_adaptive(case, _SETTINGS, transport_core="per-electron")
    n = stats["realized_electrons"]
    assert stats["stop_reason"] == "converged" and 40 <= n < 200 and n % 20 == 0
    _assert_identical(result, _fixed(case, n))


def test_same_seed_and_target_replay_the_same_count_and_output():
    case = _case()
    first, first_stats = run_case_adaptive(case, _SETTINGS, transport_core="per-electron")
    again, again_stats = run_case_adaptive(case, _SETTINGS, transport_core="per-electron")
    assert first_stats == again_stats
    _assert_identical(again, first)


def test_min_equal_max_is_the_fixed_n_run():
    case = _case()
    settings = AdaptiveSettings(
        target_rse=0.1, min_electrons=60, max_electrons=60, block_electrons=20
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LineYieldStatisticsWarning)  # ESS < 100 at 60
        result, stats = run_case_adaptive(case, settings, transport_core="per-electron")
    assert stats["realized_electrons"] == 60 and not stats["replayed_transport"]
    _assert_identical(result, _fixed(case, 60))


def test_unconverged_run_completes_statistics_limited():
    settings = AdaptiveSettings(
        target_rse=1e-4, min_electrons=20, max_electrons=40, block_electrons=20
    )
    with pytest.warns(LineYieldStatisticsWarning, match="max_electrons=40"):
        result, stats = run_case_adaptive(_case(), settings, transport_core="per-electron")
    assert stats["statistics_limited"] and stats["stop_reason"] == "max_electrons"
    assert stats["realized_electrons"] == 40
    assert result["spec"].size == result["E_grid"].size


def test_batch_means_reconstruct_the_band_of_the_final_spectra():
    settings = AdaptiveSettings(
        target_rse=1.0,
        min_electrons=60,
        max_electrons=60,
        block_electrons=20,
        batch_means_band_eV=(1000.0, 3000.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LineYieldStatisticsWarning)
        result, stats = run_case_adaptive(_case(), settings, transport_core="per-electron")
    batch = stats["batch_means"]
    assert batch["n_batches"] == 3
    for key, grid in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        energies = result[grid]
        band = (energies >= 1000.0) & (energies <= 3000.0)
        np.testing.assert_allclose(batch[key]["mean"], result[key][band], rtol=1e-9, atol=0.0)
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

    settings = AdaptiveSettings(0.1, 40, 80, 20, **_RULE_ONLY)
    monitor = StoppingMonitor(settings, lambda _start, _stop, block: {"line": block})
    calls = []

    def run_blocks(_simulate, n, *, block_electrons, on_block=None, should_stop=None, **_kw):
        calls.append(n)
        for start, stop in block_transport.electron_blocks(n, block_electrons):
            values = np.ones(stop - start)
            # The first replay invalidates the apparent convergence at 40.
            if len(calls) == 2 and start == 20:
                values = np.arange(1.0, 21.0)
            if on_block is not None:
                on_block(start, stop, values)
            if should_stop is not None and should_stop(stop):
                break
        return {"Ne": stop}

    monkeypatch.setattr(block_transport, "transport_electron_blocks", run_blocks)
    monkeypatch.setattr(block_transport, "initial_energies_keV", lambda *_: np.arange(1.0, 81.0))
    monkeypatch.setattr(runner, "_case_radiative_kwargs", lambda _case: {})
    result = block_transport.transport_case_blocks(
        False,
        simulate=None,
        case={"seed": 1, "E0_keV": 30.0},
        E_cut_by_electrons=np.ones(80),
        block_electrons=20,
        beam_kw={key: None for key in block_transport.BUNCH_KWARGS},
        monitor=monitor,
    )
    assert calls == [80, 40, 60]
    assert result["Ne"] == 60
    record = monitor.record()
    assert record["replayed_transport"]
    assert record["realized_electrons"] == 60
    assert record["statistics"]["line"]["n_electrons"] == 60
    assert record["statistics"]["line"]["mean"] == 1.0
    assert record["stop_reason"] == "converged"
