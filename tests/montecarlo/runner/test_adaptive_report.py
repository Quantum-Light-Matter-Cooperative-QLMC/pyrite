"""Explanations of adaptive stops that missed their target or guards (#1267)."""

import warnings

import numpy as np
import pytest

from pyrite._cxr_grid_policy import CxrYieldStatisticsWarning
from pyrite.montecarlo.runner.adaptive import (
    AdaptiveSettings,
    run_stopping_rule,
    warn_statistics_limited,
)
from pyrite.montecarlo.runner.adaptive_report import failure_kind, limit_summary

_RULE_ONLY = dict(max_history_share=1.0, min_effective_histories=0.0, stability_checks=0)
_GUARDS = {
    "max_electron_share": 0.05,
    "min_effective_electrons": 100.0,
    "max_variance_of_variance": 0.01,
    "stability_checks": 3,
    "stability_fraction": 0.5,
}
_ALL_PASS = {
    "relative_se": True,
    "max_electron_share": True,
    "effective_sample_size": True,
    "variance_sampling": True,
    "stability": True,
}


def _status(n=1_002_000, rse=0.0005, vov=0.0, **passes):
    return {
        "n_electrons": n,
        "relative_se": rse,
        "max_electron_share": 0.001,
        "effective_sample_size": 5e5,
        "variance_of_variance": vov,
        "stability": 0.0,
        "stability_limit": 5e-4,
        "passes": {**_ALL_PASS, **passes},
        "converged": all({**_ALL_PASS, **passes}.values()),
    }


def _record(statistics, target=0.001, maximum=1_002_000):
    return {
        "target_rse": target,
        "max_electrons": maximum,
        "guards": _GUARDS,
        "statistics": statistics,
    }


def _lognormal(n, seed=3):
    return np.random.default_rng(seed).lognormal(0.0, 1.0, n)


def test_vov_only_refusal_says_rse_passed_and_reports_the_gated_quantity():
    """A real stopping record: RSE met, VOV + 1/n above a tightened gate."""
    population = _lognormal(2000)
    settings = AdaptiveSettings(0.1, 1000, 2000, 1000, max_variance_of_variance=5e-4, **_RULE_ONLY)
    record = run_stopping_rule(settings, lambda a, b, _: {"cxr": population[a:b]})
    status = record["statistics"]["cxr"]
    assert record["stop_reason"] == "max_electrons"
    assert status["passes"]["relative_se"] and not status["passes"]["variance_sampling"]
    assert failure_kind(status) == "vov"
    summary = limit_summary(record)
    gate = status["variance_of_variance"] + 1.0 / status["n_electrons"]
    assert f"RSE {status['relative_se']:.3g} <= target 0.1 (met)" in summary
    assert f"VOV gate (VOV + 1/n) {gate:.3g} > 0.0005" in summary
    assert "missed" not in summary
    assert "looser RSE target would not lift" in summary
    assert "max electron share" not in summary and "ESS" not in summary


@pytest.mark.parametrize(
    ("energy", "cxr_rse", "brem_rse"), [(30, 0.0548115, 0.279031), (60, 0.230631, 0.007280)]
)
def test_reported_cleo_sub_examples_stay_rse_limited_without_inventing_vov(
    energy, cxr_rse, brem_rse
):
    """The #1267 warnings: both RSEs miss 0.001; their VOV results were not observed."""
    record = _record(
        {
            "cxr": _status(rse=cxr_rse, relative_se=False),
            "brem": _status(rse=brem_rse, relative_se=False),
        }
    )
    summary = limit_summary(record)
    assert failure_kind(record["statistics"]["cxr"]) == "rse"
    assert f"CXR at 1002000/1002000 electrons: RSE {cxr_rse:.3g} > target 0.001 (missed)" in (
        summary
    )
    assert f"brem at 1002000/1002000 electrons: RSE {brem_rse:.3g} > target 0.001" in summary
    assert "VOV" not in summary


def test_mixed_and_undefined_criteria_list_each_failure_with_its_threshold():
    status = _status(
        rse=None,
        relative_se=False,
        max_electron_share=False,
        variance_sampling=False,
        stability=False,
    )
    status.update(variance_of_variance=None, stability=None, max_electron_share=0.4)
    summary = limit_summary(_record({"cxr": status}))
    assert failure_kind(status) == "rse+guards"
    assert "RSE undefined" in summary
    assert "max electron share 0.4 > 0.05" in summary
    assert "VOV undefined" in summary
    assert "mean stability not evaluable" in summary
    assert "ESS" not in summary and "would not lift" not in summary

    mixed = _status(rse=0.002, vov=0.05, relative_se=False, variance_sampling=False)
    assert failure_kind(mixed) == "rse+vov"
    text = limit_summary(_record({"cxr": mixed}))
    assert "RSE 0.002 > target 0.001 (missed)" in text
    assert f"VOV gate (VOV + 1/n) {0.05 + 1 / 1_002_000:.3g} > 0.01" in text


def test_bypassed_or_disabled_guards_are_not_reported_as_failures():
    """Near-constant CV and a disabled gate pass ``variance_sampling``; ESS 0 disables ESS."""
    nearly_constant = np.ones(2000)
    nearly_constant[7] += 0.01
    # 1/n alone exceeds the 5e-4 gate; only the near-constant exception (CV <= target) passes.
    settings = AdaptiveSettings(1e-3, 1000, 2000, 1000, max_variance_of_variance=5e-4, **_RULE_ONLY)
    record = run_stopping_rule(settings, lambda a, b, _: {"cxr": nearly_constant[a:b]})
    assert record["stop_reason"] == "converged"
    assert limit_summary(record) == "CXR at 1000/2000 electrons: converged"

    population = _lognormal(2000)
    settings = AdaptiveSettings(1e-9, 1000, 2000, 1000, max_variance_of_variance=1.0, **_RULE_ONLY)
    record = run_stopping_rule(settings, lambda a, b, _: {"cxr": population[a:b]})
    assert record["statistics"]["cxr"]["passes"]["variance_sampling"]
    summary = limit_summary(record)
    assert "VOV" not in summary and "ESS" not in summary and "(missed)" in summary


def test_independent_brem_limit_names_its_own_target_count_and_cap():
    population = _lognormal(6000)
    settings = AdaptiveSettings(
        0.5,
        1000,
        6000,
        1000,
        observables=("cxr", "brem"),
        brem_target_rse=1e-4,
        brem_max_histories=3000,
        max_variance_of_variance=1.0,
        **_RULE_ONLY,
    )
    record = run_stopping_rule(
        settings, lambda a, b, _: dict.fromkeys(("cxr", "brem"), population[a:b])
    )
    assert record["statistics_limited"] and record["realized_electrons_brem"] == 3000
    summary = limit_summary(record)
    cxr_n = record["statistics"]["cxr"]["n_electrons"]
    assert f"CXR at {cxr_n}/6000 electrons: converged" in summary
    assert "brem at 3000/3000 electrons: RSE" in summary and "> target 0.0001" in summary
    case = {"name": "cleo_sub", "E0_keV": 30.0}
    with pytest.warns(CxrYieldStatisticsWarning, match=r"Ne_brem=3000 without converging"):
        warn_statistics_limited(case, settings, record)


def test_shared_limit_warning_keeps_cap_and_summary():
    population = _lognormal(2000)
    settings = AdaptiveSettings(1e-4, 1000, 2000, 1000, max_variance_of_variance=1.0, **_RULE_ONLY)
    record = run_stopping_rule(settings, lambda a, b, _: {"cxr": population[a:b]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_statistics_limited({"name": "cleo_sub", "E0_keV": 60.0}, settings, record)
    (message,) = [str(w.message) for w in caught]
    assert message.startswith(
        "cleo_sub at 60 keV: adaptive stopping reached max_histories=2000 without converging. "
        "CXR at 2000/2000 electrons: RSE "
    )
    assert "(missed)" in message and "statistics-limited" in message
