"""Held-out binary endpoints retain finite uncertainty and fixed-population gates."""

import runpy
from pathlib import Path

import numpy as np
import pytest

CHECKS = Path(__file__).parents[2] / "checks/equal_precision"


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(CHECKS))
    return runpy.run_path(str(CHECKS / "qualify_moment_fractions.py"))


def test_identical_outcomes_need_sufficient_fresh_population(diagnostic):
    bound = diagnostic["fraction_bound"]
    small = bound(np.zeros(100), np.zeros(100), alpha=0.01, comparisons=32, margin=0.002)
    large = bound(np.zeros(6144), np.zeros(6144), alpha=0.01, comparisons=32, margin=0.002)
    assert small["status"] == "unresolved"
    assert large["status"] == "passed"
    assert 0 < large["discordance_probability_upper"] < 0.002
    # Under p=.002, probability of zero mismatches is below the per-test budget.
    assert (1 - 0.002) ** 6144 < 0.01 / 32


def test_discordant_population_cannot_pass(diagnostic):
    r = diagnostic["fraction_bound"](
        np.zeros(6144), np.ones(6144), alpha=0.01, comparisons=32, margin=0.002
    )
    assert r["status"] == "unresolved"
    assert r["difference_ci"] == [-1.0, 1.0]


@pytest.mark.parametrize("candidate", [np.full(10, np.nan), np.full(10, 2), np.zeros(9)])
def test_bad_binary_scores_refused(diagnostic, candidate):
    with pytest.raises(ValueError):
        diagnostic["fraction_bound"](
            np.zeros(10), candidate, alpha=0.01, comparisons=32, margin=0.002
        )


def test_missing_groups_never_pass(diagnostic, tmp_path):
    import json

    policy = json.loads((CHECKS / "moment-heldout-788.json").read_text())
    result = diagnostic["qualify"](tmp_path, policy)
    assert result["comparisons"] == 32
    assert len(result["results"]) == 16
    assert all(not group["complete"] for group in result["results"])
    assert all(
        check["status"] == "unresolved"
        for group in result["results"]
        for check in group["checks"].values()
    )
