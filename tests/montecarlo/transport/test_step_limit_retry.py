"""Transport step-budget retry for thick MeV cases (#192)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.step_budget import retry_step_budget
from pyrite.montecarlo.transport import TransportStepLimitError


@pytest.fixture(scope="module")
def case():
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=100.0))
    return build_cases(sweep, n_electrons=4, n_electrons_brem=4)[0]


def _budget_limited(monkeypatch, fail_below):
    real = runner.simulate_trajectories
    budgets = []

    def simulate(*args, max_steps, **kwargs):
        budgets.append(max_steps)
        if max_steps < fail_below:
            raise TransportStepLimitError(1, args[1], max_steps)
        return real(*args, max_steps=max_steps, **kwargs)

    monkeypatch.setattr(runner, "simulate_trajectories", simulate)
    return budgets


def test_step_limit_doubles_the_budget_and_matches_a_direct_run(monkeypatch, case):
    direct = runner._transport_case(case, transport_core="lockstep")
    budgets = _budget_limited(monkeypatch, fail_below=4 * runner.TRANSPORT_MAX_STEPS)
    retried = runner._transport_case(case, transport_core="lockstep")
    base = runner.TRANSPORT_MAX_STEPS
    assert budgets == [base, 2 * base, 4 * base]
    for key in ("L_ang", "E_keV", "r_mid"):
        np.testing.assert_array_equal(retried["segs"][key], direct["segs"][key])


def test_step_limit_gives_up_at_the_ceiling(monkeypatch, case):
    budgets = _budget_limited(monkeypatch, fail_below=np.inf)
    with pytest.raises(TransportStepLimitError, match="incomplete electron transport"):
        runner._transport_case(case, transport_core="lockstep")
    assert budgets[-1] == runner.TRANSPORT_MAX_STEPS_CEILING


def test_smaller_initial_budget_preserves_transport(monkeypatch, case):
    reference = runner._transport_case(case, transport_core="lockstep")
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "128")
    reduced = runner._transport_case(case, transport_core="lockstep")
    for key in ("L_ang", "E_keV", "r_mid"):
        np.testing.assert_array_equal(reduced["segs"][key], reference["segs"][key])


def test_small_initial_budget_retries_without_truncating(monkeypatch):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", "128")
    budgets = []

    def run(*, max_steps):
        budgets.append(max_steps)
        if max_steps < 512:
            raise TransportStepLimitError(1, 1, max_steps)
        return "transported"

    assert retry_step_budget(run) == "transported"
    assert budgets == [128, 256, 512]


def test_retry_cannot_exceed_the_original_ceiling(monkeypatch):
    ceiling = runner.TRANSPORT_MAX_STEPS_CEILING
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", str(ceiling - 1))
    budgets = []

    def fail(*, max_steps):
        budgets.append(max_steps)
        raise TransportStepLimitError(1, 1, max_steps)

    with pytest.raises(TransportStepLimitError):
        retry_step_budget(fail)
    assert budgets == [ceiling - 1, ceiling]


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "1.5", "160001"])
def test_invalid_initial_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", value)
    with pytest.raises(ValueError, match="PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS"):
        retry_step_budget(lambda *, max_steps: None)
