"""Transport step-budget retry for thick MeV cases (#192)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
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
