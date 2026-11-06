"""Transport step-budget retry for thick MeV cases (#192)."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.campaign.trajectories import simulate_trajectories as public_simulate_trajectories
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.step_budget import retry_step_budget
from pyrite.montecarlo.transport import TransportStepLimitError, host_pass
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories


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

    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", simulate)
    return budgets


def test_step_limit_doubles_the_budget_and_matches_a_direct_run(monkeypatch, case):
    direct = runner._transport_case(case, transport_core="lockstep")
    base = 1000
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", str(base))
    budgets = _budget_limited(monkeypatch, fail_below=4 * base)
    retried = runner._transport_case(case, transport_core="lockstep")
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


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "1.5", "400001"])
def test_invalid_initial_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS", value)
    with pytest.raises(ValueError, match="PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS"):
        retry_step_budget(lambda *, max_steps: None)


def _lockstep_si(**overrides):
    return simulate_trajectories(
        E0_keV=30.0,
        Ne=20,
        thickness_ang=5.0e4,
        element="Si",
        n_atoms_per_ang3=0.04996,
        seed=11,
        transport_core="lockstep",
        elastic_model="sr",
        **overrides,
    )


def _reserved_budgets(monkeypatch, first):
    """Budgets of the fixed buffers actually reserved, from a first one of ``first``."""
    monkeypatch.setattr(host_pass, "FIXED_BUFFER_INITIAL_STEPS", first)
    budgets = []
    reserve = host_pass._row_buffers

    def record(p, c, sec_on):
        budgets.append(p.max_steps)
        return reserve(p, c, sec_on)

    monkeypatch.setattr(host_pass, "_row_buffers", record)
    return budgets


def test_default_step_limit_does_not_size_the_first_lockstep_buffers(monkeypatch):
    budgets = _reserved_budgets(monkeypatch, host_pass.FIXED_BUFFER_INITIAL_STEPS)
    _lockstep_si()
    assert budgets == [host_pass.FIXED_BUFFER_INITIAL_STEPS]


def test_lockstep_buffer_replay_is_the_run_at_the_full_limit(monkeypatch):
    direct = _lockstep_si(max_steps=20_000)
    longest = np.bincount(direct["elec_id"]).max()
    first = int(longest // 4)
    budgets = _reserved_budgets(monkeypatch, first)
    replayed = _lockstep_si(max_steps=20_000)
    assert budgets[0] == first and len(budgets) > 1 and budgets[-1] < 20_000
    for key in ("L_ang", "E_keV", "r_mid", "elec_id"):
        np.testing.assert_array_equal(replayed[key], direct[key])


def test_lockstep_buffer_replay_stops_at_the_requested_limit(monkeypatch):
    budgets = _reserved_budgets(monkeypatch, 2)
    with pytest.raises(TransportStepLimitError, match="max_steps=5"):
        _lockstep_si(max_steps=5)
    assert budgets == [2, 4, 5]


def test_every_default_step_limit_agrees():
    for function in (simulate_trajectories, public_simulate_trajectories):
        limit = inspect.signature(function).parameters["max_steps"].default
        assert limit == runner.TRANSPORT_MAX_STEPS == 400_000
