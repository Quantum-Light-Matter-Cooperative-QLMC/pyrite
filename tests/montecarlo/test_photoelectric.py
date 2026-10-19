"""EPDL photoelectric shell selection, Sauter angles and EADL energy closure.

Validation: photon-photoelectric-sampling
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.eadl_relaxation import (
    EADLRelaxation,
    EADLSubshellRelaxation,
    load_eadl_relaxation,
)
from pyrite.montecarlo.transport import photoelectric as photo
from pyrite.montecarlo.transport.photon_sampling import UniformDraws


@pytest.fixture
def atom(monkeypatch):
    def shell(d, binding, final=(), photon=(), rp=(), a1=(), a2=(), electron=(), ap=()):
        arrays = [np.array(a) for a in (final, photon, rp, a1, a2, electron, ap)]
        return EADLSubshellRelaxation(d, binding, 2, *arrays)

    relaxation = EADLRelaxation(
        4,
        (
            shell(1, 100, [2], [90], [0.4], [2], [2], [80], [0.6]),
            shell(2, 10),
        ),
    )
    monkeypatch.setattr(
        photo,
        "subshell_photoionization_cross_sections_ang2",
        lambda el, e: ((1, 2), np.array([1.0, 3.0])),
    )
    monkeypatch.setattr(
        photo,
        "photoionization_subshells",
        lambda el: tuple(
            SimpleNamespace(designator=d, binding_energy_eV=b) for d, b in [(1, 100), (2, 10)]
        ),
    )
    return relaxation


def _assert_energy(energy, event):
    budget = event.relaxation.energy_budget
    accounted = event.electron_energy_eV + budget.photon_eV + budget.electron_eV
    accounted += budget.terminal_binding_eV + budget.transition_energy_defect_eV
    assert accounted == pytest.approx(energy, rel=2e-14, abs=1e-11)
    assert np.linalg.norm(event.electron_direction) == pytest.approx(1, abs=2e-14)


def test_complete_event_explicit_draws_and_cascade(atom):
    tape = iter([0, 0, 0, 0, 0, 0.123])
    event = photo.sample_photoelectric("Be", 1000, [0, 0, 1], atom, 1, tape)
    assert event.subshell_designator == 1
    assert event.electron_energy_eV == 900
    assert event.relaxation.energy_budget.photon_eV == 90
    assert event.relaxation.energy_budget.terminal_binding_eV == 10
    assert event.draws_used == 5 and next(tape) == 0.123
    _assert_energy(1000, event)


def test_threshold_zero_energy_and_accessible_shells(atom):
    event = photo.sample_photoelectric("Be", 100, [1, 0, 0], atom, 1, [0, 0])
    assert event.electron_energy_eV == 0 and event.draws_used == 2
    np.testing.assert_array_equal(event.electron_direction, [1, 0, 0])
    rng = np.random.default_rng(34120)
    below = photo.sample_photoelectric("Be", 50, [0, 0, 1], atom, 1, iter(rng.random, None))
    assert below.subshell_designator == 2
    assert below.electron_energy_eV == 40
    _assert_energy(50, below)
    with pytest.raises(ValueError, match="positive finite total"):
        photo.sample_photoelectric("Be", 1, [0, 0, 1], atom, 1, [])


def test_shell_frequencies_and_history_replay(atom):
    rng = np.random.default_rng(34121)
    # Per-history explicit tapes have enough draws for these terminal cascades.
    tapes = rng.random((2000, 128))
    events = [photo.sample_photoelectric("Be", 1000, [0, 1, 0], atom, 1000, tape) for tape in tapes]
    k = sum(event.subshell_designator == 1 for event in events)
    assert abs(k - 500) < 6 * np.sqrt(2000 * 0.25 * 0.75)
    for i in rng.permutation(20):
        replay = photo.sample_photoelectric("Be", 1000, [0, 1, 0], atom, 1000, tapes[i])
        assert replay.subshell_designator == events[i].subshell_designator
        assert replay.draws_used == events[i].draws_used
        np.testing.assert_array_equal(replay.electron_direction, events[i].electron_direction)


@pytest.mark.parametrize("kinetic", [1e-6, 1e3, 0.5 * M_E_EV, 2 * M_E_EV, 1e8, 1e11])
def test_sauter_distribution_against_original_polar_law(kinetic):
    rng = np.random.default_rng(34122)
    draws = UniformDraws(iter(rng.random, None))
    samples = np.sort([photo._sauter_delta(kinetic, draws) for _ in range(2000)])
    # Integrate the original p(cos theta), rather than the sampler's rational
    # change of variable; log delta resolves the high-energy forward peak.
    delta = np.r_[0, np.geomspace(1e-16, 2, 60_000)]
    gamma = 1 + kinetic / M_E_EV
    beta = np.sqrt((kinetic / M_E_EV) * (2 + kinetic / M_E_EV)) / gamma
    denominator = (1 - beta) + beta * delta
    density = delta * (2 - delta) / denominator**4
    density *= 1 + gamma * (gamma - 1) * (gamma - 2) * denominator / 2
    cumulative = np.r_[0, np.cumsum(np.diff(delta) * (density[1:] + density[:-1]) / 2)]
    cdf = np.interp(samples, delta, cumulative / cumulative[-1])
    error = max(
        np.max(np.arange(1, len(cdf) + 1) / len(cdf) - cdf),
        np.max(cdf - np.arange(len(cdf)) / len(cdf)),
    )
    assert error < 2.2 / np.sqrt(len(cdf))


@pytest.mark.parametrize("element,energy", [("Si", 1000), ("Cu", 1e5), ("Au", 1e8)])
def test_real_epdl_eadl_events_close_energy(element, energy):
    relaxation = load_eadl_relaxation(element)
    rng = np.random.default_rng(34123)
    for _ in range(100):
        event = photo.sample_photoelectric(
            element, energy, np.array([1, 2, -2]) / 3, relaxation, 50, iter(rng.random, None)
        )
        _assert_energy(energy, event)
        assert event.electron_energy_eV >= 0


@pytest.mark.parametrize("draws", [[], [1], [np.nan], [0, 1], [0, 0, 0]])
def test_bad_or_exhausted_draws(atom, draws):
    with pytest.raises(ValueError, match="uniform"):
        photo.sample_photoelectric("Be", 1000, [0, 0, 1], atom, 1, draws)


def test_incompatible_relaxation_is_refused(atom):
    with pytest.raises(ValueError, match="element"):
        photo.sample_photoelectric("Si", 1000, [0, 0, 1], atom, 1, [])
    bad = replace(atom.subshells[0], binding_energy_eV=101)
    with pytest.raises(ValueError, match="binding energies"):
        photo.sample_photoelectric(
            "Be", 1000, [0, 0, 1], replace(atom, subshells=(bad, atom.subshells[1])), 1, []
        )


def test_rejection_exhaustion_is_an_error(monkeypatch, atom):
    monkeypatch.setattr(photo, "MAX_REJECTIONS", 0)
    with pytest.raises(RuntimeError, match="budget"):
        photo.sample_photoelectric("Be", 1000, [0, 0, 1], atom, 1, [0])
