"""Individual EADL vacancy histories, explicit draws and energy closure.

Validation: eadl-sampled-relaxation
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.eadl_relaxation import (
    EADL_BRANCHING_SUM_TOLERANCE,
    EADLRelaxation,
    EADLSubshellRelaxation,
    load_eadl_relaxation,
    relaxation_energy_budget,
    vacancy_cascade,
)
from pyrite.montecarlo.transport.relaxation import sample_relaxation_cascade


def _shell(designator, binding, radiative=(), auger=()):
    """Synthetic ENDF tuples (daughter, energy, p)/(daughter, daughter, energy, p)."""
    return EADLSubshellRelaxation(
        shell_designator=designator,
        binding_energy_eV=binding,
        electrons=2,
        radiative_final=np.array([r[0] for r in radiative], dtype=int),
        radiative_energy_eV=np.array([r[1] for r in radiative]),
        radiative_probability=np.array([r[2] for r in radiative]),
        auger_first=np.array([r[0] for r in auger], dtype=int),
        auger_second=np.array([r[1] for r in auger], dtype=int),
        auger_energy_eV=np.array([r[2] for r in auger]),
        auger_probability=np.array([r[3] for r in auger]),
    )


@pytest.fixture
def atom():
    # Independent single-vacancy model; repeated daughters remain distinct holes.
    # Tabulated ETR differs from binding differences, including a negative defect.
    return EADLRelaxation(
        6,
        (
            _shell(1, 100, [(2, 68, 0.25)], [(2, 2, 35, 0.75)]),
            _shell(2, 30, [(3, 18, 0.5)], [(3, 3, 11, 0.5)]),
            _shell(3, 10),
        ),
    )


def _assert_closure(sample):
    budget = sample.energy_budget
    assert budget.primary_binding_eV == pytest.approx(
        budget.photon_eV
        + budget.electron_eV
        + budget.terminal_binding_eV
        + budget.transition_energy_defect_eV,
        rel=2e-14,
        abs=1e-11,
    )


def test_radiative_history_and_draw_consumption(atom):
    draws = iter([0.1, 0.1, 0.999])
    sample = sample_relaxation_cascade(atom, 1, 1, draws)
    assert sample.draws_used == 2
    assert next(draws) == 0.999  # Terminal holes consume no random numbers.
    assert [t.kind for t in sample.transitions] == ["photon", "photon"]
    assert [t.energy_eV for t in sample.transitions] == [68, 18]
    assert [t.vacancy.parent_transition for t in sample.transitions] == [-1, 0]
    assert [(v.shell_designator, v.parent_transition) for v in sample.terminal_vacancies] == [
        (3, 1)
    ]
    assert sample.energy_budget.photon_eV == 86
    assert sample.energy_budget.electron_eV == 0
    assert sample.energy_budget.terminal_binding_eV == 10
    assert sample.energy_budget.transition_energy_defect_eV == 4
    _assert_closure(sample)


def test_auger_history_duplicate_holes_and_signed_defect(atom):
    sample = sample_relaxation_cascade(atom, 1, 1, [0.5, 0.1, 0.9])
    assert [t.kind for t in sample.transitions] == ["electron", "photon", "electron"]
    assert [t.vacancy.parent_transition for t in sample.transitions] == [-1, 0, 0]
    assert [t.daughter_designators for t in sample.transitions] == [(2, 2), (3,), (3, 3)]
    assert [(v.shell_designator, v.parent_transition) for v in sample.terminal_vacancies] == [
        (3, 1),
        (3, 2),
        (3, 2),
    ]
    assert sample.energy_budget.photon_eV == 18
    assert sample.energy_budget.electron_eV == 46
    assert sample.energy_budget.terminal_binding_eV == 30
    assert sample.energy_budget.transition_energy_defect_eV == 6
    _assert_closure(sample)


@pytest.mark.parametrize("primary,cutoff,terminal", [(1, 100, 1), (2, 30, 2), (3, 1, 3)])
def test_terminal_primary_consumes_no_draw(atom, primary, cutoff, terminal):
    sample = sample_relaxation_cascade(atom, primary, cutoff, [])
    assert sample.transitions == () and sample.draws_used == 0
    assert [v.shell_designator for v in sample.terminal_vacancies] == [terminal]
    assert sample.energy_budget.terminal_binding_eV == {1: 100, 2: 30, 3: 10}[primary]
    _assert_closure(sample)


def test_cutoff_stops_daughters_and_retains_multiplicity(atom):
    sample = sample_relaxation_cascade(atom, 1, 30, [0.25])
    assert sample.draws_used == 1
    assert [v.shell_designator for v in sample.terminal_vacancies] == [2, 2]
    assert sample.energy_budget.terminal_binding_eV == 60
    assert sample.energy_budget.transition_energy_defect_eV == 5
    _assert_closure(sample)


@pytest.mark.parametrize("draw,expected", [(0, 0), (0.25, 2), (np.nextafter(1.0, 0), 2)])
def test_transition_ties_zero_probabilities_and_rounded_sums(draw, expected):
    atom = EADLRelaxation(
        4,
        (
            _shell(1, 100, [(2, 90, 0.25), (2, 89, 0), (2, 88, 0.750001), (2, 87, 0)]),
            _shell(2, 10),
        ),
    )
    # The first boundary moves with normalization of the rounded printed FTR sum.
    if draw == 0.25:
        draw = 0.25 / 1.000001
    sample = sample_relaxation_cascade(atom, 1, 1, [draw])
    assert sample.transitions[0].transition_index == expected
    _assert_closure(sample)


def test_zero_energy_auger_is_recorded():
    atom = EADLRelaxation(4, (_shell(1, 20, auger=[(2, 2, 0, 1)]), _shell(2, 10)))
    sample = sample_relaxation_cascade(atom, 1, 1, [0])
    assert sample.transitions[0].kind == "electron"
    assert sample.transitions[0].energy_eV == 0
    assert len(sample.terminal_vacancies) == 2
    _assert_closure(sample)


def test_negative_total_defect_is_not_clamped(atom):
    sample = sample_relaxation_cascade(atom, 2, 1, [0.9])
    assert sample.energy_budget.transition_energy_defect_eV == -1
    assert sample.energy_budget.electron_eV == 11
    assert sample.energy_budget.terminal_binding_eV == 20
    _assert_closure(sample)


def test_synthetic_ensemble_matches_exact_branching_expectations(atom):
    rng = np.random.default_rng(34102)
    budgets, visits = [], []
    # At most three transitions per history; independent rows retain history streams.
    for draws in rng.random((8000, 3)):
        sample = sample_relaxation_cascade(atom, 1, 1, draws)
        _assert_closure(sample)
        budgets.append(
            [
                sample.energy_budget.photon_eV,
                sample.energy_budget.electron_eV,
                sample.energy_budget.terminal_binding_eV,
                sample.energy_budget.transition_energy_defect_eV,
            ]
        )
        holes = [t.vacancy.shell_designator for t in sample.transitions]
        holes += [v.shell_designator for v in sample.terminal_vacancies]
        visits.append([holes.count(i) for i in (1, 2, 3)])
    # K creates 0.25*1+0.75*2 = 1.75 L1 holes; each creates 1.5 L2 holes.
    np.testing.assert_allclose(np.mean(visits, axis=0), [1, 1.75, 2.625], atol=0.04, rtol=0)
    # E[photon]=.25*68+1.75*.5*18, E[electron]=.75*35+1.75*.5*11.
    # B_terminal=2.625*10, defect=100-sum(other energies).
    expected = np.array([32.75, 35.875, 26.25, 5.125])
    values = np.array(budgets)
    assert np.all(
        abs(values.mean(axis=0) - expected) < 6 * values.std(axis=0) / np.sqrt(len(values))
    )


@pytest.mark.parametrize("draws", [[], [0.1], [0.9, 0.1], [-1], [1], [np.nan], [np.inf]])
def test_invalid_or_exhausted_draws(atom, draws):
    with pytest.raises(ValueError, match="uniform"):
        sample_relaxation_cascade(atom, 1, 1, draws)


@pytest.mark.parametrize("cutoff", [0, -1, np.nan, np.inf])
def test_invalid_cutoffs(atom, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        sample_relaxation_cascade(atom, 1, cutoff, [])


def test_unknown_primary_is_refused(atom):
    with pytest.raises(ValueError, match="occupied subshell"):
        sample_relaxation_cascade(atom, 99, 1, [])


@pytest.mark.parametrize("daughter", [1, 99])
def test_nondescending_or_absent_daughter_is_refused(atom, daughter):
    bad = replace(atom.subshells[0], radiative_final=np.array([daughter]))
    with pytest.raises(ValueError, match="cascade order"):
        sample_relaxation_cascade(replace(atom, subshells=(bad, *atom.subshells[1:])), 1, 1, [0])


def test_explicit_history_streams_replay_under_reordering_and_chunking(atom):
    rng = np.random.default_rng(34104)
    draws = rng.random((32, 3))

    def sample_rows(rows):
        return [sample_relaxation_cascade(atom, 1, 1, row) for row in rows]

    whole = sample_rows(draws)
    order = rng.permutation(len(draws))
    assert sample_rows(draws[order]) == [whole[i] for i in order]
    assert sample_rows(draws[:7]) + sample_rows(draws[7:]) == whole


@pytest.mark.parametrize("probability", [-1, np.nan, np.inf, 0.5])
def test_invalid_transition_probabilities_are_refused(atom, probability):
    bad = replace(atom.subshells[0], radiative_probability=np.array([probability]))
    with pytest.raises(ValueError, match="transition probabilities"):
        sample_relaxation_cascade(replace(atom, subshells=(bad, *atom.subshells[1:])), 1, 1, [0])


@pytest.mark.parametrize("element", ["Si", "Cu", "Au"])
def test_all_occupied_primary_shells_close_at_low_and_default_cutoffs(element):
    relaxation = load_eadl_relaxation(element)
    rng = np.random.default_rng(34105 + relaxation.atomic_number)
    draws = iter(rng.random, None)
    for primary in relaxation.shell_designators:
        for cutoff in (1, 50):
            for _ in range(8):
                sample = sample_relaxation_cascade(relaxation, primary, cutoff, draws)
                _assert_closure(sample)
                assert (
                    sample.energy_budget.primary_binding_eV
                    == relaxation.subshells[
                        relaxation.shell_designators.index(primary)
                    ].binding_energy_eV
                )
                for ordinal, transition in enumerate(sample.transitions):
                    assert -1 <= transition.vacancy.parent_transition < ordinal
                for vacancy in sample.terminal_vacancies:
                    assert -1 <= vacancy.parent_transition < len(sample.transitions)


@pytest.mark.parametrize("element", ["Si", "Cu", "Au"])
def test_eadl_ensemble_matches_expected_cascade_and_energy_budget(element):
    relaxation = load_eadl_relaxation(element)
    cutoff, n = 50, 2000
    primary = relaxation.shell_designators[0]
    expected_budget = relaxation_energy_budget(relaxation, primary, cutoff)
    expected_visits = vacancy_cascade(relaxation, cutoff)[1][0]
    rng = np.random.default_rng(34103 + relaxation.atomic_number)
    draws = iter(rng.random, None)
    budgets, visits = [], []
    for _ in range(n):
        sample = sample_relaxation_cascade(relaxation, primary, cutoff, draws)
        _assert_closure(sample)
        budget = sample.energy_budget
        budgets.append(
            [
                budget.photon_eV,
                budget.electron_eV,
                budget.terminal_binding_eV,
                budget.transition_energy_defect_eV,
            ]
        )
        holes = [t.vacancy.shell_designator for t in sample.transitions]
        holes += [v.shell_designator for v in sample.terminal_vacancies]
        visits.append([holes.count(d) for d in relaxation.shell_designators])
    # The historical expectation uses printed, unnormalized FTR. The sampled
    # categorical law normalizes each row; allow that bounded rounding drift
    # along at most n_shell levels, independently of the sampling tolerance.
    rounding = 2 * len(relaxation.subshells) * EADL_BRANCHING_SUM_TOLERANCE
    expected_energy = np.array(
        [
            expected_budget.photon_eV,
            expected_budget.electron_eV,
            expected_budget.terminal_binding_eV,
            expected_budget.transition_energy_defect_eV,
        ]
    )
    for values, expected, scale in (
        (np.array(budgets), expected_energy, expected_budget.primary_binding_eV),
        (np.array(visits), expected_visits, 1),
    ):
        error = abs(values.mean(axis=0) - expected)
        tolerance = 6 * (values.std(axis=0) / np.sqrt(n) + scale / n)
        tolerance += rounding * np.maximum(abs(expected), scale)
        assert np.all(error <= tolerance), (element, error, tolerance)
