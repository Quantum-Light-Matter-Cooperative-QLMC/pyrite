"""Host reference for individual EADL vacancy cascades (#341).

No production transport caller yet. The caller supplies independent uniform
draws; this module owns no random stream, directions, lifetimes or tracking.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from typing import Literal

import numpy as np

from ..eadl_relaxation import (
    EADL_BRANCHING_SUM_TOLERANCE,
    EADLRelaxation,
    EADLSubshellRelaxation,
    RelaxationEnergyBudget,
)


@dataclass(frozen=True, slots=True)
class RelaxationVacancy:
    """One hole, with its creating transition's ordinal (-1 for the primary)."""

    shell_designator: int
    parent_transition: int


@dataclass(frozen=True, slots=True)
class RelaxationTransition:
    """One decay and its emitted particle, in EADL designators and eV.

    ``transition_index`` addresses the radiative or Auger arrays according to
    ``kind``. The emitted electron's energy is kinetic. Zero-energy EADL
    Auger electrons are retained. Daughter designators may repeat: two holes
    in the same shell remain two independent vacancies in this model.
    """

    vacancy: RelaxationVacancy
    kind: Literal["photon", "electron"]
    transition_index: int
    daughter_designators: tuple[int, ...]
    energy_eV: float


@dataclass(frozen=True, slots=True)
class SampledRelaxationCascade:
    """One sampled tree, including unrelaxed holes and a signed energy defect.

    ``energy_budget`` applies the same partition as the expected-value
    ``relaxation_energy_budget`` to this individual history. The defect is
    not deposited energy: EADL's tabulated transition energies do not exactly
    equal differences of its single-vacancy binding energies. No adjustment
    of particle energies or clamping of a negative defect is performed.
    """

    transitions: tuple[RelaxationTransition, ...]
    terminal_vacancies: tuple[RelaxationVacancy, ...]
    energy_budget: RelaxationEnergyBudget

    @property
    def draws_used(self) -> int:
        """One draw per decay; terminal vacancies consume none."""
        return len(self.transitions)


def _sample_transition(
    shell: EADLSubshellRelaxation, draw: float, vacancy: RelaxationVacancy
) -> RelaxationTransition:
    """Invert normalized EADL FTR, radiative entries before Auger entries.

    Source: ENDF-102 File 28, fractional transition probabilities FTR. Printed
    probabilities are normalized by their sum; zero entries are skipped even
    at draw=0. Assumes a decaying shell validated by the EADL loader.
    Validation: eadl-sampled-relaxation
    """
    probabilities = np.concatenate((shell.radiative_probability, shell.auger_probability))
    cumulative = np.cumsum(probabilities)
    total = cumulative[-1]
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.isfinite(total)
        or abs(total - 1) > EADL_BRANCHING_SUM_TOLERANCE
    ):
        raise ValueError("EADL transition probabilities must be non-negative and sum to one")
    threshold = min(draw * total, np.nextafter(total, 0.0))
    selected = int(np.searchsorted(cumulative, threshold, side="right"))
    n_radiative = len(shell.radiative_probability)
    if selected < n_radiative:
        return RelaxationTransition(
            vacancy,
            "photon",
            selected,
            (int(shell.radiative_final[selected]),),
            float(shell.radiative_energy_eV[selected]),
        )
    selected -= n_radiative
    return RelaxationTransition(
        vacancy,
        "electron",
        selected,
        (int(shell.auger_first[selected]), int(shell.auger_second[selected])),
        float(shell.auger_energy_eV[selected]),
    )


def _sampled_energy_budget(
    binding: Mapping[int, float],
    primary: int,
    transitions: Sequence[RelaxationTransition],
    terminal: Sequence[RelaxationVacancy],
) -> RelaxationEnergyBudget:
    """Telescope ENDF-102 File 28 EBI/ETR along a sampled vacancy tree [eV].

    Each decay reserves its daughter bindings and emits ETR. The signed
    remainder is EBI_parent - sum(EBI_daughters) - ETR. Summing cancels every
    internal vacancy; a terminal primary leaves its whole binding reserved.
    Validation: eadl-sampled-relaxation
    """
    return RelaxationEnergyBudget(
        primary_binding_eV=binding[primary],
        photon_eV=fsum(t.energy_eV for t in transitions if t.kind == "photon"),
        electron_eV=fsum(t.energy_eV for t in transitions if t.kind == "electron"),
        terminal_binding_eV=fsum(binding[v.shell_designator] for v in terminal),
        transition_energy_defect_eV=fsum(
            binding[t.vacancy.shell_designator]
            - fsum(binding[d] for d in t.daughter_designators)
            - t.energy_eV
            for t in transitions
        ),
    )


def sample_relaxation_cascade(
    relaxation: EADLRelaxation,
    primary_designator: int,
    cutoff_eV: float,
    uniforms: Iterable[float],
) -> SampledRelaxationCascade:
    """Sample one vacancy's EADL cascade from an explicit uniform sequence.

    Source: ENDF-102 File 28 (MF=28/MT=533), EBI, SUBJ, SUBK, ETR and FTR.
    A decay is chosen with probability FTR/sum(FTR); SUBK=0 leaves one hole
    and emits a photon, otherwise two holes and an Auger/Coster--Kronig
    electron. Both holes are propagated even when their designators coincide.

    Input is validated, cascade-ordered data from ``load_eadl_relaxation``.
    Assumptions: independent single-vacancy rates, neutral-atom data; no
    occupancy depletion, multiple-hole energy/rate shifts or yield overrides.
    Energies are eV. Transition energies are tabulated ETR, with a signed
    residual tracked separately from particles and terminal binding energy.
    No particle direction, lifetime, transport or deposition is sampled here.

    A shell with binding <= the finite positive cutoff, or with no recorded
    transitions, is terminal. With the cutoff above all bindings the result
    is just the primary terminal hole and consumes no uniform. Otherwise one
    finite draw in [0,1) is consumed per decay. A supplied iterator is advanced
    only that far; exhaustion raises ValueError rather than truncating a tree.
    Replay order is depth-first, first daughter before second; transitions
    retain their creating transition ordinal. There is no iteration cap:
    daughter ranks must strictly increase, so every finite tree terminates.
    Validation: eadl-sampled-relaxation
    """
    if not np.isfinite(cutoff_eV) or cutoff_eV <= 0:
        raise ValueError("relaxation cutoff must be finite and positive")
    index = {s.shell_designator: i for i, s in enumerate(relaxation.subshells)}
    if primary_designator not in index:
        raise ValueError(f"designator {primary_designator} is not an occupied subshell")
    binding = {s.shell_designator: s.binding_energy_eV for s in relaxation.subshells}
    draws = iter(uniforms)
    pending = [RelaxationVacancy(primary_designator, -1)]
    transitions: list[RelaxationTransition] = []
    terminal: list[RelaxationVacancy] = []
    while pending:
        vacancy = pending.pop()
        shell = relaxation.subshells[index[vacancy.shell_designator]]
        if shell.binding_energy_eV <= cutoff_eV or not (
            shell.radiative_probability.size + shell.auger_probability.size
        ):
            terminal.append(vacancy)
            continue
        try:
            draw = float(next(draws))
        except StopIteration as exc:
            raise ValueError(
                "uniform sequence exhausted before the relaxation cascade ended"
            ) from exc
        if not np.isfinite(draw) or not 0 <= draw < 1:
            raise ValueError("relaxation uniforms must be finite and in [0, 1)")
        transition = _sample_transition(shell, draw, vacancy)
        ordinal = len(transitions)
        for daughter in reversed(transition.daughter_designators):
            if daughter not in index or index[daughter] <= index[vacancy.shell_designator]:
                raise ValueError("EADL daughters must lie strictly later in cascade order")
            pending.append(RelaxationVacancy(daughter, ordinal))
        transitions.append(transition)
    return SampledRelaxationCascade(
        tuple(transitions),
        tuple(terminal),
        _sampled_energy_budget(binding, primary_designator, transitions, terminal),
    )
