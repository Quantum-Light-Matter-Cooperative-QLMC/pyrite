"""Validated EADL atomic-relaxation data and the deterministic vacancy cascade."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from endf_parserpy import EndfFile

from ..datasets import EADL, DatasetMismatchError, file_sha256, require_dataset
from ..materials.atomic import Z_TABLE
from .eedl_ionization import EEDL_SUBSHELL_LABELS, _readonly, _require_finite

EADL_FILENAME = EADL.filename
EADL_SHA256 = EADL.sha256

#: Tolerance on ``sum FTR = 1`` per subshell. EADL prints FTR to six
#: significant figures; the worst packaged subshell misses unity by 1.7e-6.
EADL_BRANCHING_SUM_TOLERANCE = 1.0e-5


@dataclass(frozen=True, slots=True)
class EADLSubshellRelaxation:
    """One EADL MF=28 subshell: binding energy, occupancy, and its decays.

    Radiative transition ``t`` fills this vacancy from ``radiative_final[t]``
    and emits a photon of ``radiative_energy_eV[t]``. Nonradiative transition
    ``t`` fills it from ``auger_first[t]`` and ejects an electron of
    ``auger_energy_eV[t]`` from ``auger_second[t]``, leaving one vacancy in
    each. Designators follow ENDF-6 (1=K, 2=L1, ...). Probabilities of both
    kinds together sum to one when the subshell has any transitions.
    """

    shell_designator: int
    binding_energy_eV: float
    electrons: float
    radiative_final: np.ndarray
    radiative_energy_eV: np.ndarray
    radiative_probability: np.ndarray
    auger_first: np.ndarray
    auger_second: np.ndarray
    auger_energy_eV: np.ndarray
    auger_probability: np.ndarray

    @property
    def fluorescence_yield(self) -> float:
        """EADL radiative branching ratio ``omega_i`` of this subshell."""
        return float(self.radiative_probability.sum())


@dataclass(frozen=True, slots=True)
class EADLRelaxation:
    """One element's validated EADL relaxation data, in cascade order.

    ``subshells`` is sorted by decreasing binding energy, ties broken by
    increasing ENDF designator. Every daughter vacancy of every transition
    lies strictly later in that order.
    """

    atomic_number: int
    subshells: tuple[EADLSubshellRelaxation, ...]

    @property
    def shell_designators(self) -> tuple[int, ...]:
        return tuple(shell.shell_designator for shell in self.subshells)

    @property
    def shell_labels(self) -> tuple[str, ...]:
        return tuple(EEDL_SUBSHELL_LABELS[d] for d in self.shell_designators)

    @property
    def binding_energy_eV(self) -> np.ndarray:
        return _readonly([shell.binding_energy_eV for shell in self.subshells])


def _subshell_capacity(designator: int) -> int:
    """Electron capacity ``2j + 1`` of an ENDF subshell designator.

    Within one principal shell the subshells run s1/2, p1/2, p3/2, d3/2, d5/2,
    ..., so the ``m``-th subshell has ``j = floor((m + 1) / 2) - 1/2``.
    """
    label = EEDL_SUBSHELL_LABELS[designator]
    match = re.fullmatch(r"[A-Z](\d*)", label)
    if match is None:
        raise ValueError(f"cannot parse subshell label {label!r}")
    index = int(match.group(1) or 1)
    return 2 * ((index + 1) // 2)


def _cascade_key(designator: int, binding_eV: float) -> tuple[float, int]:
    # Spin-orbit partners may share a binding energy (Mg-Si L2/L3); the ENDF
    # designator then orders them the way their transitions do.
    return (-binding_eV, designator)


def _record_value(section: Mapping[str, Any], field: str, *keys: int) -> Any:
    value: Any = section[field]
    for key in keys:
        value = value[key]
    return value


def _extract_eadl_relaxation(
    path: Path,
    atomic_number: int,
    section: Mapping[str, Any],
) -> EADLRelaxation:
    """Validate one decoded MF=28/MT=533 section and put it in cascade order.

    Validation: characteristic-radiation
    """
    where = f"{path}: EADL Z={atomic_number} MF=28/MT=533"
    try:
        n_subshell = int(section["NSS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where} has no valid NSS") from exc
    if n_subshell <= 0:
        raise ValueError(f"{where} declares no subshells")

    raw: list[tuple[int, float, float, list[tuple[int, int, float, float]]]] = []
    for position in range(1, n_subshell + 1):
        try:
            designator_float = float(_record_value(section, "SUBI", position))
            binding = _require_finite(
                _record_value(section, "EBI", position),
                f"{where} subshell {position} EBI",
                positive=True,
            )
            electrons = _require_finite(
                _record_value(section, "ELN", position),
                f"{where} subshell {position} ELN",
            )
            n_transition = int(_record_value(section, "NTR", position))
            transitions = [
                (
                    int(_record_value(section, "SUBJ", position, t)),
                    int(_record_value(section, "SUBK", position, t)),
                    _require_finite(
                        _record_value(section, "ETR", position, t),
                        f"{where} subshell {position} ETR",
                    ),
                    _require_finite(
                        _record_value(section, "FTR", position, t),
                        f"{where} subshell {position} FTR",
                    ),
                )
                for t in range(1, n_transition + 1)
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{where} subshell {position} is incomplete") from exc
        designator = round(designator_float)
        if designator != designator_float or designator not in EEDL_SUBSHELL_LABELS:
            raise ValueError(f"{where} has unsupported subshell designator {designator_float!r}")
        capacity = _subshell_capacity(designator)
        if not 0.0 < electrons <= capacity:
            raise ValueError(
                f"{where} {EEDL_SUBSHELL_LABELS[designator]} occupancy {electrons:g} "
                f"is outside (0, {capacity}]"
            )
        raw.append((designator, binding, electrons, transitions))

    designators = [designator for designator, *_rest in raw]
    if len(set(designators)) != len(designators):
        raise ValueError(f"{where} repeats a subshell designator")
    total_electrons = sum(electrons for _d, _b, electrons, _t in raw)
    if not np.isclose(total_electrons, atomic_number, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"{where} occupancies sum to {total_electrons:g}, not the neutral-atom {atomic_number}"
        )

    raw.sort(key=lambda entry: _cascade_key(entry[0], entry[1]))
    rank = {designator: position for position, (designator, *_rest) in enumerate(raw)}
    subshells: list[EADLSubshellRelaxation] = []
    for designator, binding, electrons, transitions in raw:
        label = EEDL_SUBSHELL_LABELS[designator]
        radiative = [(j, e, f) for j, k, e, f in transitions if k == 0]
        auger = [(j, k, e, f) for j, k, e, f in transitions if k != 0]
        for j, k, energy, probability in transitions:
            if probability < 0.0:
                raise ValueError(f"{where} {label} has a negative transition probability")
            for daughter in (j,) if k == 0 else (j, k):
                if daughter not in rank:
                    raise ValueError(f"{where} {label} decays into absent designator {daughter}")
                if rank[daughter] <= rank[designator]:
                    raise ValueError(
                        f"{where} {label} decays into {EEDL_SUBSHELL_LABELS[daughter]}, which "
                        "is not less tightly bound; the cascade would not be triangular"
                    )
            # EADL clamps energetically marginal super-Coster--Kronig electrons
            # to ETR = 0, but a photon must carry energy.
            if energy < 0.0 or (k == 0 and energy <= 0.0):
                raise ValueError(f"{where} {label} has a nonphysical transition energy {energy:g}")
        if transitions:
            probability_sum = sum(f for *_rest, f in transitions)
            if abs(probability_sum - 1.0) > EADL_BRANCHING_SUM_TOLERANCE:
                raise ValueError(
                    f"{where} {label} transition probabilities sum to {probability_sum:.8g}"
                )
        subshells.append(
            EADLSubshellRelaxation(
                shell_designator=designator,
                binding_energy_eV=binding,
                electrons=electrons,
                radiative_final=np.asarray([j for j, _e, _f in radiative], dtype=int),
                radiative_energy_eV=_readonly([e for _j, e, _f in radiative]),
                radiative_probability=_readonly([f for _j, _e, f in radiative]),
                auger_first=np.asarray([j for j, _k, _e, _f in auger], dtype=int),
                auger_second=np.asarray([k for _j, k, _e, _f in auger], dtype=int),
                auger_energy_eV=_readonly([e for _j, _k, e, _f in auger]),
                auger_probability=_readonly([f for _j, _k, _e, f in auger]),
            )
        )
    for shell in subshells:
        for array in (shell.radiative_final, shell.auger_first, shell.auger_second):
            array.setflags(write=False)
    return EADLRelaxation(atomic_number=atomic_number, subshells=tuple(subshells))


@cache
def _load_eadl_relaxation(
    path: Path,
    file_size: int,
    modified_time_ns: int,
    atomic_number: int,
) -> EADLRelaxation:
    """Use endf-parserpy to load one element's EADL relaxation section.

    ``file_size`` and ``modified_time_ns`` are cache-key sentinels, as for
    the EEDL loader.
    """
    del file_size, modified_time_ns
    matches = []
    with EndfFile(path, on_error="raise") as tape:
        for position in range(len(tape)):
            material = tape[position]
            if round(float(material.za) / 1000.0) != atomic_number:
                continue
            if (28, 533) in tuple(material.sections()):
                matches.append(material)
        if len(matches) > 1:
            raise ValueError(
                f"{path}: multiple EADL materials contain MF=28/MT=533 for Z={atomic_number}"
            )
        if matches:
            return _extract_eadl_relaxation(path, atomic_number, matches[0][28, 533])
    raise ValueError(
        f"{path}: no EADL MF=28/MT=533 atomic-relaxation section for Z={atomic_number}"
    )


def eadl_path() -> Path:
    """Return the installed, checksum-verified EADL tape.

    The tape is a fetched dataset (ADR-0014 class b), not packaged data.

    Raises
    ------
    pyrite.datasets.DatasetNotFoundError
        If it is not installed; the message names ``pyrite tables fetch eadl``.
    pyrite.datasets.DatasetMismatchError
        If the installed bytes differ from the vetted input.
    """
    path = require_dataset("eadl")
    _verify_eadl_file(str(path))
    return path


@cache
def _verify_eadl_file(path: str) -> None:
    """Fail closed if the installed EADL bytes differ from the vetted input."""
    actual = file_sha256(Path(path))
    if actual != EADL_SHA256:
        raise DatasetMismatchError(
            f"EADL database checksum mismatch at {path}: expected {EADL_SHA256}, got {actual}; "
            "remove it and run `pyrite tables fetch eadl`"
        )


def load_eadl_relaxation(
    element: str,
    *,
    data_dir: str | Path | None = None,
) -> EADLRelaxation:
    """Load one element's validated EADL MF=28 atomic-relaxation data.

    The fetched EPICS2025 tape (``pyrite tables fetch eadl``) is
    checksum-pinned; ``data_dir`` may instead
    name an explicit ENDF-6 file or a directory containing ``EADL2025.ALL``.
    Validation enforces ``sum FTR = 1`` per decaying subshell, daughter
    designators present and strictly less bound than the parent, positive
    photon energies, nonnegative electron energies, and subshell occupancies
    within ``2j + 1`` summing to ``Z``.

    Validation: characteristic-radiation
    """
    if not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None:
        raise ValueError("element must be a chemical symbol such as 'C' or 'Si'")
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    if data_dir is None:
        path = eadl_path()
    else:
        candidate = Path(data_dir)
        path = candidate if candidate.is_file() else candidate / EADL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no EADL atomic-relaxation database for {element}: expected {path}"
            )
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    return _load_eadl_relaxation(resolved_path, stat.st_size, stat.st_mtime_ns, atomic_number)


def _branching_scales(
    shell: EADLSubshellRelaxation,
    fluorescence_yield: float | None,
) -> tuple[float, float]:
    """Radiative and nonradiative multipliers that impose ``fluorescence_yield``.

    Keeps each branch's internal shape and the total at one. A subshell whose
    EADL yield is exactly zero or one has no branch to rescale and is kept.
    """
    if fluorescence_yield is None:
        return 1.0, 1.0
    eadl_yield = shell.fluorescence_yield
    auger_yield = float(shell.auger_probability.sum())
    if eadl_yield <= 0.0 or auger_yield <= 0.0:
        return 1.0, 1.0
    return fluorescence_yield / eadl_yield, (1.0 - fluorescence_yield) / auger_yield


def vacancy_cascade(
    relaxation: EADLRelaxation,
    cutoff_eV: float,
    *,
    fluorescence_yields: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Deterministic expected vacancy cascade in binding-energy order.

    Returns ``(daughters, visits)``, both ``(n_shell, n_shell)`` in
    ``relaxation.subshells`` order. ``daughters[i, j]`` is the expected number
    of vacancies created in ``j`` by one decay of a vacancy in ``i``: a
    radiative ``i -> j`` adds one to ``j``, a nonradiative ``i -> (j, k)`` adds
    one to each of ``j`` and ``k``. ``visits[p, i]`` is the expected number of
    vacancies that ever occupy ``i`` per primary vacancy in ``p``,

    .. math::

        V = (\mathbb{1} - D)^{-1} = \sum_{m\ge 0} D^m .

    Every daughter lies strictly later in the cascade order, so ``D`` is
    strictly upper-triangular and nilpotent: the series terminates after at
    most ``n_shell`` terms and is evaluated exactly by forward substitution,
    with no iteration cutoff or convergence tolerance.

    ``cutoff_eV`` bounds propagation: a subshell bound at or below it does not
    decay, so its row of ``D`` is zero and its vacancies are terminal. Optional
    ``fluorescence_yields`` maps a designator to a replacement ``omega_i``;
    that subshell's radiative branch is scaled to it and its nonradiative
    branch by ``(1 - omega_i) / (1 - omega_i^EADL)``.

    Assumptions: independent single-vacancy decay (EADL rates are for one
    hole; multiple-vacancy shifts and rate changes are ignored).

    Limiting case: with ``cutoff_eV`` above every binding energy, ``D = 0`` and
    ``V`` is the identity -- the direct-vacancy model.

    Validation: characteristic-radiation
    """
    cutoff = _require_finite(cutoff_eV, "relaxation cutoff", positive=True)
    index = {
        designator: position for position, designator in enumerate(relaxation.shell_designators)
    }
    n_shell = len(index)
    daughters = np.zeros((n_shell, n_shell), dtype=float)
    yields = {} if fluorescence_yields is None else fluorescence_yields
    for row, shell in enumerate(relaxation.subshells):
        if shell.binding_energy_eV <= cutoff:
            continue
        radiative_scale, auger_scale = _branching_scales(shell, yields.get(shell.shell_designator))
        for final, probability in zip(
            shell.radiative_final, shell.radiative_probability, strict=True
        ):
            daughters[row, index[int(final)]] += radiative_scale * probability
        for first, second, probability in zip(
            shell.auger_first, shell.auger_second, shell.auger_probability, strict=True
        ):
            daughters[row, index[int(first)]] += auger_scale * probability
            daughters[row, index[int(second)]] += auger_scale * probability
    visits = np.eye(n_shell, dtype=float)
    for parent in range(n_shell):
        # Column `parent` of every row is final once all earlier shells are done.
        visits[:, parent + 1 :] += np.outer(visits[:, parent], daughters[parent, parent + 1 :])
    return _readonly(daughters), _readonly(visits)


@dataclass(frozen=True, slots=True)
class RelaxationEnergyBudget:
    """Expected energy partition of one primary vacancy's EADL cascade, in eV.

    ``primary_binding_eV == photon_eV + electron_eV + terminal_binding_eV +
    transition_energy_defect_eV`` up to the EADL branching-sum tolerance.
    ``terminal_binding_eV`` is the binding energy still held by vacancies that
    do not decay (below the cutoff or without EADL transitions).
    ``transition_energy_defect_eV`` is the expected sum of ``EBI_i - EBI_j -
    EBI_k - ETR`` over decays: EADL transition energies are computed
    separately from its single-vacancy binding energies, so they do not
    difference exactly.
    """

    primary_binding_eV: float
    photon_eV: float
    electron_eV: float
    terminal_binding_eV: float
    transition_energy_defect_eV: float


def relaxation_energy_budget(
    relaxation: EADLRelaxation,
    primary_designator: int,
    cutoff_eV: float,
) -> RelaxationEnergyBudget:
    """Partition one primary vacancy's binding energy over its EADL cascade.

    Validation: characteristic-radiation
    """
    index = {
        designator: position for position, designator in enumerate(relaxation.shell_designators)
    }
    if primary_designator not in index:
        raise ValueError(f"designator {primary_designator} is not an occupied subshell")
    _daughters, visits = vacancy_cascade(relaxation, cutoff_eV)
    population = visits[index[primary_designator]]
    binding = {s.shell_designator: s.binding_energy_eV for s in relaxation.subshells}
    photon = electron = terminal = defect = 0.0
    for shell, count in zip(relaxation.subshells, population, strict=True):
        decays = shell.binding_energy_eV > cutoff_eV and (
            shell.radiative_probability.size + shell.auger_probability.size
        )
        if not decays:
            terminal += count * shell.binding_energy_eV
            continue
        for final, energy, probability in zip(
            shell.radiative_final,
            shell.radiative_energy_eV,
            shell.radiative_probability,
            strict=True,
        ):
            photon += count * probability * energy
            defect += count * probability * (shell.binding_energy_eV - binding[int(final)] - energy)
        for first, second, energy, probability in zip(
            shell.auger_first,
            shell.auger_second,
            shell.auger_energy_eV,
            shell.auger_probability,
            strict=True,
        ):
            electron += count * probability * energy
            defect += (
                count
                * probability
                * (shell.binding_energy_eV - binding[int(first)] - binding[int(second)] - energy)
            )
    return RelaxationEnergyBudget(
        primary_binding_eV=binding[primary_designator],
        photon_eV=photon,
        electron_eV=electron,
        terminal_binding_eV=terminal,
        transition_energy_defect_eV=defect,
    )


__all__ = [
    "EADL_BRANCHING_SUM_TOLERANCE",
    "eadl_path",
    "EADL_FILENAME",
    "EADL_SHA256",
    "EADLRelaxation",
    "EADLSubshellRelaxation",
    "RelaxationEnergyBudget",
    "load_eadl_relaxation",
    "relaxation_energy_budget",
    "vacancy_cascade",
]
