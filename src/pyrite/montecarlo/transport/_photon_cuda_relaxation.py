"""Immutable canonical EPDL/EADL staging for CUDA photoelectric events."""

from functools import cache

import numpy as np

from ...materials.photon_cross_sections import _atomic_number
from ...materials.photon_differential import photoionization_subshells
from ..eadl_relaxation import EADL_BRANCHING_SUM_TOLERANCE, load_eadl_relaxation


def _shell_arrays(shell, index, parent):
    """Radiative-before-Auger records and local CDF, retaining original ordinals."""
    nr = len(shell.radiative_probability)
    probabilities = np.concatenate((shell.radiative_probability, shell.auger_probability))
    cumulative = np.cumsum(probabilities)
    if probabilities.size and (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.isfinite(cumulative[-1])
        or abs(cumulative[-1] - 1) > EADL_BRANCHING_SUM_TOLERANCE
    ):
        raise ValueError("EADL transition probabilities must be non-negative and sum to one")
    first = [index[int(d)] for d in (*shell.radiative_final, *shell.auger_first)]
    second = [-1] * nr + [index[int(d)] for d in shell.auger_second]
    if any(d <= parent for d in (*first, *[d for d in second if d >= 0])):
        raise ValueError("EADL daughters must lie strictly later in cascade order")
    return dict(
        probability=cumulative,
        first=np.asarray(first, dtype=np.int32),
        second=np.asarray(second, dtype=np.int32),
        kind=np.r_[np.zeros(nr, np.int32), np.ones(len(second) - nr, np.int32)],
        energy=np.concatenate((shell.radiative_energy_eV, shell.auger_energy_eV)),
        parent=np.full(len(first), parent, np.int32),
    )


@cache
def host_relaxation_table(element):
    """Canonical pinned tables; zero probability rows and repeated holes survive."""
    model = load_eadl_relaxation(element)
    if model.atomic_number != _atomic_number(element):
        raise ValueError("photoelectric element does not match EADL data")
    shells = model.subshells
    if not 1 <= len(shells) <= 64:
        raise ValueError("CUDA vacancy workspace supports 1..64 ordered EADL shells")
    index = {shell.shell_designator: i for i, shell in enumerate(shells)}
    binding = np.array([shell.binding_energy_eV for shell in shells])
    epdl = photoionization_subshells(element)
    if any(
        s.designator not in index or s.binding_energy_eV != binding[index[s.designator]]
        for s in epdl
    ):
        raise ValueError("EPDL and EADL binding energies must match")
    parts = [_shell_arrays(shell, index, i) for i, shell in enumerate(shells)]
    arrays = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    arrays.update(
        offsets=np.cumsum([0, *[len(p["first"]) for p in parts]], dtype=np.int32),
        binding=binding,
        designators=np.array([shell.shell_designator for shell in shells], dtype=np.int32),
        primary_shells=np.array([index[s.designator] for s in epdl], dtype=np.int32),
    )
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


@cache
def device_relaxation_table(cp, element, device):
    del device  # cache key: never reuse an upload from a different CUDA device
    return {name: cp.asarray(value) for name, value in host_relaxation_table(element).items()}
