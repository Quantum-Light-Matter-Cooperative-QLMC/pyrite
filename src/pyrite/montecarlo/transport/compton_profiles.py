"""Per-subshell impulse-approximation inputs for Compton Doppler broadening (#1231).

``compton_profile_j0.csv`` (in-package, written by
``scripts/generate_compton_profiles.py``) holds one-electron Hartree--Fock
Compton profiles at zero projected momentum, ``J_i(0)`` in inverse atomic
units, from Biggs, Mendelsohn & Mann, At. Data Nucl. Data Tables 16, 201
(1975), as distributed by xraylib 4.3.0 (BSD-3-Clause). Rows are keyed by EADL
subshell designator. Occupation numbers and binding energies come from the
pinned EADL2025 relaxation data, so a Compton vacancy names an EADL subshell.
"""

import hashlib
from dataclasses import dataclass
from functools import cache

import numpy as np

from ...paths import data_dir
from ..eadl_relaxation import load_eadl_relaxation

COMPTON_PROFILE_SHA256 = "31671e65c882134a3fa591f0497a524846137a97c3e60e06317f08e228a35e81"
_TABLE = ("photon_cross_sections", "compton_profile_j0.csv")  # under data_dir()


@dataclass(frozen=True, slots=True)
class ComptonShells:
    """One element's occupied EADL subshells in cascade order (read-only arrays).

    ``electrons`` are EADL occupation numbers, ``binding_energy_eV`` EADL
    binding energies [eV] and ``profile_j0_au`` the one-electron ``J_i(0)``
    [1/(atomic unit of momentum)].
    """

    designators: np.ndarray
    electrons: np.ndarray
    binding_energy_eV: np.ndarray
    profile_j0_au: np.ndarray


@cache
def _profile_table() -> dict[int, dict[int, float]]:
    data = data_dir().joinpath(*_TABLE).read_bytes()
    if hashlib.sha256(data).hexdigest() != COMPTON_PROFILE_SHA256:
        raise ValueError("packaged Compton-profile table does not match its SHA-256 pin")
    table: dict[int, dict[int, float]] = {}
    lines = [line for line in data.decode().splitlines() if not line.startswith("#")]
    for line in lines[1:]:
        z, designator, _label, value, _source = line.split(",")
        table.setdefault(int(z), {})[int(designator)] = float(value)
    return table


def _readonly(values, dtype=float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


@cache
def compton_shells(element: str) -> ComptonShells:
    """Join EADL occupations/bindings with the pinned Biggs ``J_i(0)`` values.

    The subshell sets must agree exactly, and every value must be finite and
    positive; otherwise a ``ValueError`` names the element.

    Validation: photon-compton-scattering
    """
    relaxation = load_eadl_relaxation(element)
    profiles = _profile_table().get(relaxation.atomic_number, {})
    shells = relaxation.subshells
    if set(profiles) != {shell.shell_designator for shell in shells}:
        raise ValueError(f"Compton-profile subshells for {element} do not match EADL")
    result = ComptonShells(
        _readonly([s.shell_designator for s in shells], np.int64),
        _readonly([s.electrons for s in shells]),
        _readonly([s.binding_energy_eV for s in shells]),
        _readonly([profiles[s.shell_designator] for s in shells]),
    )
    if not all(
        np.isfinite(a).all() and np.all(a > 0)
        for a in (result.electrons, result.binding_energy_eV, result.profile_j0_au)
    ):
        raise ValueError(f"Compton-shell data for {element} must be finite and positive")
    return result
