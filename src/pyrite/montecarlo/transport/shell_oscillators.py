"""Host-side PENELOPE-2024 shell oscillators for one material formula unit.

This prepares inputs for a shell GOS model; it evaluates no cross section and
selects no transport mode.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xraydb
from scipy.optimize import brentq

from ... import DATA_DIR
from ..shell_configuration import AtomicShell

CONDUCTION_BAND_PATH = DATA_DIR / "conduction_band.toml"
DEFAULT_CONDUCTION_THRESHOLD_EV = 15.0


@dataclass(frozen=True, slots=True)
class ConductionBand:
    """Plasmon-like conduction-band parameters per formula unit.

    ``resonance_eV`` of ``None`` selects the free-electron plasmon energy.
    """

    electrons_per_formula: float
    resonance_eV: float | None
    source: str
    formula: Mapping[int, float] | None = None


@dataclass(frozen=True, slots=True)
class ShellOscillator:
    """One oscillator; the conduction band has ``atomic_number == 0``."""

    atomic_number: int
    label: str
    strength: float
    ionization_energy_eV: float
    resonance_energy_eV: float


@dataclass(frozen=True, slots=True)
class MaterialShellOscillators:
    """Shell oscillators satisfying the dipole sum rule and mean excitation energy."""

    oscillators: tuple[ShellOscillator, ...]
    electrons_per_formula: float
    mean_excitation_eV: float
    plasma_energy_eV: float
    sternheimer_factor: float
    conduction_source: str
    conduction_shells: tuple[tuple[int, str], ...]


def load_conduction_bands(path: str | Path | None = None) -> dict[str, ConductionBand]:
    """Read measured conduction-band parameters keyed by catalog key."""
    source = CONDUCTION_BAND_PATH if path is None else Path(path)
    with source.open("rb") as handle:
        table = tomllib.load(handle)
    bands = {}
    for key, entry in table.items():
        try:
            formula = {xraydb.atomic_number(el): float(n) for el, n in entry["formula"].items()}
            band = ConductionBand(
                float(entry["electrons_per_formula"]),
                float(entry["resonance_eV"]),
                f"{entry['source']}; doi:{entry['doi']}",
                formula,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid conduction-band entry {key!r}") from exc
        values = (band.electrons_per_formula, band.resonance_eV or 0.0)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(f"{source}: {key!r} needs finite positive electrons and resonance")
        bands[key] = band
    return bands


def build_shell_oscillators(
    composition: Mapping[int, float],
    mean_excitation_eV: float,
    plasma_energy_eV: float,
    shells: Mapping[int, tuple[AtomicShell, ...]],
    conduction: ConductionBand | None = None,
    *,
    default_threshold_eV: float = DEFAULT_CONDUCTION_THRESHOLD_EV,
) -> MaterialShellOscillators:
    """Build conduction-band and bound-shell oscillators for one formula unit.

    Source: PENELOPE-2024 §3.2.1. The conduction band has
    ``U_cb = 0``; without measured data its strength counts electrons with
    ``U < default_threshold_eV`` and ``W_cb = sqrt(f_cb/Z) Omega_p`` (Eq. 3.62).
    A measured band instead consumes whole shells in increasing ``U`` until
    ``f_cb`` electrons are reached; a boundary that splits a shell or an
    equal-``U`` group raises. Bound shells use
    ``W_k = sqrt((a U_k)^2 + 2 f_k Omega_p^2 / (3 Z))`` (Eq. 3.63), with one
    Sternheimer factor ``a`` solved from
    ``Z ln I = f_cb ln W_cb + sum_k f_k ln W_k`` (Eq. 3.64).

    Assumptions: free-atom ``pdatconf.p14`` shells, Bragg additivity for
    compounds, and ``Omega_p`` from the total electron density. Limit: a
    single bound shell with ``f = Z`` and no conduction band gives ``W = I``.
    Non-finite inputs, ``W_cb >= I``, and any bound ``W_k <= U_k`` raise.

    Validation: penelope-shell-oscillators
    """
    if not composition or any(not count > 0.0 for count in composition.values()):
        raise ValueError("composition needs positive formula counts")
    if not all(np.isfinite(v) and v > 0.0 for v in (mean_excitation_eV, plasma_energy_eV)):
        raise ValueError("mean excitation and plasma energies must be finite and positive")
    if conduction is not None:
        measured = (conduction.electrons_per_formula, conduction.resonance_eV or 1.0)
        if not all(np.isfinite(v) and v > 0.0 for v in measured):
            raise ValueError("measured conduction band needs finite positive parameters")
        if conduction.formula is not None and dict(conduction.formula) != {
            z: float(n) for z, n in composition.items()
        }:
            raise ValueError("conduction-band formula does not match the composition")
    if missing := sorted(set(composition) - set(shells)):
        raise ValueError(f"no atomic shells for Z={missing}")
    expanded = sorted(
        (
            (shell.ionization_energy_eV, z, shell.designator, count * shell.occupation, shell)
            for z, count in composition.items()
            for shell in shells[z]
        ),
        key=lambda item: item[:3],
    )
    total = sum(item[3] for item in expanded)
    if conduction is None:
        in_band = [item for item in expanded if item[0] < default_threshold_eV]
        source = f"PENELOPE-2024 default: U < {default_threshold_eV:g} eV, Eq. 3.62"
    else:
        in_band, consumed = [], 0.0
        tolerance = 1e-9 * total
        for item in expanded:
            if consumed >= conduction.electrons_per_formula - tolerance:
                break
            in_band.append(item)
            consumed += item[3]
        if abs(consumed - conduction.electrons_per_formula) > tolerance:
            raise ValueError("conduction electrons do not end on a whole-shell boundary")
        rest = expanded[len(in_band) :]
        if in_band and rest and rest[0][0] == in_band[-1][0]:
            raise ValueError("conduction electrons split shells with equal ionization energy")
        source = conduction.source
    f_cb = sum(item[3] for item in in_band)
    bound = (
        expanded[len(in_band) :]
        if conduction is not None
        else [item for item in expanded if item[0] >= default_threshold_eV]
    )
    if f_cb > 0.0:
        measured = None if conduction is None else conduction.resonance_eV
        w_cb = np.sqrt(f_cb / total) * plasma_energy_eV if measured is None else measured
    else:
        w_cb = 0.0
    if w_cb >= mean_excitation_eV:
        raise ValueError("conduction-band resonance must lie below the mean excitation energy")
    u = np.array([item[0] for item in bound])
    f = np.array([item[3] for item in bound])
    lorentz = 2.0 * f * plasma_energy_eV**2 / (3.0 * total)
    target = total * np.log(mean_excitation_eV) - (f_cb * np.log(w_cb) if f_cb > 0.0 else 0.0)

    def closure(a: float) -> float:
        return float(np.sum(f * 0.5 * np.log((a * u) ** 2 + lorentz)) - target)

    if not bound:
        raise ValueError("no bound shell remains to satisfy the mean excitation energy")
    if closure(0.0) >= 0.0:
        raise ValueError("mean excitation energy is below the zero-binding limit")
    upper = 1.0
    while closure(upper) < 0.0:
        upper *= 2.0
    a = brentq(closure, 0.0, upper, xtol=1e-14, rtol=1e-14)
    w = np.sqrt((a * u) ** 2 + lorentz)
    if np.any(w <= u):
        raise ValueError("a bound-shell resonance falls at or below its ionization energy")
    oscillators = [ShellOscillator(0, "cb", f_cb, 0.0, w_cb)] if f_cb > 0.0 else []
    oscillators += [
        ShellOscillator(item[1], item[4].label, item[3], item[0], float(wk))
        for item, wk in zip(bound, w, strict=True)
    ]
    return MaterialShellOscillators(
        tuple(oscillators),
        total,
        mean_excitation_eV,
        plasma_energy_eV,
        float(a),
        source,
        tuple((item[1], item[4].label) for item in in_band),
    )
