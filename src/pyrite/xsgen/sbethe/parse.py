"""Parse native-grid stopping-power and cross-section tables written by SBETHE.

Each output carries its own energy grid and each is kept on it, per the
decision that tables are stored on the code's native dense grid and resampled
at load. ``stp.dat`` begins at the corrected-Bethe validity floor (ECUT, about
1 keV for electrons); ``stp-low.dat`` covers the empirical extrapolation below
it. They are different quantities and are not concatenated here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_SCALARS = {
    "electrons_per_molecule": r"Electrons/molecule\s*\.+\s*([0-9.E+-]+)",
    "molecular_weight_g_mol": r"Molecular weight\s*\.+\s*([0-9.E+-]+)",
    "density_g_cm3": r"Density\s*\.+\s*([0-9.E+-]+)",
    "mean_excitation_eV": r"Mean excitation energy\s*\.+\s*([0-9.E+-]+)",
}


def _scalar(text: str, name: str) -> float:
    match = re.search(_SCALARS[name], text)
    if match is None:
        raise ValueError(f"SBETHE output is missing {name}")
    return float(match.group(1).replace("D", "E"))


def _rows(data: str | bytes, columns: int, what: str, *, strict: bool = True) -> np.ndarray:
    """Return the numeric block of one ``#``-commented SBETHE output.

    ``strict`` requires a strictly increasing first column. The oscillator
    table sets it ``False``: its abscissa is transferred energy, and it repeats
    a value at each shell binding edge to carry the step in the
    oscillator-strength density there.
    """
    text = data.decode("ascii") if isinstance(data, bytes) else data
    rows: list[list[float]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != columns:
            raise ValueError(f"SBETHE {what} row has {len(fields)} columns, expected {columns}")
        try:
            rows.append([float(field.replace("D", "E")) for field in fields])
        except ValueError as exc:
            raise ValueError(f"SBETHE {what} row is not numeric: {line!r}") from exc
    if len(rows) < 2:
        raise ValueError(f"SBETHE {what} contains fewer than two rows")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"SBETHE {what} contains non-finite values")
    energy = values[:, 0]
    steps = np.diff(energy)
    if energy[0] <= 0 or np.any(steps < 0) or (strict and np.any(steps <= 0)):
        raise ValueError(f"SBETHE {what} energy grid must be positive and increasing")
    return values


@dataclass(frozen=True)
class SbetheStopping:
    """The ``stp.dat`` collision stopping-power table and its header scalars."""

    energy_ev: np.ndarray
    stopping_ev_per_angstrom: np.ndarray
    stopping_mev_cm2_per_g: np.ndarray
    stopping_no_shell_ev_per_angstrom: np.ndarray
    stopping_no_shell_mev_cm2_per_g: np.ndarray
    stopping_cs_ev_cm2: np.ndarray
    electrons_per_molecule: float
    molecular_weight_g_mol: float
    density_g_cm3: float
    mean_excitation_eV: float


@dataclass(frozen=True)
class SbetheIntegratedCS:
    """The ``asymptotic.dat`` integrated cross sections.

    ``sigma^0``, ``sigma^1`` and ``sigma^2`` are the ``CS0A``/``CS1A``/``CS2A``
    moments the soft/hard inelastic transport work consumes.
    """

    energy_ev: np.ndarray
    sigma0_cm2: np.ndarray
    sigma1_ev_cm2: np.ndarray
    sigma2_ev2_cm2: np.ndarray
    mean_free_path_g_cm2: np.ndarray
    stopping_mev_cm2_per_g: np.ndarray
    straggling_mev2_cm2_per_g: np.ndarray


@dataclass(frozen=True)
class SbetheOscillator:
    """The ``OOS.dat`` optical oscillator-strength distribution."""

    energy_ev: np.ndarray
    oos_per_ev: np.ndarray
    cumulative: np.ndarray
    electrons_per_molecule: float
    mean_excitation_eV: float


def parse_stopping(data: str | bytes) -> SbetheStopping:
    """Parse a complete vendor-format ``stp.dat``."""
    text = data.decode("ascii") if isinstance(data, bytes) else data
    values = _rows(text, 6, "stopping table")
    energy, stp_a, stp_mtu, raw_a, raw_mtu, stp_cs = values.T
    if np.any(stp_a <= 0) or np.any(stp_mtu <= 0) or np.any(stp_cs <= 0):
        raise ValueError("SBETHE stopping powers must be positive")
    return SbetheStopping(
        energy_ev=energy,
        stopping_ev_per_angstrom=stp_a,
        stopping_mev_cm2_per_g=stp_mtu,
        stopping_no_shell_ev_per_angstrom=raw_a,
        stopping_no_shell_mev_cm2_per_g=raw_mtu,
        stopping_cs_ev_cm2=stp_cs,
        electrons_per_molecule=_scalar(text, "electrons_per_molecule"),
        molecular_weight_g_mol=_scalar(text, "molecular_weight_g_mol"),
        density_g_cm3=_scalar(text, "density_g_cm3"),
        mean_excitation_eV=_scalar(text, "mean_excitation_eV"),
    )


def parse_integrated(data: str | bytes) -> SbetheIntegratedCS:
    """Parse a complete vendor-format ``asymptotic.dat``."""
    values = _rows(data, 7, "integrated cross sections")
    energy, sigma0, sigma1, sigma2, mfp, stopping, straggling = values.T
    if np.any(sigma0 <= 0) or np.any(sigma1 <= 0) or np.any(mfp <= 0):
        raise ValueError("SBETHE integrated cross sections must be positive")
    return SbetheIntegratedCS(
        energy_ev=energy,
        sigma0_cm2=sigma0,
        sigma1_ev_cm2=sigma1,
        sigma2_ev2_cm2=sigma2,
        mean_free_path_g_cm2=mfp,
        stopping_mev_cm2_per_g=stopping,
        straggling_mev2_cm2_per_g=straggling,
    )


def parse_oscillator(data: str | bytes) -> SbetheOscillator:
    """Parse a complete vendor-format ``OOS.dat``."""
    text = data.decode("ascii") if isinstance(data, bytes) else data
    values = _rows(text, 3, "oscillator table", strict=False)
    energy, oos, cumulative = values.T
    if np.any(oos < 0):
        raise ValueError("SBETHE oscillator strengths must be non-negative")
    if np.any(np.diff(cumulative) < 0):
        raise ValueError("SBETHE cumulative oscillator strength must be non-decreasing")
    return SbetheOscillator(
        energy_ev=energy,
        oos_per_ev=oos,
        cumulative=cumulative,
        electrons_per_molecule=_scalar(text, "electrons_per_molecule"),
        mean_excitation_eV=_scalar(text, "mean_excitation_eV"),
    )


def table_arrays(
    stopping: SbetheStopping,
    integrated: SbetheIntegratedCS,
    oscillator: SbetheOscillator,
) -> dict[str, np.ndarray]:
    """Stack the parsed outputs into the native-grid table-store schema.

    The three grids differ and each keeps its own axis: ``stp.dat`` starts at
    the corrected-Bethe floor, ``asymptotic.dat`` extends below it, and
    ``OOS.dat`` is in transferred energy, not projectile energy.
    """
    if abs(stopping.electrons_per_molecule - oscillator.electrons_per_molecule) > 1e-6 * max(
        stopping.electrons_per_molecule, 1.0
    ):
        raise ValueError("SBETHE outputs disagree on electrons per molecule")
    return {
        "stopping_energy_eV": stopping.energy_ev,
        "stopping_eV_per_angstrom": stopping.stopping_ev_per_angstrom,
        "stopping_MeV_cm2_per_g": stopping.stopping_mev_cm2_per_g,
        "stopping_no_shell_eV_per_angstrom": stopping.stopping_no_shell_ev_per_angstrom,
        "stopping_no_shell_MeV_cm2_per_g": stopping.stopping_no_shell_mev_cm2_per_g,
        "stopping_cs_eV_cm2": stopping.stopping_cs_ev_cm2,
        "integrated_energy_eV": integrated.energy_ev,
        "sigma0_cm2": integrated.sigma0_cm2,
        "sigma1_eV_cm2": integrated.sigma1_ev_cm2,
        "sigma2_eV2_cm2": integrated.sigma2_ev2_cm2,
        "mean_free_path_g_cm2": integrated.mean_free_path_g_cm2,
        "integrated_stopping_MeV_cm2_per_g": integrated.stopping_mev_cm2_per_g,
        "straggling_MeV2_cm2_per_g": integrated.straggling_mev2_cm2_per_g,
        "oos_energy_eV": oscillator.energy_ev,
        "oos_per_eV": oscillator.oos_per_ev,
        "oos_cumulative": oscillator.cumulative,
        "electrons_per_molecule": np.asarray(stopping.electrons_per_molecule),
        "molecular_weight_g_mol": np.asarray(stopping.molecular_weight_g_mol),
        "density_g_cm3": np.asarray(stopping.density_g_cm3),
        "mean_excitation_eV": np.asarray(stopping.mean_excitation_eV),
    }
