"""EEDL and Bote--Salvat shell ionization against measured cross sections.

Compares both characteristic-scoring ionization models with measured
electron-impact cross sections, transcribed with per-row primary citations into
``reference_data/inner_shell_ionization/`` (#92): thin-film K-shell data for the
catalogue transport elements, and the Pb and Th L1--L3 subshell data of
Rahangdale et al. (2015). Each model is evaluated at the measured incident
energy exactly as characteristic scoring evaluates it, by lin-lin interpolation
of the subshell rows of ``load_characteristic_cross_sections`` with zero outside
the row, so the Bote--Salvat side is the tabulated backend, not the raw fit.

Ratios ``model/measured`` are summarised per shell and overvoltage band by their
geometric mean, with the fraction of points the model meets within the
measurement's own one-sigma uncertainty. Overvoltage is ``E/I`` with the
compilation's Larkins energy for K rows and the EEDL edge for L rows.

Assumptions: thin targets (single collision, no backscatter or fluorescence
enhancement beyond the authors' corrections); free-atom models compared with
solid-target measurements; quoted uncertainties are one sigma. Limiting case:
both models vanish below their own edges, so rows with ``E`` between the two
edges compare a zero with a finite measurement and are kept, not dropped.

Validation: bote-salvat-ionization-backend
"""

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..montecarlo.spectrum.characteristic import (
    CharacteristicCrossSectionTable,
    load_characteristic_cross_sections,
)

REFERENCE_DIR = Path(__file__).with_name("reference_data") / "inner_shell_ionization"
K_SHELL_CSV = REFERENCE_DIR / "k_shell_thin_film.csv"
L_SUBSHELL_CSV = REFERENCE_DIR / "l_subshell_pb_th.csv"
REFERENCES_CSV = REFERENCE_DIR / "references.csv"
#: Overvoltage band edges: near threshold, the ionization peak, the fall-off,
#: and the Bethe region past the fit's U = 16 branch switch.
OVERVOLTAGE_BANDS = (1.0, 1.5, 4.0, 20.0, np.inf)
_BARN_CM2 = 1.0e-24


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured shell ionization cross section [barn] at ``energy_keV``."""

    atomic_number: int
    symbol: str
    shell: str
    energy_keV: float
    overvoltage: float
    sigma_b: float
    dsigma_b: float | None
    ref: str


@dataclass(frozen=True, slots=True)
class BandSummary:
    """Model/measured agreement for one shell family and overvoltage band.

    ``*_ratio`` is the geometric mean of ``model/measured`` over points the
    model populates; ``*_within`` is the fraction of points with an uncertainty
    that the model meets within one sigma.
    """

    shell: str
    band: tuple[float, float]
    n: int
    eedl_ratio: float
    bote_salvat_ratio: float
    eedl_within: float
    bote_salvat_within: float


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def load_references() -> dict[str, dict[str, str]]:
    """Primary citations keyed by the ``ref`` column of the measurement files."""
    return {row["ref"]: row for row in _rows(REFERENCES_CSV)}


def load_k_shell_measurements() -> list[Measurement]:
    """Thin-film K-shell measurements; ``U`` is the compilation's ``E/I_K``."""
    return [
        Measurement(
            int(row["Z"]),
            row["symbol"],
            "K",
            float(row["E_keV"]),
            float(row["U"]),
            float(row["sigma_K_b"]),
            float(row["dsigma_K_b"]) if row["dsigma_K_b"] else None,
            row["ref"],
        )
        for row in _rows(K_SHELL_CSV)
    ]


def load_l_subshell_measurements() -> list[Measurement]:
    """Pb and Th L1--L3 measurements; ``U`` is ``E`` over the EEDL edge."""
    measurements = []
    for row in _rows(L_SUBSHELL_CSV):
        table = load_characteristic_cross_sections(row["symbol"])
        edge = table.shell_binding_energy_eV[table.ionization_shell_labels.index(row["shell"])]
        energy = float(row["E_keV"])
        measurements.append(
            Measurement(
                int(row["Z"]),
                row["symbol"],
                row["shell"],
                energy,
                1.0e3 * energy / float(edge),
                float(row["sigma_b"]),
                float(row["dsigma_b"]),
                row["ref"],
            )
        )
    return measurements


def _table_sigma_b(table: CharacteristicCrossSectionTable, shell: str, energy_eV: float):
    """Scoring's lin-lin, zero-outside value of one subshell row [barn]."""
    index = table.ionization_shell_labels.index(shell)
    grid = table.projectile_energy_eV_by_shell[index]
    if not grid[0] <= energy_eV <= grid[-1]:
        return 0.0
    values = table.ionization_cross_sections_cm2_by_shell[index]
    return float(np.interp(energy_eV, grid, values)) / _BARN_CM2


def model_cross_sections_b(measurements: Iterable[Measurement]) -> np.ndarray:
    """``(n, 2)`` EEDL and Bote--Salvat cross sections [barn] at each measurement.

    Validation: bote-salvat-ionization-backend
    """
    values = []
    for m in measurements:
        tables = [
            load_characteristic_cross_sections(m.symbol, ionization_model=model)
            for model in ("eedl", "bote-salvat")
        ]
        values.append([_table_sigma_b(t, m.shell, 1.0e3 * m.energy_keV) for t in tables])
    return np.array(values, dtype=float).reshape(-1, 2)


def summarise(
    measurements: Sequence[Measurement],
    model_b: np.ndarray,
    bands: Sequence[float] = OVERVOLTAGE_BANDS,
) -> list[BandSummary]:
    """Per shell family and overvoltage band agreement of both models.

    Validation: bote-salvat-ionization-backend
    """
    shell = np.array([m.shell for m in measurements])
    overvoltage = np.array([m.overvoltage for m in measurements])
    sigma = np.array([m.sigma_b for m in measurements])
    dsigma = np.array([np.nan if m.dsigma_b is None else m.dsigma_b for m in measurements])
    summaries = []
    for label in dict.fromkeys(shell):
        for low, high in zip(bands[:-1], bands[1:], strict=True):
            keep = (shell == label) & (overvoltage >= low) & (overvoltage < high)
            if not keep.any():
                continue
            ratios, within = [], []
            for column in model_b[keep].T:
                positive = column > 0.0
                ratio = column[positive] / sigma[keep][positive]
                ratios.append(float(np.exp(np.mean(np.log(ratio)))) if ratio.size else np.nan)
                quoted = np.isfinite(dsigma[keep])
                hit = np.abs(column - sigma[keep])[quoted] <= dsigma[keep][quoted]
                within.append(float(hit.mean()) if hit.size else np.nan)
            summaries.append(
                BandSummary(str(label), (low, high), int(keep.sum()), *ratios, *within)
            )
    return summaries


__all__ = [
    "K_SHELL_CSV",
    "L_SUBSHELL_CSV",
    "OVERVOLTAGE_BANDS",
    "REFERENCES_CSV",
    "BandSummary",
    "Measurement",
    "load_k_shell_measurements",
    "load_l_subshell_measurements",
    "load_references",
    "model_cross_sections_b",
    "summarise",
]
