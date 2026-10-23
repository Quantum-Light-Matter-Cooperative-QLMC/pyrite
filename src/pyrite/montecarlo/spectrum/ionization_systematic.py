"""EEDL versus Bote--Salvat characteristic production as a systematic diagnostic.

Validation: bote-salvat-ionization-backend
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .characteristic import (
    CharacteristicCrossSectionTable,
    FluorescenceYieldSource,
    load_characteristic_cross_sections,
)


@dataclass(frozen=True, slots=True)
class IonizationModelDisagreement:
    """Per-line characteristic production under both ionization models.

    ``eedl_cm2`` and ``bote_salvat_cm2`` have shape ``(n_energy, n_line)``:
    the production cross section ``sum_i sigma_i Y_i,line`` [cm²] per atom at
    each incident energy. ``relative_difference`` is
    ``bote_salvat / eedl - 1``, NaN where EEDL production is zero.
    """

    element: str
    incident_energy_eV: np.ndarray
    line_labels: tuple[str, ...]
    eedl_cm2: np.ndarray
    bote_salvat_cm2: np.ndarray
    relative_difference: np.ndarray


def _line_production_cm2(table: CharacteristicCrossSectionTable, energy_eV: np.ndarray):
    """``sum_i sigma_i(E) Y_i,line`` with scoring's lin-lin, zero-outside rule."""
    sigma = np.column_stack(
        [
            np.where(
                (energy_eV >= grid[0]) & (energy_eV <= grid[-1]),
                np.interp(energy_eV, grid, values),
                0.0,
            )
            for grid, values in zip(
                table.projectile_energy_eV_by_shell,
                table.ionization_cross_sections_cm2_by_shell,
                strict=True,
            )
        ]
    ).reshape(energy_eV.size, len(table.ionization_shell_labels))
    return sigma @ table.line_yield_per_vacancy


def ionization_model_disagreement(
    element: str,
    incident_energy_eV: Sequence[float],
    *,
    fluorescence_yields: FluorescenceYieldSource = "eadl",
) -> IonizationModelDisagreement:
    """Compare per-line characteristic production under EEDL and Bote--Salvat.

    Both sides share the EADL cascade, line set and fluorescence yields, so the
    difference isolates the primary-vacancy model and can be quoted as a
    model systematic on a characteristic-line yield at a given beam energy. It
    is a thin-target (single-energy) quantity; a thick-target systematic needs
    the scored spectrum under each model.

    Validation: bote-salvat-ionization-backend
    """
    energy = np.atleast_1d(np.asarray(incident_energy_eV, dtype=float))
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("incident energies must be a finite positive 1-D sequence")
    eedl_table = load_characteristic_cross_sections(
        element, fluorescence_yields=fluorescence_yields
    )
    bote_table = load_characteristic_cross_sections(
        element, fluorescence_yields=fluorescence_yields, ionization_model="bote-salvat"
    )
    eedl = _line_production_cm2(eedl_table, energy)
    bote = _line_production_cm2(bote_table, energy)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(eedl > 0.0, bote / eedl - 1.0, np.nan)
    return IonizationModelDisagreement(
        element, energy, eedl_table.line_labels, eedl, bote, relative
    )


__all__ = ["IonizationModelDisagreement", "ionization_model_disagreement"]
