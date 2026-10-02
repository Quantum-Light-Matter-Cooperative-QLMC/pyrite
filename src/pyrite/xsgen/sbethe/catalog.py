"""Resolve catalog compositions into reproducible SBETHE material inputs."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.constants import Avogadro

from ...materials import CATALOG
from ...materials._transport_data import TRANSPORT_ELEMENTS
from .._errors import TableNotFoundError
from ..store import StoredTable, resolve


@dataclass(frozen=True)
class CatalogMaterial:
    """Physical material inputs consumed by one SBETHE run."""

    key: str
    composition: dict[int, float]
    density_g_cm3: float
    mean_excitation_eV: float
    band_gap_eV: float | None = None


def material_inputs_from_composition(
    key: str,
    composition: tuple[tuple[str, float], ...] | list[tuple[str, float]],
    *,
    band_gap_eV: float | None = None,
) -> CatalogMaterial:
    """Derive SBETHE density and mean excitation energy from number density.

    Source: the mass-density definition and the Bragg logarithmic additivity
    rule for compound mean excitation energies (ICRU Report 37). Mass density
    follows exactly from the catalog's elemental number densities and standard
    atomic weights. The material mean excitation energy uses the
    electron-fraction Bragg logarithmic mixture,
    ``ln(I) = sum(n_i Z_i ln(I_i)) / sum(n_i Z_i)``, with the same elemental
    ICRU/PDG values used by the legacy transport reference model.

    Assumptions: independent-atom Bragg additivity and the catalog's bulk
    number density.  A one-element composition reduces exactly to its elemental
    density and mean excitation energy.

    Validation: sbethe-material-inputs
    """
    if not composition:
        raise ValueError("material composition is empty")

    by_z: dict[int, float] = {}
    mass_sum = 0.0
    electron_sum = 0.0
    log_i_sum = 0.0
    for element, number_density in composition:
        n_i = float(number_density)
        if not np.isfinite(n_i) or n_i <= 0.0:
            raise ValueError(f"number density for {element!r} must be finite and positive")
        try:
            params = TRANSPORT_ELEMENTS[element]
        except KeyError:
            raise ValueError(f"no SBETHE transport inputs for element {element!r}") from None
        z_i = int(params["Z"])
        a_i = float(params["A"])
        i_ev = 1000.0 * float(params["J_keV"])
        by_z[z_i] = by_z.get(z_i, 0.0) + n_i
        mass_sum += n_i * a_i
        electron_weight = n_i * z_i
        electron_sum += electron_weight
        log_i_sum += electron_weight * math.log(i_ev)

    # SBETHE's keyboard composition is a molecular stoichiometry, not a number
    # density.  CIF-derived densities share one common cell-volume factor, so
    # dividing by the smallest population recovers the formula counts.  Snap
    # numerical CIF noise back to exact integers when it is already within the
    # parser tolerance; keep genuinely non-integral occupancies proportional.
    scale = min(by_z.values())
    stoichiometry = {z: value / scale for z, value in by_z.items()}
    stoichiometry = {
        z: float(round(value)) if np.isclose(value, round(value), rtol=0.0, atol=1.0e-8) else value
        for z, value in stoichiometry.items()
    }

    density_g_cm3 = mass_sum / (Avogadro * 1.0e-24)
    # ``math``, not ``np``: this value is hashed into the table key, and
    # numpy's log/exp pick CPU-dependent kernels that differ in the last bit
    # (``pdse2`` re-keyed on the CI runner), whereas glibc's are the same on
    # every x86-64 Linux host. Locally both agree for every catalogue key.
    mean_excitation_eV = math.exp(log_i_sum / electron_sum)
    return CatalogMaterial(
        key=str(key),
        composition=stoichiometry,
        density_g_cm3=float(density_g_cm3),
        mean_excitation_eV=mean_excitation_eV,
        band_gap_eV=band_gap_eV,
    )


def catalog_material(key: str) -> CatalogMaterial:
    """Return SBETHE inputs for a catalog material, crystal, or medium key."""
    resolved_key = str(key)
    if resolved_key in CATALOG.materials:
        resolved_key = CATALOG.material(resolved_key).crystal_key
    if resolved_key in CATALOG.crystals:
        composition = CATALOG.crystal(resolved_key).composition
    elif resolved_key in CATALOG.media:
        composition = CATALOG.media[resolved_key].composition
    else:
        choices = sorted(set(CATALOG.materials) | set(CATALOG.crystals) | set(CATALOG.media))
        raise ValueError(f"unknown catalog material {key!r}; choose one of: {', '.join(choices)}")
    return material_inputs_from_composition(str(key), composition)


def resolve_composition_table(
    key: str,
    composition: tuple[tuple[str, float], ...] | list[tuple[str, float]],
) -> StoredTable:
    """Resolve the exact SBETHE table for one catalog-derived composition."""
    from .generate import material_request

    material = material_inputs_from_composition(key, composition)
    request = material_request(
        material.key,
        material.composition,
        density_g_cm3=material.density_g_cm3,
        mean_excitation_eV=material.mean_excitation_eV,
        band_gap_eV=material.band_gap_eV,
    )
    table = resolve(request.key)
    if table is None:
        raise TableNotFoundError(
            f"no SBETHE table for material {key!r}; generate the catalog table with "
            f"'pyrite tables generate --code sbethe --material {key}'"
        )
    return table


def resolve_catalog_table(key: str) -> StoredTable:
    """Resolve the stored SBETHE table matching a catalog key's current identity."""
    resolved_key = str(key)
    if resolved_key in CATALOG.materials:
        resolved_key = CATALOG.material(resolved_key).crystal_key
    if resolved_key in CATALOG.crystals:
        composition = CATALOG.crystal(resolved_key).composition
    elif resolved_key in CATALOG.media:
        composition = CATALOG.media[resolved_key].composition
    else:
        choices = sorted(set(CATALOG.materials) | set(CATALOG.crystals) | set(CATALOG.media))
        raise ValueError(f"unknown catalog material {key!r}; choose one of: {', '.join(choices)}")
    return resolve_composition_table(str(key), composition)


__all__ = [
    "CatalogMaterial",
    "catalog_material",
    "material_inputs_from_composition",
    "resolve_catalog_table",
    "resolve_composition_table",
]
