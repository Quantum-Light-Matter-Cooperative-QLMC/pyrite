"""Argument normalisation shared by :func:`simulate_trajectories`.

Split from ``api.py`` to keep that module under the line budget.
"""

import numpy as np

from ...materials.attenuation import _normalize_composition


def resolve_layers_and_elastic(
    layers,
    *,
    thickness_ang,
    element,
    n_atoms_per_ang3,
    composition,
    elastic_model,
    elastic_tables,
    energy_model,
):
    """Return ``(layers, elastic_tables)`` after validating the model names.

    ``layers=None`` becomes a single layer spanning the slab (bit-for-bit the
    old transport); ``elastic_model='elsepa'`` resolves tables from the stack."""
    if layers is None:
        layers = [
            (
                0.0,
                float(thickness_ang),
                _normalize_composition(element, n_atoms_per_ang3, composition),
            )
        ]

    if elastic_model not in ("mott", "sr", "elsepa"):
        raise ValueError("elastic_model must be 'mott', 'sr', or 'elsepa'")
    if elastic_model != "elsepa" and elastic_tables is not None:
        raise ValueError("elastic_tables is only valid with elastic_model='elsepa'")
    if elastic_model == "elsepa" and elastic_tables is None:
        from ...xsgen.elsepa.catalog import resolve_stack_tables

        elastic_tables = resolve_stack_tables(layers)
    if energy_model not in ("frozen", "midpoint"):
        raise ValueError("energy_model must be 'frozen' or 'midpoint'")
    return layers, elastic_tables


def resolve_cutoffs_by_electron(E_cut_by_electrons, E_cut_keV, Ne):
    """Per-electron cutoff energies [keV] as a validated ``(Ne,)`` float array."""
    if E_cut_by_electrons is None:
        E_cut_by_electrons = np.full(Ne, float(E_cut_keV), dtype=np.float64)
    else:
        E_cut_by_electrons = np.asarray(E_cut_by_electrons, dtype=np.float64)

        if E_cut_by_electrons.shape != (Ne,):
            raise ValueError(
                f"E_cut_by_electrons must have shape ({Ne},), got {E_cut_by_electrons.shape}"
            )

    if not np.all(np.isfinite(E_cut_by_electrons)) or not np.all(E_cut_by_electrons > 0.0):
        raise ValueError("electron cutoff energies must be finite and strictly positive")
    return E_cut_by_electrons
