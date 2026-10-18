"""Argument normalisation shared by :func:`simulate_trajectories`.

Split from ``api.py`` to keep that module under the line budget.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ...materials.attenuation import _normalize_composition


@dataclass(frozen=True)
class TransportInputs:
    """Explicit :func:`simulate_trajectories` call: one field per parameter.

    The cascade re-enters the public function with :meth:`as_kwargs`, so a
    recursive call receives exactly these inputs and no host-side locals.
    """

    E0_keV: Any
    Ne: Any
    thickness_ang: Any
    element: Any
    n_atoms_per_ang3: Any
    E_cut_keV: Any
    seed: Any
    max_steps: Any
    elastic_model: Any
    beam_dir: Any
    composition: Any
    layers: Any
    beam_fwhm_mm: Any
    beam_fwhm_y_mm: Any
    bunch_length_fs: Any
    long_shape: Any
    long_offsets_fs: Any
    longitudinal_distribution: Any
    transverse_distribution: Any
    energy_spread_frac: Any
    crystal_width_mm: Any
    crystal_height_mm: Any
    tilt_polar_rad: Any
    tilt_azim_rad: Any
    groove: Any
    E_cut_by_electrons: Any
    transport_core: Any
    per_electron_config: Any
    keep_segments_on_device: Any
    transport_lut_config: Any
    collect_diagnostics: Any
    energy_model: Any
    max_dE_frac: Any
    straggling: Any
    stopping_tables: Any
    inelastic_model: Any
    inelastic_cutoff_eV: Any
    inelastic_materials: Any
    elastic_tables: Any
    radiative_model: Any
    radiative_cutoff_eV: Any
    bremslib_tables: Any
    gdf_source: Any
    secondary_threshold_eV: Any
    max_secondary_generations: Any
    max_secondary_tracks: Any
    pair_production_model: Any
    positron_transport: Any
    atomic_electron_deflection: Any
    transport_progress: Any
    _secondary: Any
    _electron_start: Any
    _energy_range_keV: Any
    bunch_front_tilt: Any

    def as_kwargs(self):
        """Shallow ``{parameter: value}`` mapping; arrays and tables are shared."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


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
