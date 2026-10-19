"""Adapters between the configured sweep/settings pair and the public objects.

Permanent infrastructure, not a shim awaiting removal. `adapt_configured_sweep`
is what `pyrite scan` and `pyrite material blaze` run through on every
invocation, via `api.build_configured_cases`. The `*_from_legacy` helpers keep
that name because their *input* is the older representation; nothing about the
callers is legacy. The public `Sweep.from_legacy()` door was removed in 0.5.0.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from .._numerics import _emission_selection
from .geometry import Footprint, Slab, Stack
from .model import Analysis, CxrReflections, Numerics, Physics, Scene, Sweep


def _values(value: Any) -> tuple[Any, ...]:
    array = np.atleast_1d(value)
    return tuple(item.item() if hasattr(item, "item") else item for item in array)


def _axis(axes: dict[str, tuple[Any, ...]], path: str, value: Any) -> Any:
    values = _values(value)
    if len(values) > 1:
        axes[path] = values
    return values[0]


def sweep_from_legacy(old_sweep: Any, settings: Any) -> Sweep:
    """Convert scene-valued grids while preserving their product order."""
    if not hasattr(old_sweep, "beam") or not hasattr(old_sweep, "target"):
        raise TypeError("old_sweep must be a campaign.sweep.Sweep")
    axes: dict[str, tuple[Any, ...]] = {}
    energy_values = _values(old_sweep.beam.energy_keV)
    beam = replace(old_sweep.beam, energy_keV=energy_values[0])
    target = old_sweep.target
    if isinstance(target, Slab):
        footprint = target.footprint
        footprint_values = None
        if footprint is not None:
            footprint_values = (_values(footprint.width_mm), _values(footprint.height_mm))
            footprint = Footprint(footprint_values[0][0], footprint_values[1][0])
        target = replace(
            target,
            thickness_ang=_axis(axes, "target.thickness_ang", target.thickness_ang),
            tilt_deg=_axis(axes, "target.tilt_deg", target.tilt_deg),
            tilt_azim_deg=_axis(axes, "target.tilt_azim_deg", target.tilt_azim_deg),
            footprint=footprint,
        )
        if footprint_values is not None:
            if len(footprint_values[0]) > 1:
                axes["target.footprint.width_mm"] = footprint_values[0]
            if len(footprint_values[1]) > 1:
                axes["target.footprint.height_mm"] = footprint_values[1]
    elif isinstance(target, Stack):
        layers = list(target.layers)
        layers[0] = replace(
            layers[0],
            thickness_ang=_axis(axes, "target.layers[0].thickness_ang", layers[0].thickness_ang),
        )
        footprint = target.footprint
        footprint_values = None
        if footprint is not None:
            footprint_values = (_values(footprint.width_mm), _values(footprint.height_mm))
            footprint = Footprint(footprint_values[0][0], footprint_values[1][0])
        target = replace(
            target,
            layers=tuple(layers),
            footprint=footprint,
            tilt_deg=_axis(axes, "target.tilt_deg", target.tilt_deg),
            tilt_azim_deg=_axis(axes, "target.tilt_azim_deg", target.tilt_azim_deg),
        )
        if footprint_values is not None:
            if len(footprint_values[0]) > 1:
                axes["target.footprint.width_mm"] = footprint_values[0]
            if len(footprint_values[1]) > 1:
                axes["target.footprint.height_mm"] = footprint_values[1]
    else:
        raise TypeError("old_sweep.target must be a Slab or Stack")
    if len(energy_values) > 1:
        axes["beam.energy_keV"] = energy_values
    scene = Scene(
        beam=beam,
        target=target,
        detector=old_sweep.detector,
        physics=physics_from_legacy(settings),
        numerics=numerics_from_legacy(old_sweep, settings),
    )
    return Sweep(base=scene, axes=axes)


def adapt_configured_sweep(old_sweep: Any, settings: Any) -> Sweep:
    """Convert the pair, retaining the exact source expansion for its callers."""
    converted = sweep_from_legacy(old_sweep, settings)
    object.__setattr__(converted, "_legacy_source", (old_sweep, settings))
    return converted


def physics_from_legacy(settings: Any) -> Physics:
    """Resolve profile model choices to the public physics owner."""
    return Physics(
        emission=_emission_selection(getattr(settings, "emission", "incoherent")),
        brem_source=getattr(settings, "brem_source", "mc"),
        straggling=getattr(settings, "straggling", False),
        inelastic_model=getattr(settings, "inelastic_model", "auto"),
        inelastic_cutoff_eV=getattr(settings, "inelastic_cutoff_eV", None),
        secondary_threshold_eV=getattr(settings, "secondary_threshold_eV", None),
        elastic_model=getattr(settings, "elastic_model", "elsepa"),
        bremsstrahlung_model=getattr(settings, "bremsstrahlung_model", "auto"),
        radiative_model=getattr(settings, "radiative_model", "uncoupled"),
        radiative_cutoff_eV=getattr(settings, "radiative_cutoff_eV", None),
        pair_production_model=getattr(settings, "pair_production_model", None),
        positron_transport=getattr(settings, "positron_transport", False),
        atomic_electron_deflection=getattr(settings, "atomic_electron_deflection", "kawrakow"),
        channeling_model=getattr(settings, "channeling_model", "off"),
        channeling_plane=getattr(settings, "channeling_plane", None),
        transition_radiation=getattr(settings, "transition_radiation", "off"),
        transition_lattice_origin_ang=getattr(settings, "transition_lattice_origin_ang", None),
    )


def numerics_from_legacy(old_sweep: Any, settings: Any) -> Numerics:
    """Resolve the first legacy sampling-budget point and all fixed controls."""
    cxr = old_sweep.n_electrons
    brem = old_sweep.n_electrons_brem
    return Numerics(
        cxr_trials=int(getattr(settings, "n_electrons", 450) if cxr is None else _values(cxr)[0]),
        brem_trials=int(
            getattr(settings, "n_electrons_brem", 100) if brem is None else _values(brem)[0]
        ),
        channeling_basis_size=getattr(settings, "channeling_basis_size", 16),
        channeling_bloch_cells=getattr(settings, "channeling_bloch_cells", 16),
        channeling_dz_ang=getattr(settings, "channeling_dz_ang", 250.0),
        channeling_zone_max_ang=getattr(settings, "channeling_zone_max_ang", None),
        precision=getattr(settings, "precision", None),
        energy_bins=old_sweep.energy_bins,
        cxr_chunk=old_sweep.spec_chunk,
        brem_chunk=old_sweep.brem_chunk,
        energy_model=getattr(settings, "energy_model", "frozen"),
        max_dE_frac=getattr(settings, "max_dE_frac", 0.0),
        cxr_reflections=CxrReflections(
            n_families=old_sweep.n_families,
            max_reflections=old_sweep.max_reflections,
            mosaic_nodes=old_sweep.mosaic_nodes,
            mosaic_route=old_sweep.mosaic_route,
        ),
    )


def analysis_from_legacy(settings: Any) -> Analysis:
    """Translate the presentation-only legacy settings fields."""
    return Analysis(
        beam_current_na=getattr(settings, "beam_current_na", 5.0),
        apply_detector_qe=getattr(settings, "apply_detector_qe", False),
        convolve_with_det=getattr(settings, "convolve_with_det", False),
    )


__all__ = [
    "adapt_configured_sweep",
    "analysis_from_legacy",
    "numerics_from_legacy",
    "sweep_from_legacy",
]
