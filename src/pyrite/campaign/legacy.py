"""D7 adapters between the legacy sweep/settings pair and public objects."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .geometry import Footprint, Slab, Stack
from .model import Analysis, Convergence, Numerics, Scene, Sweep


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
            thickness_ang=_axis(
                axes, "target.layers[0].thickness_ang", layers[0].thickness_ang
            ),
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
        emission=getattr(settings, "emission", "incoherent"),
        xray_dispersion=getattr(settings, "xray_dispersion", "vacuum"),
        brem_source=getattr(settings, "brem_source", "mc"),
    )
    return Sweep(base=scene, axes=axes)


def numerics_from_legacy(old_sweep: Any, settings: Any) -> Numerics:
    """Resolve the first legacy sampling-budget point and all fixed controls."""
    line = old_sweep.n_electrons
    brem = old_sweep.n_electrons_brem
    return Numerics(
        n_electrons=int(
            getattr(settings, "n_electrons", 450) if line is None else _values(line)[0]
        ),
        n_electrons_brem=int(
            getattr(settings, "n_electrons_brem", 100) if brem is None else _values(brem)[0]
        ),
        spec_chunk=old_sweep.spec_chunk,
        brem_chunk=old_sweep.brem_chunk,
        convergence=Convergence(
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


__all__ = ["analysis_from_legacy", "numerics_from_legacy", "sweep_from_legacy"]
