"""Automatic CXR reflection policy at the campaign lowering boundary (#1173)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._numerics import CxrReflections, Numerics, Physics
from ..materials import CATALOG

if TYPE_CHECKING:
    from .sweep import Sweep


def reflection_controls(numerics: Numerics, physics: Physics, channeling: object) -> CxrReflections:
    """Fall back to four families only for an unsupported automatic selection."""
    controls = numerics.cxr_reflections
    # Streamed scoring selects on the electron prefix since #1281, so it no
    # longer falls back.
    if (
        controls.n_families is None
        and controls.line_yield_tolerance is None
        and (
            physics.coherent_emission
            or numerics.temporal_profile
            or numerics.line_roulette is not None
            or channeling is not None
            or physics.transition_radiation == "two-beam"
        )
    ):
        return replace(controls, n_families=4)
    return controls


def resolve_reflection_sweep(
    sweep: Sweep, coherent_emission: object, temporal_profile: object
) -> tuple[Sweep, bool]:
    """Resolve the automatic tolerance while preserving explicit and pinned sets."""
    from .sweep import _stack_line_radiators, sweep_crystal_params

    automatic = (
        sweep.n_families is None
        and sweep.max_reflections is None
        and sweep.line_yield_tolerance is None
        and not coherent_emission
        and not temporal_profile
        and not any(
            CATALOG.crystal(crystal).hkl_list
            for crystal, _ in [
                (sweep_crystal_params(sweep)["crystal"], None),
                *_stack_line_radiators(sweep),
            ]
        )
    )
    return replace(
        sweep,
        n_families=sweep.n_families or 4,
        line_yield_tolerance=1e-3 if automatic else sweep.line_yield_tolerance,
    ), automatic
