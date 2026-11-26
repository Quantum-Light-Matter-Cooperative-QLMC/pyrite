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
    if controls.fiber is not None:
        # The fiber domain owns the line reflections; the four catalog basal
        # families only name the diffracted-brem and Kossel rows (#1483). The
        # case carries the fiber record instead (:func:`fiber_case_fields`).
        return replace(controls, fiber=None, n_families=4)
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


def fiber_case_fields(numerics: Numerics) -> dict[str, object]:
    """Case keys of the opt-in full-fiber domain (#1483); ``{}`` when basal-only.

    The fiber domain has no qualified automatic or stored line grid: stored
    catalog rows and automatic seeding know only the basal ``(00l)`` lines,
    so an explicit ``EnergyBins.cxr`` grid is required. Lines are scored as
    bin means (``line_quadrature="bin-mean"``).

    Validation: cxr-fiber-intensity-average
    """
    fiber = numerics.cxr_reflections.fiber
    if fiber is None:
        return {}
    bins = numerics.energy_bins
    if bins is None or bins.cxr is None or numerics.line_resolution is not None:
        raise ValueError(
            "full-fiber CXR requires an explicit line grid (Numerics.energy_bins with "
            "EnergyBins.cxr, no line_resolution): stored and automatic line grids "
            "know only the basal (00l) lines (#1483)"
        )
    from .._fiber_average import FiberAverage
    from ..montecarlo.spectrum.cxr.fiber import FiberCXRModel

    # Bin means, as the paired scorer and the #1480 arms integrate the lines.
    fields: dict[str, object] = {
        "cxr_fiber": FiberCXRModel.from_numerics(fiber).case_record(),
        "line_quadrature": "bin-mean",
    }
    if fiber.reflection_batch_size != FiberAverage(cutoff_invang=1.0).reflection_batch_size:
        fields["cxr_fiber_batch_size"] = fiber.reflection_batch_size
    return fields
