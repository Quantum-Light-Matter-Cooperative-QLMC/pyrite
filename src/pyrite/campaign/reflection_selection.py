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
        # sweep carries the fiber numerics instead (:func:`fiber_case_fields`).
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
        sweep.cxr_fiber is None
        and sweep.n_families is None
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


def sweep_fiber(sweep: Sweep):
    """``sweep.cxr_fiber`` as a :class:`FiberAverage` (or ``None``), validated.

    A profile table becomes a ``FiberAverage``. The fiber domain owns the line
    reflections, so it excludes an explicit family count other than the four
    basal DBR/Kossel families, a reflection cap and a measured tolerance.
    """
    from .._cxr_reflections import fiber_average

    fiber = fiber_average(sweep.cxr_fiber)
    if fiber is not None and (
        sweep.n_families not in (None, 4)
        or sweep.max_reflections is not None
        or sweep.line_yield_tolerance is not None
    ):
        raise ValueError(
            "cxr_fiber owns the reflection domain; n_families, max_reflections and "
            "line_yield_tolerance must be unset"
        )
    return fiber


def fiber_case_fields(sweep: Sweep, composition, bremsstrahlung_model: str) -> dict[str, object]:
    """Case keys of the opt-in full-fiber domain (#1483); ``{}`` when basal-only.

    Lines are scored as bin means (``line_quadrature="bin-mean"``), as the
    paired scorer and the #1480 arms integrate them. An explicit
    ``EnergyBins.cxr`` grid is final and carries that key here; otherwise the
    automatic line-grid policy resolves over the fiber domain's roll-averaged
    population and its bin-mean payload carries it (#1489). Stored catalog
    rows know only the basal ``(00l)`` lines and are never used.

    A fiber case requires the BremsLib continuum when it is constructed, so
    ``"auto"`` resolves here; other cases resolve it after construction.

    Validation: cxr-fiber-intensity-average
    """
    fiber = sweep.cxr_fiber
    if fiber is None:
        return {}
    from .._fiber_average import FiberAverage
    from ..montecarlo.spectrum.cxr.fiber import FiberCXRModel

    fields: dict[str, object] = {"cxr_fiber": FiberCXRModel.from_numerics(fiber).case_record()}
    if sweep.energy_bins.cxr is not None and not sweep.cxr_grid_policy:
        fields["line_quadrature"] = "bin-mean"
    if fiber.reflection_batch_size != FiberAverage(cutoff_invang=1.0).reflection_batch_size:
        fields["cxr_fiber_batch_size"] = fiber.reflection_batch_size
    if bremsstrahlung_model == "auto":
        from ..xsgen.bremslib.tables import resolve_bremsstrahlung_model

        elements = [str(row[0]) for row in composition]
        fields["bremsstrahlung_model"] = resolve_bremsstrahlung_model("auto", elements)
    return fields


def project_fiber_identity(sweep_payload: dict, sweep: Sweep) -> None:
    """Hash the opt-in full-fiber domain (#1483) as the record its cases carry.

    The model marker, tensor temperature and thermal source give a changed model
    a new stem. The execution-only batch size stays out. A basal sweep carries no
    key, so its digest is unchanged.
    """
    if sweep.cxr_fiber is None:
        sweep_payload.pop("cxr_fiber", None)
        return
    from ..montecarlo.spectrum.cxr.fiber import FiberCXRModel

    sweep_payload["cxr_fiber"] = FiberCXRModel.from_numerics(sweep.cxr_fiber).case_record()


#: Provenance of a full-fiber case's automatic line axis: bandwidth bounded by
#: the fiber domain's reflections, lines scored as bin means (#1489).
FIBER_DOMAIN_BANDWIDTH_MODEL = "full-fiber-domain-v1"


def fiber_line_axis(sweep: Sweep) -> tuple[list | None, dict | None, dict[str, str]]:
    """``(domain rows, per-call policy, provenance)`` of a sweep's line axis (#1489).

    A basal sweep returns ``(None, sweep.cxr_grid_policy, {})``. A full-fiber
    sweep's domain rows, not its catalog ``hkl_list``, bound the automatic
    bandwidth (``kinematic_cxr_stop_eV``); |g| is roll invariant, so the bound
    holds at every roll node. Fiber lines are scored as bin means and the
    resolution step and point budget follow the quadrature, so an unset
    quadrature becomes ``bin-mean``. Feature windows are refused: their
    kinematic seeds read the case's basal ``hkl_list``, not the domain.

    Validation: cxr-fiber-intensity-average
    Validation: line-grid-kinematic-bandwidth
    """
    per_call = sweep.cxr_grid_policy
    if sweep.cxr_fiber is None:
        return None, per_call, {}
    from ..montecarlo.spectrum.cxr.fiber import FiberCXRModel

    if (per_call or {}).get("windows"):
        raise ValueError("full-fiber CXR does not support feature windows (#1489)")
    hkls = list(FiberCXRModel.from_numerics(sweep.cxr_fiber).prepare().hkls)
    sources = {"bandwidth_reflections": FIBER_DOMAIN_BANDWIDTH_MODEL}
    if not (per_call or {}).get("quadrature"):
        per_call = {**(per_call or {}), "quadrature": "bin-mean"}
        sources["quadrature"] = FIBER_DOMAIN_BANDWIDTH_MODEL
    return hkls, per_call, sources


def with_sources(policy, sources: dict[str, str]):
    """``policy`` with ``sources`` merged into its provenance; unchanged when empty."""
    if not sources:
        return policy
    merged = {**dict(policy.sources), **sources}
    return replace(policy, sources=tuple(sorted(merged.items())))
