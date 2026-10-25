"""Internal simulation composition shared by API and case/observation lowering.

Scene's embedded run fields remain compatibility inputs during the migration.
Resolve them once, before material preparation, and keep requested settings
separate from effective quantum-adapted settings. This is not a public RunSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from .._numerics import Numerics, Physics, resolve_physics
from ..instrument import Acquisition, Filter, FilterPlate, NativeAcquisition, PlanarDetector
from ..quantum import QuantumSetup

if TYPE_CHECKING:
    from ..detectors import Detector
    from .model import Scene


class _Unset(Enum):
    VALUE = "not supplied"


UNSET = _Unset.VALUE


@dataclass(frozen=True)
class ResolvedSimulation:
    """Requested compatibility scene plus effective model/numerical controls."""

    scene: Scene
    physics: Physics
    numerics: Numerics


def resolve_models(scene: Scene, numerics: Numerics) -> tuple[Physics, Numerics]:
    """Adapt quantum settings once and validate source/model applicability."""
    physics = resolve_physics(scene.physics, numerics)
    if scene.quantum is not None:
        physics, numerics = scene.quantum.lower(physics, numerics)
    physics.validate_transport(numerics.energy_model)
    if physics.channeling_model == "axial":
        from .axial_spec import refuse_unverified_axial_run

        refuse_unverified_axial_run()
    if scene.beam.state is not None:
        from ..montecarlo.channeling.planar_injection import source_record
        from ..montecarlo.transverse import TransverseDistribution

        if physics.channeling_model != "planar":
            raise ValueError("Beam.state production requires planar channeling")
        if isinstance(scene.beam.transverse, TransverseDistribution):
            raise ValueError("Beam.state cannot combine with classical Twiss slopes")
        if scene.beam.energy_spread_frac not in (None, 0.0):
            raise ValueError("Beam.state requires a monoenergetic source")
        source_record(scene.beam.state)
    if numerics.precision is not None and isinstance(scene.detector, PlanarDetector):
        raise ValueError("adaptive precision does not support physical detectors; use fixed counts")
    return physics, numerics


def resolve_simulation(
    beam: Any,
    target: Any = None,
    detector: Detector | PlanarDetector | None = None,
    *,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    filters: tuple[FilterPlate | Filter, ...] | None = None,
    acquisition: Acquisition | NativeAcquisition | None = None,
    require_detector: bool = False,
) -> ResolvedSimulation:
    """Resolve Scene/component compatibility and explicit run overrides.

    Physics/numerics retain their historical ``None``-means-inherit behavior.
    Omitted quantum inherits Scene.quantum; explicit ``None`` disables it.
    No transport, table loading, response generation or output writes occur.
    """
    from .model import Scene

    if isinstance(beam, Scene):
        if (
            target is not None
            or detector is not None
            or filters is not None
            or acquisition is not None
        ):
            raise TypeError("with a Scene, omit target, detector, filters and acquisition")
        scene = beam
        numerics = scene.numerics if numerics is None else numerics
        physics = scene.physics if physics is None else physics
        quantum = scene.quantum if quantum is UNSET else quantum
    else:
        if target is None:
            raise TypeError("simulate requires a target")
        numerics = Numerics() if numerics is None else numerics
        quantum = None if quantum is UNSET else quantum
        scene = None
    if not isinstance(numerics, Numerics):
        raise TypeError("numerics must be a Numerics")
    physics = resolve_physics(physics, numerics)
    if quantum is not None and not isinstance(quantum, QuantumSetup):
        raise TypeError("quantum must be a QuantumSetup or None")
    if scene is None:
        scene = Scene(
            beam,
            target,
            detector,
            filters=() if filters is None else filters,
            acquisition=acquisition,
            physics=physics,
            numerics=numerics,
            quantum=quantum,
        )
    else:
        scene = replace(scene, physics=physics, numerics=numerics, quantum=quantum)
    if require_detector and scene.detector is None:
        raise TypeError("simulate requires a detector; use directional_source for source maps")
    effective_physics, effective_numerics = resolve_models(scene, numerics)
    return ResolvedSimulation(scene, effective_physics, effective_numerics)
