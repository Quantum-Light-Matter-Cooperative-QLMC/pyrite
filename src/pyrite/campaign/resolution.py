"""Internal simulation composition shared by API and case/observation lowering.

Scene's embedded run fields remain compatibility inputs during the migration.
Resolve them once, before material preparation, and keep requested settings
separate from effective quantum-adapted settings. This is not a public RunSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._numerics import Numerics, Physics, resolve_physics
from .._spec_defaults import UNSET, _Unset
from ..instrument import Acquisition, Filter, FilterPlate, NativeAcquisition, PlanarDetector
from ..observations.spec import Observation, bind_observation
from ..quantum import QuantumSetup

if TYPE_CHECKING:
    from ..detectors import Detector
    from ..execution.protocol import Executor
    from .model import Scene


@dataclass(frozen=True)
class ResolvedSimulation:
    """Requested compatibility scene plus effective model/numerical controls."""

    scene: Scene
    physics: Physics
    numerics: Numerics
    input_scene: Scene
    observation: Observation | None = None
    executor: Executor | None = None

    @property
    def composition_provenance(self) -> dict[str, Any]:
        """Keep physical inputs and independent effective configuration available."""
        provenance: dict[str, Any] = {
            "input_scene": self.input_scene,
            "physics": self.physics,
            "quantum": self.scene.quantum,
            "acquisition": self.scene.acquisition,
        }
        if self.executor is not None:
            from .._executor_contract import execution_provenance

            provenance["execution"] = execution_provenance(self.executor)
        if self.observation is None:
            return provenance
        return {**provenance, "observation_spec": self.observation}


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


def _bind_scene_observation(
    scene: Scene, observation: Observation | None
) -> tuple[Scene, Observation | None]:
    """Create the legacy execution view without changing the supplied scene."""
    if observation is None:
        return scene, None
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation or None")
    detector, acquisition, normalized = bind_observation(
        observation, scene.detector, scene.acquisition
    )
    if scene.filters and not isinstance(detector, PlanarDetector):
        raise ValueError("virtual angular observations do not support Scene.filters")
    return replace(scene, detector=detector, acquisition=acquisition), normalized


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
    observation: Observation | None = None,
    executor: Executor | None = None,
    require_detector: bool = False,
) -> ResolvedSimulation:
    """Resolve Scene/component compatibility and explicit run overrides.

    Physics/numerics retain their historical ``None``-means-inherit behavior.
    Omitted quantum inherits Scene.quantum; explicit ``None`` disables it.
    No transport, table loading, response generation or output writes occur.
    """
    from .model import Scene

    if isinstance(beam, Scene):
        if any(value is not None for value in (target, detector, filters, acquisition)):
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
    if executor is not None:
        from .._executor_contract import validate_executor

        validate_executor(executor, numerics)
    physics = resolve_physics(physics, numerics)
    if quantum is not None and not isinstance(quantum, QuantumSetup):
        raise TypeError("quantum must be a QuantumSetup or None")
    input_scene = scene
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
    input_scene = scene if input_scene is None else input_scene
    scene, observation = _bind_scene_observation(scene, observation)
    if require_detector and scene.detector is None:
        raise TypeError("simulate requires a detector; use directional_source for source maps")
    effective_physics, effective_numerics = resolve_models(scene, numerics)
    return ResolvedSimulation(
        scene, effective_physics, effective_numerics, input_scene, observation, executor
    )
