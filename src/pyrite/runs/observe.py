"""Produce or reuse one persisted pixel-detector observation.

A stored observation is reused, without transport, when its true-spatial
factors are exactly those this scene would produce: the same source case, pixel
geometry, ordered filters, angular scorer, representative directions, and
attenuation coefficients recomputed on the stored energy grids. Only the
read-time layers -- response, acquisition, and beam normalization -- are then
rescored. Any geometry, filter, or angular-sampling change needs new transport,
because intrinsic checkpoints keep one observation direction and cannot be
re-angled.
"""

from dataclasses import dataclass

from .. import api
from .._numerics import Physics
from ..campaign.model import Numerics, Scene
from ..campaign.resolution import UNSET, _Unset
from ..observations import ObservationStore, StoredObservation, observation_from_result
from ..observations.spec import Observation
from ..quantum import QuantumSetup


@dataclass(frozen=True)
class ObservationRun:
    """Outcome of :func:`produce_observation`.

    Parameters
    ----------
    observation
        The stored observation matching the requested scene.
    transported
        ``True`` when electron transport ran; ``False`` when stored true
        factors were rescored.
    """

    observation: StoredObservation
    transported: bool


def produce_observation(
    scene: Scene,
    numerics: Numerics,
    store: ObservationStore,
    *,
    physics: Physics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> ObservationRun:
    """Return the stored observation for ``scene``, transporting only if needed.

    Parameters
    ----------
    scene
        Scene with a pixelated :class:`~pyrite.PlanarDetector`, an explicit
        response, and an :class:`~pyrite.Acquisition`.
    numerics
        Transport numerics; part of the source identity.
    store
        Destination store. The result is indexed under its source content key.
    physics
        Optional override of the scene's legacy model selection.
    quantum
        Independent quantum setup. Omission inherits ``Scene.quantum``;
        explicit ``None`` disables it, as in :func:`pyrite.simulate`.
    observation
        Independent detector response, acquisition and scoring request.

    Returns
    -------
    ObservationRun
        The persisted observation and whether transport ran.
    """
    reused = api.observation_plan(
        scene, numerics, physics=physics, quantum=quantum, observation=observation
    ).find_reusable(store)
    if reused is not None:
        store.put(reused)
        return ObservationRun(reused, transported=False)
    result = api.simulate(
        scene,
        numerics=numerics,
        physics=physics,
        quantum=quantum,
        observation=observation,
    )
    produced = observation_from_result(result)
    store.put(produced)
    return ObservationRun(produced, transported=True)


__all__ = ["ObservationRun", "produce_observation"]
