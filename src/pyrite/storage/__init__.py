"""Persistence facade for checkpoints, observations, and execution artifacts.

Storage objects describe bytes on disk, not new scientific result types.
Classes and functions are imported lazily from their original implementations;
this preserves existing file schemas, pickle names, and cache identities. The
campaign coordinator still owns checkpoint *publication* and resume logic.

Use :mod:`pyrite.results` for the in-memory numerical values and reductions.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.checkpoints.persistence import (
        checkpoint_manifest as checkpoint_manifest,
    )
    from pyrite.checkpoints.persistence import (
        checkpoint_path_for as checkpoint_path_for,
    )
    from pyrite.checkpoints.persistence import (
        default_checkpoint_dir as default_checkpoint_dir,
    )
    from pyrite.checkpoints.persistence import (
        load_checkpoint as load_checkpoint,
    )
    from pyrite.execution.artifacts import (
        ArtifactIntegrityError as ArtifactIntegrityError,
    )
    from pyrite.execution.artifacts import (
        CaseArtifactRef as CaseArtifactRef,
    )
    from pyrite.execution.artifacts import (
        CaseArtifactStore as CaseArtifactStore,
    )
    from pyrite.execution.artifacts import (
        DirectionalArtifactRef as DirectionalArtifactRef,
    )
    from pyrite.execution.artifacts import (
        directional_content_key as directional_content_key,
    )
    from pyrite.execution.artifacts import (
        load_directional_result as load_directional_result,
    )
    from pyrite.observations.store import (
        ObservationStore as ObservationStore,
    )
    from pyrite.observations.store import (
        ObservationStoreError as ObservationStoreError,
    )
    from pyrite.observations.store import (
        StoredObservation as StoredObservation,
    )
    from pyrite.observations.store import (
        default_observation_root as default_observation_root,
    )
    from pyrite.observations.store import (
        observation_from_result as observation_from_result,
    )

_EXPORTS = {
    "checkpoint_manifest": ("pyrite.checkpoints.persistence", "checkpoint_manifest"),
    "checkpoint_path_for": ("pyrite.checkpoints.persistence", "checkpoint_path_for"),
    "default_checkpoint_dir": ("pyrite.checkpoints.persistence", "default_checkpoint_dir"),
    "load_checkpoint": ("pyrite.checkpoints.persistence", "load_checkpoint"),
    "ObservationStore": ("pyrite.observations.store", "ObservationStore"),
    "ObservationStoreError": ("pyrite.observations.store", "ObservationStoreError"),
    "StoredObservation": ("pyrite.observations.store", "StoredObservation"),
    "default_observation_root": ("pyrite.observations.store", "default_observation_root"),
    "observation_from_result": ("pyrite.observations.store", "observation_from_result"),
    "ArtifactIntegrityError": ("pyrite.execution.artifacts", "ArtifactIntegrityError"),
    "CaseArtifactRef": ("pyrite.execution.artifacts", "CaseArtifactRef"),
    "CaseArtifactStore": ("pyrite.execution.artifacts", "CaseArtifactStore"),
    "DirectionalArtifactRef": ("pyrite.execution.artifacts", "DirectionalArtifactRef"),
    "directional_content_key": ("pyrite.execution.artifacts", "directional_content_key"),
    "load_directional_result": ("pyrite.execution.artifacts", "load_directional_result"),
}

__all__ = [
    "checkpoint_manifest",
    "checkpoint_path_for",
    "default_checkpoint_dir",
    "load_checkpoint",
    "ObservationStore",
    "ObservationStoreError",
    "StoredObservation",
    "default_observation_root",
    "observation_from_result",
    "ArtifactIntegrityError",
    "CaseArtifactRef",
    "CaseArtifactStore",
    "DirectionalArtifactRef",
    "directional_content_key",
    "load_directional_result",
]


def __getattr__(name: str) -> Any:
    """Resolve one approved storage API without importing scheduler backends."""
    binding = _EXPORTS.get(name)
    if binding is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = binding
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
