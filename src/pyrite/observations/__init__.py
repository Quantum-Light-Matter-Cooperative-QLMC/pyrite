"""Measurement, acquisition, scoring plans, and persisted observations.

Observation plans preserve the existing factorized pixel-scoring and identity
contracts. Physical detector definitions belong to ``pyrite.detectors``;
scientific result values belong to ``pyrite.results``. Imports stay lazy so
measurement specifications do not initialize transport or storage engines.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.instrument.acquisition import AcquisitionBatch as AcquisitionBatch
    from pyrite.instrument.model import PixelScorer as PixelScorer
    from pyrite.instrument.observation import (
        Acquisition as Acquisition,
    )
    from pyrite.instrument.observation import (
        NativeAcquisition as NativeAcquisition,
    )
    from pyrite.instrument.observation import (
        ResolvedObservation as ResolvedObservation,
    )

    from .inventory import (
        ObservationEntry as ObservationEntry,
    )
    from .inventory import (
        ObservationInventory as ObservationInventory,
    )
    from .inventory import (
        observation_inventory as observation_inventory,
    )
    from .plan import (
        ObservationPlan as ObservationPlan,
    )
    from .plan import (
        PixelSampling as PixelSampling,
    )
    from .plan import (
        SweepObservation as SweepObservation,
    )
    from .spec import Observation as Observation
    from .store import (
        ObservationStore as ObservationStore,
    )
    from .store import (
        ObservationStoreError as ObservationStoreError,
    )
    from .store import (
        StoredObservation as StoredObservation,
    )
    from .store import (
        default_observation_root as default_observation_root,
    )
    from .store import (
        observation_from_result as observation_from_result,
    )

# Existing root aliases now also have a canonical observations import.
_EXPORTS = {
    "Acquisition": ("pyrite.instrument", "Acquisition"),
    "NativeAcquisition": ("pyrite.instrument", "NativeAcquisition"),
    "ResolvedObservation": ("pyrite.instrument", "ResolvedObservation"),
}

# Preexisting observations package exports retain their import paths and identity.
_EXISTING_EXPORTS = {
    # Advanced-stable historical imports remain aliases, not withdrawn APIs.
    "AcquisitionBatch": ("pyrite.instrument", "AcquisitionBatch"),
    "PixelScorer": ("pyrite.instrument", "PixelScorer"),
    "Observation": ("pyrite.observations.spec", "Observation"),
    "ObservationEntry": ("pyrite.observations.inventory", "ObservationEntry"),
    "ObservationInventory": ("pyrite.observations.inventory", "ObservationInventory"),
    "ObservationPlan": ("pyrite.observations.plan", "ObservationPlan"),
    "ObservationStore": ("pyrite.observations.store", "ObservationStore"),
    "ObservationStoreError": ("pyrite.observations.store", "ObservationStoreError"),
    "StoredObservation": ("pyrite.observations.store", "StoredObservation"),
    "default_observation_root": ("pyrite.observations.store", "default_observation_root"),
    "PixelSampling": ("pyrite.observations.plan", "PixelSampling"),
    "SweepObservation": ("pyrite.observations.plan", "SweepObservation"),
    "observation_from_result": ("pyrite.observations.store", "observation_from_result"),
    "observation_inventory": ("pyrite.observations.inventory", "observation_inventory"),
}

__all__ = [
    "Acquisition",
    "AcquisitionBatch",
    "NativeAcquisition",
    "Observation",
    "ObservationEntry",
    "ObservationInventory",
    "ObservationPlan",
    "ObservationStore",
    "ObservationStoreError",
    "PixelSampling",
    "PixelScorer",
    "ResolvedObservation",
    "StoredObservation",
    "SweepObservation",
    "default_observation_root",
    "observation_from_result",
    "observation_inventory",
]


def __getattr__(name: str) -> Any:
    """Resolve acquisition and observation APIs without eager engine imports."""
    binding = _EXPORTS.get(name) or _EXISTING_EXPORTS.get(name)
    if binding is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = binding
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
