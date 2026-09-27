"""Persisted, factorized pixel-detector observations.

An observation store lives beside, not inside, the intrinsic checkpoint tree:
``<workspace>/observations/<stem>/``. Large true-spatial factors are immutable
HDF5 objects named by their true-spatial digest; each response/acquisition
choice is a small JSON record named by its observation digest; a mutable index
maps source content keys to observation digests. ``plan`` is the one place
pixel sampling, factor assembly, and layered identity are built for every
producer.
"""

from .plan import ObservationPlan, PixelSampling, SweepObservation
from .store import (
    ObservationStore,
    ObservationStoreError,
    StoredObservation,
    default_observation_root,
    observation_from_result,
)

__all__ = [
    "ObservationPlan",
    "ObservationStore",
    "ObservationStoreError",
    "StoredObservation",
    "default_observation_root",
    "PixelSampling",
    "SweepObservation",
    "observation_from_result",
]
