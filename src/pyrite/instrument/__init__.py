"""Physical downstream photon geometry."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "REALIZATION_RNG": ".acquisition",
    "AcquisitionBatch": ".acquisition",
    "combine_acquisitions": ".acquisition",
    "electron_count": ".acquisition",
    "score_acquisition": ".acquisition",
    "primary_transmission": ".attenuation",
    "FilterPlate": ".model",
    "PixelGrid": ".model",
    "PixelScorer": ".model",
    "PlanarDetector": ".model",
    "PlanarPose": ".model",
    "Acquisition": ".observation",
    "ObservationIdentity": ".observation",
    "ResolvedObservation": ".observation",
    "observation_identity": ".observation",
}

__all__ = [
    "FilterPlate",
    "Acquisition",
    "AcquisitionBatch",
    "ObservationIdentity",
    "PixelGrid",
    "PixelScorer",
    "PlanarDetector",
    "PlanarPose",
    "ResolvedObservation",
    "REALIZATION_RNG",
    "combine_acquisitions",
    "electron_count",
    "observation_identity",
    "primary_transmission",
    "score_acquisition",
]


def __getattr__(name: str) -> Any:
    """Load scoring operators separately from physical geometry records."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unloaded exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
