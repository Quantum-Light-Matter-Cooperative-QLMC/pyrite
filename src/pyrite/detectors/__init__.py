"""Detector and detector-adjacent forward models."""

from importlib import import_module
from typing import Any

from .spec import (
    Detector,
    DetectorResponse,
    DetectorSpec,
    EagleXO,
    EnergyBins,
    IdealPhotonCounter,
    LegacyEDS,
    NativeSpectrum,
    Timepix3,
)

__all__ = [
    "Detector",
    "DetectorResponse",
    "DetectorSpec",
    "EagleXO",
    "EnergyBins",
    "IdealPhotonCounter",
    "LegacyEDS",
    "NativeSpectrum",
    "Timepix3",
    "convolve_detector",
    "detector_efficiency",
    "eaglexo_response",
    "grating",
    "timepix_response",
]

_MODULE_EXPORTS = frozenset({"eaglexo_response", "grating", "timepix_response"})


def __getattr__(name: str) -> Any:
    """Load detector response modules without creating catalog import cycles."""
    if name in {"convolve_detector", "detector_efficiency"}:
        value = getattr(import_module(".response", __name__), name)
        globals()[name] = value
        return value
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include unloaded response exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
