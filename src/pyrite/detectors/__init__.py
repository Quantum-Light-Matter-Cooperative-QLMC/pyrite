"""Detector and detector-adjacent forward models."""

from importlib import import_module
from types import ModuleType

from .response import convolve_detector, detector_efficiency
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


def __getattr__(name: str) -> ModuleType:
    """Load detector response modules without creating catalog import cycles."""
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module
