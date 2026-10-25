"""Detector models, acceptance, geometry, and sensor-response interfaces.

Physical detector geometry is public here; implementation bindings continue to
live in ``pyrite.instrument`` during the compatibility transition.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .acceptance import AngularAcceptance, SensorAcceptance
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

if TYPE_CHECKING:
    from pyrite._planar_geometry import (
        PixelGrid as PixelGrid,
    )
    from pyrite._planar_geometry import (
        PlanarPose as PlanarPose,
    )
    from pyrite.instrument.model import (
        FilterPlate as FilterPlate,
    )
    from pyrite.instrument.model import (
        PlanarDetector as PlanarDetector,
    )


__all__ = [
    "AngularAcceptance",
    "SensorAcceptance",
    "Detector",
    "DetectorResponse",
    "DetectorSpec",
    "EagleXO",
    "EnergyBins",
    "FilterPlate",
    "IdealPhotonCounter",
    "LegacyEDS",
    "NativeSpectrum",
    "PixelGrid",
    "PlanarDetector",
    "PlanarPose",
    "Timepix3",
    "convolve_detector",
    "detector_efficiency",
    "eaglexo_response",
    "grating",
    "timepix_response",
]

# Governed root-export bindings. Already-local symbols are imported above; the
# physical geometry names are resolved lazily to avoid detector/instrument cycles.
_EXPORTS = {
    "AngularAcceptance": ("pyrite.detectors", "AngularAcceptance"),
    "Detector": ("pyrite.detectors", "Detector"),
    "EnergyBins": ("pyrite.detectors", "EnergyBins"),
    "FilterPlate": ("pyrite.instrument", "FilterPlate"),
    "IdealPhotonCounter": ("pyrite.detectors", "IdealPhotonCounter"),
    "PixelGrid": ("pyrite.instrument", "PixelGrid"),
    "PlanarDetector": ("pyrite.instrument", "PlanarDetector"),
    "PlanarPose": ("pyrite.instrument", "PlanarPose"),
    "SensorAcceptance": ("pyrite.detectors", "SensorAcceptance"),
}

_MODULE_EXPORTS = frozenset({"eaglexo_response", "grating", "timepix_response"})


def __getattr__(name: str) -> Any:
    """Resolve physical detector geometry and response functions lazily."""
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
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
