"""Physical downstream photon geometry."""

from .attenuation import primary_transmission
from .model import FilterPlate, PixelGrid, PixelScorer, PlanarDetector, PlanarPose

__all__ = [
    "FilterPlate",
    "PixelGrid",
    "PixelScorer",
    "PlanarDetector",
    "PlanarPose",
    "primary_transmission",
]
