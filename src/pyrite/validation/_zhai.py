"""Canonical detector and cache metadata for maintained Zhai validations."""

from __future__ import annotations

from dataclasses import asdict
from math import radians
from typing import Any

from ..detectors import Detector

ZHAI_CACHE_SCHEMA = 4
ZHAI_CACHE_FORMAT = "cxr.zhai-cache.v4"
ZHAI_DETECTOR = Detector(
    observation_angle_deg=119.0,
    polar_acceptance_deg=16.6,
    solid_angle_sr=0.066,
)


def detector_metadata(detector: Detector = ZHAI_DETECTOR) -> dict[str, Any]:
    """Return portable detector provenance plus resolved historical case fields."""
    return {
        "spec": asdict(detector),
        "case": {
            "theta_obs_rad": radians(detector.observation_angle_deg),
            "dtheta_obs_rad": (
                None
                if detector.polar_acceptance_deg is None
                else radians(detector.polar_acceptance_deg)
            ),
            "domega_sr": detector.solid_angle_sr,
        },
    }
