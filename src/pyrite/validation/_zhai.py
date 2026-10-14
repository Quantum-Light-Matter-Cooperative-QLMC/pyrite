"""Canonical detector and cache metadata for maintained Zhai validations."""

from dataclasses import asdict
from math import radians
from typing import Any

from ..detectors import AngularAcceptance, Detector

ZHAI_CACHE_SCHEMA = 4
ZHAI_CACHE_FORMAT = "cxr.zhai-cache.v4"
ZHAI_DETECTOR = Detector(
    observation_angle_deg=119.0,
    acceptance=AngularAcceptance(polar_deg=16.6, solid_angle_sr=0.066),
)


def detector_metadata(detector: Detector = ZHAI_DETECTOR) -> dict[str, Any]:
    """Return portable detector provenance plus resolved historical case fields."""
    return {
        "spec": {
            # Keep the cache-schema-4 vocabulary after detector-grid removal.
            "energy_bins": {"cxr": None, "cxr_by_energy": None, "brem": None},
            "_energy_bins_explicit": False,
            **{
                key: value
                for key, value in asdict(detector).items()
                if key not in {"acceptance", "azimuthal_acceptance_deg", "_legacy_acceptance"}
            },
        },
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
