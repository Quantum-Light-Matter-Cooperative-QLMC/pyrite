"""Portable detector geometry and reserved detector-description metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real


def _number(name: str, value: object, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < minimum or (maximum is not None and number > maximum):
        upper = "" if maximum is None else f" and <= {maximum:g}"
        raise ValueError(f"{name} must be >= {minimum:g}{upper}")
    return number


def _positive_optional(
    name: str, value: object | None, *, allow_zero: bool = False
) -> float | None:
    if value is None:
        return None
    number = _number(name, value, minimum=0.0)
    if not allow_zero and number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _portable_identifier(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string resource identifier")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty resource identifier")
    if (
        identifier.startswith(("/", "~"))
        or "\\" in identifier
        or re.match(r"^[A-Za-z]:", identifier)
        or ".." in identifier.split("/")
    ):
        raise ValueError(
            f"{name} must be a portable registry/resource identifier, not an absolute "
            "or parent-relative path"
        )
    return identifier


@dataclass(frozen=True)
class DetectorSpec:
    """Resolved detector geometry plus inert, portable hardware metadata.

    ``polar_acceptance_deg`` is the detector's full polar span and maps to the
    historical ``dtheta_obs_rad`` case field. ``solid_angle_sr`` maps to the
    historical ``domega_sr`` field. When either is ``None``, case construction
    retains the current Timepix3 geometry fallback.

    The remaining fields are serializable identity metadata only. No response,
    efficiency, pixel, thickness, distance, or threshold behavior changes until
    a named detector adapter explicitly consumes the corresponding field.
    ``qe_curve`` and ``response_model`` are portable registry/resource
    identifiers, never embedded arrays or machine-specific absolute paths.
    """

    observation_angle_deg: float = 90.0
    polar_acceptance_deg: float | None = None
    solid_angle_sr: float | None = None
    response_model: str | None = None
    qe_curve: str | None = None
    pixel_pitch_um: float | None = None
    sensor_thickness_um: float | None = None
    distance_mm: float | None = None
    threshold_eV: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_angle_deg",
            _number(
                "observation_angle_deg",
                self.observation_angle_deg,
                minimum=0.0,
                maximum=180.0,
            ),
        )
        acceptance = _positive_optional("polar_acceptance_deg", self.polar_acceptance_deg)
        if acceptance is not None and acceptance > 180.0:
            raise ValueError("polar_acceptance_deg is a full polar span and must be <= 180 degrees")
        object.__setattr__(self, "polar_acceptance_deg", acceptance)
        solid_angle = _positive_optional("solid_angle_sr", self.solid_angle_sr)
        if solid_angle is not None and solid_angle > 4.0 * math.pi:
            raise ValueError("solid_angle_sr must be <= 4*pi sr")
        object.__setattr__(self, "solid_angle_sr", solid_angle)
        object.__setattr__(
            self, "response_model", _portable_identifier("response_model", self.response_model)
        )
        object.__setattr__(self, "qe_curve", _portable_identifier("qe_curve", self.qe_curve))
        object.__setattr__(
            self, "pixel_pitch_um", _positive_optional("pixel_pitch_um", self.pixel_pitch_um)
        )
        object.__setattr__(
            self,
            "sensor_thickness_um",
            _positive_optional("sensor_thickness_um", self.sensor_thickness_um),
        )
        object.__setattr__(self, "distance_mm", _positive_optional("distance_mm", self.distance_mm))
        object.__setattr__(
            self,
            "threshold_eV",
            _positive_optional("threshold_eV", self.threshold_eV, allow_zero=True),
        )
