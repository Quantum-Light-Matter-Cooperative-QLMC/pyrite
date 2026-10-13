"""Closed scalar-detector acceptance descriptions, independent of response."""

import math
from dataclasses import dataclass
from numbers import Real


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class AngularAcceptance:
    """Spherical patch specified by exactly two full spans or solid angle.

    ``polar_deg`` and ``azimuthal_deg`` are full angular spans in degrees;
    ``solid_angle_sr`` is in steradians. The missing quantity is resolved at
    the detector's observation angle, not at construction.
    """

    polar_deg: float | None = None
    azimuthal_deg: float | None = None
    solid_angle_sr: float | None = None

    def __post_init__(self) -> None:
        if (
            sum(
                value is not None
                for value in (self.polar_deg, self.azimuthal_deg, self.solid_angle_sr)
            )
            != 2
        ):
            raise ValueError(
                "AngularAcceptance requires exactly two of: polar_deg + azimuthal_deg, "
                "polar_deg + solid_angle_sr, or azimuthal_deg + solid_angle_sr"
            )
        for name, maximum in (
            ("polar_deg", 180.0),
            ("azimuthal_deg", 360.0),
            ("solid_angle_sr", 4 * math.pi),
        ):
            value = getattr(self, name)
            if value is not None:
                value = _positive(name, value)
                if value > maximum:
                    raise ValueError(f"{name} must be <= {maximum:g}")
                object.__setattr__(self, name, value)

    def resolve(self, observation_angle_deg: float) -> tuple[float, float, float]:
        """Return full polar/azimuthal spans (deg) and solid angle (sr).

        Source: solid-angle definition dΩ = sin θ dθ dφ, integrated over a
        centred spherical patch. Assumes θ ± Δθ/2 lies in [0, π] and Δφ ≤ 2π.
        Small-patch limit: Ω → Δθ Δφ sin θ (angles in radians).
        Validation: detector-angular-acceptance
        """
        if not math.isfinite(observation_angle_deg) or not 0 <= observation_angle_deg <= 180:
            raise ValueError("observation_angle_deg must be finite and in [0, 180]")
        theta = math.radians(observation_angle_deg)
        max_polar = 2 * min(observation_angle_deg, 180 - observation_angle_deg)
        if max_polar == 0:
            raise ValueError("a positive centred angular patch cannot lie at a polar pole")
        sine = math.sin(theta)
        polar = self.polar_deg
        azimuthal = self.azimuthal_deg
        omega = self.solid_angle_sr
        if polar is None:
            assert omega is not None and azimuthal is not None
            ratio = omega / (2 * math.radians(azimuthal) * sine)
            bound = math.sin(math.radians(max_polar) / 2)
            if ratio > bound and not math.isclose(ratio, bound, rel_tol=1e-14):
                raise ValueError(
                    "solid_angle_sr cannot be reached by a centred polar span at this observation angle"
                )
            polar = math.degrees(2 * math.asin(min(ratio, bound)))
        if polar > max_polar and not math.isclose(polar, max_polar, rel_tol=1e-14):
            raise ValueError("observation_angle_deg ± polar_deg/2 must lie in [0, 180]")
        band = 2 * sine * math.sin(math.radians(polar) / 2)
        if azimuthal is None:
            assert omega is not None
            azimuthal = math.degrees(omega / band)
        if azimuthal > 360 and not math.isclose(azimuthal, 360, rel_tol=1e-14):
            raise ValueError("resolved azimuthal_deg must be <= 360")
        azimuthal = min(azimuthal, 360.0)
        if omega is None:
            omega = math.radians(azimuthal) * band
        return polar, azimuthal, omega


@dataclass(frozen=True)
class SensorAcceptance:
    """Source-facing rectangle at radial ``distance_mm``, all lengths in mm.

    ``height_mm`` runs along the polar direction; ``width_mm`` along the
    local azimuthal tangent. No pixels, pose, or response are implied.
    """

    distance_mm: float
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        for name in ("distance_mm", "width_mm", "height_mm"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))

    def resolve(self) -> tuple[float, float, float]:
        """Return local full polar/azimuthal spans (deg) and exact Ω (sr).

        Source: PNNL-19616 §2.3.2a (centred special case); projected-area integral dΩ = d dx dy/(d²+x²+y²)^(3/2)
        over a centred source-facing rectangle. Angles are local meridional
        spans, not a spherical-patch approximation. Small-sensor limits:
        Ω → wh/d², Δθ → h/d, Δφ → w/d (angles in radians).
        Validation: detector-sensor-acceptance
        """

        # Normalize each opening separately: a common scale can erase both
        # distance and height for an extremely wide, narrow rectangle.
        def opening(length: float) -> float:
            scale = max(length, self.distance_mm)
            return math.degrees(2 * math.atan2(length / scale, 2 * (self.distance_mm / scale)))

        scale = max(self.distance_mm, self.width_mm, self.height_mm)
        radius = math.hypot(
            2 * (self.distance_mm / scale), self.width_mm / scale, self.height_mm / scale
        )
        # Form the arctangent argument in log space, so a small factor and a
        # large factor can cancel without either product under/overflowing.
        log_ratio = (
            math.log(self.width_mm)
            + math.log(self.height_mm)
            - math.log(2)
            - math.log(self.distance_mm)
            - math.log(scale)
            - math.log(radius)
        )
        omega = 4 * math.atan2(math.exp(min(log_ratio, 0)), math.exp(min(-log_ratio, 0)))
        return opening(self.height_mm), opening(self.width_mm), omega
