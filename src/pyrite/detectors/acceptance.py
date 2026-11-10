"""Closed scalar-detector acceptance descriptions, independent of response."""

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING, Any, Self, cast

from .._spec_defaults import UNSET as _UNSET
from .._spec_defaults import _Unset
from .._units import _is_pint_quantity, _positive_length_quantity, normalize_unit

if TYPE_CHECKING:
    from pint import Quantity


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _explicit_angular_quantity(
    value: object, *, field: str, unit: str, kind: str, maximum: float, positive: bool
) -> float:
    """Check a scalar angular Pint input without importing Pint on numeric paths."""
    import numpy as np

    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires an explicit Pint {kind.replace('_', ' ')} quantity")
    original = np.asarray(cast(Any, value).magnitude)
    if original.ndim != 0 or original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar {kind.replace('_', ' ')} quantity")
    converted = normalize_unit(value, unit, field=field, expected=kind.replace("_", " "), kind=kind)
    magnitude = np.asarray(converted)
    if magnitude.ndim != 0 or magnitude.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar {kind.replace('_', ' ')} quantity")
    number = float(magnitude)
    if not np.isfinite(magnitude) or not math.isfinite(number) or (original != 0 and number == 0):
        raise ValueError(f"{field} is not representable as a finite {unit} scalar")
    if number < 0 or (positive and number == 0) or number > maximum:
        lower = "positive" if positive else "nonnegative"
        raise ValueError(f"{field} must be {lower} and <= {maximum:g} {unit}")
    return number


def _opening_deg(length: float, distance: float) -> float:
    """Full opening angle (deg) of a centred ``length`` at radial ``distance``."""
    # Normalize each opening separately: a common scale can erase both
    # distance and height for an extremely wide, narrow rectangle.
    scale = max(length, distance)
    return math.degrees(2 * math.atan2(length / scale, 2 * (distance / scale)))


def _rectangle_solid_angle(distance: float, width: float, height: float) -> float:
    """Exact solid angle (sr) of a centred source-facing ``width`` x ``height`` rectangle."""
    scale = max(distance, width, height)
    radius = math.hypot(2 * (distance / scale), width / scale, height / scale)
    # Form the arctangent argument in log space, so a small factor and a
    # large factor can cancel without either product under/overflowing.
    log_ratio = (
        math.log(width)
        + math.log(height)
        - math.log(2)
        - math.log(distance)
        - math.log(scale)
        - math.log(radius)
    )
    return 4 * math.atan2(math.exp(min(log_ratio, 0)), math.exp(min(-log_ratio, 0)))


@dataclass(frozen=True, init=False)
class AngularAcceptance:
    """Spherical patch specified by exactly two full spans or solid angle.

    ``polar_deg`` and ``azimuthal_deg`` are full angular spans in degrees;
    ``solid_angle_sr`` is in steradians. The missing quantity is resolved at
    the detector's observation angle, not at construction.

    Keyword-only ``polar``, ``azimuthal`` and ``solid_angle`` require explicit
    scalar Pint quantities. Matching numeric/semantic keywords conflict.
    Their properties return the *specified* values (None for the unresolved
    member), not a value calculated for any observation angle. Use ``resolve``
    for numeric degree/sr geometry. Replacements preserve the specified pair;
    switching pairs requires a new acceptance value.
    """

    polar_deg: float | None = None
    azimuthal_deg: float | None = None
    solid_angle_sr: float | None = None

    def __init__(
        self,
        polar_deg: float | None | _Unset = _UNSET,
        azimuthal_deg: float | None | _Unset = _UNSET,
        solid_angle_sr: float | None | _Unset = _UNSET,
        *,
        polar: object = _UNSET,
        azimuthal: object = _UNSET,
        solid_angle: object = _UNSET,
    ) -> None:
        for semantic, suffix, numeric, quantity, unit, kind, maximum in (
            ("polar", "polar_deg", polar_deg, polar, "degree", "angle", 180.0),
            ("azimuthal", "azimuthal_deg", azimuthal_deg, azimuthal, "degree", "angle", 360.0),
            (
                "solid_angle",
                "solid_angle_sr",
                solid_angle_sr,
                solid_angle,
                "steradian",
                "solid_angle",
                4 * math.pi,
            ),
        ):
            if quantity is not _UNSET:
                if numeric is not _UNSET:
                    raise TypeError(f"AngularAcceptance received both {suffix} and {semantic}")
                numeric = _explicit_angular_quantity(
                    quantity,
                    field=f"AngularAcceptance.{semantic}",
                    unit=unit,
                    kind=kind,
                    maximum=maximum,
                    positive=True,
                )
            elif numeric is _UNSET:
                numeric = None
            object.__setattr__(self, suffix, numeric)
        self.__post_init__()

    @property
    def polar(self) -> "Quantity | None":
        """Specified full polar span as a fresh degree quantity, or None."""
        if self.polar_deg is None:
            return None
        from ..units import Q_

        return Q_(self.polar_deg, "degree")

    @property
    def azimuthal(self) -> "Quantity | None":
        """Specified full azimuthal span as a fresh degree quantity, or None."""
        if self.azimuthal_deg is None:
            return None
        from ..units import Q_

        return Q_(self.azimuthal_deg, "degree")

    @property
    def solid_angle(self) -> "Quantity | None":
        """Specified solid angle as a fresh steradian quantity, or None."""
        if self.solid_angle_sr is None:
            return None
        from ..units import Q_

        return Q_(self.solid_angle_sr, "steradian")

    def with_polar(self, polar: object) -> Self:
        """Replace the specified full polar span, keeping the chosen pair."""
        if self.polar_deg is None:
            raise ValueError("polar was not specified; construct a new two-field acceptance")
        return replace(
            self,
            polar_deg=_explicit_angular_quantity(
                polar,
                field="AngularAcceptance.polar",
                unit="degree",
                kind="angle",
                maximum=180.0,
                positive=True,
            ),
        )

    def with_azimuthal(self, azimuthal: object) -> Self:
        """Replace the specified full azimuthal span, keeping the chosen pair."""
        if self.azimuthal_deg is None:
            raise ValueError("azimuthal was not specified; construct a new two-field acceptance")
        return replace(
            self,
            azimuthal_deg=_explicit_angular_quantity(
                azimuthal,
                field="AngularAcceptance.azimuthal",
                unit="degree",
                kind="angle",
                maximum=360.0,
                positive=True,
            ),
        )

    def with_solid_angle(self, solid_angle: object) -> Self:
        """Replace the specified solid angle, keeping the chosen pair."""
        if self.solid_angle_sr is None:
            raise ValueError("solid_angle was not specified; construct a new two-field acceptance")
        return replace(
            self,
            solid_angle_sr=_explicit_angular_quantity(
                solid_angle,
                field="AngularAcceptance.solid_angle",
                unit="steradian",
                kind="solid_angle",
                maximum=4 * math.pi,
                positive=True,
            ),
        )

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
                target = "steradian" if name == "solid_angle_sr" else "degree"
                kind = "solid_angle" if name == "solid_angle_sr" else "angle"
                value = normalize_unit(
                    value,
                    target,
                    field=f"AngularAcceptance.{name}",
                    expected=kind.replace("_", " "),
                    kind=kind,
                )
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


@dataclass(frozen=True, init=False)
class SensorAcceptance:
    """Source-facing rectangle at radial ``distance_mm``, all lengths in mm.

    ``height_mm`` runs along the polar direction; ``width_mm`` along the
    local azimuthal tangent. No pixels, pose, or response are implied.

    Keyword-only ``distance``, ``width`` and ``height`` require positive finite
    scalar Pint lengths. Each conflicts with its corresponding numeric field.
    Their properties return fresh mm quantities; ``with_distance``, ``with_width``
    and ``with_height`` provide quantity-aware replacement. Positional numeric
    arguments, dataclass fields and persistence retain mm. ``dataclasses.replace``
    uses those numeric fields; semantic keywords conflict with the stored fields.
    ``resolve()`` still returns numeric degree spans and steradians.
    """

    distance_mm: float
    width_mm: float
    height_mm: float

    def __init__(
        self,
        distance_mm: float | _Unset = _UNSET,
        width_mm: float | _Unset = _UNSET,
        height_mm: float | _Unset = _UNSET,
        *,
        distance: object = _UNSET,
        width: object = _UNSET,
        height: object = _UNSET,
    ) -> None:
        for name, numeric, quantity in (
            ("distance", distance_mm, distance),
            ("width", width_mm, width),
            ("height", height_mm, height),
        ):
            if quantity is not _UNSET:
                if numeric is not _UNSET:
                    raise TypeError(f"SensorAcceptance received both {name}_mm and {name}")
                numeric = float(
                    _positive_length_quantity(
                        quantity, unit="mm", field=f"SensorAcceptance.{name}", shape=()
                    )
                )
            elif numeric is _UNSET:
                raise TypeError(f"SensorAcceptance requires {name}_mm or {name}")
            object.__setattr__(self, f"{name}_mm", numeric)
        self.__post_init__()

    @property
    def distance(self) -> "Quantity":
        """Radial source-to-sensor distance as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.distance_mm, "mm")

    @property
    def width(self) -> "Quantity":
        """Full local azimuthal width as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.width_mm, "mm")

    @property
    def height(self) -> "Quantity":
        """Full local polar height as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.height_mm, "mm")

    def with_distance(self, distance: object) -> Self:
        """Replace radial distance using an explicit scalar length quantity."""
        return replace(
            self,
            distance_mm=_positive_length_quantity(
                distance, unit="mm", field="SensorAcceptance.distance", shape=()
            ),
        )

    def with_width(self, width: object) -> Self:
        """Replace full azimuthal width using an explicit scalar length quantity."""
        return replace(
            self,
            width_mm=_positive_length_quantity(
                width, unit="mm", field="SensorAcceptance.width", shape=()
            ),
        )

    def with_height(self, height: object) -> Self:
        """Replace full polar height using an explicit scalar length quantity."""
        return replace(
            self,
            height_mm=_positive_length_quantity(
                height, unit="mm", field="SensorAcceptance.height", shape=()
            ),
        )

    def __post_init__(self) -> None:
        for name in ("distance_mm", "width_mm", "height_mm"):
            value = normalize_unit(
                getattr(self, name),
                "mm",
                field=f"SensorAcceptance.{name}",
                expected="length",
            )
            object.__setattr__(self, name, _positive(name, value))

    def resolve(self) -> tuple[float, float, float]:
        """Return local full polar/azimuthal spans (deg) and exact Ω (sr).

        Source: PNNL-19616 §2.3.2a (centred special case); projected-area integral dΩ = d dx dy/(d²+x²+y²)^(3/2)
        over a centred source-facing rectangle. Angles are local meridional
        spans, not a spherical-patch approximation. Small-sensor limits:
        Ω → wh/d², Δθ → h/d, Δφ → w/d (angles in radians).
        Validation: detector-sensor-acceptance
        """
        return (
            _opening_deg(self.height_mm, self.distance_mm),
            _opening_deg(self.width_mm, self.distance_mm),
            _rectangle_solid_angle(self.distance_mm, self.width_mm, self.height_mm),
        )


def _roll_quantity(value: object, *, field: str) -> float:
    """Check a scalar Pint angle of any sign, returning finite degrees."""
    import numpy as np

    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires an explicit Pint angle quantity")
    original = np.asarray(cast(Any, value).magnitude)
    if original.ndim != 0 or original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar angle quantity")
    converted = normalize_unit(value, "degree", field=field, expected="angle", kind="angle")
    magnitude = np.asarray(converted)
    if magnitude.ndim != 0 or magnitude.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar angle quantity")
    number = float(magnitude)
    if not np.isfinite(magnitude) or not math.isfinite(number) or (original != 0 and number == 0):
        raise ValueError(f"{field} is not representable as a finite degree scalar")
    return number


def _normalized_roll_deg(value: object) -> float:
    """Return a finite real roll folded into [0, 180) degrees."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("roll_deg must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("roll_deg must be finite")
    folded = number % 180.0
    # ``-1e-20 % 180`` rounds to 180.0; keep the half-open interval.
    return 0.0 if folded >= 180.0 else folded


def _cos_sin_deg(angle_deg: float) -> tuple[float, float]:
    """Cosine and sine of an angle in [0, 90] degrees, exact at both ends."""
    if angle_deg == 0.0:
        return 1.0, 0.0
    if angle_deg == 90.0:
        return 0.0, 1.0
    angle = math.radians(angle_deg)
    return math.cos(angle), math.sin(angle)


@dataclass(frozen=True, init=False)
class SlitAcceptance:
    """Source-facing rectangular slit at radial ``distance_mm``, lengths in mm.

    ``width_mm`` is the narrow opening and ``length_mm`` the long axis (a
    width larger than the length is allowed). ``roll_deg`` is the angle from
    the ``length`` axis to the local azimuthal tangent, so ``roll_deg=0`` puts
    the narrow width along the polar direction and ``90`` along the azimuthal
    tangent. It is stored folded into [0, 180) degrees; ψ and 180° - ψ are
    mirror-image slits with equal resolved spans. No line curvature, source size, diffraction, pose, or
    response is implied.

    Keyword-only ``distance``, ``width`` and ``length`` require positive finite
    scalar Pint lengths and ``roll`` a real finite scalar Pint angle. Each
    conflicts with its numeric field. Properties return fresh quantities, and
    ``with_distance``, ``with_width``, ``with_length`` and ``with_roll``
    provide quantity-aware replacement. Dataclass fields and persistence retain
    mm and degrees. ``resolve()`` returns numeric degree spans and steradians
    on the same scalar acceptance fields as :class:`SensorAcceptance`.
    """

    distance_mm: float
    width_mm: float
    length_mm: float
    roll_deg: float = 0.0

    def __init__(
        self,
        distance_mm: float | _Unset = _UNSET,
        width_mm: float | _Unset = _UNSET,
        length_mm: float | _Unset = _UNSET,
        roll_deg: float | _Unset = _UNSET,
        *,
        distance: object = _UNSET,
        width: object = _UNSET,
        length: object = _UNSET,
        roll: object = _UNSET,
    ) -> None:
        for name, numeric, quantity in (
            ("distance", distance_mm, distance),
            ("width", width_mm, width),
            ("length", length_mm, length),
        ):
            if quantity is not _UNSET:
                if numeric is not _UNSET:
                    raise TypeError(f"SlitAcceptance received both {name}_mm and {name}")
                numeric = float(
                    _positive_length_quantity(
                        quantity, unit="mm", field=f"SlitAcceptance.{name}", shape=()
                    )
                )
            elif numeric is _UNSET:
                raise TypeError(f"SlitAcceptance requires {name}_mm or {name}")
            object.__setattr__(self, f"{name}_mm", numeric)
        if roll is not _UNSET:
            if roll_deg is not _UNSET:
                raise TypeError("SlitAcceptance received both roll_deg and roll")
            roll_deg = _roll_quantity(roll, field="SlitAcceptance.roll")
        elif roll_deg is _UNSET:
            roll_deg = 0.0
        object.__setattr__(self, "roll_deg", roll_deg)
        self.__post_init__()

    @property
    def distance(self) -> "Quantity":
        """Radial source-to-slit distance as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.distance_mm, "mm")

    @property
    def width(self) -> "Quantity":
        """Narrow slit opening as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.width_mm, "mm")

    @property
    def length(self) -> "Quantity":
        """Long slit axis as a fresh scalar mm quantity."""
        from ..units import Q_

        return Q_(self.length_mm, "mm")

    @property
    def roll(self) -> "Quantity":
        """Normalized roll in [0, 180) as a fresh scalar degree quantity."""
        from ..units import Q_

        return Q_(self.roll_deg, "degree")

    def with_distance(self, distance: object) -> Self:
        """Replace radial distance using an explicit scalar length quantity."""
        return replace(
            self,
            distance_mm=_positive_length_quantity(
                distance, unit="mm", field="SlitAcceptance.distance", shape=()
            ),
        )

    def with_width(self, width: object) -> Self:
        """Replace the narrow opening using an explicit scalar length quantity."""
        return replace(
            self,
            width_mm=_positive_length_quantity(
                width, unit="mm", field="SlitAcceptance.width", shape=()
            ),
        )

    def with_length(self, length: object) -> Self:
        """Replace the long axis using an explicit scalar length quantity."""
        return replace(
            self,
            length_mm=_positive_length_quantity(
                length, unit="mm", field="SlitAcceptance.length", shape=()
            ),
        )

    def with_roll(self, roll: object) -> Self:
        """Replace the roll using an explicit real scalar angle quantity."""
        return replace(self, roll_deg=_roll_quantity(roll, field="SlitAcceptance.roll"))

    def __post_init__(self) -> None:
        for name in ("distance_mm", "width_mm", "length_mm"):
            value = normalize_unit(
                getattr(self, name),
                "mm",
                field=f"SlitAcceptance.{name}",
                expected="length",
            )
            object.__setattr__(self, name, _positive(name, value))
        roll = normalize_unit(
            self.roll_deg,
            "degree",
            field="SlitAcceptance.roll_deg",
            expected="angle",
            kind="angle",
        )
        object.__setattr__(self, "roll_deg", _normalized_roll_deg(roll))

    def resolve(self) -> tuple[float, float, float]:
        """Return local full polar/azimuthal spans (deg) and exact Ω (sr).

        Source: the centred-rectangle projected-area integral of
        :meth:`SensorAcceptance.resolve` (PNNL-19616 §2.3.2a, centred special
        case). Rotating the slit about the line of sight preserves every
        distance, so Ω = 4 arctan[wℓ/(2L√(4L²+w²+ℓ²))] is independent of
        roll. With ψ the angle from the length axis to the azimuthal tangent,
        the rolled rectangle projects onto the polar and azimuthal tangent
        axes with full extents h = w|cos ψ| + ℓ|sin ψ| and
        b = ℓ|cos ψ| + w|sin ψ|; the spans are the opening angles
        2 arctan[h/(2L)] and 2 arctan[b/(2L)] of those extents, as for
        ``SensorAcceptance``. Assumes a point source on the slit normal and
        uses the central direction times Ω downstream. Limits: ψ = 0 equals
        ``SensorAcceptance(height=w, width=ℓ)``, ψ = 90° equals
        ``SensorAcceptance(height=ℓ, width=w)``; for a small slit Ω → wℓ/L²
        and Δθ → h/L, Δφ → b/L (angles in radians).
        Validation: detector-slit-acceptance
        """
        cosine, sine = _cos_sin_deg(min(self.roll_deg, 180.0 - self.roll_deg))
        polar_extent = self.width_mm * cosine + self.length_mm * sine
        azimuthal_extent = self.length_mm * cosine + self.width_mm * sine
        return (
            _opening_deg(polar_extent, self.distance_mm),
            _opening_deg(azimuthal_extent, self.distance_mm),
            _rectangle_solid_angle(self.distance_mm, self.length_mm, self.width_mm),
        )
