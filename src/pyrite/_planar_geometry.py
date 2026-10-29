"""Planar pose, pixel-grid, and point-source ray geometry.

A leaf below :mod:`pyrite.montecarlo`, :mod:`pyrite.detectors`, and
:mod:`pyrite.instrument`: Monte Carlo tiles a detector face and the detector
responses take a face's solid angle without importing the instrument model.
The public names stay importable from :mod:`pyrite.instrument` and
:mod:`pyrite.instrument.geometry`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from ._spec_defaults import UNSET, _Unset
from ._units import _is_pint_quantity, _quantity_spelling, _scalar_quantity, normalize_unit

if TYPE_CHECKING:
    from pint import Quantity

_AXIS_ATOL = 1.0e-10


def _finite_vector(name: str, value: object, size: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a {size}-component real vector") from exc
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _positive_pair(name: str, value: object) -> tuple[float, float]:
    numbers = _finite_vector(name, value, 2)
    if np.any(numbers <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return float(numbers[0]), float(numbers[1])


def _length_vector(value: object, *, field: str, size: int) -> tuple[float, ...]:
    """Own an explicit ``size``-component length quantity as numeric mm floats."""
    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with length units")
    original = np.asarray(cast(Any, value).magnitude)
    if original.shape != (size,) or original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real {size}-component length quantity")
    magnitude = np.asarray(normalize_unit(value, "mm", field=field, expected="length"))
    with np.errstate(over="ignore", under="ignore"):
        stored = magnitude.astype(float)
    if not np.all(np.isfinite(stored)) or np.any((magnitude != 0) & (stored == 0)):
        raise ValueError(f"{field} must be finite and representable in numeric mm storage")
    return tuple(float(x) for x in stored)


def _angle_deg(value: object, *, field: str) -> float:
    return _scalar_quantity(
        value, unit="degree", field=field, expected="angle", kind="angle", signed=True
    )


def _mm_quantity(value: object) -> Quantity:
    from .units import Q_

    return Q_(np.array(value, dtype=float), "mm")


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _pose_center_mm(value: object) -> tuple[float, ...]:
    return _length_vector(value, field="PlanarPose.center", size=3)


@dataclass(frozen=True, init=False)
class PlanarPose:
    """Pose of a bounded plane in the target-centred lab frame.

    ``center_mm`` is measured from the target reference emission point. Local
    ``+z`` is ``normal`` and points downstream, from the source side toward the
    detector side. Local ``+x`` is ``x_axis`` and local ``+y`` is
    ``normal cross x_axis``. The source must therefore lie in a plane object's
    negative-local-z half-space; scene validation enforces that relationship.

    Parameters
    ----------
    center_mm
        Plane-centre ``(x, y, z)`` coordinates in mm from the target reference.
    normal
        Local ``+z`` direction. It is normalized during construction.
    x_axis
        Local ``+x`` direction, orthogonal to ``normal``. It is normalized
        during construction.
    center
        Keyword-only three-component Pint length quantity, mutually exclusive
        with ``center_mm``. Read-back returns a fresh quantity in mm; use
        :meth:`with_center` for quantity-aware replacement. Directions are
        dimensionless unit vectors, not quantities.
    """

    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __init__(
        self,
        center_mm: tuple[float, float, float] | _Unset = UNSET,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        *,
        center: object = UNSET,
    ) -> None:
        center_mm = _quantity_spelling(
            "PlanarPose", "center_mm", center_mm, "center", center, _pose_center_mm
        )
        if center_mm is UNSET:
            raise TypeError("PlanarPose requires center_mm or center")
        object.__setattr__(self, "center_mm", center_mm)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "x_axis", x_axis)
        self.__post_init__()

    @property
    def center(self) -> Quantity:
        """Plane centre as a fresh three-component quantity in mm."""
        return _mm_quantity(self.center_mm)

    def with_center(self, center: object) -> Self:
        """Return a pose with a validated centre quantity and unchanged axes."""
        return replace(self, center_mm=_pose_center_mm(center))

    def __post_init__(self) -> None:
        center = _finite_vector(
            "PlanarPose.center_mm",
            normalize_unit(self.center_mm, "mm", field="PlanarPose.center_mm", expected="length"),
            3,
        )
        normal = _finite_vector("PlanarPose.normal", self.normal, 3)
        x_axis = _finite_vector("PlanarPose.x_axis", self.x_axis, 3)
        normal_norm = float(np.linalg.norm(normal))
        x_norm = float(np.linalg.norm(x_axis))
        if normal_norm == 0.0:
            raise ValueError("PlanarPose.normal must be nonzero")
        if x_norm == 0.0:
            raise ValueError("PlanarPose.x_axis must be nonzero")
        normal /= normal_norm
        x_axis /= x_norm
        if abs(float(normal @ x_axis)) > _AXIS_ATOL:
            raise ValueError("PlanarPose.normal and x_axis must be orthogonal")
        object.__setattr__(self, "center_mm", tuple(float(x) for x in center))
        object.__setattr__(self, "normal", tuple(float(x) for x in normal))
        object.__setattr__(self, "x_axis", tuple(float(x) for x in x_axis))

    @property
    def y_axis(self) -> tuple[float, float, float]:
        """Return the canonical local ``+y = +z cross +x`` axis."""
        axis = np.cross(np.asarray(self.normal), np.asarray(self.x_axis))
        return float(axis[0]), float(axis[1]), float(axis[2])

    @classmethod
    def from_observation(
        cls,
        distance_mm: float | _Unset = UNSET,
        polar_deg: float | _Unset = UNSET,
        azimuth_deg: float | _Unset = UNSET,
        roll_deg: float | _Unset = UNSET,
        offset_mm: tuple[float, float] | _Unset = UNSET,
        *,
        distance: object = UNSET,
        polar: object = UNSET,
        azimuth: object = UNSET,
        roll: object = UNSET,
        offset: object = UNSET,
    ) -> PlanarPose:
        """Place a plane at a spherical observation direction.

        Local ``+z`` points from the source toward the unshifted plane centre;
        a detector's source-facing sensitive face is consequently its local
        ``-z`` face. At zero roll, local ``+x`` increases polar angle and local
        ``+y`` increases azimuth. Positive roll rotates ``+x`` toward ``+y``
        about local ``+z``.

        Parameters
        ----------
        distance_mm
            Positive source-to-plane distance in mm before transverse offset.
        polar_deg, azimuth_deg
            Spherical observation direction in degrees.
        roll_deg
            Clockwise local-axis rotation in degrees about ``+z``.
        offset_mm
            Local ``(x, y)`` centre offset in mm.
        distance, polar, azimuth, roll, offset
            Keyword-only Pint quantity spellings (positive scalar length,
            scalar plane angles, two-component length), each mutually
            exclusive with its numeric counterpart.

        Returns
        -------
        PlanarPose
            Normalized pose in the target-centred lab frame.
        """
        owner = "PlanarPose.from_observation"

        def resolve(numeric_name, numeric, semantic_name, semantic, convert, default):
            value = _quantity_spelling(
                owner, numeric_name, numeric, semantic_name, semantic, convert
            )
            return default if value is UNSET else value

        def angle(name):
            return lambda q: _angle_deg(q, field=f"{owner}.{name}")

        distance_mm = resolve(
            "distance_mm",
            distance_mm,
            "distance",
            distance,
            lambda q: _scalar_quantity(q, unit="mm", field=f"{owner}.distance", expected="length"),
            UNSET,
        )
        if distance_mm is UNSET or (polar_deg is UNSET and polar is UNSET):
            raise TypeError(f"{owner} requires a distance and a polar angle")
        polar_deg = resolve("polar_deg", polar_deg, "polar", polar, angle("polar"), UNSET)
        azimuth_deg = resolve("azimuth_deg", azimuth_deg, "azimuth", azimuth, angle("azimuth"), 0.0)
        roll_deg = resolve("roll_deg", roll_deg, "roll", roll, angle("roll"), 0.0)
        offset_mm = resolve(
            "offset_mm",
            offset_mm,
            "offset",
            offset,
            lambda q: _length_vector(q, field=f"{owner}.offset", size=2),
            (0.0, 0.0),
        )
        distance = _positive_real(
            "distance_mm",
            normalize_unit(distance_mm, "mm", field="distance_mm", expected="length"),
        )
        polar_deg = normalize_unit(
            polar_deg, "degree", field="polar_deg", expected="angle", kind="angle"
        )
        azimuth_deg = normalize_unit(
            azimuth_deg, "degree", field="azimuth_deg", expected="angle", kind="angle"
        )
        roll_deg = normalize_unit(
            roll_deg, "degree", field="roll_deg", expected="angle", kind="angle"
        )
        offset_mm = normalize_unit(offset_mm, "mm", field="offset_mm", expected="length")
        for name, value in (
            ("polar_deg", polar_deg),
            ("azimuth_deg", azimuth_deg),
            ("roll_deg", roll_deg),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(polar_deg) <= 180.0:
            raise ValueError("polar_deg must be between 0 and 180 degrees")
        offset = _finite_vector("offset_mm", offset_mm, 2)
        polar = math.radians(float(polar_deg))
        azimuth = math.radians(float(azimuth_deg))
        roll = math.radians(float(roll_deg))
        normal = np.array(
            [
                math.sin(polar) * math.cos(azimuth),
                math.sin(polar) * math.sin(azimuth),
                math.cos(polar),
            ]
        )
        polar_axis = np.array(
            [
                math.cos(polar) * math.cos(azimuth),
                math.cos(polar) * math.sin(azimuth),
                -math.sin(polar),
            ]
        )
        azimuth_axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        x_axis = math.cos(roll) * polar_axis + math.sin(roll) * azimuth_axis
        y_axis = np.cross(normal, x_axis)
        center = distance * normal + offset[0] * x_axis + offset[1] * y_axis
        return cls(tuple(center), tuple(normal), tuple(x_axis))


def _positive_length_pair(value: object, *, field: str) -> tuple[float, ...]:
    """Own an explicit two-component positive length quantity as numeric mm."""
    pair = _length_vector(value, field=field, size=2)
    if any(x <= 0.0 for x in pair):
        raise ValueError(f"{field} values must be positive")
    return pair


def _pitch_mm(value: object) -> tuple[float, ...]:
    return _positive_length_pair(value, field="PixelGrid.pitch")


@dataclass(frozen=True, init=False)
class PixelGrid:
    """Centred rectangular physical-pixel grid.

    Shape is ``(ny, nx)`` and pitch is ``(pitch_y_mm, pitch_x_mm)``. Array
    indices are ``[row_y, column_x]``; index zero lies on the negative local
    y/x side.

    Parameters
    ----------
    shape
        Positive ``(ny, nx)`` pixel counts.
    pitch_mm
        Positive ``(pitch_y, pitch_x)`` values in mm.
    pitch
        Keyword-only two-component ``(pitch_y, pitch_x)`` Pint length
        quantity, mutually exclusive with ``pitch_mm``. Read-back returns a
        fresh quantity in mm; use :meth:`with_pitch` for replacement.
    """

    shape: tuple[int, int]
    pitch_mm: tuple[float, float]

    def __init__(
        self,
        shape: tuple[int, int],
        pitch_mm: tuple[float, float] | _Unset = UNSET,
        *,
        pitch: object = UNSET,
    ) -> None:
        pitch_mm = _quantity_spelling("PixelGrid", "pitch_mm", pitch_mm, "pitch", pitch, _pitch_mm)
        if pitch_mm is UNSET:
            raise TypeError("PixelGrid requires pitch_mm or pitch")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "pitch_mm", pitch_mm)
        self.__post_init__()

    @property
    def pitch(self) -> Quantity:
        """``(pitch_y, pitch_x)`` as a fresh quantity in mm."""
        return _mm_quantity(self.pitch_mm)

    @property
    def active_size(self) -> Quantity:
        """Full active ``(width_x, height_y)`` as a fresh quantity in mm."""
        return _mm_quantity(self.active_size_mm)

    def with_pitch(self, pitch: object) -> Self:
        """Return a grid with a validated pitch quantity and unchanged shape."""
        return replace(self, pitch_mm=_pitch_mm(pitch))

    def __post_init__(self) -> None:
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise TypeError("PixelGrid.shape must be a two-integer tuple") from exc
        if len(shape) != 2 or any(
            isinstance(x, bool) or not isinstance(x, Integral) for x in shape
        ):
            raise TypeError("PixelGrid.shape must contain exactly two integers")
        if any(int(x) <= 0 for x in shape):
            raise ValueError("PixelGrid.shape values must be positive")
        object.__setattr__(self, "shape", (int(shape[0]), int(shape[1])))
        object.__setattr__(
            self,
            "pitch_mm",
            _positive_pair(
                "PixelGrid.pitch_mm",
                normalize_unit(self.pitch_mm, "mm", field="PixelGrid.pitch_mm", expected="length"),
            ),
        )

    @property
    def active_size_mm(self) -> tuple[float, float]:
        """Return full active ``(width_x_mm, height_y_mm)``."""
        ny, nx = self.shape
        pitch_y, pitch_x = self.pitch_mm
        return nx * pitch_x, ny * pitch_y

    @classmethod
    def timepix3_chip(cls) -> PixelGrid:
        """One 256 by 256 Timepix3 ASIC with 55 micrometre pitch."""
        return cls(shape=(256, 256), pitch_mm=(0.055, 0.055))


def _readonly_float_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    array.setflags(write=False)
    return array


def solid_angle_sr(width_mm: float, height_mm: float, distance_mm: float) -> float:
    """Return the exact on-axis solid angle [sr] of a rectangular face.

    The source lies on the rectangle normal through its centre. The result
    tends to ``width * height / distance²`` in the far field.
    """
    a, b, d = 0.5 * width_mm, 0.5 * height_mm, float(distance_mm)
    return float(4.0 * np.arctan(a * b / (d * np.sqrt(a * a + b * b + d * d))))


def _triangle_solid_angle_sr(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the unsigned solid angle [sr] of a source-to-corner triangle."""
    denominator = (
        np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
        + np.dot(a, b) * np.linalg.norm(c)
        + np.dot(b, c) * np.linalg.norm(a)
        + np.dot(c, a) * np.linalg.norm(b)
    )
    return float(2.0 * np.arctan2(abs(np.dot(a, np.cross(b, c))), denominator))


def _rectangle_solid_angle_sr(
    center_mm: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    width_mm: float,
    height_mm: float,
) -> float:
    """Return the exact point-source solid angle [sr] of a planar rectangle."""
    distance = float(np.linalg.norm(center_mm))
    normal = np.cross(x_axis, y_axis)
    if np.isclose(abs(np.dot(center_mm / distance, normal)), 1.0, rtol=0.0, atol=1.0e-14):
        return solid_angle_sr(width_mm, height_mm, distance)

    half_width, half_height = 0.5 * width_mm, 0.5 * height_mm
    corners = (
        center_mm - half_width * x_axis - half_height * y_axis,
        center_mm + half_width * x_axis - half_height * y_axis,
        center_mm + half_width * x_axis + half_height * y_axis,
        center_mm - half_width * x_axis + half_height * y_axis,
    )
    return _triangle_solid_angle_sr(corners[0], corners[1], corners[2]) + _triangle_solid_angle_sr(
        corners[0], corners[2], corners[3]
    )


@dataclass(frozen=True, eq=False)
class PixelRays:
    """Point-source rays and differential acceptance for a detector grid."""

    centers_mm: np.ndarray
    directions_lab: np.ndarray
    distance_mm: np.ndarray
    solid_angle_sr: np.ndarray

    def __post_init__(self) -> None:
        centers = _readonly_float_array(self.centers_mm)
        directions = _readonly_float_array(self.directions_lab)
        distance = _readonly_float_array(self.distance_mm)
        solid_angle = _readonly_float_array(self.solid_angle_sr)
        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError("PixelRays.centers_mm must have shape (ny, nx, 3)")
        if directions.shape != centers.shape:
            raise ValueError("PixelRays.directions_lab must match centers_mm")
        if distance.shape != centers.shape[:2] or solid_angle.shape != distance.shape:
            raise ValueError("PixelRays scalar fields must have shape (ny, nx)")
        if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=0.0, atol=1.0e-9):
            raise ValueError("PixelRays.directions_lab must be unit vectors")
        object.__setattr__(self, "centers_mm", centers)
        object.__setattr__(self, "directions_lab", directions)
        object.__setattr__(self, "distance_mm", distance)
        object.__setattr__(self, "solid_angle_sr", solid_angle)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the physical/scoring grid shape ``(ny, nx)``."""
        return self.distance_mm.shape  # type: ignore[return-value]


def planar_rays(
    pose: PlanarPose,
    *,
    pixels: PixelGrid | None = None,
    size_mm: tuple[float, float] | None = None,
) -> PixelRays:
    """Construct lab-frame source-to-pixel-centre rays for a posed plane.

    Pixels are point samples with their full pitch area. An unpixelated
    plane of ``size_mm = (width_x, height_y)`` instead uses its exact
    finite-rectangle solid angle.

    Validation: positioned-filter-attenuation
    """
    center = np.asarray(pose.center_mm)
    x_axis = np.asarray(pose.x_axis)
    y_axis = np.asarray(pose.y_axis)
    normal = np.asarray(pose.normal)
    if pixels is None:
        if size_mm is None:
            raise ValueError("an unpixelated plane needs size_mm")
        x_offsets = np.array([0.0])
        y_offsets = np.array([0.0])
        pixel_area_mm2 = size_mm[0] * size_mm[1]
    else:
        ny, nx = pixels.shape
        pitch_y, pitch_x = pixels.pitch_mm
        x_offsets = (np.arange(nx) - (nx - 1) / 2.0) * pitch_x
        y_offsets = (np.arange(ny) - (ny - 1) / 2.0) * pitch_y
        pixel_area_mm2 = pitch_x * pitch_y
    centers = (
        center[None, None, :]
        + y_offsets[:, None, None] * y_axis[None, None, :]
        + x_offsets[None, :, None] * x_axis[None, None, :]
    )
    distance = np.linalg.norm(centers, axis=-1)
    directions = centers / distance[..., None]
    obliquity = directions @ normal
    if np.any(obliquity <= 0.0):
        raise ValueError("detector pixels must see the source through their local -z face")
    if pixels is None:
        assert size_mm is not None
        solid_angle = np.array([[_rectangle_solid_angle_sr(center, x_axis, y_axis, *size_mm)]])
    else:
        solid_angle = pixel_area_mm2 * obliquity / distance**2
    return PixelRays(centers, directions, distance, solid_angle)
