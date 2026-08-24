"""Bounded downstream photon geometry and spatial-scoring requests.

These objects describe physical planes after photon emission. They are not
target geometry and never enter the electron-transport navigator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from ..detectors.spec import Detector, DetectorResponse, EnergyBins
from ..materials import CATALOG, MediumSpec

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


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
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
    """

    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        center = _finite_vector("PlanarPose.center_mm", self.center_mm, 3)
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
        distance_mm: float,
        polar_deg: float,
        azimuth_deg: float = 0.0,
        roll_deg: float = 0.0,
        offset_mm: tuple[float, float] = (0.0, 0.0),
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

        Returns
        -------
        PlanarPose
            Normalized pose in the target-centred lab frame.
        """
        distance = _positive_real("distance_mm", distance_mm)
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


@dataclass(frozen=True)
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
    """

    shape: tuple[int, int]
    pitch_mm: tuple[float, float]

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
        object.__setattr__(self, "pitch_mm", _positive_pair("PixelGrid.pitch_mm", self.pitch_mm))

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


@dataclass(frozen=True)
class FilterPlate:
    """Represent a finite rectangular attenuating plate.

    Parameters
    ----------
    material
        Catalog crystal/media key or an explicit immutable medium composition.
    thickness_mm
        Positive full thickness along the pose normal in mm.
    size_mm
        Positive full ``(width_x, height_y)`` in mm.
    pose
        Plate-centre position and local axes.
    name
        Optional non-empty identifier, unique among filters in one scene.
    """

    material: str | MediumSpec
    thickness_mm: float
    size_mm: tuple[float, float]
    pose: PlanarPose
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.material, str):
            known = CATALOG.media.keys() | CATALOG.crystals.keys()
            if self.material not in known:
                raise ValueError(f"unknown FilterPlate material {self.material!r}")
        elif isinstance(self.material, MediumSpec):
            if not self.material.composition:
                raise ValueError("FilterPlate MediumSpec composition must not be empty")
            for element, density in self.material.composition:
                if not element or not math.isfinite(float(density)) or float(density) <= 0.0:
                    raise ValueError(
                        "FilterPlate MediumSpec composition must have positive densities"
                    )
        else:
            raise TypeError("FilterPlate.material must be a catalog key or MediumSpec")
        object.__setattr__(
            self, "thickness_mm", _positive_real("FilterPlate.thickness_mm", self.thickness_mm)
        )
        object.__setattr__(self, "size_mm", _positive_pair("FilterPlate.size_mm", self.size_mm))
        if not isinstance(self.pose, PlanarPose):
            raise TypeError("FilterPlate.pose must be a PlanarPose")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("FilterPlate.name must be a non-empty string or None")

    def corners_mm(self) -> np.ndarray:
        """Return the eight finite-box corners in lab coordinates.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(8, 3)`` and units mm.
        """
        center = np.asarray(self.pose.center_mm)
        x_axis = np.asarray(self.pose.x_axis)
        y_axis = np.asarray(self.pose.y_axis)
        normal = np.asarray(self.pose.normal)
        width, height = self.size_mm
        return np.asarray(
            [
                center
                + sx * width / 2.0 * x_axis
                + sy * height / 2.0 * y_axis
                + sz * self.thickness_mm / 2.0 * normal
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )


@dataclass(frozen=True)
class PlanarDetector:
    """Describe physical planar detector geometry and optional response.

    Parameters
    ----------
    pose
        Detector-centre position and local axes; local ``+z`` points downstream.
    size_mm
        Full ``(width_x, height_y)`` in mm. Required without ``pixels`` and,
        when both are given, must match the pixel-grid active size.
    pixels
        Optional physical pixel grid used for spatial scoring.
    energy_bins
        Intrinsic line and background photon-energy grids.
    response
        Optional read-time detector response implementing ``score``.

    Notes
    -----
    Geometry changes source observation and accepted solid angle. ``response``
    changes only read-time scoring.
    """

    pose: PlanarPose
    size_mm: tuple[float, float] | None = None
    pixels: PixelGrid | None = None
    energy_bins: EnergyBins = field(default_factory=EnergyBins)
    response: DetectorResponse | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose, PlanarPose):
            raise TypeError("PlanarDetector.pose must be a PlanarPose")
        size = (
            None if self.size_mm is None else _positive_pair("PlanarDetector.size_mm", self.size_mm)
        )
        if self.pixels is not None and not isinstance(self.pixels, PixelGrid):
            raise TypeError("PlanarDetector.pixels must be a PixelGrid or None")
        if self.pixels is None and size is None:
            raise ValueError("PlanarDetector requires size_mm or pixels")
        if self.pixels is not None:
            pixel_size = self.pixels.active_size_mm
            if size is not None and not np.allclose(size, pixel_size, rtol=0.0, atol=1.0e-12):
                raise ValueError("PlanarDetector.size_mm must equal the PixelGrid active size")
            size = pixel_size
        assert size is not None
        object.__setattr__(self, "size_mm", size)
        if not isinstance(self.energy_bins, EnergyBins):
            raise TypeError("PlanarDetector.energy_bins must be an EnergyBins")
        if self.response is not None and not hasattr(self.response, "score"):
            raise TypeError("PlanarDetector.response must implement score()")
        center = np.asarray(self.pose.center_mm)
        normal = np.asarray(self.pose.normal)
        if float(center @ normal) <= 0.0:
            raise ValueError(
                "PlanarDetector local +z must point downstream, with the source in its negative-z half-space"
            )

    @classmethod
    def timepix3_chip(
        cls,
        pose: PlanarPose,
        *,
        energy_bins: EnergyBins | None = None,
        response: DetectorResponse | None = None,
    ) -> PlanarDetector:
        """Construct one physical Timepix3 ASIC without implying a response."""
        return cls(
            pose=pose,
            pixels=PixelGrid.timepix3_chip(),
            energy_bins=EnergyBins() if energy_bins is None else energy_bins,
            response=response,
        )

    def score(
        self,
        energy_eV: np.ndarray,
        intrinsic_density: np.ndarray,
        *,
        fwhm_eV: float | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Apply only the read-time response; geometry is handled upstream."""
        if self.response is None:
            return np.asarray(intrinsic_density) * scale
        return self.response.score(
            np.asarray(energy_eV),
            np.asarray(intrinsic_density),
            fwhm_eV=fwhm_eV,
            scale=scale,
        )

    def scalar_detector(self) -> Detector:
        """Project physical geometry onto unchanged legacy source-case fields."""
        center = np.asarray(self.pose.center_mm)
        distance = float(np.linalg.norm(center))
        direction = center / distance
        polar_deg = math.degrees(math.acos(float(np.clip(direction[2], -1.0, 1.0))))
        size = self.size_mm
        assert size is not None
        width, height = size
        x_axis = np.asarray(self.pose.x_axis)
        y_axis = np.asarray(self.pose.y_axis)
        corners = np.asarray(
            [
                center + sx * width / 2.0 * x_axis + sy * height / 2.0 * y_axis
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )
        corner_dirs = corners / np.linalg.norm(corners, axis=1)[:, None]
        polar = np.arccos(np.clip(corner_dirs[:, 2], -1.0, 1.0))
        polar_span_deg = math.degrees(float(np.max(polar) - np.min(polar)))
        area = width * height
        solid_angle = area * abs(float(direction @ np.asarray(self.pose.normal))) / distance**2
        return Detector(
            observation_angle_deg=polar_deg,
            polar_acceptance_deg=max(polar_span_deg, np.finfo(float).eps),
            solid_angle_sr=solid_angle,
            energy_bins=self.energy_bins,
            response=self.response,
        )


@dataclass(frozen=True)
class PixelScorer:
    """Request factorized spatial output on a planar detector grid.

    Parameters
    ----------
    angular_shape
        Positive ``(n_polar, n_azimuth)`` count of source-direction tiles. Each
        count must not exceed the corresponding detector pixel count.
    """

    angular_shape: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        try:
            shape = tuple(self.angular_shape)
        except TypeError as exc:
            raise TypeError("PixelScorer.angular_shape must be a two-integer tuple") from exc
        if len(shape) != 2 or any(
            isinstance(x, bool) or not isinstance(x, Integral) for x in shape
        ):
            raise TypeError("PixelScorer.angular_shape must contain exactly two integers")
        if any(int(x) <= 0 for x in shape):
            raise ValueError("PixelScorer.angular_shape values must be positive")
        object.__setattr__(self, "angular_shape", (int(shape[0]), int(shape[1])))


def validate_downstream_scene(filters: tuple[FilterPlate, ...], detector: PlanarDetector) -> None:
    """Validate finite downstream volumes without requiring central-ray hits."""
    detector_center = np.asarray(detector.pose.center_mm)
    detector_normal = np.asarray(detector.pose.normal)
    detector_projection = float(detector_center @ detector_normal)
    tolerance = 1.0e-12 * max(1.0, abs(detector_projection))
    names = [plate.name for plate in filters if plate.name is not None]
    if len(names) != len(set(names)):
        raise ValueError("FilterPlate names must be unique within a Scene")
    for plate in filters:
        plate_center = np.asarray(plate.pose.center_mm)
        plate_normal = np.asarray(plate.pose.normal)
        if float(plate_center @ plate_normal) <= tolerance:
            raise ValueError(
                "FilterPlate local +z must point downstream, with the source in its negative-z half-space"
            )
        projections = plate.corners_mm() @ detector_normal
        if float(np.min(projections)) <= tolerance:
            raise ValueError("the full FilterPlate volume must lie downstream of the source")
        if float(np.max(projections)) >= detector_projection - tolerance:
            raise ValueError("the full FilterPlate volume must lie before the detector plane")


__all__ = [
    "FilterPlate",
    "PixelGrid",
    "PixelScorer",
    "PlanarDetector",
    "PlanarPose",
]
