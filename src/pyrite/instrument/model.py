"""Bounded downstream photon geometry and spatial-scoring requests.

These objects describe physical planes after photon emission. They are not
target geometry and never enter the electron-transport navigator.
"""

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np

from .._planar_geometry import PixelGrid, PlanarPose, _positive_pair, _positive_real
from ..detectors.spec import Detector, DetectorResponse, EnergyBins
from ..materials import CATALOG, MediumSpec


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
        True-energy grids for the line and background source spectra.
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


#: Pixel reconstruction modes accepted by :class:`PixelScorer`.
RECONSTRUCTIONS = ("nearest_tile", "bilinear_tile")


@dataclass(frozen=True)
class PixelScorer:
    """Request factorized spatial output on a planar detector grid.

    Parameters
    ----------
    angular_shape
        Positive ``(n_polar, n_azimuth)`` count of source-direction tiles. Each
        count must not exceed the corresponding detector pixel count.
    reconstruction
        ``"nearest_tile"`` gives each pixel its own tile's intrinsic spectrum.
        ``"bilinear_tile"`` blends the neighbouring tiles' spectra with convex
        weights (:func:`~pyrite.instrument.geometry.angular_tile_weights`).
        Blending mixes spectra at fixed energy, so a line whose energy shifts
        with angle broadens or doubles instead of shifting.
    """

    angular_shape: tuple[int, int] = (1, 1)
    reconstruction: Literal["nearest_tile", "bilinear_tile"] = "nearest_tile"

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
        if self.reconstruction not in RECONSTRUCTIONS:
            raise ValueError(
                "PixelScorer.reconstruction must be one of " + ", ".join(map(repr, RECONSTRUCTIONS))
            )


def _plates_overlap(first: FilterPlate, second: FilterPlate) -> bool:
    """Return whether two plate interiors intersect (face contact is allowed).

    Separating-axis test for two oriented boxes: the 3 + 3 face normals and the
    9 edge-edge cross products. Overlap needs positive projected overlap on
    every axis; touching faces project to zero overlap and so do not count.
    """
    axes_a = (first.pose.x_axis, first.pose.y_axis, first.pose.normal)
    axes_b = (second.pose.x_axis, second.pose.y_axis, second.pose.normal)
    corners_a, corners_b = first.corners_mm(), second.corners_mm()
    candidates = [np.asarray(a, dtype=float) for a in (*axes_a, *axes_b)]
    candidates += [np.cross(a, b) for a in axes_a for b in axes_b]
    scale = max(float(np.ptp(corners_a)), float(np.ptp(corners_b)), 1.0)
    for axis in candidates:
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:  # parallel edges: no new separating direction
            continue
        axis = axis / norm
        proj_a, proj_b = corners_a @ axis, corners_b @ axis
        overlap = min(proj_a.max(), proj_b.max()) - max(proj_a.min(), proj_b.min())
        if overlap <= 1.0e-12 * scale:
            return False
    return True


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
    for i, first in enumerate(filters):
        for second in filters[i + 1 :]:
            if _plates_overlap(first, second):
                raise ValueError(
                    "FilterPlate volumes must not overlap: the optical depth sums per-plate "
                    f"chords, so shared volume would be counted twice ({first.name!r}, {second.name!r})"
                )


__all__ = [
    "FilterPlate",
    "PixelGrid",
    "PixelScorer",
    "PlanarDetector",
    "PlanarPose",
    "RECONSTRUCTIONS",
]
