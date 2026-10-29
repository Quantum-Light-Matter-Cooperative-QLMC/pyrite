"""Bounded downstream photon geometry and spatial-scoring requests.

These objects describe physical planes after photon emission. They are not
target geometry and never enter the electron-transport navigator.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import InitVar, dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from .._planar_geometry import (
    PixelGrid,
    PlanarPose,
    _mm_quantity,
    _positive_length_pair,
    _positive_pair,
    _positive_real,
)
from .._spec_defaults import UNSET, _Unset
from .._units import _quantity_spelling, _scalar_quantity, normalize_unit
from ..detectors.spec import Detector, DetectorResponse
from ..materials import CATALOG, MediumSpec
from ..materials._composition import has_positive_number_densities, resolve_composition

if TYPE_CHECKING:
    from pint import Quantity

_PLANAR_COMPATIBILITY: dict[int, tuple[weakref.ReferenceType, tuple[object, ...]]] = {}


def _remember_planar_compatibility(detector, values) -> None:
    identity = id(detector)

    def discard(reference) -> None:
        stored = _PLANAR_COMPATIBILITY.get(identity)
        if stored is not None and stored[0] is reference:
            _PLANAR_COMPATIBILITY.pop(identity, None)

    reference = weakref.ref(detector, discard)
    _PLANAR_COMPATIBILITY[identity] = (reference, values)


def _restore_planar_detector(pose, size_mm, pixels, response, scorer, origin):
    return PlanarDetector(
        pose,
        size_mm=size_mm,
        pixels=pixels,
        response=response,
        scorer=scorer,
        _scorer_origin=origin,
    )


def _plate_thickness_mm(value: object) -> float:
    return _scalar_quantity(value, unit="mm", field="FilterPlate.thickness", expected="length")


def _plate_size_mm(value: object) -> tuple[float, ...]:
    return _positive_length_pair(value, field="FilterPlate.size")


def _detector_size_mm(value: object) -> tuple[float, ...]:
    return _positive_length_pair(value, field="PlanarDetector.size")


@dataclass(frozen=True, init=False)
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
    thickness, size
        Keyword-only Pint length quantities (positive scalar; positive
        two-component ``(width_x, height_y)``), each mutually exclusive with
        its numeric counterpart. Read-back returns fresh quantities in mm; use
        :meth:`with_thickness` / :meth:`with_size` for replacement.
    """

    material: str | MediumSpec
    thickness_mm: float
    size_mm: tuple[float, float]
    pose: PlanarPose
    name: str | None = None

    def __init__(
        self,
        material: str | MediumSpec,
        thickness_mm: float | _Unset = UNSET,
        size_mm: tuple[float, float] | _Unset = UNSET,
        pose: PlanarPose | _Unset = UNSET,
        name: str | None = None,
        *,
        thickness: object = UNSET,
        size: object = UNSET,
    ) -> None:
        thickness_mm = _quantity_spelling(
            "FilterPlate", "thickness_mm", thickness_mm, "thickness", thickness, _plate_thickness_mm
        )
        size_mm = _quantity_spelling(
            "FilterPlate", "size_mm", size_mm, "size", size, _plate_size_mm
        )
        if thickness_mm is UNSET or size_mm is UNSET or pose is UNSET:
            raise TypeError("FilterPlate requires a thickness, a size and a pose")
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "thickness_mm", thickness_mm)
        object.__setattr__(self, "size_mm", size_mm)
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "name", name)
        self.__post_init__()

    @property
    def thickness(self) -> Quantity:
        """Full thickness along the pose normal as a fresh quantity in mm."""
        return _mm_quantity(self.thickness_mm)

    @property
    def size(self) -> Quantity:
        """Full ``(width_x, height_y)`` as a fresh quantity in mm."""
        return _mm_quantity(self.size_mm)

    def with_thickness(self, thickness: object) -> Self:
        """Return a plate with a validated thickness quantity."""
        return replace(self, thickness_mm=_plate_thickness_mm(thickness))

    def with_size(self, size: object) -> Self:
        """Return a plate with a validated ``(width_x, height_y)`` quantity."""
        return replace(self, size_mm=_plate_size_mm(size))

    def __post_init__(self) -> None:
        if isinstance(self.material, str):
            try:
                resolve_composition(self.material, catalog=CATALOG)
            except KeyError:
                raise ValueError(f"unknown FilterPlate material {self.material!r}") from None
        elif isinstance(self.material, MediumSpec):
            if not self.material.composition:
                raise ValueError("FilterPlate MediumSpec composition must not be empty")
            if not has_positive_number_densities(self.material.composition):
                raise ValueError("FilterPlate MediumSpec composition must have positive densities")
        else:
            raise TypeError("FilterPlate.material must be a catalog key or MediumSpec")
        object.__setattr__(
            self,
            "thickness_mm",
            _positive_real(
                "FilterPlate.thickness_mm",
                normalize_unit(
                    self.thickness_mm,
                    "mm",
                    field="FilterPlate.thickness_mm",
                    expected="length",
                ),
            ),
        )
        object.__setattr__(
            self,
            "size_mm",
            _positive_pair(
                "FilterPlate.size_mm",
                normalize_unit(self.size_mm, "mm", field="FilterPlate.size_mm", expected="length"),
            ),
        )
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


def _legacy_pixel_scorer(pixels, scorer, origin):
    """Resolve embedded scoring while retaining omission through dataclass replacement.

    The origin is an InitVar/private attribute, never a dataclass identity field.
    A replacement that changes the effective scorer is an explicit new selection.
    """
    if scorer is not None and not isinstance(scorer, PixelScorer):
        raise TypeError("PlanarDetector.scorer must be a PixelScorer or None")
    explicit = scorer is not None
    if origin is not None and scorer is origin[1]:
        explicit = origin[0]
    if pixels is None:
        if scorer is not None:
            raise ValueError("PlanarDetector.scorer requires pixels")
        return None, (False, None)
    resolved = PixelScorer() if scorer is None else scorer
    if any(
        requested > available
        for requested, available in zip(resolved.angular_shape, pixels.shape, strict=True)
    ):
        raise ValueError("pixel scorer angular shape cannot exceed detector pixel shape")
    return resolved, (explicit, resolved)


@dataclass(frozen=True, init=False)
class PlanarDetector:
    """Describe physical planar detector geometry.

    Parameters
    ----------
    pose
        Detector-centre position and local axes; local ``+z`` points downstream.
    size_mm
        Full ``(width_x, height_y)`` in mm. Required without ``pixels`` and,
        when both are given, must match the pixel-grid active size.
    pixels
        Optional physical pixel grid used for spatial scoring.
    response
        Legacy response accepted for compatibility. Prefer ``Observation``.
    scorer
        Legacy factorized sampling request. Prefer ``Observation.sampling``.
    size
        Keyword-only positive two-component ``(width_x, height_y)`` Pint
        length quantity, mutually exclusive with ``size_mm``. Read-back returns
        a fresh quantity in mm; use :meth:`with_size` for replacement.

    Notes
    -----
    Geometry changes source observation and accepted solid angle. Response and
    scoring constructor arguments remain accessible for compatibility but are
    excluded from the canonical physical dataclass value. A pixelated detector
    always returns spatial output from :func:`pyrite.simulate`.
    """

    pose: PlanarPose
    size_mm: tuple[float, float] | None = None
    pixels: PixelGrid | None = None
    response: InitVar[DetectorResponse | None] = None
    scorer: InitVar[PixelScorer | None] = None
    _scorer_origin: InitVar[tuple[bool, PixelScorer | None] | None] = field(
        default=None, kw_only=True
    )

    def __init__(
        self,
        pose: PlanarPose,
        size_mm: tuple[float, float] | None = None,
        pixels: PixelGrid | None = None,
        response: DetectorResponse | None = None,
        scorer: PixelScorer | None = None,
        *,
        _scorer_origin: tuple[bool, PixelScorer | None] | None = None,
        size: object = UNSET,
    ) -> None:
        resolved = _quantity_spelling(
            "PlanarDetector",
            "size_mm",
            UNSET if size_mm is None else size_mm,
            "size",
            size,
            _detector_size_mm,
        )
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "size_mm", None if resolved is UNSET else resolved)
        object.__setattr__(self, "pixels", pixels)
        self.__post_init__(response, scorer, _scorer_origin)

    @property
    def size(self) -> Quantity:
        """Full active ``(width_x, height_y)`` as a fresh quantity in mm."""
        return _mm_quantity(self.size_mm)

    def with_size(self, size: object) -> Self:
        """Return a detector with a validated size and unchanged compatibility state."""
        return replace(
            self,
            size_mm=_detector_size_mm(size),
            response=self.response,
            scorer=self.scorer,
            _scorer_origin=self._scorer_origin,
        )

    def __post_init__(self, response, scorer, _scorer_origin=None) -> None:
        if not isinstance(self.pose, PlanarPose):
            raise TypeError("PlanarDetector.pose must be a PlanarPose")
        size = (
            None
            if self.size_mm is None
            else _positive_pair(
                "PlanarDetector.size_mm",
                normalize_unit(
                    self.size_mm,
                    "mm",
                    field="PlanarDetector.size_mm",
                    expected="length",
                ),
            )
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
        if response is not None and not hasattr(response, "score"):
            raise TypeError("PlanarDetector.response must implement score()")
        scorer, origin = _legacy_pixel_scorer(self.pixels, scorer, _scorer_origin)
        _remember_planar_compatibility(self, (response, scorer, origin))
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
        response: DetectorResponse | None = None,
        scorer: PixelScorer | None = None,
    ) -> PlanarDetector:
        """Construct one physical Timepix3 ASIC without implying a response."""
        return cls(
            pose=pose,
            pixels=PixelGrid.timepix3_chip(),
            response=response,
            scorer=scorer,
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

    def __getattr__(self, name: str):
        names = ("response", "scorer", "_scorer_origin")
        if name not in names:
            raise AttributeError(name)
        stored = _PLANAR_COMPATIBILITY.get(id(self))
        if stored is None or stored[0]() is not self:
            return (None, None, (False, None))[names.index(name)]
        return stored[1][names.index(name)]

    def __reduce__(self):
        return (
            _restore_planar_detector,
            (self.pose, self.size_mm, self.pixels, self.response, self.scorer, self._scorer_origin),
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
            _legacy_acceptance=(max(polar_span_deg, np.finfo(float).eps), solid_angle),
            response=self.response,
        )


for _compatibility_name in ("response", "scorer", "_scorer_origin"):
    delattr(PlanarDetector, _compatibility_name)


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
