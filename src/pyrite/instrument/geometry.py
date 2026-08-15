"""Analytic point-source geometry for bounded downstream photon objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FilterPlate, PlanarDetector


def _readonly_float_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    array.setflags(write=False)
    return array


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
        object.__setattr__(self, "centers_mm", centers)
        object.__setattr__(self, "directions_lab", directions)
        object.__setattr__(self, "distance_mm", distance)
        object.__setattr__(self, "solid_angle_sr", solid_angle)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the physical/scoring grid shape ``(ny, nx)``."""
        return self.distance_mm.shape  # type: ignore[return-value]


def planar_detector_rays(detector: PlanarDetector) -> PixelRays:
    """Construct lab-frame source-to-pixel-centre rays.

    Pixels are point samples with their full pitch area. An unpixelated
    detector is one centre sample carrying the full active area.
    """
    if not isinstance(detector, PlanarDetector):
        raise TypeError("detector must be a PlanarDetector")
    center = np.asarray(detector.pose.center_mm)
    x_axis = np.asarray(detector.pose.x_axis)
    y_axis = np.asarray(detector.pose.y_axis)
    normal = np.asarray(detector.pose.normal)
    if detector.pixels is None:
        x_offsets = np.array([0.0])
        y_offsets = np.array([0.0])
        assert detector.size_mm is not None
        pixel_area_mm2 = detector.size_mm[0] * detector.size_mm[1]
    else:
        ny, nx = detector.pixels.shape
        pitch_y, pitch_x = detector.pixels.pitch_mm
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
    solid_angle = pixel_area_mm2 * obliquity / distance**2
    return PixelRays(centers, directions, distance, solid_angle)


def ray_box_path_lengths(rays: PixelRays, plate: FilterPlate) -> np.ndarray:
    """Return exact source-to-pixel path length through one finite plate [mm].

    The point-source ray is clipped both to the oriented rectangular box and to
    the finite source--pixel segment. Parallel axes use an explicit inside/
    outside test; side escape and oblique thickness follow from the same slab
    interval calculation.
    """
    if not isinstance(rays, PixelRays):
        raise TypeError("rays must be PixelRays")
    if not isinstance(plate, FilterPlate):
        raise TypeError("plate must be a FilterPlate")
    directions = rays.directions_lab.reshape(-1, 3)
    distance = rays.distance_mm.reshape(-1)
    center = np.asarray(plate.pose.center_mm)
    basis = np.column_stack(
        (
            np.asarray(plate.pose.x_axis),
            np.asarray(plate.pose.y_axis),
            np.asarray(plate.pose.normal),
        )
    )
    origin_local = -center @ basis
    direction_local = directions @ basis
    half_extent = np.array(
        [plate.size_mm[0] / 2.0, plate.size_mm[1] / 2.0, plate.thickness_mm / 2.0]
    )
    direction_tolerance = 64.0 * np.finfo(float).eps
    length_tolerance = direction_tolerance * max(1.0, float(np.max(half_extent)))
    enter = np.zeros_like(distance)
    exit = distance.copy()
    missed = np.zeros(distance.shape, dtype=bool)
    for axis in range(3):
        component = direction_local[:, axis]
        parallel = np.abs(component) <= direction_tolerance
        missed |= parallel & (
            (origin_local[axis] < -half_extent[axis] - length_tolerance)
            | (origin_local[axis] > half_extent[axis] + length_tolerance)
        )
        lower = np.full_like(component, -np.inf)
        upper = np.full_like(component, np.inf)
        nonparallel = ~parallel
        first = np.empty_like(component)
        second = np.empty_like(component)
        np.divide(
            -half_extent[axis] - origin_local[axis],
            component,
            out=first,
            where=nonparallel,
        )
        np.divide(
            half_extent[axis] - origin_local[axis],
            component,
            out=second,
            where=nonparallel,
        )
        lower[nonparallel] = np.minimum(first[nonparallel], second[nonparallel])
        upper[nonparallel] = np.maximum(first[nonparallel], second[nonparallel])
        enter = np.maximum(enter, lower)
        exit = np.minimum(exit, upper)
    length = np.maximum(exit - enter, 0.0)
    length[missed | (length <= length_tolerance)] = 0.0
    return length.reshape(rays.shape)


def filter_path_lengths(rays: PixelRays, filters: tuple[FilterPlate, ...]) -> np.ndarray:
    """Stack exact per-pixel plate lengths as ``(ny, nx, n_filter)``."""
    if not filters:
        empty = np.empty((*rays.shape, 0), dtype=float)
        empty.setflags(write=False)
        return empty
    paths = np.stack([ray_box_path_lengths(rays, plate) for plate in filters], axis=-1)
    paths.setflags(write=False)
    return paths


def angular_tiles(rays: PixelRays, angular_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return fine-pixel tile indices and solid-angle-weighted lab directions."""
    ay, ax = angular_shape
    ny, nx = rays.shape
    if ay <= 0 or ax <= 0 or ay > ny or ax > nx:
        raise ValueError("angular_shape must be positive and not exceed the pixel shape")
    row_groups = np.array_split(np.arange(ny), ay)
    column_groups = np.array_split(np.arange(nx), ax)
    tile_index = np.empty((ny, nx), dtype=np.int64)
    directions = np.empty((ay * ax, 3), dtype=float)
    tile = 0
    for rows in row_groups:
        for columns in column_groups:
            yy, xx = np.ix_(rows, columns)
            tile_index[yy, xx] = tile
            weights = rays.solid_angle_sr[yy, xx]
            weighted = np.sum(weights[..., None] * rays.directions_lab[yy, xx, :], axis=(0, 1))
            directions[tile] = weighted / np.linalg.norm(weighted)
            tile += 1
    tile_index.setflags(write=False)
    directions.setflags(write=False)
    return tile_index, directions


__all__ = [
    "PixelRays",
    "angular_tiles",
    "filter_path_lengths",
    "planar_detector_rays",
    "ray_box_path_lengths",
]
