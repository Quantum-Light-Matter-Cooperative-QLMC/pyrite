"""Analytic point-source geometry for bounded downstream photon objects."""

import numpy as np

from .._planar_geometry import PixelRays, planar_rays, solid_angle_sr
from .model import FilterPlate, PlanarDetector


def planar_detector_rays(detector: PlanarDetector) -> PixelRays:
    """Construct lab-frame source-to-pixel-centre rays.

    Pixels are point samples with their full pitch area. An unpixelated
    detector instead uses its exact finite-rectangle solid angle; see
    :func:`pyrite._planar_geometry.planar_rays`.
    """
    if not isinstance(detector, PlanarDetector):
        raise TypeError("detector must be a PlanarDetector")
    return planar_rays(detector.pose, pixels=detector.pixels, size_mm=detector.size_mm)


def ray_box_path_lengths(rays: PixelRays, plate: FilterPlate) -> np.ndarray:
    """Return exact source-to-pixel path length through one finite plate [mm].

    The point-source ray is clipped both to the oriented rectangular box and to
    the finite source--pixel segment. Parallel axes use an explicit inside/
    outside test; side escape and oblique thickness follow from the same slab
    interval calculation.

    Validation: positioned-filter-attenuation
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


#: Fraction of a knot interval within which a pixel snaps exactly onto a knot,
#: absorbing round-off in the plane projection of a one-pixel tile.
_KNOT_SNAP = 1.0e-9


def _axis_weights(
    coordinate: np.ndarray, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower knot, upper knot, and upper weight of clamped linear interpolation."""
    if knots.size == 1:
        zero = np.zeros(coordinate.shape, dtype=np.int64)
        return zero, zero, np.zeros(coordinate.shape)
    lower = np.clip(np.searchsorted(knots, coordinate, side="right") - 1, 0, knots.size - 2)
    upper = lower + 1
    fraction = np.clip((coordinate - knots[lower]) / (knots[upper] - knots[lower]), 0.0, 1.0)
    fraction[fraction < _KNOT_SNAP] = 0.0
    fraction[fraction > 1.0 - _KNOT_SNAP] = 1.0
    return lower, upper, fraction


def angular_tile_weights(
    detector: PlanarDetector,
    directions_lab: np.ndarray,
    angular_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel tiles and convex weights blending neighbouring tile spectra.

    Each tile representative direction is projected onto the detector plane
    along its source ray. Knots are the tile-column means of the projected
    local ``x`` and the tile-row means of the projected local ``y``, so the
    blend is a separable piecewise-linear interpolation in detector-local
    coordinates evaluated at each pixel centre. Pixels outside the outermost
    knots take the edge value (clamped, never extrapolated), and a one-tile
    axis is constant. A representative direction projects to a convex
    combination of its own pixel centres, so knots are strictly increasing,
    and a one-pixel tile's knot is that pixel's centre: with
    ``angular_shape`` equal to the pixel shape every pixel takes exactly its
    own tile.

    Parameters
    ----------
    detector
        Pixelated planar detector the tiles were drawn on.
    directions_lab
        ``(n_tile, 3)`` representative lab directions from :func:`angular_tiles`.
    angular_shape
        ``(n_tile_rows, n_tile_columns)`` used by :func:`angular_tiles`.

    Returns
    -------
    tile, weight
        ``(ny, nx, 4)`` tile indices and non-negative weights summing to 1.

    Validation: pixel-angular-interpolation
    """
    if detector.pixels is None:
        raise ValueError("angular tile interpolation requires a pixelated detector")
    ay, ax = angular_shape
    directions = np.asarray(directions_lab, dtype=float)
    if directions.shape != (ay * ax, 3):
        raise ValueError("directions_lab must have shape (n_tile, 3) matching angular_shape")
    ny, nx = detector.pixels.shape
    pitch_y, pitch_x = detector.pixels.pitch_mm
    center = np.asarray(detector.pose.center_mm)
    x_axis = np.asarray(detector.pose.x_axis)
    y_axis = np.asarray(detector.pose.y_axis)
    normal = np.asarray(detector.pose.normal)
    hits = directions * ((center @ normal) / (directions @ normal))[:, None] - center
    local_x = (hits @ x_axis).reshape(ay, ax)
    local_y = (hits @ y_axis).reshape(ay, ax)
    lower_c, upper_c, fraction_x = _axis_weights(
        (np.arange(nx) - (nx - 1) / 2.0) * pitch_x, np.mean(local_x, axis=0)
    )
    lower_r, upper_r, fraction_y = _axis_weights(
        (np.arange(ny) - (ny - 1) / 2.0) * pitch_y, np.mean(local_y, axis=1)
    )
    rows = (lower_r[:, None], lower_r[:, None], upper_r[:, None], upper_r[:, None])
    columns = (lower_c[None, :], upper_c[None, :], lower_c[None, :], upper_c[None, :])
    tile = np.stack([r * ax + c for r, c in zip(rows, columns, strict=True)], axis=-1)
    wy, wx = fraction_y[:, None], fraction_x[None, :]
    weight = np.stack(
        np.broadcast_arrays((1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx), axis=-1
    )
    tile = np.ascontiguousarray(tile, dtype=np.int64)
    weight = np.ascontiguousarray(weight)
    tile.setflags(write=False)
    weight.setflags(write=False)
    return tile, weight


__all__ = [
    "PixelRays",
    "angular_tile_weights",
    "angular_tiles",
    "filter_path_lengths",
    "planar_detector_rays",
    "ray_box_path_lengths",
    "solid_angle_sr",
]
