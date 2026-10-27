"""Host first-collision reference for the optional photon tracker (#341).

This module is not called by production transport. It consumes explicit draws;
it owns no RNG, scattering kinematics, descendant queue, or detector scoring.
Coordinates are in the sample frame, matching the analytic stack geometry.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...materials.photon_cross_sections import (
    EPDL_E_MAX_EV,
    EPDL_E_MIN_EV,
    PHOTON_CHANNELS,
    photon_cross_sections_ang2,
)
from ..geometry import validate_transverse_dimensions

PHOTON_CHANNEL_ORDER = tuple(PHOTON_CHANNELS)


def _validate_layers(layers: Sequence) -> None:
    """Require ordered, non-overlapping slabs and finite number densities."""
    previous = -np.inf
    for top, bottom, composition in layers:
        if not np.isfinite([top, bottom]).all() or top < previous or bottom <= top:
            raise ValueError("layers must be finite, ordered and non-overlapping with top < bottom")
        previous = bottom
        for _, density in composition:
            if not np.isfinite(density) or density < 0.0:
                raise ValueError("number densities must be finite and non-negative")


def _validate_rays(origins, directions):
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must both have shape (N, 3)")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("origins and directions must be finite")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("directions must be unit vectors")
    return origins, directions


def ray_layer_intervals(
    origins, directions, layers: Sequence, *, width_ang=None, height_ang=None
) -> tuple[np.ndarray, np.ndarray]:
    """Forward entry/exit distances [Å] for each ray and layer, shape ``(N, K)``.

    Layers are ``(top_ang, bottom_ang, [(element, density_ang3), ...])`` in
    increasing z; gaps and empty compositions represent vacuum. The common
    footprint is centered on x=y=0. Both transverse sizes must be supplied
    together, or both omitted for infinite slabs. Outside origins are never
    clamped; a miss or tangency has zero path length. For parallel rays an
    axis interval is half-open [lower, upper), giving shared faces one owner.
    Non-parallel boundary rays belong to the material they enter, with no
    artificial displacement. No rotation, grooves, or overlapping bodies.

    Source: analytic ray/box slab intersection; PENELOPE-2024 §1.4.4
    interface continuation. Vacuum or zero intersection gives zero depth.
    Validation: photon-channel-first-interaction
    """
    origins, directions = _validate_rays(origins, directions)
    _validate_layers(layers)
    width_ang, height_ang = validate_transverse_dimensions(width_ang, height_ang, unit="Ang")
    return _layer_intervals(origins, directions, layers, width_ang, height_ang)


def _layer_intervals(origins, directions, layers, width_ang, height_ang, *, xp=np):
    """Clip validated rays/layers on the selected array backend."""
    bounds = [
        (2, xp.asarray([layer[0] for layer in layers]), xp.asarray([layer[1] for layer in layers]))
    ]
    if width_ang is not None:
        assert height_ang is not None  # validated as a pair above
        bounds.extend([(0, -width_ang / 2, width_ang / 2), (1, -height_ang / 2, height_ang / 2)])
    return _clip_axes(origins, directions, bounds, len(layers), xp=xp)


def _clip_axes(origins, directions, bounds, k: int, *, xp=np):
    """Intersect forward rays with per-axis [lower, upper) bands for K bodies.

    Validation: photon-scene-navigation
    """
    shape = (origins.shape[0], k)
    start, end = xp.zeros(shape), xp.full(shape, np.inf)
    for axis, lower, upper in bounds:
        coordinate, direction = origins[:, axis, None], directions[:, axis, None]
        parallel = direction == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lower - coordinate) / direction, (upper - coordinate) / direction
        inside = (coordinate >= lower) & (coordinate < upper)
        near = xp.where(parallel, -np.inf, xp.minimum(a, b))
        far = xp.where(parallel, xp.where(inside, np.inf, -np.inf), xp.maximum(a, b))
        start, end = xp.maximum(start, near), xp.minimum(end, far)
    hit = end > start
    return xp.where(hit, start, 0.0), xp.where(hit, end, 0.0)


@dataclass(frozen=True, slots=True)
class PhotonBox:
    """Homogeneous oriented finite box, e.g. a filter plate, in the sample frame.

    ``axes`` rows are the orthonormal local x, y, z directions; ``half_extent_ang``
    the positive half sizes along them [Å]; ``composition`` ordered
    ``(element, atoms/Å³)`` pairs, empty for vacuum.

    Validation: photon-scene-navigation
    """

    center_ang: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    half_extent_ang: tuple[float, float, float]
    composition: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        center = np.asarray(self.center_ang, dtype=float)
        axes = np.asarray(self.axes, dtype=float)
        half = np.asarray(self.half_extent_ang, dtype=float)
        if center.shape != (3,) or axes.shape != (3, 3) or half.shape != (3,):
            raise ValueError("box center, axes and half extents must be (3,), (3, 3), (3,)")
        if not (np.isfinite(center).all() and np.isfinite(axes).all()):
            raise ValueError("box center and axes must be finite")
        if not np.allclose(axes @ axes.T, np.eye(3), rtol=0.0, atol=1e-12):
            raise ValueError("box axes must be orthonormal")
        if not np.isfinite(half).all() or np.any(half <= 0.0):
            raise ValueError("box half extents must be finite and positive")
        _validate_layers([(0.0, 1.0, self.composition)])


def ray_box_intervals(origins, directions, boxes: Sequence[PhotonBox]):
    """Forward entry/exit distances [Å] for each ray and oriented box, ``(N, B)``.

    Rays are expressed in each box's local frame and clipped with the same
    half-open [lower, upper) parallel-face rule as :func:`ray_layer_intervals`.
    Source: analytic ray/oriented-box slab intersection; rotation preserves
    path length. Validation: photon-scene-navigation
    """
    origins, directions = _validate_rays(origins, directions)
    return _box_intervals(origins, directions, boxes)


def _box_intervals(origins, directions, boxes, *, xp=np):
    """Clip validated rays/boxes on the selected array backend."""
    start = xp.zeros((origins.shape[0], len(boxes)))
    end = xp.zeros_like(start)
    for j, box in enumerate(boxes):
        axes = xp.asarray(box.axes, dtype=float)
        half = xp.asarray(box.half_extent_ang, dtype=float)
        local_o = (origins - xp.asarray(box.center_ang, dtype=float)) @ axes.T
        local_d = directions @ axes.T
        bounds = [(axis, -half[axis], half[axis]) for axis in range(3)]
        lo, hi = _clip_axes(local_o, local_d, bounds, 1, xp=xp)
        start[:, j], end[:, j] = lo[:, 0], hi[:, 0]
    return start, end


def _coefficients(composition, energies):
    """Element-major, channel-minor macroscopic rates [Å⁻¹], shape (N, I*C)."""
    columns = []
    for element, density in composition:
        sigma = photon_cross_sections_ang2(element, energies)
        columns.extend(float(density) * sigma[channel] for channel in PHOTON_CHANNEL_ORDER)
    rates = np.stack(columns, axis=1) if columns else np.zeros((energies.size, 0))
    if not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("photon cross sections must be finite and non-negative")
    return rates


def _collision_distances(start, end, mu, optical_depth, *, xp=np):
    """Invert accumulated depth along positive material intervals."""
    with np.errstate(invalid="ignore"):
        depths = xp.where((end > start) & (mu > 0.0), mu * (end - start), 0.0)
    order = xp.argsort(start, axis=1, kind="stable")
    rows = xp.arange(start.shape[0])
    ordered = depths[rows[:, None], order]
    cumulative = xp.cumsum(ordered, axis=1)
    # Strict comparison assigns an exact interface to the following interval.
    reached = (cumulative > optical_depth[:, None]) & (ordered > 0.0)
    interacts = reached.any(axis=1)
    slot = xp.argmax(reached, axis=1)
    layer = xp.where(interacts, order[rows, slot], -1)
    before = xp.where(slot > 0, cumulative[rows, xp.maximum(slot - 1, 0)], 0.0)
    selected = xp.flatnonzero(interacts)
    distance = xp.full(rows.size, np.inf)
    distance[selected] = (
        start[selected, layer[selected]]
        + (optical_depth[selected] - before[selected]) / mu[selected, layer[selected]]
    )
    return distance, layer


def photon_first_interactions(
    origins, directions, energies_eV, uniforms, layers: Sequence, *, width_ang=None, height_ang=None
) -> dict[str, np.ndarray]:
    """Sample all five EPDL channels and their atoms along stack/prism rays.

    Inputs are sample-frame origins/directions ``(N, 3)``, energies ``(N,)``
    [eV], and independent uniforms ``(N, 2)`` in [0, 1). Draw 0 sets
    ``tau=-log1p(-u0)``; draw 1 selects the element/channel with probability
    ``density * sigma_channel / mu_total``. Integer ``channel`` indexes
    ``PHOTON_CHANNEL_ORDER``; ``element`` indexes the hit layer's composition.
    Escape returns distance=inf and layer=channel=element=-1. Empty batches
    and an empty (vacuum) stack are valid. No outgoing particles are sampled.

    Source: PENELOPE-2024 §1.4.2–1.4.4, Eqs. 1.106–1.111, 1.119,
    1.124–1.125. Rates are constant in each homogeneous layer, photon energy
    is unchanged in a flight, and cross sections are per atom [Å²]. Zero
    density gives escape, and an infinite positive-rate slab always collides.
    Production pair/annihilation sampling retains its historical entry point.
    Validation: photon-channel-first-interaction
    """
    origins, directions, energies, uniforms = _validate_flights(
        origins, directions, energies_eV, uniforms
    )
    start, end = ray_layer_intervals(
        origins, directions, layers, width_ang=width_ang, height_ang=height_ang
    )
    return _interval_first_interactions(
        start, end, [layer[2] for layer in layers], energies, uniforms
    )


def scene_first_interactions(
    origins,
    directions,
    energies_eV,
    uniforms,
    layers: Sequence,
    boxes: Sequence[PhotonBox] = (),
    *,
    width_ang=None,
    height_ang=None,
) -> dict[str, np.ndarray]:
    """First collision through the stack/prism and non-overlapping oriented boxes.

    As :func:`photon_first_interactions`, with ``layer`` indexing regions in
    the order ``layers`` then ``boxes``. Vacuum gaps between bodies, re-entry
    after scattering and arbitrary box orientation follow from summing optical
    depth over every forward interval; a straight ray crosses each convex
    region at most once. Callers validate disjointness with
    :func:`validate_photon_regions`. With no boxes the result is identical to
    :func:`photon_first_interactions`.
    Validation: photon-scene-navigation
    """
    origins, directions, energies, uniforms = _validate_flights(
        origins, directions, energies_eV, uniforms
    )
    start, end = ray_layer_intervals(
        origins, directions, layers, width_ang=width_ang, height_ang=height_ang
    )
    if boxes:
        box_start, box_end = ray_box_intervals(origins, directions, boxes)
        start, end = np.hstack((start, box_start)), np.hstack((end, box_end))
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    return _interval_first_interactions(start, end, compositions, energies, uniforms)


def _validate_flights(origins, directions, energies_eV, uniforms):
    origins, directions = _validate_rays(origins, directions)
    energies, uniforms = np.asarray(energies_eV, dtype=float), np.asarray(uniforms, dtype=float)
    n = origins.shape[0]
    if energies.shape != (n,) or uniforms.shape != (n, 2):
        raise ValueError("energies must have shape (N,) and uniforms (N, 2)")
    if not np.isfinite(energies).all() or np.any(
        (energies < EPDL_E_MIN_EV) | (energies > EPDL_E_MAX_EV)
    ):
        raise ValueError("photon energy is outside the EPDL2025 attenuation domain")
    if not np.isfinite(uniforms).all() or np.any((uniforms < 0.0) | (uniforms >= 1.0)):
        raise ValueError("uniforms must be finite and in [0, 1)")
    return origins, directions, energies, uniforms


def _layer_regions(layers, width_ang, height_ang):
    """Each stack layer as (center, axes, half extents), infinite laterally if unbounded."""
    lateral = (np.inf, np.inf) if width_ang is None else (width_ang / 2, height_ang / 2)
    return [
        (
            np.array([0.0, 0.0, (top + bottom) / 2]),
            np.eye(3),
            np.array([*lateral, (bottom - top) / 2]),
        )
        for top, bottom, _ in layers
    ]


def _project(region, axis) -> float:
    """Box support radius [Å], retaining every nonzero axis projection.

    Source: Eberly §§2.1–2.2.1, support sum of half extents times absolute
    axis dot products. A small angle can project a long side significantly.
    Exactly zero factors contribute zero even for an infinite slab; any
    nonzero factor on an infinite extent gives infinite support.
    Limiting case: an axis-aligned box returns its matching half extent.

    Validation: photon-scene-navigation
    """
    _, axes, half = region
    weights = np.abs(axes @ axis)
    used = weights > 0.0
    return float(np.sum(half[used] * weights[used]))


def _overlap(first, second) -> bool:
    """Separating-axis test for convex boxes; face contact is not overlap.

    Source: separating-axis theorem, Eberly §§2.1–2.2.1. Compare projected
    relative centres with the sum of support radii [Å]. A common translation
    leaves this comparison unchanged when the displacement remains representable.
    Contact tolerance is 32 binary64 epsilons of the lengths on this axis;
    absolute coordinates and unrelated lateral extents cannot erase thin bodies.
    Infinite projected slabs do not supply a separating axis.

    Validation: photon-scene-navigation
    """
    candidates = [*first[1], *second[1]]
    candidates += [np.cross(a, b) for a in first[1] for b in second[1]]
    displacement = first[0] - second[0]
    for axis in candidates:
        norm = float(np.hypot.reduce(axis))
        if norm == 0.0:
            continue
        axis = axis / norm
        radius = _project(first, axis) + _project(second, axis)
        if np.isinf(radius):
            continue
        distance = abs(float(displacement @ axis))
        tolerance = 32 * np.finfo(float).eps * max(distance, radius)
        if distance >= radius - tolerance:
            return False
    return True


def validate_photon_regions(
    layers: Sequence,
    boxes: Sequence[PhotonBox] = (),
    *,
    width_ang=None,
    height_ang=None,
    surfaces: Sequence = (),
) -> None:
    """Refuse boxes overlapping the stack or each other, and surfaces inside material.

    ``surfaces`` are extra ``(center, axes, half_extent)`` regions (e.g. a
    zero-thickness aperture) that may touch but not enter any body.
    Validation: photon-scene-navigation
    """
    _validate_layers(layers)
    width_ang, height_ang = validate_transverse_dimensions(width_ang, height_ang, unit="Ang")
    stack = _layer_regions(layers, width_ang, height_ang)
    solids = [
        tuple(np.asarray(v, dtype=float) for v in (b.center_ang, b.axes, b.half_extent_ang))
        for b in boxes
    ]
    for i, box in enumerate(solids):
        if any(_overlap(box, other) for other in stack + solids[:i]):
            raise ValueError("photon boxes must not overlap the target stack or each other")
    for surface in surfaces:
        if any(_overlap(surface, other) for other in stack + solids):
            raise ValueError("a scoring surface must lie outside all modeled material")


def _interval_first_interactions(start, end, compositions, energies, uniforms, *, xp=np):
    """Optical-depth inversion and channel/atom choice over precomputed intervals.

    Validation: photon-scene-navigation
    """
    n = energies.shape[0]
    distance = xp.full(n, np.inf)
    layer, channel, element = (xp.full(n, -1, dtype=np.int64) for _ in range(3))
    if n and compositions:
        rates = [_coefficients(composition, energies) for composition in compositions]
        mu = np.stack([r.sum(axis=1) for r in rates], axis=1)
        if not np.isfinite(mu).all():
            raise ValueError("total photon rates must be finite")
        # Table evaluation and draw conversion stay on the host. Device geometry,
        # depth inversion and channel selection use the same equations below.
        mu = xp.asarray(mu)
        rates = [xp.asarray(rate) for rate in rates]
        optical_depth = xp.asarray(-np.log1p(-uniforms[:, 0]))
        uniforms = xp.asarray(uniforms)
        distance, layer = _collision_distances(start, end, mu, optical_depth, xp=xp)
        for region in range(len(compositions)):
            rows = xp.flatnonzero(layer == region)
            if rows.size == 0:
                continue
            cumulative = xp.cumsum(rates[region][rows], axis=1)
            total = cumulative[:, -1]
            threshold = xp.minimum(uniforms[rows, 1] * total, xp.nextafter(total, 0.0))
            selected = xp.sum(cumulative <= threshold[:, None], axis=1)
            element[rows], channel[rows] = xp.divmod(selected, len(PHOTON_CHANNEL_ORDER))
    return {"distance_ang": distance, "layer": layer, "channel": channel, "element": element}
