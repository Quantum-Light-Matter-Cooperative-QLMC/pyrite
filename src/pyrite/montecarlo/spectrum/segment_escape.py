"""Photon escape attenuation averaged along each straight transport segment.

An incoherent emitter (characteristic lines, bremsstrahlung) radiates
uniformly along a segment of length ``L``, so its escaping yield carries the
segment mean of ``exp(-tau(s))``, not ``exp(-tau)`` at the midpoint. Inside
one layer the escape optical depth is linear along a straight segment -- the
emitting layer's path changes linearly with depth and every other crossed
layer's path is constant -- so with ``tau_0``, ``tau_1`` the endpoint depths

    <exp(-tau)> = (1/L) int_0^L exp(-tau(s)) ds
                = exp(-min(tau_0, tau_1)) * g(|tau_1 - tau_0|),
    g(x) = (1 - exp(-x)) / x,   g(0) = 1.

Both factors are at most one, so the form cannot overflow however opaque the
layer. The midpoint rule is its ``|tau_1 - tau_0| -> 0`` limit, and it
undercounts by Jensen's inequality whenever a line's attenuation length is
comparable to a segment (issue #176). Where the escape path is only piecewise
linear along a segment -- a finite footprint whose nearest exit face changes,
or a grooved entrance face -- endpoint depths alone do not determine the
integral. The helper splits at those changes and integrates each linear piece.

Validation: segment-escape-average
"""

import numpy as np

from ..._backend import REAL
from ...materials.attenuation import _layer_dz, _layer_path_length
from ..geometry import first_prism_exit
from ..groove import escape_distance_ang


def _escape_paths_at(r, n_hat, segments, layers, groove, xp):
    """Escape path [Ang] per layer from points ``r`` along ``n_hat``; shape ``(N, K)``.

    ``K`` is the number of layers, or 1 for a single slab. Mirrors the
    midpoint geometry of the spectrum routes: groove escape, the finite
    rectangular footprint, or the laterally infinite slab.
    """
    z = r[:, 2]
    n_z = float(n_hat[2])
    thickness = float(segments["thickness_ang"])
    width = segments.get("crystal_width_ang")
    height = segments.get("crystal_height_ang")
    if groove is not None:
        path = xp.asarray(escape_distance_ang(r[:, 0], z, groove), dtype=REAL)
        return xp.maximum(path, 0.0)[:, None]
    if width is not None and height is not None:
        exit_distance, _ = first_prism_exit(
            r,
            xp.asarray(n_hat, dtype=REAL),
            z_min_ang=0.0,
            z_max_ang=thickness,
            width_ang=width,
            height_ang=height,
            xp=xp,
        )
        exit_distance = xp.maximum(exit_distance, 0.0)
        if layers is None:
            return exit_distance[:, None]
        return xp.stack(
            [
                _layer_path_length(z, n_z, exit_distance, float(z_top), float(z_bot))
                for z_top, z_bot, _ in layers
            ],
            axis=1,
        )
    inv_nz = 1.0 / max(abs(n_z), 1.0e-12)
    if layers is None:
        depth = xp.clip(z, 0.0, thickness)
        path = depth * inv_nz if n_z < 0 else (thickness - depth) * inv_nz
        return path[:, None]
    return xp.stack(
        [_layer_dz(z, n_z, float(z_top), float(z_bot)) * inv_nz for z_top, z_bot, _ in layers],
        axis=1,
    )


def segment_escape_gradient(segments, n_hat, *, groove=None, xp=np):
    """Gradient of each affine piece's escape distance, in the sample frame.

    Planar ray intersection gives ``grad L = -e/(e.n_hat)``; this is the
    first-order Snell correction from tangential continuity. Pieces must have
    been cut at face switches before calling. Groove rays follow the working
    facet normal, whose no-re-entry geometry gives ``grad L = -n_hat``.
    A z-only slab uses the same formula, independent of its thickness.
    Validation: xray-in-medium-resonance
    """
    n = xp.asarray(n_hat, dtype=REAL)
    count = int(xp.asarray(segments["L_ang"]).size)
    if groove is not None:
        return xp.broadcast_to(-n, (count, 3))
    if segments.get("crystal_width_ang") is None:
        normal = xp.broadcast_to(xp.asarray([0.0, 0.0, 1.0], dtype=REAL), (count, 3))
    else:
        _, face = first_prism_exit(
            xp.asarray(segments["r_mid"], dtype=REAL),
            n,
            z_min_ang=0.0,
            z_max_ang=float(segments["thickness_ang"]),
            width_ang=segments.get("crystal_width_ang"),
            height_ang=segments.get("crystal_height_ang"),
            xp=xp,
        )
        normal = xp.eye(3, dtype=REAL)[face // 2]
    gradient = -normal / xp.sum(normal * n, axis=1)[:, None]
    if (
        segments.get("crystal_width_ang") is None
        and segments.get("r_mid") is not None
        and segments.get("thickness_ang") is not None
    ):
        # L is clipped to zero outside a slab; synthetic transparent anchors
        # also use exterior pieces. Their escape phase has zero gradient.
        z = xp.asarray(segments["r_mid"])[:, 2]
        exterior = (z < 0.0) | (z > float(segments["thickness_ang"]))
        gradient = xp.where(exterior[:, None], 0.0, gradient)
    return gradient


def _cut_at_zero(start, end, xp):
    """Fraction where an affine quantity changes sign, or 1 if it does not."""
    denominator = start - end
    safe = xp.where(denominator == 0.0, 1.0, denominator)
    fraction = start / safe
    return xp.where((fraction > 0.0) & (fraction < 1.0), fraction, 1.0)


def _groove_segment_paths(r0, r1, groove, xp):
    """Split at relief-plane and valley-depth jumps in periodic groove escape."""
    n_rows = int(r0.shape[0])
    if n_rows == 0:
        empty = xp.empty(0, dtype=np.float64)
        return xp.empty(0, dtype=np.int64), empty, empty[:, None], empty[:, None]
    st = float(np.sin(groove.tilt_polar_rad))
    ct = float(np.cos(groove.tilt_polar_rad))
    cell = float(groove.spacing_ang) * ct
    depth = float(groove.depth_ang)
    d0 = r0[:, 0] * ct - r0[:, 2] * st
    d1 = r1[:, 0] * ct - r1[:, 2] * st
    first_cell = xp.floor(xp.minimum(d0, d1) / cell)
    n_planes = int(xp.max(xp.abs(d1 - d0) / cell)) + 2
    cuts = [xp.zeros(n_rows, dtype=np.float64), xp.ones(n_rows, dtype=np.float64)]
    for offset in range(1, n_planes + 1):
        plane = (first_cell + offset) * cell
        cuts.append(_cut_at_zero(d0 - plane, d1 - plane, xp))
    ordered = xp.sort(xp.stack(cuts, axis=1), axis=1)
    valid = ordered[:, 1:] - ordered[:, :-1] > 1.0e-12
    owner, part = xp.nonzero(valid)
    left = ordered[owner, part]
    right = ordered[owner, part + 1]
    cell_index = xp.floor((d0[owner] + (d1[owner] - d0[owner]) * (left + right) / 2.0) / cell)
    z_start = r0[owner, 2] + (r1[owner, 2] - r0[owner, 2]) * left
    z_end = r0[owner, 2] + (r1[owner, 2] - r0[owner, 2]) * right
    d_start = d0[owner] + (d1[owner] - d0[owner]) * left
    d_end = d0[owner] + (d1[owner] - d0[owner]) * right
    valley_start = z_start - st * ((cell_index + 1.0) * cell - d_start)
    valley_end = z_end - st * ((cell_index + 1.0) * cell - d_end)
    first_band = xp.floor(xp.minimum(valley_start, valley_end) / depth)
    n_bands = int(xp.max(xp.abs(valley_end - valley_start) / depth)) + 2
    subcuts = [xp.zeros(owner.size, dtype=np.float64), xp.ones(owner.size, dtype=np.float64)]
    for offset in range(1, n_bands + 1):
        plane = (first_band + offset) * depth
        subcuts.append(_cut_at_zero(valley_start - plane, valley_end - plane, xp))
    subordered = xp.sort(xp.stack(subcuts, axis=1), axis=1)
    subvalid = subordered[:, 1:] - subordered[:, :-1] > 1.0e-12
    parent, subpart = xp.nonzero(subvalid)
    out_owner = owner[parent]
    t_start = left[parent] + (right[parent] - left[parent]) * subordered[parent, subpart]
    t_end = left[parent] + (right[parent] - left[parent]) * subordered[parent, subpart + 1]
    # At a band boundary the endpoint formula must use the one-sided band of
    # this open interval, not floor() evaluated on the shared boundary.
    valley_mid = (
        valley_start[parent]
        + (valley_end[parent] - valley_start[parent])
        * (subordered[parent, subpart] + subordered[parent, subpart + 1])
        / 2.0
    )
    band = xp.floor(valley_mid / depth)
    cell_number = cell_index[parent]
    path_start = (cell_number + 1.0) * cell - (
        d0[out_owner] + (d1[out_owner] - d0[out_owner]) * t_start
    )
    path_end = (cell_number + 1.0) * cell - (
        d0[out_owner] + (d1[out_owner] - d0[out_owner]) * t_end
    )
    path_start += xp.maximum(band, 0.0) * cell
    path_end += xp.maximum(band, 0.0) * cell
    return out_owner, t_end - t_start, path_start[:, None], path_end[:, None]


def segment_escape_paths(segments, index, n_hat, *, layers=None, groove=None, xp=np):
    """Split segments where a slab/box escape optical depth changes slope.

    Return ``(owner, fraction, path_start, path_end)``. Each output row is one
    linear piece of an input row, and ``owner`` indexes the selected input rows.
    ``fraction`` is its share of the original electron path length. Splitting
    at source-layer boundaries, photon exit-layer boundaries, and competing
    box faces makes endpoint integration exact for planar layers and boxes.
    Groove relief uses one-sided values at its periodic path jumps.

    Validation: segment-escape-average
    """
    if segments.get("v_hat") is None:
        raise ValueError("segment escape integration requires per-segment directions v_hat")
    r_mid = xp.asarray(segments["r_mid"], dtype=np.float64)[index]
    v_hat = xp.asarray(segments["v_hat"], dtype=np.float64)[index]
    half = 0.5 * xp.asarray(segments["L_ang"], dtype=np.float64)[index][:, None]
    r0 = r_mid - half * v_hat
    r1 = r_mid + half * v_hat
    if groove is not None:
        return _groove_segment_paths(r0, r1, groove, xp)
    n_rows = int(r_mid.shape[0])
    if layers is None and (
        segments.get("crystal_width_ang") is None or segments.get("crystal_height_ang") is None
    ):
        owner = xp.arange(n_rows)
        fraction = xp.ones(n_rows, dtype=np.float64)
        return (
            owner,
            fraction,
            _escape_paths_at(r0, n_hat, segments, None, None, xp),
            _escape_paths_at(r1, n_hat, segments, None, None, xp),
        )
    cuts = [xp.zeros(n_rows, dtype=np.float64), xp.ones(n_rows, dtype=np.float64)]
    boundaries = [] if layers is None else [float(layer[0]) for layer in layers[1:]]
    for boundary in boundaries:
        cuts.append(_cut_at_zero(r0[:, 2] - boundary, r1[:, 2] - boundary, xp))

    width = segments.get("crystal_width_ang")
    height = segments.get("crystal_height_ang")
    if groove is None and width is not None and height is not None:
        face_distances = []
        for axis, lower, upper in (
            (0, -float(width) / 2.0, float(width) / 2.0),
            (1, -float(height) / 2.0, float(height) / 2.0),
            (2, 0.0, float(segments["thickness_ang"])),
        ):
            direction = float(n_hat[axis])
            if abs(direction) <= 1.0e-12:
                continue
            face = upper if direction > 0.0 else lower
            face_distances.append(
                ((face - r0[:, axis]) / direction, (face - r1[:, axis]) / direction)
            )
        for i, (d0, d1) in enumerate(face_distances):
            for other0, other1 in face_distances[i + 1 :]:
                cuts.append(_cut_at_zero(d0 - other0, d1 - other1, xp))
            for boundary in boundaries:
                cuts.append(
                    _cut_at_zero(
                        r0[:, 2] + float(n_hat[2]) * d0 - boundary,
                        r1[:, 2] + float(n_hat[2]) * d1 - boundary,
                        xp,
                    )
                )

    ordered = xp.sort(xp.stack(cuts, axis=1), axis=1)
    fraction_all = ordered[:, 1:] - ordered[:, :-1]
    owner, part = xp.nonzero(fraction_all > 1.0e-12)
    fraction = fraction_all[owner, part]
    # A piece endpoint can lie on a crystal face -- every track starts on the
    # entrance plane -- where the box exit distance is degenerate: the face the
    # photon leaves through sits at distance 0 where face switches meet. The
    # paths are affine on each piece, so sample at
    # 1/4 and 3/4 and extrapolate to the one-sided endpoint values exactly.
    delta = r1[owner] - r0[owner]
    left = ordered[owner, part, None]
    width = ordered[owner, part + 1, None] - left
    quarter = _escape_paths_at(
        r0[owner] + (left + 0.25 * width) * delta, n_hat, segments, layers, None, xp
    )
    three_quarter = _escape_paths_at(
        r0[owner] + (left + 0.75 * width) * delta, n_hat, segments, layers, None, xp
    )
    path0 = xp.maximum(1.5 * quarter - 0.5 * three_quarter, 0.0)
    path1 = xp.maximum(1.5 * three_quarter - 0.5 * quarter, 0.0)
    return owner, fraction, path0, path1


def mean_transmission(tau_start, tau_end, xp=np):
    """Segment mean of ``exp(-tau)`` for ``tau`` linear between the endpoints.

    ``exp(-min(tau)) * (1 - exp(-d)) / d`` with ``d = |tau_end - tau_start|``,
    and its series ``1 - d/2 + d^2/6`` below ``d = 1e-4``, where the direct
    ratio loses digits. Validation: segment-escape-average
    """
    lo = xp.minimum(tau_start, tau_end)
    d = xp.abs(tau_end - tau_start)
    safe = xp.where(d > 1.0e-4, d, 1.0)
    ratio = xp.where(d > 1.0e-4, -xp.expm1(-safe) / safe, 1.0 - 0.5 * d + d * d / 6.0)
    return xp.exp(-lo) * ratio


def segment_escape_pieces(segments, n_hat, *, layers=None, groove=None, xp=np):
    """Linear escape pieces of every segment, padded to one row per segment.

    Returns ``(fraction, path_start, path_end)`` with shapes ``(N, P)``,
    ``(N, P, K)`` and ``(N, P, K)``: ``N`` segments, ``P`` the largest piece
    count, ``K`` layers (1 for a single slab). Padding slots carry zero
    fraction and zero paths, so they add nothing in
    :func:`piece_mean_transmission`. This layout suits consumers whose ``mu``
    varies per (segment, reflection) and so cannot use the flat ``owner`` rows
    of :func:`segment_escape_paths`. Validation: segment-escape-average
    """
    n_rows = int(xp.asarray(segments["L_ang"]).shape[0])
    owner, fraction, path_start, path_end = segment_escape_paths(
        segments, xp.arange(n_rows), n_hat, layers=layers, groove=groove, xp=xp
    )
    counts = xp.bincount(owner, minlength=n_rows)
    n_pieces = max(1, int(counts.max())) if n_rows else 1
    # ``owner`` is ascending (row-major nonzero), so a piece's slot is its
    # offset from its owner's first piece.
    slot = xp.arange(owner.size) - (xp.cumsum(counts) - counts)[owner]
    n_layers = int(path_start.shape[1])
    frac = xp.zeros((n_rows, n_pieces), dtype=REAL)
    start = xp.zeros((n_rows, n_pieces, n_layers), dtype=REAL)
    end = xp.zeros((n_rows, n_pieces, n_layers), dtype=REAL)
    frac[owner, slot] = fraction
    start[owner, slot] = path_start
    end[owner, slot] = path_end
    return frac, start, end


def piece_mean_transmission(fraction, path_start, path_end, mu, xp=np):
    """Segment mean of ``exp(-tau)`` over padded linear pieces.

    ``fraction`` and ``path_*`` are row-selected outputs of
    :func:`segment_escape_pieces`; ``mu`` has shape ``(N, M, K)`` -- per row,
    per column (e.g. reflection), per layer. Returns ``(N, M)``, the
    fraction-weighted sum of :func:`mean_transmission` over each row's pieces.
    Validation: segment-escape-average
    """
    total = None
    for p in range(int(fraction.shape[1])):
        tau_start = xp.sum(path_start[:, None, p, :] * mu, axis=-1)
        tau_end = xp.sum(path_end[:, None, p, :] * mu, axis=-1)
        term = fraction[:, p, None] * mean_transmission(tau_start, tau_end, xp=xp)
        total = term if total is None else total + term
    return total


__all__ = [
    "mean_transmission",
    "piece_mean_transmission",
    "segment_escape_paths",
    "segment_escape_pieces",
]
