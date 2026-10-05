"""Numeric trajectory scene geometry shared by display and interchange backends.

Lengths remain in the caller's units. Sample coordinates use the entrance
origin; ``case_rotation`` maps them to lab axes without a translation.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .montecarlo.geometry import project_beam_entry, sample_to_lab_R
from .montecarlo.groove import GrooveSpec, surface_depth_ang

_IDENTITY_R = np.eye(3)
_GROOVE_MAX_PERIODS = 200


def case_rotation(case: Mapping[str, Any]) -> np.ndarray:
    """Sample -> lab rotation for this case's tilt (``v_lab = R @ v_sample``);
    see :func:`pyrite.montecarlo.geometry.sample_to_lab_R`."""
    return sample_to_lab_R(
        np.deg2rad(case.get("tilt_deg", 0.0)), np.deg2rad(case.get("tilt_azim_deg", 0.0))
    )


def rotate(points: ArrayLike, R: np.ndarray) -> np.ndarray:
    """Apply the sample -> lab rotation ``R`` (``v_lab = R @ v_sample``) to an
    ``(N, 3)`` array of points/vectors. NaN rows (segment separators) stay NaN."""
    return np.asarray(points, dtype=float) @ R.T


def groove_spec(case: Mapping[str, Any]) -> GrooveSpec | None:
    """Blazed :class:`~pyrite.montecarlo.groove.GrooveSpec` for a case carrying a
    ``groove_spacing_ang`` knob, or ``None`` when it is absent.

    Mirrors ``montecarlo.runner._transport_case``'s spec construction exactly
    (``blazed_groove_spec(spacing, theta_obs_rad, tilt_polar_rad,
    tilt_azim_rad)``) so the penetration figures transport electrons through the
    SAME relief facets the spectrum runner does -- no new physics, no new
    convention. ``blazed_groove_spec`` validates the restricted geometry
    (theta_obs = 90 deg, tilt_azim = 180 deg, 0 < tilt_polar < 90 deg) and
    raises ``ValueError`` otherwise; callers that build cases via
    ``sweep.build_cases`` never hit that because it rejects the same geometries
    up front."""
    spacing = case.get("groove_spacing_ang")
    if spacing is None:
        return None
    from .montecarlo.groove import blazed_groove_spec

    return blazed_groove_spec(
        spacing,
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )


def groove_profile_knots(
    x_lo_ang: float, x_hi_ang: float, spec: GrooveSpec, *, max_periods: int = 200
) -> tuple[np.ndarray, np.ndarray] | None:
    """Minimal sample-frame sawtooth vertices ``(x_ang, z_ang)`` covering
    ``[x_lo_ang, x_hi_ang]``: two knots per period (apex at ``z = 0``, valley
    floor at ``z = depth``), so a corrugated surface needs only ~2 vertices per
    groove instead of a dense sweep.

    Returns ``None`` when the requested span exceeds ``max_periods`` grooves --
    the caller then falls back to the flat entrance face (drawing thousands of
    teeth is neither legible nor cheap). ``x`` is returned strictly increasing so
    the vertices trace the profile directly as a polyline."""
    lam = spec.spacing_ang
    k0 = int(np.floor(x_lo_ang / lam))
    k1 = int(np.ceil(x_hi_ang / lam))
    if k1 - k0 > max_periods:
        return None
    x_valley = spec.depth_ang * np.tan(spec.tilt_polar_rad)
    xs = np.empty(2 * (k1 - k0 + 1))
    ks = np.arange(k0, k1 + 1)
    xs[0::2] = ks * lam  # apexes (z = 0)
    xs[1::2] = ks * lam + x_valley  # valley floors (z = depth)
    zs = np.asarray(surface_depth_ang(xs, spec))
    return xs, zs


def crystal_mesh(
    lox: float, hix: float, loy: float, hiy: float, thick: float, *, R: np.ndarray = _IDENTITY_R
) -> tuple[np.ndarray, np.ndarray]:
    """Return rotated box vertices and two triangles per face in caller units."""
    corners = rotate(
        np.array(
            [
                [lox, loy, 0.0],
                [hix, loy, 0.0],
                [hix, hiy, 0.0],
                [lox, hiy, 0.0],
                [lox, loy, thick],
                [hix, loy, thick],
                [hix, hiy, thick],
                [lox, hiy, thick],
            ]
        ),
        R,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return corners, triangles


def groove_mesh(
    case: Mapping[str, Any],
    lox: float,
    hix: float,
    loy: float,
    hiy: float,
    u: float,
    *,
    R: np.ndarray = _IDENTITY_R,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Corrugated blazed-groove entrance surface as a ``Mesh3d`` ribbon, or
    ``None`` when the case is ungrooved or the extent spans too many teeth.

    The sawtooth is invariant along y (the grooves run along y) and periodic in
    the sample-frame lateral coordinate x, so the ribbon is the profile
    ``groove_profile_z(x)`` (in display units) extruded across the displayed y
    span ``[loy, hiy]``. Apexes sit at ``x = k * spacing`` (``z = 0``); valleys
    reach the groove depth. Every vertex is rotated through the sample -> lab
    rotation ``R`` (see :func:`case_rotation`) exactly like the crystal mesh and the
    tracks, so on a tilted slab the corrugation reads correctly rather than
    face-on.

    Falls back to ``None`` (leaving the flat crystal entrance face already drawn
    by :func:`crystal_mesh`) when the displayed lateral extent would need more
    than :data:`_GROOVE_MAX_PERIODS` teeth -- e.g. a realistic mm-scale footprint
    over a micron-scale spacing.
    """
    spec = groove_spec(case)
    if spec is None:
        return None
    knots = groove_profile_knots(lox * u, hix * u, spec, max_periods=_GROOVE_MAX_PERIODS)
    if knots is None:
        return None
    xs_ang, zs_ang = knots
    xs = xs_ang / u  # sample-frame display units, same as lox/hix
    zs = zs_ang / u
    n = len(xs)
    # Two y-rows (front loy, back hiy) of the same x/z profile; triangulate each
    # x-interval into the quad (front_i, front_i+1, back_i+1, back_i).
    verts = rotate(
        np.column_stack(
            (
                np.concatenate((xs, xs)),
                np.concatenate((np.full(n, loy), np.full(n, hiy))),
                np.concatenate((zs, zs)),
            )
        ),
        R,
    )
    a = np.arange(n - 1)  # front-row left vertices
    i = np.concatenate((a, a))
    j = np.concatenate((a + 1, a + 1 + n))
    k = np.concatenate((a + 1 + n, a + n))
    return verts, np.column_stack((i, j, k))


def crystal_footprint_extent(
    case: Mapping[str, Any], u: float
) -> tuple[float, float, float, float]:
    """True lateral crystal extent ``(lox, hix, loy, hiy)`` in display units.

    Uses the case's finite footprint (``crystal_width_mm`` x ``crystal_height_mm``,
    full dimensions centred on the transverse origin); a missing/None dimension
    falls back to the 5 mm default. 1 mm = 1e7 Ang.
    """
    width_mm = case.get("crystal_width_mm") or 5.0
    height_mm = case.get("crystal_height_mm") or 5.0
    hx = 0.5 * float(width_mm) * 1e7 / u
    hy = 0.5 * float(height_mm) * 1e7 / u
    return -hx, hx, -hy, hy


def beam_footprint_outline(
    case: Mapping[str, Any],
    u: float,
    fwhm_mm: float | None,
    *,
    R: np.ndarray = _IDENTITY_R,
    n_points: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Closed ``(x, y, z)`` outline [display units] of the collimated beam spot
    where it strikes the tilted entrance face ``z = 0`` (SAMPLE frame), for the
    realistic-scale view, rotated into the lab frame by ``R``.

    The lab beam is a round spot travelling along ``+z_lab``; on a face tilted by
    (``tilt_deg``, ``tilt_azim_deg``) its footprint elongates by ``1 / cos(tilt)``
    along the azimuth -- the grazing-incidence stretch that can outgrow the finite
    crystal. Reuse :func:`pyrite.montecarlo.geometry.project_beam_entry` so the
    drawn outline matches transport's own entry mapping exactly. Returns ``None``
    for the legacy point beam (``fwhm_mm`` falsy / absent).

    ``fwhm_mm`` is the ALREADY-RESOLVED lab-plane spot FWHM (the caller's
    ``beam_fwhm_mm`` override or the case's own default -- resolved once by the
    caller so this draws exactly the spot that was transported, never re-reading
    ``case`` itself). Draw the outline at the FWHM contour (lab radius =
    ``fwhm_mm / 2``, mm -> Ang factor 1e7). Sample a ring of ``n_points`` lab
    offsets ``(u_i, v_i)`` on that circle, project them, and divide the
    resulting sample-frame (x, y) by ``u`` for display units before rotating.
    """
    if not fwhm_mm:
        return None
    radius_ang = 0.5 * float(fwhm_mm) * 1e7  # FWHM-contour radius, mm -> Ang
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)  # closed loop
    offsets_uv = radius_ang * np.column_stack((np.cos(phi), np.sin(phi)))
    entry = project_beam_entry(
        offsets_uv,
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    pts = rotate(np.column_stack((entry[:, 0] / u, entry[:, 1] / u, np.zeros(len(entry)))), R)
    return pts[:, 0], pts[:, 1], pts[:, 2]
