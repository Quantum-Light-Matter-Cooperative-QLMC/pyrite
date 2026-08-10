"""
montecarlo.groove

Blazed sawtooth groove profile on the crystal beam-entrance face, for
escape-path engineering of the coherent line yield. Restricted geometry:
theta_obs = 90 deg, tilt_azim = 180 deg, 0 < tilt_polar < 90 deg. In the
sample frame (entrance face z = 0, depth +z) this puts

    beam  b = ( sin tp, 0,  cos tp )
    n_hat n = ( cos tp, 0, -sin tp ),     b . n = 0.

Grooves run along y with period ``spacing_ang`` (apexes at x = k*spacing,
z = 0). The WORKING facet is the plane family ``n . r = k*spacing*cos(tp)``
(perpendicular to the observation direction, parallel to the beam); the
RELIEF facet is ``b . r = k*spacing*sin(tp)`` (perpendicular to the beam,
parallel to the exit rays -- zero shadowing by construction). Closing the
unit cell fixes the groove depth

    h = spacing * sin(tp) * cos(tp).

Source: elementary ray-plane intersection on a periodic sawtooth (no
literature equation); facet-choice rationale in
docs/validation/geometry/blazed-groove-geometry.md.
Assumptions: profile invariant along y; laterally infinite slab; photons
travel straight along n_hat (incoherent Beer-Lambert transport -- no wave
optics, consistent with mc_spectrum).

For arbitrary scattered-electron direction d, later facet crossings use
``r(s) = p + s*d`` substituted into both plane families above. A candidate is
physical only when ``s > 0`` and its depth lies in ``[0, h]``; a two-sided
surface-predicate check distinguishes material-to-vacuum exits from
vacuum-to-material re-entries. Exact event handling is specified in
docs/validation/geometry/blazed-groove-geometry.md and is
implemented below.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

_THETA_TOL = 1e-9


@dataclass(frozen=True)
class GrooveSpec:
    """Blazed sawtooth groove profile (validated; build via blazed_groove_spec)."""

    spacing_ang: float
    depth_ang: float
    tilt_polar_rad: float


def blazed_groove_spec(spacing_ang, theta_obs_rad, tilt_polar_rad, tilt_azim_rad):
    """
    Construct the blazed GrooveSpec for the restricted geometry, deriving the
    depth h = spacing*sin(tp)*cos(tp) that closes the sawtooth unit cell
    (working facet perpendicular to n_hat, relief facet perpendicular to the
    beam -- see the module docstring).

    Limiting case: tilt_polar_rad -> 0 or 90 deg degenerates the sawtooth
    (h -> 0) AND breaks the entrance/exit face assignment, so both are
    rejected rather than silently producing a flat profile.

    Validation: blazed-groove-geometry
    """
    if not np.isclose(theta_obs_rad, np.pi / 2, atol=_THETA_TOL):
        raise ValueError(
            "blazed grooves require theta_obs = 90 deg exactly (beam "
            "perpendicular to observation; relief facet parallel to exit rays)"
        )
    if not np.isclose(tilt_azim_rad, np.pi, atol=_THETA_TOL):
        raise ValueError("blazed grooves require tilt_azim = 180 deg")
    tp = float(tilt_polar_rad)
    if not 0.0 < tp < np.pi / 2:
        raise ValueError("blazed grooves require 0 < tilt_polar < 90 deg")
    lam = float(spacing_ang)
    if lam <= 0.0:
        raise ValueError("groove spacing_ang must be positive")
    return GrooveSpec(
        spacing_ang=lam,
        depth_ang=lam * np.sin(tp) * np.cos(tp),
        tilt_polar_rad=tp,
    )


def surface_depth_ang(x: float | np.ndarray, spec: GrooveSpec) -> float | np.ndarray:
    """
    Sawtooth entrance-surface depth [Ang] at lateral coordinate ``x`` [Ang].

    Derivation: reduce x modulo the period, then solve the working-facet plane
    for z before the valley and the relief-facet plane for z after it.
    Assumptions: exact periodic profile, invariant along y, with facet geometry
    defined in the module docstring. Limiting case: at each periodic apex the
    depth is zero; both facet branches meet at ``spec.depth_ang``.

    Validation: blazed-groove-geometry
    """
    tp = spec.tilt_polar_rad
    u = np.mod(x, spec.spacing_ang)
    x_valley = spec.depth_ang * np.tan(tp)
    return np.where(
        u <= x_valley,
        u / np.tan(tp),
        (spec.spacing_ang - u) * np.tan(tp),
    )


def in_material(
    position: np.ndarray,
    thickness_ang: float,
    spec: GrooveSpec,
    width_ang: float | None = None,
    height_ang: float | None = None,
) -> np.bool_ | np.ndarray:
    """
    Test whether sample-frame positions lie inside grooved slab material.

    Source: intersection of sawtooth half-space ``z >= surface_depth_ang(x)``,
    back face ``z <= thickness_ang``, and optional centered x/y footprint.
    Assumptions: boundaries belong to material and slab back face is planar.
    Limiting case: omitted width/height gives laterally infinite periodic slab.

    Validation: blazed-groove-geometry
    """
    p = np.asarray(position, dtype=float)
    inside = (p[..., 2] >= surface_depth_ang(p[..., 0], spec)) & (p[..., 2] <= thickness_ang)
    if width_ang is not None:
        inside &= np.abs(p[..., 0]) <= width_ang / 2
    if height_ang is not None:
        inside &= np.abs(p[..., 1]) <= height_ang / 2
    return inside


@njit(cache=True)
def _surface_depth_scalar_numba(x, spacing, depth, st, ct):
    """Scalar sawtooth surface depth for compiled transport geometry."""
    u = np.mod(x, spacing)
    tan_tp = st / ct
    x_valley = depth * tan_tp
    if u <= x_valley:
        return u / tan_tp
    return (spacing - u) * tan_tp


@njit(cache=True)
def _groove_in_material_scalar_numba(x, z, thickness, spacing, depth, st, ct):
    """Scalar material predicate used by the compiled facet-event search."""
    return z >= _surface_depth_scalar_numba(x, spacing, depth, st, ct) and z <= thickness


@njit(cache=True)
def _first_surface_event_scalar_numba(
    px,
    pz,
    dx,
    dz,
    spacing,
    depth,
    st,
    ct,
    transition_code,
):
    """Compiled scalar equivalent of :func:`first_surface_event`.

    ``transition_code`` is 0 for either transition, 1 for material->vacuum
    (exit), and 2 for vacuum->material (entry).  The candidate construction,
    closed facet-depth band, geometry-scaled tolerances, and two-sided material
    predicate intentionally mirror the validated Python implementation.
    """
    machine_eps = 2.220446049250313e-16
    eps = max(32.0 * machine_eps * spacing, 1.0e-12 * spacing)
    band_tol = 32.0 * machine_eps * max(spacing, depth)

    if dz == 0.0:
        if pz < 0.0 or pz > depth:
            return np.inf
        band_start = eps
        band_end = np.inf
    else:
        s_at_zero = -pz / dz
        s_at_depth = (depth - pz) / dz
        band_start = max(eps, min(s_at_zero, s_at_depth))
        band_end = max(s_at_zero, s_at_depth)
        if band_end <= eps:
            return np.inf

    best = np.inf

    # family 0: working facets n=(ct,0,-st), spacing*ct, exit sign +1
    # family 1: relief facets  b=(st,0, ct), spacing*st, exit sign -1
    for family in range(2):
        if family == 0:
            nx = ct
            nz = -st
            plane_spacing = spacing * ct
            exit_rate_sign = 1.0
        else:
            nx = st
            nz = ct
            plane_spacing = spacing * st
            exit_rate_sign = -1.0

        rate = nx * dx + nz * dz
        if rate == 0.0:
            continue
        origin = nx * px + nz * pz
        transformed = (origin + band_start * rate) / plane_spacing
        floor_index = int(np.floor(transformed))
        ceil_index = int(np.ceil(transformed))

        # The validated implementation checks +/- 1 around both floor and ceil.
        for base_choice in range(2):
            base = floor_index if base_choice == 0 else ceil_index
            for offset in range(-1, 2):
                period_index = base + offset
                s = (period_index * plane_spacing - origin) / rate
                if s <= eps or s > band_end or s >= best:
                    continue

                qx = px + s * dx
                qz = pz + s * dz
                if qz < -band_tol or qz > depth + band_tol:
                    continue

                before_x = qx - eps * dx
                before_z = qz - eps * dz
                after_x = qx + eps * dx
                after_z = qz + eps * dz

                before = _groove_in_material_scalar_numba(
                    before_x, before_z, np.inf, spacing, depth, st, ct
                )
                after = _groove_in_material_scalar_numba(
                    after_x, after_z, np.inf, spacing, depth, st, ct
                )

                is_exit = before and not after
                is_entry = (not before) and after
                if before == after and 0.0 < qz < depth:
                    signed_rate = exit_rate_sign * rate
                    is_exit = signed_rate > 0.0
                    is_entry = signed_rate < 0.0

                accept = False
                if transition_code == 0:
                    accept = is_exit or is_entry
                elif transition_code == 1:
                    accept = is_exit
                else:  # transition_code == 2
                    accept = is_entry

                if accept:
                    best = s

    return best


def first_surface_event(
    position: np.ndarray,
    direction: np.ndarray,
    spec: GrooveSpec,
    transition: str | None = None,
) -> float:
    """Return nearest strictly forward exact sawtooth crossing distance [Ang].

    The public wrapper retains validation and ``GrooveSpec`` handling; the hot
    scalar ray/facet search is implemented by a Numba helper so transport can
    call the exact same geometry without leaving compiled code.

    ``transition`` may select material-to-vacuum ``"exit"`` or
    vacuum-to-material ``"entry"``; ``None`` accepts either transition.

    Validation: blazed-groove-geometry
    """
    if transition not in (None, "exit", "entry"):
        raise ValueError("transition must be None, 'exit', or 'entry'")

    p = np.asarray(position, dtype=float)
    d = np.asarray(direction, dtype=float)
    if p.shape != (3,) or d.shape != (3,):
        raise ValueError("position and direction must be three-vectors")

    code = 0 if transition is None else (1 if transition == "exit" else 2)
    tp = spec.tilt_polar_rad
    st = np.sin(tp)
    ct = np.cos(tp)
    return float(
        _first_surface_event_scalar_numba(
            p[0],
            p[2],
            d[0],
            d[2],
            float(spec.spacing_ang),
            float(spec.depth_ang),
            float(st),
            float(ct),
            code,
        )
    )


def escape_distance_ang(x, z, spec):
    """
    Straight-line path length [Ang] inside the grooved material from emission
    point (x, z) to the surface, along the observation direction
    n = (cos tp, 0, -sin tp).

    Derivation: exit rays are parallel to the relief facets, so a ray only
    ever crosses working-facet planes ``n . r = k*spacing*cos(tp)``.
    Successive plane crossings are spaced ``c = spacing*cos(tp)`` in path and
    exactly ``h`` in depth, so exactly one crossing depth lands in the
    physical facet band [0, h) -- crossings deeper than h are interior points
    below the valley floor. Hence, with d = n . (x, 0, z):

        s1 = c - mod(d, c);  z1 = z - s1*sin(tp)
        L  = s1 + max(floor(z1/h), 0) * c

    numpy ufuncs only, so cupy arrays dispatch through __array_ufunc__.
    Limiting cases: h -> 0 at fixed z gives L -> z/sin(tp), the flat
    entrance-face path of mc_spectrum's ``z_mid / (-n_hat[2])`` branch; a
    point just inside its own working facet gives L -> 0.

    This first working-facet crossing is the complete material path, not a
    first-exit approximation: ray direction n is parallel to every relief
    facet and points outward through the working-facet family, so it cannot
    cross from vacuum back into material after escape. Arbitrary electron
    directions do not share this property and require explicit later-facet
    re-entry handling.

    Validation: blazed-groove-geometry
    """
    tp = spec.tilt_polar_rad
    st, ct = np.sin(tp), np.cos(tp)
    c = spec.spacing_ang * ct
    s1 = c - np.mod(x * ct - z * st, c)
    z1 = z - s1 * st
    m = np.maximum(np.floor(z1 / spec.depth_ang), 0.0)
    return s1 + m * c


def entry_points(x0, spec):
    """
    Electron entry point on the relief facet for a collimated ray whose
    flat-face z=0 intersection is (x0, y0, 0).

    Derivation: at z = 0 the material is only the measure-zero apex lines, so
    every ray continues along b = (sin tp, 0, cos tp) into the groove cut and
    enters through the first relief-facet plane ``b . r = k*spacing*sin(tp)``
    (rays are parallel to the working facets and never cross them). The same
    one-crossing-in-band argument as escape_distance_ang applies with plane
    spacing ``spacing*sin(tp)`` and depth step h:

        s1 = mod(-x0*sin(tp), spacing*sin(tp))
        entry = (x0 + s1*sin(tp), y0, s1*cos(tp)),  z_entry in [0, h)

    Limiting case: spacing -> 0 (h -> 0) gives z_entry -> 0, the flat face.

    Validation: blazed-groove-geometry

    Returns (x_entry, z_entry), shaped like x0 (y is untouched by the
    profile and handled by the caller).
    """
    tp = spec.tilt_polar_rad
    st, ct = np.sin(tp), np.cos(tp)
    s1 = np.mod(-x0 * st, spec.spacing_ang * st)
    return x0 + s1 * st, s1 * ct
