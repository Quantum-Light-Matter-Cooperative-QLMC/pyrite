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
docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md.
Assumptions: profile invariant along y; laterally infinite slab; photons
travel straight along n_hat (incoherent Beer-Lambert transport -- no wave
optics, consistent with mc_spectrum).
"""

from dataclasses import dataclass

import numpy as np

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
