"""
plots.plotly.camera

Explicit 3D scene-camera control for the Plotly viewers: turn a human
azimuth/elevation/zoom (or a named preset) into the ``layout.scene.camera``
dict Plotly wants, so the view chosen in an app is also the view a snapshot or
an offscreen render uses.

The viewers need this because a *dragged* camera never reaches Python: marimo
renders every figure through its ``marimo-plotly`` component, whose relayout
handler forwards only ``dragmode`` and 2D ``xaxis``/``yaxis`` keys, so a 3D
``scene.camera`` relayout is dropped. Dragging still orbits the live scene; a
camera built here is the one that survives into the figure spec, and therefore
into the modebar PNG and into
:func:`pyrite.plots.plotly.render.render_reveal_animation`.

Pure geometry over plain dicts -- no Plotly import, no physics, so no
validation-ledger marker is needed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Plotly's own default 3D eye, and the same point in spherical terms. A viewer
# that has not overridden the camera is sitting exactly here, so this is the
# "default" preset's target for figures that ship no camera of their own.
DEFAULT_EYE: tuple[float, float, float] = (1.25, 1.25, 1.25)

# Elevation is clamped just short of the poles: at exactly +/-90 deg the eye is
# collinear with the default up vector (+z) and Plotly's view matrix degenerates
# (the scene flips or blanks). 89.9 deg is visually a top/bottom view.
_MAX_ELEVATION_DEG = 89.9

# Named viewpoints as (orbit_axis, azimuth_deg, elevation_deg). Axis presets
# look ALONG the named axis toward the origin -- "down z" is the top view. They
# are Cartesian, not lattice, directions: for the catalog's crystals c lies
# along z, so "down z" is the down-the-stacking-axis view, while a and b
# generally are not x/y. "down z" orbits about x so it lands ON the z axis
# rather than against the clamp a z-orbit would hit there.
CAMERA_PRESETS: dict[str, tuple[str, float, float]] = {
    "isometric": ("z", 45.0, 35.264389682754654),  # eye = (1, 1, 1) direction
    "down x": ("z", 0.0, 0.0),
    "down y": ("z", 90.0, 0.0),
    "down z": ("x", 90.0, 0.0),
}

# Orbit axis -> the two axes the azimuth sweeps through, in cyclic (x, y, z)
# order. Azimuth orbits ABOUT this axis and elevation lifts toward it, so the
# axis is also the camera's up vector: the horizon stays level against it.
_ORBIT_FRAMES: dict[str, tuple[str, str]] = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def camera_eye(azimuth_deg, elevation_deg, distance, orbit_axis="z"):
    """Plotly ``scene.camera.eye`` on the sphere of radius ``distance``.

    ``azimuth_deg`` orbits about ``orbit_axis`` and ``elevation_deg`` lifts
    toward it, clamped to +/-89.9 deg (see ``_MAX_ELEVATION_DEG``). Orbiting
    about ``"z"`` is the usual turntable; ``"x"`` or ``"y"`` re-poles the sphere
    so a sweep passes over the z pole instead of stopping short of it -- the
    clamp only ever blocks the axis you are orbiting about, so between the three
    axes every direction is reachable head-on.

    ``distance`` is in Plotly's normalized scene units, where its own default
    eye sits at 1.25*sqrt(3) ~= 2.17 from the origin. Note that distance frames
    a PERSPECTIVE scene only: an orthographic projection ignores it (see
    :func:`data_aspect_ratio`).
    """
    azimuth = np.radians(float(azimuth_deg))
    elevation = np.radians(
        float(np.clip(float(elevation_deg), -_MAX_ELEVATION_DEG, _MAX_ELEVATION_DEG))
    )
    radius = float(distance)
    horizontal = radius * np.cos(elevation)
    first, second = _ORBIT_FRAMES[orbit_axis]
    return {
        first: float(horizontal * np.cos(azimuth)),
        second: float(horizontal * np.sin(azimuth)),
        orbit_axis: float(radius * np.sin(elevation)),
    }


def eye_angles(eye: Sequence[float]) -> tuple[float, float, float]:
    """Inverse of :func:`camera_eye`: ``(azimuth_deg, elevation_deg, distance)``.

    Lets a figure that already carries a hand-tuned eye publish its view as
    angles, so a control seeded from it reproduces that exact camera instead of
    a rounded approximation of it.
    """
    x, y, z = (float(component) for component in eye)
    distance = float(np.linalg.norm([x, y, z]))
    if distance == 0.0:
        return 0.0, 0.0, 0.0
    return (
        float(np.degrees(np.arctan2(y, x))),
        float(np.degrees(np.arcsin(z / distance))),
        distance,
    )


def scene_camera(azimuth_deg, elevation_deg, distance, *, orbit_axis="z", orthographic=False):
    """Full ``layout.scene.camera`` dict for the given view.

    The up vector follows ``orbit_axis``, so the horizon stays level against
    whichever axis the view orbits about. ``orthographic`` selects the
    projection that removes perspective parallax, so parallel scene directions
    stay parallel at every depth; otherwise Plotly's default perspective
    projection is named explicitly, which keeps a toggle honest in both
    directions.
    """
    up = {"x": 0.0, "y": 0.0, "z": 0.0}
    up[orbit_axis] = 1.0
    return {
        "eye": camera_eye(azimuth_deg, elevation_deg, distance, orbit_axis),
        "up": up,
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "projection": {"type": "orthographic" if orthographic else "perspective"},
    }


def resolve_scene_camera(
    preset,
    azimuth_deg,
    elevation_deg,
    zoom=1.0,
    *,
    default_angles,
    orbit_axis="z",
    orthographic=False,
):
    """``scene.camera`` for a preset-or-custom control set.

    ``default_angles`` is the host figure's own ``(azimuth_deg, elevation_deg,
    distance)`` -- from :func:`eye_angles` where the figure hand-tunes its eye,
    or :data:`DEFAULT_ANGLES` where it leaves Plotly's default in place.
    ``preset`` picks the direction: ``"default"`` keeps the figure's own,
    ``"custom"`` takes ``azimuth_deg``/``elevation_deg`` about ``orbit_axis``,
    and any key of :data:`CAMERA_PRESETS` takes that named viewpoint (each
    carries the orbit axis that reaches it head-on). ``zoom`` scales the eye
    distance in every case (>1 moves closer), so it stays live under presets --
    but only a perspective scene reads that distance.
    """
    default_azimuth, default_elevation, default_distance = default_angles
    if preset == "custom":
        axis, azimuth, elevation = orbit_axis, float(azimuth_deg), float(elevation_deg)
    elif preset in CAMERA_PRESETS:
        axis, azimuth, elevation = CAMERA_PRESETS[preset]
    else:
        axis, azimuth, elevation = "z", default_azimuth, default_elevation
    return scene_camera(
        azimuth,
        elevation,
        default_distance / float(zoom),
        orbit_axis=axis,
        orthographic=orthographic,
    )


def scene_extents(fig):
    """Per-axis ``(min, max)`` over every 3D trace's coordinates in ``fig``.

    Falls back to ``(0.0, 1.0)`` on an axis no trace populates, so a figure with
    nothing plotted still yields a usable ratio instead of a NaN one.
    """
    extents = []
    for axis in ("x", "y", "z"):
        values = []
        for trace in fig.data:
            coords = getattr(trace, axis, None)
            if coords is None:
                continue
            finite = np.asarray(
                [c for c in coords if c is not None], dtype=float
            )  # None separates line segments
            finite = finite[np.isfinite(finite)]
            if finite.size:
                values.append((finite.min(), finite.max()))
        if values:
            extents.append((min(lo for lo, _ in values), max(hi for _, hi in values)))
        else:
            extents.append((0.0, 1.0))
    return extents


def data_aspect_ratio(fig, zoom=1.0):
    """``scene.aspectratio`` matching ``fig``'s data proportions, scaled by ``zoom``.

    This is how an ORTHOGRAPHIC scene zooms. Its projection box is fixed --
    plotly.js builds it as ``ortho(-aspect, aspect, -1, 1, near, far)``,
    independent of the camera -- so moving the eye does nothing at all, and
    Plotly's own scroll handler instead scales ``scene.aspectratio`` (its wheel
    listener multiplies the live ratio by 1.1 per notch under ``_ortho``).

    The returned ratio is the data extents normalized to a largest component of
    1, which reproduces ``aspectmode="data"`` proportions; ``zoom`` then scales
    all three together, which magnifies without distorting. Callers must set
    ``aspectmode="manual"`` alongside it, since any other mode makes Plotly
    recompute the ratio and discard this one.
    """
    spans = np.array([max(hi - lo, 0.0) for lo, hi in scene_extents(fig)], dtype=float)
    largest = spans.max() if spans.max() > 0.0 else 1.0
    spans = np.where(spans > 0.0, spans, largest)
    ratio = spans / largest * float(zoom)
    return {"x": float(ratio[0]), "y": float(ratio[1]), "z": float(ratio[2])}


# Spherical form of Plotly's default eye; the seed for figures with no camera.
DEFAULT_ANGLES: tuple[float, float, float] = eye_angles(DEFAULT_EYE)

__all__: Sequence[str] = [
    "CAMERA_PRESETS",
    "DEFAULT_ANGLES",
    "DEFAULT_EYE",
    "camera_eye",
    "data_aspect_ratio",
    "eye_angles",
    "resolve_scene_camera",
    "scene_camera",
    "scene_extents",
]
