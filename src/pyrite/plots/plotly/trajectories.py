"""Browser-native 3D electron-penetration rendering with Plotly.

The transport and coordinate preparation remain owned by
:mod:`pyrite.plots.mpl.trajectories`; this module only builds an interactive view of
the true sample-frame segment endpoints.  It is intentionally imported directly
rather than re-exported from :mod:`pyrite.plots`, whose legacy export set is
frozen.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ...montecarlo.geometry import project_beam_entry, sample_to_lab_R
from ..mpl.trajectories import (
    _case_of,
    _groove_spec,
    _trajectory_data,
    groove_profile_knots,
)

_CRYSTAL = "#B9D9EB"
_CRYSTAL_EDGE = "#EDF6F9"
_BEAM = "#E76F51"
_DETECTOR = "#42C7C7"
_GROOVE = "#E9C46A"
_FIELD = "#17202A"
_GRID = "#34495E"

# Cap on how many groove periods the corrugated entrance surface will draw
# before falling back to the flat face (see _groove_surface_mesh): a realistic
# mm-scale footprint over a micron-scale spacing spans thousands of teeth, which
# is neither legible nor cheap as a mesh.
_GROOVE_MAX_PERIODS = 200

# Zoomed (non-realistic) view spot FWHM [mm]: a micron-scale beam so the incident
# bundle has visible width at the fitted, sub-micron cascade scale.
_ZOOM_BEAM_FWHM_MM = 1.0e-3

_IDENTITY_R = np.eye(3)

# Zoomed-view camera eye. Direction matches the long-standing hand-tuned angle
# (-0.8, 1.5, -0.4); norm is rescaled to 1.25 (down from that vector's own
# ~1.75) so the fitted scene box from `_zoom_scene_ranges` fills the frame and
# decoration traces do not inflate Plotly's default range. Plotly's own default
# eye norm is 1.25*sqrt(3) ~= 2.17, for scale.
_CAMERA_EYE_DIRECTION = np.array([-0.8, 1.5, -0.4])
_CAMERA_EYE = tuple((_CAMERA_EYE_DIRECTION / np.linalg.norm(_CAMERA_EYE_DIRECTION) * 1.25).tolist())


def _case_R(case):
    """Sample -> lab rotation for this case's tilt (``v_lab = R @ v_sample``);
    see :func:`pyrite.montecarlo.geometry.sample_to_lab_R`."""
    return sample_to_lab_R(
        np.deg2rad(case.get("tilt_deg", 0.0)), np.deg2rad(case.get("tilt_azim_deg", 0.0))
    )


def _rotate(points, R):
    """Apply the sample -> lab rotation ``R`` (``v_lab = R @ v_sample``) to an
    ``(N, 3)`` array of points/vectors. NaN rows (segment separators) stay NaN."""
    return np.asarray(points, dtype=float) @ R.T


def track_vertices_3d(data, *, t_fs=None, reveal_until_fs=None):
    """Return NaN-separated physical segments for one Plotly ``Scatter3d``.

    Each transport segment contributes ``start, end, NaN``.  Separators prevent
    Plotly from joining different segments or electrons.  Energy and electron ID
    repeat at both endpoints, preserving segment-wise color and hover metadata.

    ``t_fs`` (per-segment START age, fs, same order as ``start_xyz``/``end_xyz``)
    and ``reveal_until_fs`` together implement playback: when BOTH are given, a
    segment contributes only if its start age ``t_fs <= reveal_until_fs`` -- whole
    segments are kept or dropped, never sub-segment-interpolated. Either left
    ``None`` (the default) reproduces today's full-reveal behaviour unchanged.
    """
    start = np.asarray(data["start_xyz"], dtype=float)
    end = np.asarray(data["end_xyz"], dtype=float)
    energy = np.asarray(data["E"], dtype=float)
    elec_id = np.asarray(data["elec_id"], dtype=int)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("start_xyz and end_xyz must both have shape (N, 3)")
    if len(start) != len(energy) or len(start) != len(elec_id):
        raise ValueError("3D segment, energy, and electron arrays must have equal length")
    if t_fs is not None:
        t_fs = np.asarray(t_fs, dtype=float)
        if len(t_fs) != len(start):
            raise ValueError("t_fs must have the same length as the segment arrays")
        if reveal_until_fs is not None:
            keep = t_fs <= reveal_until_fs
            start, end, energy, elec_id = start[keep], end[keep], energy[keep], elec_id[keep]

    xyz = np.full((3 * len(start), 3), np.nan)
    xyz[0::3] = start
    xyz[1::3] = end
    colors = np.full(3 * len(start), np.nan)
    colors[0::3] = energy
    colors[1::3] = energy
    ids = np.full(3 * len(start), np.nan)
    ids[0::3] = elec_id
    ids[1::3] = elec_id
    return xyz, colors, ids


def _tracks_trace(case, data, unit, *, R=_IDENTITY_R, reveal_until_fs=None):
    """Electron-track ``Scatter3d`` at one reveal cutoff (see
    :func:`track_vertices_3d`). Shared by the full-reveal figure
    (:func:`trajectory_volume_figure_from_data`) and by every animation frame
    -- the reveal-until-fs prerendered animation (see
    :mod:`pyrite.plots.plotly.render`) calls this once per frame with a
    different ``reveal_until_fs``, so the vertex-filtering logic lives in
    exactly one place.

    ``R`` (default identity) rotates the plotted ``x``/``y``/``z`` from the
    sample frame into the lab frame (see :func:`_case_R`); the hover ``depth``
    stays the SAMPLE-frame ``z`` (true penetration depth) regardless of ``R``."""
    xyz, energy, elec_id = track_vertices_3d(
        data, t_fs=data["t_fs"], reveal_until_fs=reveal_until_fs
    )
    depth = xyz[:, 2]
    xyz = _rotate(xyz, R)
    custom = np.column_stack((energy, elec_id, depth))
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines",
        line={
            "color": energy,
            "colorscale": "Turbo",
            "cmin": 0.0,
            "cmax": float(case["E0_keV"]),
            "width": 4,
            "colorbar": {
                "title": {"text": "electron<br>energy (keV)"},
                "thickness": 14,
                "len": 0.62,
                "x": 1.02,
            },
        },
        customdata=custom,
        hovertemplate=(
            "electron %{customdata[1]:.0f}<br>"
            "energy %{customdata[0]:.3g} keV<br>"
            f"depth %{{customdata[2]:.4g}} {unit}<extra></extra>"
        ),
        name="electron tracks",
    )


def vacuum_legs_trace(
    data,
    *,
    reveal_until_fs=None,
    R=_IDENTITY_R,
    unit="",
    cmax=None,
):
    """Faint non-radiating groove-gap flights at one reveal cutoff."""
    start = np.asarray(data.get("vacuum_start_xyz", np.empty((0, 3))), dtype=float)
    end = np.asarray(data.get("vacuum_end_xyz", np.empty((0, 3))), dtype=float)
    energy = np.asarray(data.get("vacuum_E", np.empty(0)), dtype=float)
    t_fs = np.asarray(data.get("vacuum_t_fs", np.empty(0)), dtype=float)
    elec_id = np.asarray(data.get("vacuum_elec_id", np.empty(0)), dtype=np.int64)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("vacuum_start_xyz and vacuum_end_xyz must both have shape (N, 3)")
    if not (len(start) == len(energy) == len(t_fs) == len(elec_id)):
        raise ValueError("vacuum segment, energy, age, and electron arrays must have equal length")
    if reveal_until_fs is not None:
        keep = t_fs <= reveal_until_fs
        start, end, energy, t_fs, elec_id = (
            start[keep],
            end[keep],
            energy[keep],
            t_fs[keep],
            elec_id[keep],
        )

    xyz = np.full((3 * len(start), 3), np.nan)
    xyz[0::3], xyz[1::3] = start, end
    vertex_energy = np.full(3 * len(start), np.nan)
    vertex_energy[0::3], vertex_energy[1::3] = energy, energy
    vertex_time = np.full(3 * len(start), np.nan)
    vertex_time[0::3], vertex_time[1::3] = t_fs, t_fs
    vertex_id = np.full(3 * len(start), np.nan)
    vertex_id[0::3], vertex_id[1::3] = elec_id, elec_id
    xyz = _rotate(xyz, R)
    custom = np.column_stack((vertex_energy, vertex_time, vertex_id))
    color_max = float(cmax) if cmax is not None else (float(np.max(energy)) if energy.size else 1.0)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines",
        line={
            "color": vertex_energy,
            "colorscale": "Turbo",
            "cmin": 0.0,
            "cmax": color_max,
            "width": 3,
            "showscale": False,
        },
        customdata=custom,
        hovertemplate=(
            "electron %{customdata[2]:.0f}<br>"
            "energy %{customdata[0]:.3g} keV<br>"
            "start age %{customdata[1]:.3g} fs<br>"
            f"vacuum leg ({unit})<extra></extra>"
        ),
        name="vacuum legs",
        opacity=0.4,
    )


def _transverse_window(data):
    """Robust square x/y display window fitted to tracks, not crystal footprint."""
    points = np.vstack((data["start_xyz"], data["end_xyz"]))
    transverse = points[:, :2]
    transverse = transverse[np.isfinite(transverse).all(axis=1)]
    thick = float(data["thick"])
    radius = float(np.percentile(np.abs(transverse), 99.0)) if transverse.size else 0.0
    radius = max(1.25 * radius, 0.18 * thick, np.finfo(float).eps)
    return -radius, radius


def _display_extent(case, data, *, realistic):
    """Lateral/depth display window + axis unit for one dataset: ``(lox, hix,
    loy, hiy, thick, unit, span)``.

    Every trace builder for a given ``(case, data, realistic)`` -- the static
    box/planes/arrows AND the reveal-dependent tracks/exit-path traces built
    per animation frame -- must agree on the SAME window, so it is computed
    once here rather than re-derived ad hoc in each caller."""
    lo, hi = _transverse_window(data)
    thick = float(data["thick"])
    unit = "µm" if data["u"] == 1e4 else "nm"
    if realistic:
        lox, hix, loy, hiy = _crystal_footprint_extent(case, data["u"])
    else:
        lox = loy = lo
        hix = hiy = hi
    span = max(hix - lox, hiy - loy, thick)
    return lox, hix, loy, hiy, thick, unit, span


def _zoom_scene_ranges(lox, hix, loy, hiy, thick, span, R):
    """Padded lab-frame axis ranges ``(x_range, y_range, z_range)`` fitted to
    the display box, for the zoomed (``realistic=False``) scene.

    The display box ``[lox, hix] x [loy, hiy] x [0, thick]`` is defined in the
    SAMPLE frame; a tilted case rotates it into the lab frame via ``R``, so a
    tilt can swing a corner outside an axis-aligned range computed from the
    unrotated box alone. Ranges must instead be the min/max over all 8
    ROTATED corners, or a tilted slab clips against its own axis range.
    Padding: 8% of span on x/y, 15% on z (small upstream/downstream slivers
    so the beam-entry stub and exit face read past the box edge)."""
    corners = _rotate(
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
    xmin, ymin, zmin = corners.min(axis=0)
    xmax, ymax, zmax = corners.max(axis=0)
    xy_pad = 0.08 * span
    z_pad = 0.15 * span
    return (
        (xmin - xy_pad, xmax + xy_pad),
        (ymin - xy_pad, ymax + xy_pad),
        (zmin - z_pad, zmax + z_pad),
    )


def _crystal_mesh(lox, hix, loy, hiy, thick, *, R=_IDENTITY_R):
    corners = _rotate(
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
    x, y, z = corners[:, 0].tolist(), corners[:, 1].tolist(), corners[:, 2].tolist()
    # Two triangles per face.  Slight transparency keeps internal tracks legible.
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=_CRYSTAL,
        opacity=0.16,
        flatshading=True,
        hoverinfo="skip",
        name="crystal volume",
        showlegend=True,
    )


def _groove_surface_mesh(case, lox, hix, loy, hiy, u, *, R=_IDENTITY_R):
    """Corrugated blazed-groove entrance surface as a ``Mesh3d`` ribbon, or
    ``None`` when the case is ungrooved or the extent spans too many teeth.

    The sawtooth is invariant along y (the grooves run along y) and periodic in
    the sample-frame lateral coordinate x, so the ribbon is the profile
    ``groove_profile_z(x)`` (in display units) extruded across the displayed y
    span ``[loy, hiy]``. Apexes sit at ``x = k * spacing`` (``z = 0``); valleys
    reach the groove depth. Every vertex is rotated through the sample -> lab
    rotation ``R`` (see :func:`_case_R`) exactly like the crystal mesh and the
    tracks, so on a tilted slab the corrugation reads correctly rather than
    face-on.

    Falls back to ``None`` (leaving the flat crystal entrance face already drawn
    by :func:`_crystal_mesh`) when the displayed lateral extent would need more
    than :data:`_GROOVE_MAX_PERIODS` teeth -- e.g. a realistic mm-scale footprint
    over a micron-scale spacing.
    """
    spec = _groove_spec(case)
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
    verts = _rotate(
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
    return go.Mesh3d(
        x=verts[:, 0].tolist(),
        y=verts[:, 1].tolist(),
        z=verts[:, 2].tolist(),
        i=i.tolist(),
        j=j.tolist(),
        k=k.tolist(),
        color=_GROOVE,
        opacity=0.55,
        flatshading=True,
        hoverinfo="skip",
        name="groove profile",
        showlegend=True,
    )


def _plane_outline(z, lox, hix, loy, hiy, *, R=_IDENTITY_R, name, dash="solid"):
    pts = _rotate(
        np.array([[lox, loy, z], [hix, loy, z], [hix, hiy, z], [lox, hiy, z], [lox, loy, z]]),
        R,
    )
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="lines",
        line={"color": _CRYSTAL_EDGE, "width": 3, "dash": dash},
        hovertemplate=f"{name}<br>depth={z:.4g}<extra></extra>",
        name=name,
        showlegend=False,
    )


def _direction_arrow(direction, length, *, color, name, reverse=False):
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    start = -length * unit if reverse else np.zeros(3)
    end = np.zeros(3) if reverse else length * unit
    line = go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines",
        line={"color": color, "width": 8},
        hoverinfo="skip",
        name=name,
    )
    cone = go.Cone(
        x=[end[0]],
        y=[end[1]],
        z=[end[2]],
        u=[unit[0]],
        v=[unit[1]],
        w=[unit[2]],
        anchor="tip",
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode="absolute",
        sizeref=0.18 * length,
        hoverinfo="skip",
        name=name,
        showlegend=False,
    )
    return line, cone


def _crystal_footprint_extent(case, u):
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


def _beam_footprint_outline(case, u, fwhm_mm, *, R=_IDENTITY_R, n_points=96):
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
    pts = _rotate(np.column_stack((entry[:, 0] / u, entry[:, 1] / u, np.zeros(len(entry)))), R)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def _incident_beam_lines(data, length, *, R=_IDENTITY_R):
    """NaN-separated incident-beam segments -- one short stub per electron drawn
    UPSTREAM along the beam direction into its true entry point on the ``z = 0``
    face, so the beam reads as a bundle of incoming particles rather than a single
    arrow. Entry points are each electron's first-segment start; electrons whose
    Gaussian draw missed a finite crystal are already absent from ``data`` (dropped
    as ``n_missed`` by transport), so this never draws a particle that missed.
    ``R`` (default identity) rotates the plotted points into the lab frame.

    Each stub runs back along that electron's OWN incident direction, not the
    nominal axis: with a finite emittance the two differ by the sampled slope,
    and drawing the nominal one would hide exactly the divergence the plot is
    there to show. A collimated beam has every direction equal to the axis, so
    this is the same picture as before wherever no phase space is set."""
    sid = np.asarray(data["elec_id"])
    start = np.asarray(data["start_xyz"], dtype=float)
    ids, first = np.unique(sid, return_index=True)  # one entry per electron
    entry = start[first]
    beam = np.asarray(data["initial_v_hat"], dtype=float)[ids]
    beam = beam / np.linalg.norm(beam, axis=1, keepdims=True)
    upstream = entry - length * beam  # where each incoming ray starts, off the face
    n = len(entry)
    xyz = np.full((3 * n, 3), np.nan)  # start, entry, NaN per electron
    xyz[0::3] = upstream
    xyz[1::3] = entry
    xyz = _rotate(xyz, R)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines",
        line={"color": _BEAM, "width": 3},
        hovertemplate="incident beam<extra></extra>",
        name="incident beam",
    )


def _exit_paths_3d(
    data, length, *, R=_IDENTITY_R, cmax, tol_frac=1e-6, reveal_until_fs=None, empty_ok=False
):
    """ONE NaN-separated ``Scatter3d`` of solid exit-path continuations, one per
    electron that leaves through the TOP (``z ~ 0``, backscattered) or BOTTOM
    (``z ~ thick``, transmitted) face. Side exits (terminal ``z`` strictly
    between the two faces -- absorbed, or the rare true side leak) are not drawn.

    The terminal segment of each electron is its largest-``t_fs`` (oldest-age)
    entry in ``data``. Its exit direction is the terminal segment's own
    ``end_xyz - start_xyz`` (normalized), continued straight for ``length``
    display units past ``end_xyz`` -- an extrapolation, not a further-simulated
    path. ``tol_frac * thick`` is the face-membership tolerance: transported
    ``z`` at the exit face agrees with ``0``/``thick`` only up to the segment
    integrator's floating-point precision, so an exact ``==`` comparison would
    spuriously miss real exits; ``1e-6`` of the slab thickness is generous
    against that roundoff while still well inside any physical slab.

    Each dash is colored by its electron's terminal-segment energy on the SAME
    Turbo scale as the in-crystal tracks (``cmin=0``, ``cmax=E0_keV``), so an
    exit continuation reads as a faint prolongation of the track that produced
    it rather than a disconnected gray stub. ``showscale`` stays off so the
    tracks own the single energy colorbar. The line is ``"solid"``: any dashed
    ``scatter3d.line.dash`` style rendered with gaps wide enough to look like the
    continuation began in mid-air, so it draws continuous from the exit point.

    ``reveal_until_fs`` (playback cutoff, ``None`` -> unfiltered) additionally
    gates each electron's dash on its terminal segment's own start age: the dash
    appears only once that segment has itself been revealed (``t_fs <=
    reveal_until_fs``), so the exit continuation never appears ahead of the track
    that produced it.

    Returns ``None`` when no electron exits through the top or bottom face,
    UNLESS ``empty_ok`` is set: then an EMPTY same-styled ``Scatter3d`` is
    returned instead. This is for per-frame animation construction, where a
    frame must always be able to update this trace's fixed index even when
    its own reveal cutoff has not yet exposed any exit.

    Face membership uses the SAMPLE-frame ``z`` (true penetration depth)
    regardless of ``R``; ``R`` (default identity) only rotates the final
    plotted points into the lab frame.
    """
    elec_id = np.asarray(data["elec_id"])
    start = np.asarray(data["start_xyz"], dtype=float)
    end = np.asarray(data["end_xyz"], dtype=float)
    energy = np.asarray(data["E"], dtype=float)
    t_fs = np.asarray(data["t_fs"], dtype=float)
    thick = float(data["thick"])
    if elec_id.size == 0:
        return None

    # terminal (max t_fs) segment per electron: sort by (electron, age), then
    # take the last row of each contiguous electron run.
    order = np.lexsort((t_fs, elec_id))
    sorted_ids = elec_id[order]
    run_end = np.flatnonzero(np.diff(sorted_ids) != 0)
    terminal = order[np.concatenate((run_end, [len(order) - 1]))]

    term_start = start[terminal]
    term_end = end[terminal]
    term_energy = energy[terminal]
    term_t_fs = t_fs[terminal]
    tol = max(tol_frac * thick, np.finfo(float).eps)
    exits = (np.abs(term_end[:, 2]) <= tol) | (np.abs(term_end[:, 2] - thick) <= tol)
    if reveal_until_fs is not None:
        exits = exits & (term_t_fs <= reveal_until_fs)
    if not np.any(exits):
        if not empty_ok:
            return None
        xyz = np.empty((0, 3))
        color = np.empty(0)
    else:
        term_start = term_start[exits]
        term_end = term_end[exits]
        term_energy = term_energy[exits]

        direction = term_end - term_start
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm[norm[:, 0] == 0.0] = 1.0  # degenerate zero-length terminal segment: skip normalizing
        unit = direction / norm
        far = term_end + length * unit

        n = len(term_end)
        xyz = np.full((3 * n, 3), np.nan)
        xyz[0::3] = term_end
        xyz[1::3] = far
        # per-vertex color = owning electron's exit energy (3 verts/electron incl. the
        # NaN separator, whose color value is inert since its position breaks the line).
        color = np.repeat(term_energy, 3)
    xyz = _rotate(xyz, R)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines",
        line={
            "color": color,
            "colorscale": "Turbo",
            "cmin": 0.0,
            "cmax": cmax,
            "showscale": False,
            "width": 3,
            "dash": "solid",
        },
        opacity=0.4,
        hovertemplate="exit path<extra></extra>",
        name="exit path",
    )


def _resolve_beam_fwhm(case, beam_fwhm_mm):
    """Effective realistic-mode beam FWHM [mm]: the explicit ``beam_fwhm_mm``
    override if given, else the case's own ``beam_fwhm_mm`` (falling back to
    ``1.0``). Shared by the transport call and the drawn footprint outline so
    both use the same spot size."""
    return beam_fwhm_mm if beam_fwhm_mm is not None else (case.get("beam_fwhm_mm") or 1.0)


def trajectory_volume_data(rec_or_case, *, Ne=40, seed=0, realistic=False, beam_fwhm_mm=None):
    """Transport the penetration-tab dataset ONCE for a given parameter set and
    return the raw ``_trajectory_data`` dict, without building any figure.

    This is the single source of the ``realistic`` (finite crystal + physical
    beam spot) vs. zoomed (``_ZOOM_BEAM_FWHM_MM`` micron spot) transport branch,
    shared by :func:`trajectory_volume_figure` and :func:`case_t_max` so the two
    never drift. The marimo playback cell caches the returned dict keyed on the
    parameter set and reuses it across every animation frame -- a 60-frame scrub
    then re-runs only the cheap reveal filter and trace assembly per frame
    (:func:`trajectory_volume_figure_from_data`), not full Monte Carlo transport
    per tick.

    Not new physics -- delegates to the already-ledgered ``_trajectory_data``.
    """
    case = _case_of(rec_or_case)
    if realistic:
        # true finite crystal + physical beam spot: transport samples each entry
        # from the Gaussian, projects it onto the tilted face, and drops off-crystal
        # entries (n_missed) so they never render.
        fwhm_mm = _resolve_beam_fwhm(case, beam_fwhm_mm)
        return _trajectory_data(
            case,
            Ne,
            seed,
            beam_fwhm_mm=fwhm_mm,
            crystal_width_mm=case.get("crystal_width_mm") or 5.0,
            crystal_height_mm=case.get("crystal_height_mm") or 5.0,
        )
    # zoomed view: a micron-scale spot so the bundle has visible width at the
    # fitted window scale, but no finite footprint (nothing to miss).
    return _trajectory_data(case, Ne, seed, beam_fwhm_mm=_ZOOM_BEAM_FWHM_MM)


def trajectory_volume_figure(
    rec_or_case, *, Ne=40, seed=0, realistic=False, beam_fwhm_mm=None, reveal_until_fs=None
):
    """Interactive 3D cutaway of electron tracks inside the crystal slab.

    Axes are LAB-frame coordinates (see :func:`pyrite.montecarlo.geometry.
    sample_to_lab_R`): z is the fixed beam axis, x is the in-plane direction 90
    deg from the beam within the beam-detector plane (the detector direction
    always has zero lab-y component by convention, so x and the detector
    direction coincide exactly at ``theta_obs = 90 deg``), and y is out of that
    plane. The crystal itself is rotated into this frame too, so a tilted slab
    reads as visibly tilted rather than always face-on. The incident beam is
    drawn as a BUNDLE of particles: each electron enters at its own
    Gaussian-sampled, tilt-projected point on the sample's ``z = 0`` face, and a
    short red stub upstream of that point marks the incoming ray (replacing the
    old single beam arrow).

    By default (``realistic=False``) the crystal x/y extent is a fitted display
    window around the simulated tracks, not a claim about physical lateral
    footprint; depth and internal layer-interface positions retain their true
    scale. The spot is a micron-scale (``_ZOOM_BEAM_FWHM_MM``) Gaussian so the
    bundle has visible width at that zoom, with no finite footprint to miss.

    ``realistic=True`` instead draws the box at the case's TRUE lateral footprint
    (``crystal_width_mm`` x ``crystal_height_mm``, default 5x5 mm), transports a
    physical beam spot, and overlays the FWHM entry footprint on the ``z = 0``
    face. At a large polar tilt that footprint stretches by ``1 / cos(tilt)``
    along the azimuth and can exceed the crystal -- the finite-crystal
    grazing-incidence overlap loss; electrons landing off the crystal are
    dropped by transport and never drawn. At true 5 mm scale the ~micron
    cascade collapses toward the origin, as expected. ``beam_fwhm_mm`` (only
    meaningful when ``realistic=True``) overrides the case's own
    ``beam_fwhm_mm``; ``None`` (default) preserves the current default of
    ``case.get("beam_fwhm_mm") or 1.0``. The zoomed view always uses
    ``_ZOOM_BEAM_FWHM_MM`` and ignores this argument.

    Electrons that exit through the top (``z = 0``, backscattered) or bottom
    (``z = thick``, transmitted) face get a short dashed "exit path" continuing
    straight past their last transported point, in their terminal segment's own
    direction; side exits are not drawn (see :func:`_exit_paths_3d`).

    ``reveal_until_fs`` (``None`` by default -> unchanged full reveal) drives
    playback: only segments whose start age ``t_fs <= reveal_until_fs`` are drawn
    for the in-crystal tracks (:func:`track_vertices_3d`), and an electron's exit
    dash appears only once its terminal segment has itself been revealed (see
    :func:`_exit_paths_3d`). Use :func:`dataset_t_max` and :func:`frame_reveal_fs`
    to derive a cutoff for a given animation frame.
    """
    data = trajectory_volume_data(
        rec_or_case, Ne=Ne, seed=seed, realistic=realistic, beam_fwhm_mm=beam_fwhm_mm
    )
    return trajectory_volume_figure_from_data(
        rec_or_case,
        data,
        realistic=realistic,
        beam_fwhm_mm=beam_fwhm_mm,
        reveal_until_fs=reveal_until_fs,
    )


def trajectory_volume_figure_from_data(
    rec_or_case, data, *, realistic=False, beam_fwhm_mm=None, reveal_until_fs=None
):
    """Assemble the 3D cutaway figure from an ALREADY-transported ``data`` dict
    (see :func:`trajectory_volume_data`) plus a playback reveal cutoff.

    Split out of :func:`trajectory_volume_figure` so the marimo playback cell can
    transport once per parameter set and rebuild the figure per frame from the
    cached dataset -- only the ``reveal_until_fs`` filter and trace assembly rerun
    per frame, not Monte Carlo transport. ``realistic`` and ``beam_fwhm_mm`` must
    match the values ``data`` was transported with; here they drive only the
    drawn crystal footprint and beam-spot outline, not transport.
    """
    case = _case_of(rec_or_case)
    R = _case_R(case)
    lox, hix, loy, hiy, thick, unit, span = _display_extent(case, data, realistic=realistic)
    tracks = _tracks_trace(case, data, unit, R=R, reveal_until_fs=reveal_until_fs)

    fig = go.Figure([_crystal_mesh(lox, hix, loy, hiy, thick, R=R), tracks])
    if len(data.get("vacuum_start_xyz", ())):
        fig.add_trace(
            vacuum_legs_trace(
                data,
                reveal_until_fs=reveal_until_fs,
                R=R,
                unit=unit,
                cmax=float(case["E0_keV"]),
            )
        )
    fig.add_trace(_plane_outline(0.0, lox, hix, loy, hiy, R=R, name="entrance face"))
    fig.add_trace(_plane_outline(thick, lox, hix, loy, hiy, R=R, name="exit face"))
    groove = _groove_surface_mesh(case, lox, hix, loy, hiy, data["u"], R=R)
    if groove is not None:
        fig.add_trace(groove)
    for index, depth in enumerate(data.get("layer_bounds", ()), start=1):
        fig.add_trace(
            _plane_outline(
                depth, lox, hix, loy, hiy, R=R, name=f"layer interface {index}", dash="dash"
            )
        )
    if realistic:
        outline = _beam_footprint_outline(
            case, data["u"], _resolve_beam_fwhm(case, beam_fwhm_mm), R=R
        )
        if outline is not None:
            fx, fy, fz = outline
            fig.add_trace(
                go.Scatter3d(
                    x=fx,
                    y=fy,
                    z=fz,
                    mode="lines",
                    line={"color": _BEAM, "width": 5},
                    hovertemplate="beam footprint (entrance face)<extra></extra>",
                    name="beam footprint",
                )
            )
    fig.add_trace(_incident_beam_lines(data, 0.28 * span, R=R))
    exit_paths = _exit_paths_3d(
        data, 0.28 * span, R=R, cmax=float(case["E0_keV"]), reveal_until_fs=reveal_until_fs
    )
    if exit_paths is not None:
        fig.add_trace(exit_paths)
    for trace in _direction_arrow(
        R @ np.asarray(data["detector"], dtype=float),
        0.28 * span,
        color=_DETECTOR,
        name="detector direction",
    ):
        fig.add_trace(trace)

    material = case["name"].split()[0]
    fig.update_layout(
        title={
            "text": (
                f"{material} · {case['E0_keV']:g} keV · "
                f"θ<sub>tilt</sub>={case.get('tilt_deg', 0.0):g}° · "
                f"φ<sub>azim</sub>={case.get('tilt_azim_deg', 0.0):g}°"
            ),
            "x": 0.02,
        },
        template="plotly_dark",
        paper_bgcolor=_FIELD,
        plot_bgcolor=_FIELD,
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "color": "#EDF6F9"},
        legend={"orientation": "h", "y": 1.02, "x": 0.02},
        margin={"l": 8, "r": 86, "t": 72, "b": 8},
        # 650 left dead vertical letterbox below the aspect-locked scene before
        # reaching the animation slider (aspectmode="data" fits the cube by
        # width, not height); 560 tightens that gap.
        height=560,
        scene=_scene_layout(lox, hix, loy, hiy, thick, span, R, unit, realistic=realistic),
        uirevision="penetration-volume",
    )
    return fig


def _scene_layout(lox, hix, loy, hiy, thick, span, R, unit, *, realistic):
    """Build the ``scene`` layout dict for :func:`trajectory_volume_figure_from_data`.

    ``realistic=True`` keeps the original auto-range behavior (``aspectmode
    ="data"``, no explicit ranges) unchanged. ``realistic=False`` (zoomed)
    instead fits explicit ranges to the display box (see
    :func:`_zoom_scene_ranges`) and switches to ``aspectmode="manual"`` with
    an ``aspectratio`` proportional to those range widths, normalized so the
    largest axis is 1 -- ``"data"`` mode and explicit ranges interact poorly
    in Plotly, so manual is the reliable way to reproduce ``"data"``
    proportions while still respecting the fitted ranges."""
    scene = {
        "xaxis": {
            "title": f"lab x, beam·detector plane ({unit})",
            "gridcolor": _GRID,
            "zeroline": False,
        },
        "yaxis": {
            "title": f"lab y, out of plane ({unit})",
            "gridcolor": _GRID,
            "zeroline": False,
        },
        "zaxis": {
            "title": f"lab z, beam axis ({unit})",
            "gridcolor": _GRID,
            "zeroline": False,
        },
        "aspectmode": "data",
        "camera": {"eye": {"x": _CAMERA_EYE[0], "y": _CAMERA_EYE[1], "z": _CAMERA_EYE[2]}},
        "bgcolor": _FIELD,
    }
    if not realistic:
        x_range, y_range, z_range = _zoom_scene_ranges(lox, hix, loy, hiy, thick, span, R)
        widths = np.array(
            [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
        )
        ratio = widths / widths.max()
        scene["xaxis"]["range"] = x_range
        scene["yaxis"]["range"] = y_range
        scene["zaxis"]["range"] = z_range
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = {"x": ratio[0], "y": ratio[1], "z": ratio[2]}
    return scene


def frame_reveal_fs(frame_index, t_max, n_frames=60):
    """Reveal cutoff [fs] for animation ``frame_index`` of ``n_frames``, linear
    from ``0`` (frame 0) to ``t_max`` (the last frame, ``n_frames - 1``). Pure --
    no marimo/plotly import -- so it is unit-testable and reusable by
    :func:`pyrite.plots.plotly.render.render_reveal_animation` without
    dragging in UI or figure state."""
    return frame_index / (n_frames - 1) * t_max


def dataset_t_max(data):
    """Oldest finite material/vacuum segment start age [fs], or ``0.0`` for an
    empty/all-non-finite dataset -- the natural playback upper bound."""
    t_fs = np.concatenate(
        (
            np.asarray(data.get("t_fs", []), dtype=float),
            np.asarray(data.get("vacuum_t_fs", []), dtype=float),
        )
    )
    finite = t_fs[np.isfinite(t_fs)]
    return float(np.max(finite)) if finite.size else 0.0


def case_t_max(rec_or_case, *, Ne, seed, realistic=False, beam_fwhm_mm=None):
    """Oldest per-segment start age [fs] for this case/``Ne``/``seed``/``realistic``/
    ``beam_fwhm_mm`` combination, without building a figure.

    Runs ``_trajectory_data`` the SAME way :func:`trajectory_volume_figure` does
    (identical ``beam_fwhm_mm`` resolution and ``realistic``-mode crystal
    footprint) and returns :func:`dataset_t_max` of the result, so a caller can
    derive a :func:`frame_reveal_fs` cutoff for every animation frame without
    re-running transport once per frame -- call this once per parameter set
    instead, and reuse the returned ``T_max`` across the whole
    ``0..n_frames-1`` scrub range.
    :func:`pyrite.plots.plotly.render.render_reveal_animation` instead
    takes an already-transported ``data`` dict and calls
    :func:`dataset_t_max` on it directly, since its caller (the marimo
    penetration tab) already caches that dict.

    Not new physics -- delegates to the same ``_trajectory_data`` transport call
    already used and ledgered by :func:`trajectory_volume_figure`; no separate
    validation entry needed.
    """
    return dataset_t_max(
        trajectory_volume_data(
            rec_or_case, Ne=Ne, seed=seed, realistic=realistic, beam_fwhm_mm=beam_fwhm_mm
        )
    )
