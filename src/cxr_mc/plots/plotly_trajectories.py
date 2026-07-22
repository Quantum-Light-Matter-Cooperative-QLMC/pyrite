"""Browser-native 3D electron-penetration rendering with Plotly.

The transport and coordinate preparation remain owned by
:mod:`cxr_mc.plots.trajectories`; this module only builds an interactive view of
the true sample-frame segment endpoints.  It is intentionally imported directly
rather than re-exported from :mod:`cxr_mc.plots`, whose legacy export set is
frozen.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..montecarlo.geometry import project_beam_entry
from .trajectories import _case_of, _trajectory_data

_CRYSTAL = "#B9D9EB"
_CRYSTAL_EDGE = "#EDF6F9"
_BEAM = "#E76F51"
_DETECTOR = "#42C7C7"
_FIELD = "#17202A"
_GRID = "#34495E"

# Zoomed (non-realistic) view spot FWHM [mm]: a micron-scale beam so the incident
# bundle has visible width at the fitted, sub-micron cascade scale.
_ZOOM_BEAM_FWHM_MM = 1.0e-3

# Both views transport 4x the requested electrons: the incident bundle reads as a
# beam only when densely populated, and at grazing tilt a large fraction of the
# realistic spot misses the finite crystal (n_missed), thinning it further.
_NE_SCALE = 4


def track_vertices_3d(data):
    """Return NaN-separated physical segments for one Plotly ``Scatter3d``.

    Each transport segment contributes ``start, end, NaN``.  Separators prevent
    Plotly from joining different segments or electrons.  Energy and electron ID
    repeat at both endpoints, preserving segment-wise color and hover metadata.
    """
    start = np.asarray(data["start_xyz"], dtype=float)
    end = np.asarray(data["end_xyz"], dtype=float)
    energy = np.asarray(data["E"], dtype=float)
    elec_id = np.asarray(data["elec_id"], dtype=int)
    if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
        raise ValueError("start_xyz and end_xyz must both have shape (N, 3)")
    if len(start) != len(energy) or len(start) != len(elec_id):
        raise ValueError("3D segment, energy, and electron arrays must have equal length")

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


def _transverse_window(data):
    """Robust square x/y display window fitted to tracks, not crystal footprint."""
    points = np.vstack((data["start_xyz"], data["end_xyz"]))
    transverse = points[:, :2]
    transverse = transverse[np.isfinite(transverse).all(axis=1)]
    thick = float(data["thick"])
    radius = float(np.percentile(np.abs(transverse), 99.0)) if transverse.size else 0.0
    radius = max(1.25 * radius, 0.18 * thick, np.finfo(float).eps)
    return -radius, radius


def _crystal_mesh(lox, hix, loy, hiy, thick):
    x = [lox, hix, hix, lox, lox, hix, hix, lox]
    y = [loy, loy, hiy, hiy, loy, loy, hiy, hiy]
    z = [0.0, 0.0, 0.0, 0.0, thick, thick, thick, thick]
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


def _plane_outline(z, lox, hix, loy, hiy, *, name, dash="solid"):
    return go.Scatter3d(
        x=[lox, hix, hix, lox, lox],
        y=[loy, loy, hiy, hiy, loy],
        z=[z] * 5,
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


def _beam_footprint_outline(case, u, *, n_points=96):
    """Closed ``(x, y)`` outline [display units] of the collimated beam spot where
    it strikes the tilted entrance face ``z = 0``, for the realistic-scale view.

    The lab beam is a round spot travelling along ``+z_lab``; on a face tilted by
    (``tilt_deg``, ``tilt_azim_deg``) its footprint elongates by ``1 / cos(tilt)``
    along the azimuth -- the grazing-incidence stretch that can outgrow the finite
    crystal. Reuse :func:`cxr_mc.montecarlo.geometry.project_beam_entry` so the
    drawn outline matches transport's own entry mapping exactly. Returns ``None``
    for the legacy point beam (``beam_fwhm_mm`` falsy / absent).

    ``beam_fwhm_mm`` is the lab-plane spot FWHM; draw the outline at the FWHM
    contour (lab radius = ``beam_fwhm_mm / 2``, mm -> Ang factor 1e7). Sample a
    ring of ``n_points`` lab offsets ``(u_i, v_i)`` on that circle, project them,
    and divide the resulting sample-frame (x, y) by ``u`` for display units.
    """
    fwhm_mm = case.get("beam_fwhm_mm")
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
    return entry[:, 0] / u, entry[:, 1] / u


def _incident_beam_lines(data, length):
    """NaN-separated incident-beam segments -- one short stub per electron drawn
    UPSTREAM along the beam direction into its true entry point on the ``z = 0``
    face, so the beam reads as a bundle of incoming particles rather than a single
    arrow. Entry points are each electron's first-segment start; electrons whose
    Gaussian draw missed a finite crystal are already absent from ``data`` (dropped
    as ``n_missed`` by transport), so this never draws a particle that missed."""
    sid = np.asarray(data["elec_id"])
    start = np.asarray(data["start_xyz"], dtype=float)
    _, first = np.unique(sid, return_index=True)  # one entry per electron
    entry = start[first]
    beam = np.asarray(data["beam"], dtype=float)
    beam = beam / np.linalg.norm(beam)
    upstream = entry - length * beam  # where each incoming ray starts, off the face
    n = len(entry)
    xyz = np.full((3 * n, 3), np.nan)  # start, entry, NaN per electron
    xyz[0::3] = upstream
    xyz[1::3] = entry
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="lines",
        line={"color": _BEAM, "width": 3},
        hovertemplate="incident beam<extra></extra>",
        name="incident beam",
    )


def trajectory_volume_figure(rec_or_case, *, Ne=40, seed=0, realistic=False):
    """Interactive 3D cutaway of electron tracks inside the crystal slab.

    Axes are sample-frame coordinates. The incident beam is drawn as a BUNDLE of
    particles: each electron enters at its own Gaussian-sampled, tilt-projected
    point on the ``z = 0`` face, and a short red stub upstream of that point marks
    the incoming ray (replacing the old single beam arrow).

    By default (``realistic=False``) the crystal x/y extent is a fitted display
    window around the simulated tracks, not a claim about physical lateral
    footprint; depth and internal layer-interface positions retain their true
    scale. The spot is a micron-scale (``_ZOOM_BEAM_FWHM_MM``) Gaussian so the
    bundle has visible width at that zoom, with no finite footprint to miss.

    ``realistic=True`` instead draws the box at the case's TRUE lateral footprint
    (``crystal_width_mm`` x ``crystal_height_mm``, default 5x5 mm), transports a
    physical beam spot (``beam_fwhm_mm``, default 1 mm), and overlays the FWHM
    entry footprint on the ``z = 0`` face. At a large polar tilt that footprint
    stretches by ``1 / cos(tilt)`` along the azimuth and can exceed the crystal --
    the finite-crystal grazing-incidence overlap loss; electrons landing off the
    crystal are dropped by transport and never drawn. At true 5 mm scale the
    ~micron cascade collapses toward the origin, as expected.
    """
    case = _case_of(rec_or_case)
    Ne = Ne * _NE_SCALE  # both bundles need enough particles to read as a beam
    if realistic:
        # true finite crystal + physical beam spot: transport samples each entry
        # from the Gaussian, projects it onto the tilted face, and drops off-crystal
        # entries (n_missed) so they never render.
        data = _trajectory_data(
            case,
            Ne,
            seed,
            beam_fwhm_mm=case.get("beam_fwhm_mm") or 1.0,
            crystal_width_mm=case.get("crystal_width_mm") or 5.0,
            crystal_height_mm=case.get("crystal_height_mm") or 5.0,
        )
    else:
        # zoomed view: a micron-scale spot so the bundle has visible width at the
        # fitted window scale, but no finite footprint (nothing to miss).
        data = _trajectory_data(case, Ne, seed, beam_fwhm_mm=_ZOOM_BEAM_FWHM_MM)
    xyz, energy, elec_id = track_vertices_3d(data)
    lo, hi = _transverse_window(data)
    thick = float(data["thick"])
    unit = "µm" if data["u"] == 1e4 else "nm"
    if realistic:
        lox, hix, loy, hiy = _crystal_footprint_extent(case, data["u"])
    else:
        lox = loy = lo
        hix = hiy = hi
    span = max(hix - lox, hiy - loy, thick)

    custom = np.column_stack((energy, elec_id, xyz[:, 2]))
    tracks = go.Scatter3d(
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

    fig = go.Figure([_crystal_mesh(lox, hix, loy, hiy, thick), tracks])
    fig.add_trace(_plane_outline(0.0, lox, hix, loy, hiy, name="entrance face"))
    fig.add_trace(_plane_outline(thick, lox, hix, loy, hiy, name="exit face"))
    for index, depth in enumerate(data.get("layer_bounds", ()), start=1):
        fig.add_trace(
            _plane_outline(depth, lox, hix, loy, hiy, name=f"layer interface {index}", dash="dash")
        )
    if realistic:
        outline = _beam_footprint_outline(case, data["u"])
        if outline is not None:
            fx, fy = outline
            fig.add_trace(
                go.Scatter3d(
                    x=fx,
                    y=fy,
                    z=[0.0] * len(fx),
                    mode="lines",
                    line={"color": _BEAM, "width": 5},
                    hovertemplate="beam footprint (z=0)<extra></extra>",
                    name="beam footprint",
                )
            )
    fig.add_trace(_incident_beam_lines(data, 0.28 * span))
    for trace in _direction_arrow(
        data["detector"], 0.28 * span, color=_DETECTOR, name="detector direction"
    ):
        fig.add_trace(trace)

    material = case["name"].split()[0]
    fig.update_layout(
        title={
            "text": (
                f"{material} · {case['E0_keV']:g} keV · "
                f"θ<sub>tilt</sub>={case.get('tilt_deg', 0.0):g}°"
            ),
            "x": 0.02,
        },
        template="plotly_dark",
        paper_bgcolor=_FIELD,
        plot_bgcolor=_FIELD,
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "color": "#EDF6F9"},
        legend={"orientation": "h", "y": 1.02, "x": 0.02},
        margin={"l": 8, "r": 86, "t": 72, "b": 8},
        height=650,
        scene={
            "xaxis": {"title": f"sample x ({unit})", "gridcolor": _GRID, "zeroline": False},
            "yaxis": {"title": f"sample y ({unit})", "gridcolor": _GRID, "zeroline": False},
            "zaxis": {
                "title": f"penetration depth ({unit})",
                "gridcolor": _GRID,
                "zeroline": False,
            },
            "aspectmode": "data",
            "camera": {"eye": {"x": -0.8, "y": 1.5, "z": -0.4}},
            "bgcolor": _FIELD,
        },
        uirevision="penetration-volume",
    )
    return fig
