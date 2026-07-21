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

from .trajectories import _case_of, _trajectory_data

_CRYSTAL = "#B9D9EB"
_CRYSTAL_EDGE = "#EDF6F9"
_BEAM = "#E76F51"
_DETECTOR = "#42C7C7"
_FIELD = "#17202A"
_GRID = "#34495E"


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


def _crystal_mesh(lo, hi, thick):
    x = [lo, hi, hi, lo, lo, hi, hi, lo]
    y = [lo, lo, hi, hi, lo, lo, hi, hi]
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


def _plane_outline(z, lo, hi, *, name, dash="solid"):
    return go.Scatter3d(
        x=[lo, hi, hi, lo, lo],
        y=[lo, lo, hi, hi, lo],
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


def trajectory_volume_figure(rec_or_case, *, Ne=40, seed=0):
    """Interactive 3D cutaway of electron tracks inside the crystal slab.

    Axes are sample-frame coordinates.  Crystal x/y extent is a fitted display
    window around simulated tracks, not a claim about physical lateral footprint.
    Depth and internal layer-interface positions retain their true scale.
    """
    case = _case_of(rec_or_case)
    data = _trajectory_data(case, Ne, seed)
    xyz, energy, elec_id = track_vertices_3d(data)
    lo, hi = _transverse_window(data)
    thick = float(data["thick"])
    unit = "µm" if data["u"] == 1e4 else "nm"
    span = max(hi - lo, thick)

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

    fig = go.Figure([_crystal_mesh(lo, hi, thick), tracks])
    fig.add_trace(_plane_outline(0.0, lo, hi, name="entrance face"))
    fig.add_trace(_plane_outline(thick, lo, hi, name="exit face"))
    for index, depth in enumerate(data.get("layer_bounds", ()), start=1):
        fig.add_trace(
            _plane_outline(depth, lo, hi, name=f"layer interface {index}", dash="dash")
        )
    for trace in _direction_arrow(data["beam"], 0.28 * span, color=_BEAM, name="beam", reverse=True):
        fig.add_trace(trace)
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
