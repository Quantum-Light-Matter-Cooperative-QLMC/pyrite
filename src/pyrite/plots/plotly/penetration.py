"""Interactive 2D electron-penetration views with Plotly.

The interactive counterparts of :mod:`pyrite.plots.mpl.trajectories`, sharing
its transport and projection (``_trajectory_data`` / ``_trajectory_cases``) and
the renderer-neutral tables in :mod:`pyrite.plots._trajectory_frames`. Only the
renderer differs; physics and units are identical.

* :func:`penetration_survival_chart` -- surviving-electron fraction versus depth,
  one monotone line per beam energy.
* :func:`trajectory_chart` -- a low-electron-count vector cross-section in the
  beam-detector plane: energy-coloured tracks, slab faces, beam and detector
  arrows, groove profile and vacuum flights. The dense datashader raster stays on
  Matplotlib.

Plotly cannot vary one 2D line's colour along its length, so track segments are
grouped into ``_ENERGY_BANDS`` energy bands, one NaN-separated line trace per
band coloured at the band centre, plus a marker layer at segment midpoints that
carries the continuous colour bar and per-segment hover.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from .._common import _case_of
from .._frames import (
    _square_frame,
    _trajectory_cases,
    _trajectory_data,
    _trajectory_frame,
    survival_frame,
)
from .._trajectory_frames import groove_profile_xy, track_segments_frame, vacuum_segments_frame
from .trajectories import _BEAM, _DETECTOR, _FIELD, _GRID, _GROOVE

_TEXT = "#EDF6F9"
_SLAB = "rgba(187,187,187,0.22)"
_FACE = "#666666"
_VACUUM = "rgba(142,154,175,0.35)"
_ENERGY_BANDS = 24


def _dark_layout(fig, *, title, width, height, x_title, y_title):
    axis = {"gridcolor": _GRID, "zerolinecolor": _GRID, "linecolor": _GRID, "color": _TEXT}
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        paper_bgcolor=_FIELD,
        plot_bgcolor=_FIELD,
        font={"color": _TEXT},
        xaxis={**axis, "title": x_title},
        yaxis={**axis, "title": y_title},
        legend={"font": {"color": _TEXT}},
        margin={"l": 64, "r": 24, "t": 64, "b": 56},
    )
    return fig


def _resolve_saved(cases_or_results):
    from ...montecarlo.trajectory_selection import TrajectorySelection
    from ..trajectory_capture import saved_trajectory_data

    if isinstance(cases_or_results, TrajectorySelection):
        case, saved = saved_trajectory_data(cases_or_results)
        return [case], saved
    return cases_or_results, None


def penetration_survival_chart(
    cases_or_results,
    *,
    Ne=500,
    seed=0,
    n_bins=80,
    tilt=None,
    depth_frac=True,
    width=720,
    height=420,
):
    """Surviving-electron fraction (% of N0) versus penetration depth, one line
    per beam energy at one polar tilt (nearest to ``tilt``; default closest to
    normal incidence). Returns a Plotly ``Figure``, or ``None`` without cases."""
    cases_or_results, saved = _resolve_saved(cases_or_results)
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    t = min(tilts, key=lambda x: abs(x - (0.0 if tilt is None else tilt)))
    data_by_energy = {}
    for E0 in sorted({c["E0_keV"] for c in cases}):
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is not None:
            data_by_energy[E0] = saved if saved is not None else _trajectory_data(c, Ne, seed)
    if not data_by_energy:
        return None
    df = survival_frame(data_by_energy, n_bins=n_bins, depth_frac=depth_frac)
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = "um" if case0["thickness_ang"] >= 1e4 else "nm"
    fig = go.Figure()
    for energy, part in df.groupby("energy", sort=False):
        fig.add_trace(
            go.Scatter(
                x=part["depth"].to_numpy(dtype=float),
                y=part["survival"].to_numpy(dtype=float),
                mode="lines",
                line={"width": 1.9},
                name=str(energy),
                hovertemplate=(
                    f"{energy}<br>depth: %{{x:.4g}}<br>surviving: %{{y:.3g}}%<extra></extra>"
                ),
            )
        )
    _dark_layout(
        fig,
        title=(
            f"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} um, "
            f"theta_tilt={t:g} deg -- electron penetration / survival"
        ),
        width=width,
        height=height,
        x_title="depth / thickness" if depth_frac else f"penetration depth ({ulab})",
        y_title="surviving electrons (% of N0)",
    )
    fig.update_layout(legend_title_text="beam energy", yaxis_range=[0, 100])
    return fig


def _segments_xy(df):
    """NaN-separated x/y arrays drawing every row of ``df`` as one segment."""
    n = len(df)
    xs = np.full(3 * n, np.nan)
    ys = np.full(3 * n, np.nan)
    xs[0::3], xs[1::3] = df["x"].to_numpy(dtype=float), df["x2"].to_numpy(dtype=float)
    ys[0::3], ys[1::3] = df["y"].to_numpy(dtype=float), df["y2"].to_numpy(dtype=float)
    return xs, ys


def _track_traces(seg_df, *, E_cut, E0, unit):
    """Energy-band line traces plus a midpoint marker layer (colour bar, hover)."""
    lo, hi = float(E_cut), float(max(E0, E_cut + 1e-9))
    energy = seg_df["E"].to_numpy(dtype=float)
    frac = np.clip((energy - lo) / (hi - lo), 0.0, 1.0)
    band = np.minimum((frac * _ENERGY_BANDS).astype(int), _ENERGY_BANDS - 1)
    traces = []
    for b in range(_ENERGY_BANDS):
        part = seg_df[band == b]
        if part.empty:
            continue
        xs, ys = _segments_xy(part)
        color = sample_colorscale("Turbo", [(b + 0.5) / _ENERGY_BANDS])[0]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color, "width": 0.9},
                opacity=0.85,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    hover = "energy: %{marker.color:.3g} keV"
    customdata = None
    if "track_id" in seg_df:
        customdata = seg_df[["track_id", "parent_id", "generation"]].to_numpy()
        hover = (
            "track %{customdata[0]} (parent %{customdata[1]}, generation %{customdata[2]})"
            "<br>" + hover
        )
    traces.append(
        go.Scatter(
            x=0.5 * (seg_df["x"].to_numpy(dtype=float) + seg_df["x2"].to_numpy(dtype=float)),
            y=0.5 * (seg_df["y"].to_numpy(dtype=float) + seg_df["y2"].to_numpy(dtype=float)),
            mode="markers",
            marker={
                "size": 3,
                "color": energy,
                "colorscale": "Turbo",
                "cmin": lo,
                "cmax": hi,
                "opacity": 0.0,
                "colorbar": {"title": {"text": "electron energy (keV)", "side": "right"}},
            },
            customdata=customdata,
            hovertemplate=hover + f"<br>({unit})<extra></extra>",
            showlegend=False,
            name="tracks",
        )
    )
    return traces


def _slab_shapes(data, frame):
    """Slab shading, entrance/exit faces and internal layer boundaries."""
    xlo, xhi, ylo, yhi = frame
    nslab = np.asarray(data["nslab"], dtype=float)
    thick = float(data["thick"])
    tang = np.array([-nslab[1], nslab[0]])
    W = 3.0 * max(xhi - xlo, yhi - ylo)
    shapes = []
    if abs(nslab[1]) > 1e-4:
        xs = np.linspace(xlo, xhi, 120)
        yf = np.clip((0.0 - xs * nslab[0]) / nslab[1], ylo, yhi)
        yb = np.clip((thick - xs * nslab[0]) / nslab[1], ylo, yhi)
        lower, upper = np.minimum(yf, yb), np.maximum(yf, yb)
        path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, lower, strict=True))
        path += " L " + " L ".join(f"{x},{y}" for x, y in zip(xs[::-1], upper[::-1], strict=True))
        shapes.append({"type": "path", "path": path + " Z", "fillcolor": _SLAB, "line_width": 0})
    else:
        x0, x1 = min(0.0, thick) / nslab[0], max(0.0, thick) / nslab[0]
        shapes.append(
            {"type": "rect", "x0": x0, "x1": x1, "y0": ylo, "y1": yhi, "fillcolor": _SLAB}
            | {"line_width": 0}
        )
    for offset, dash, width in [(0.0, "solid", 1.2), (thick, "solid", 1.2)] + [
        (float(z), "dash", 0.8) for z in data.get("layer_bounds", ())
    ]:
        p0, p1 = -W * tang + offset * nslab, W * tang + offset * nslab
        shapes.append(
            {"type": "line", "x0": p0[0], "y0": p0[1], "x1": p1[0], "y1": p1[1]}
            | {"line": {"color": _FACE, "width": width, "dash": dash}}
        )
    return shapes


def trajectory_chart(rec_or_case, *, Ne=40, seed=0, E_cut=5.0, frame=None, width=480, data=None):
    """Interactive vector view of one penetration cross-section.

    Energy-coloured tracks (turbo, ``E_cut`` to the beam energy, keV) in the
    beam-detector plane with the crystal faces, beam (red) and detector arrows,
    and any groove profile or vacuum flights. Keep ``Ne`` modest: these are
    vector marks, not a raster. Returns a Plotly ``Figure``, or ``None`` when the
    cascade is empty.
    """
    from ...montecarlo.trajectory_selection import TrajectorySelection
    from ..trajectory_capture import saved_trajectory_data

    if isinstance(rec_or_case, TrajectorySelection):
        rec_or_case, data = saved_trajectory_data(rec_or_case)
    case = _case_of(rec_or_case)
    if data is None:
        data = _trajectory_data(case, Ne, seed)
    seg_df = track_segments_frame(data)
    if seg_df.empty:
        return None
    if frame is None:
        frame = _square_frame(_trajectory_frame([data["pts"]]))
    xlo, xhi, ylo, yhi = frame
    E0 = case["E0_keV"]
    ndet = np.asarray(data["ndet"], dtype=float)
    aL = 0.16 * (xhi - xlo)

    fig = go.Figure()
    vacuum_df = vacuum_segments_frame(data)
    if not vacuum_df.empty:
        xs, ys = _segments_xy(vacuum_df)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": _VACUUM, "width": 0.8},
                name="vacuum flight",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for trace in _track_traces(seg_df, E_cut=E_cut, E0=E0, unit=data["ulab"]):
        fig.add_trace(trace)
    groove = groove_profile_xy(case, data, frame)
    if groove is not None:
        fig.add_trace(
            go.Scatter(
                x=groove[0],
                y=groove[1],
                mode="lines",
                line={"color": _GROOVE, "width": 1.4},
                opacity=0.95,
                name="groove profile",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    shapes = _slab_shapes(data, frame)
    shapes += [
        {"type": "line", "x0": -aL, "y0": 0.0, "x1": 0.0, "y1": 0.0, "line": {"color": _BEAM}}
        | {"line_width": 2.0},
        {"type": "line", "x0": 0.0, "y0": 0.0, "x1": ndet[0] * aL, "y1": ndet[1] * aL}
        | {"line": {"color": _DETECTOR, "width": 2.0}},
    ]
    _dark_layout(
        fig,
        title=(
            f"{case['name'].split()[0]}, {E0:g} keV, theta={case.get('tilt_deg', 0.0):g} deg, "
            f"phi={case.get('tilt_azim_deg', 0.0):g} deg -- {data['Ne']} e- "
            f"({data['eta']:.0f}% back, {data['thru']:.0f}% through)"
        ),
        width=width,
        height=width,
        x_title=f"along beam ({data['ulab']})",
        y_title=f"transverse ({data['ulab']})",
    )
    fig.update_layout(
        shapes=[{**shape, "layer": "below"} for shape in shapes],
        xaxis_range=[xlo, xhi],
        yaxis_range=[ylo, yhi],
        yaxis_scaleanchor="x",
        dragmode="pan",
        hovermode="closest",
    )
    return fig
