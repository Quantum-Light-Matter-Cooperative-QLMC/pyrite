"""Experimental single-cell Plotly map for the marimo analysis app (#622).

The scoring and gating policy lives in :mod:`pyrite.plots._frames`.  A nearly
transparent Scatter overlay supplies single-click targets because Plotly's
heatmap selection can require a drag gesture.  This is a pilot, not a change to
the default Altair map widget.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .._frames import _axis_label


def single_panel_frame(frame: pd.DataFrame, energy: float) -> pd.DataFrame:
    """Retain exactly one energy panel and selectable, finite-value cells."""
    if frame.empty:
        return frame.copy()
    panel = pd.to_numeric(frame["panel_raw"], errors="coerce")
    keep = np.isclose(panel.to_numpy(dtype=float), float(energy), rtol=1e-12, atol=1e-12)
    return frame.loc[keep].copy()


def _cell(frame: pd.DataFrame, x: float, y: float) -> pd.DataFrame:
    """Match one *valid* plotted cell, never an empty/masked map location."""
    if frame.empty or not (isfinite(x) and isfinite(y)):
        return frame.iloc[0:0]
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)
    values = frame["value"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isclose(xs, x, rtol=1e-10, atol=1e-10)
    valid &= np.isclose(ys, y, rtol=1e-10, atol=1e-10)
    return frame.loc[valid]


def cell_from_points(points: object, frame: pd.DataFrame) -> dict[str, object] | None:
    """Decode marimo Plotly ``.value``/``on_change`` into a unique valid cell.

    Reject ambiguous brush/multi-select results and coordinates not present in
    the current grid.  Trust the current frame for record name and panel, not
    the browser's optional customdata payload.
    """
    if not isinstance(points, list) or not points:
        return None
    found: list[pd.Series] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        matches = _cell(frame, x, y)
        if len(matches) != 1:
            continue
        found.append(matches.iloc[0])
    if not found:
        return None
    coords = {(float(row["x"]), float(row["y"])) for row in found}
    if len(coords) != 1:
        return None
    row = found[0]
    return {
        "x": float(row["x"]),
        "y": float(row["y"]),
        "name": str(row["name"]),
        "panel_raw": float(row["panel_raw"]),
    }


def selected_cell_frame(frame: pd.DataFrame, selection: object) -> pd.DataFrame:
    """Revalidate a saved selection after changing quantity, theme or panel."""
    if not isinstance(selection, Mapping):
        return frame.iloc[0:0]
    try:
        x, y = float(selection["x"]), float(selection["y"])
        panel = float(selection["panel_raw"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return frame.iloc[0:0]
    matching = _cell(frame, x, y)
    if len(matching) != 1 or not np.isclose(
        float(matching.iloc[0]["panel_raw"]), panel, rtol=1e-12, atol=1e-12
    ):
        return frame.iloc[0:0]
    # The source record matters: selecting different best cases at the same
    # coordinates must not silently rebind the old selection to a new case.
    if str(matching.iloc[0]["name"]) != selection.get("name"):
        return frame.iloc[0:0]
    return matching


def selectable_map_figure(
    frame: pd.DataFrame,
    *,
    label: str,
    cmap: str = "viridis",
    color_domain: tuple[float, float] | None = None,
    width: int = 540,
    height: int = 440,
) -> go.Figure | None:
    """Build a bounded Plotly heatmap with Scatter-based click targets.

    ``frame`` has already been reduced to one panel by ``single_panel_frame``.
    Every overlay point maps one-to-one to a row of finite data.  Missing or
    flux-gated cells have neither a click target nor a numerical zero.
    """
    if frame.empty:
        return None
    xs = sorted(float(v) for v in frame["x"].unique())
    ys = sorted(float(v) for v in frame["y"].unique())
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}
    z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    markers_x: list[float] = []
    markers_y: list[float] = []
    names: list[str] = []
    for row in frame.itertuples(index=False):
        x, y, value = float(row.x), float(row.y), float(row.value)
        if not isfinite(value):
            continue
        z[y_index[y], x_index[x]] = value
        markers_x.append(x)
        markers_y.append(y)
        names.append(str(row.name))

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return None
    cmin, cmax = (
        color_domain
        if color_domain is not None
        else (float(finite.min()), float(finite.max()))
    )
    if cmin == cmax:
        cmax = cmin + 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs, y=ys, z=z, coloraxis="coloraxis", xgap=1, ygap=1,
            hoverongaps=False,
            hovertemplate="azimuth: %{x:.6g}°<br>tilt: %{y:.6g}°<br>value: %{z:.6g}<extra></extra>",
            name=label,
        )
    )
    # Plotly heatmaps in selection dragmode may not generate a single-click
    # event.  A topmost Scatter trace with clickmode='event+select' does.
    # Keep the targets smaller than a heatmap cell so adjacent cells cannot
    # eclipse one another. For very dense maps, zoom before clicking.
    px = max(3.0, min(14.0, 0.75 * min((width - 160) / len(xs), (height - 135) / len(ys))))
    fig.add_trace(
        go.Scatter(
            x=markers_x, y=markers_y, mode="markers", name="Select cell",
            customdata=names,
            marker={"symbol": "square", "size": px, "color": "rgba(0,0,0,0.01)"},
            hovertemplate=(
                "%{customdata}<br>azimuth: %{x:.6g}°"
                "<br>tilt: %{y:.6g}°<extra>click to select</extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"{label} (experimental: click the center of a cell)",
        width=width, height=height,
        xaxis={"title": _axis_label("tilt_azim_deg"), "type": "linear"},
        yaxis={"title": _axis_label("tilt_deg"), "type": "linear"},
        coloraxis={"colorscale": cmap, "cmin": cmin, "cmax": cmax,
                   "colorbar": {"title": label}},
        clickmode="event+select", dragmode="pan", showlegend=False,
        uirevision="pyrite-map", margin={"t": 65},
    )
    return fig
