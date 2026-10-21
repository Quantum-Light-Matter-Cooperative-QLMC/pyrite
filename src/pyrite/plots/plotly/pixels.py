"""Plotly pixel detector views using the existing bounded image/histogram frames.

The image stays a regular 2-D array (at most 128 x 128 display bins by default),
with exact pixel-edge clipping, row zero at the top, and raw-value hover.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def _signed_ticks(raw):
    """Readable physical-count ticks for Plotly's explicitly transformed symlog."""
    vals = np.asarray(raw, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return [0.0], ["0"]
    maximum = float(np.max(np.abs(finite)))
    if maximum == 0:
        return [0.0], ["0"]
    magnitudes = [0.0] + [10.0 ** e for e in range(0, int(np.floor(np.log10(maximum))) + 1)]
    levels = sorted(set([-v for v in magnitudes if np.min(finite) < 0] + magnitudes))
    levels = [v for v in levels if np.min(finite) <= v <= np.max(finite)]
    if 0 not in levels:
        levels.append(0.0)
    levels.sort()
    return _signed_log(np.array(levels)).tolist(), [f"{v:g}" for v in levels]


def detector_image_chart(
    frame,
    *,
    title: str,
    value_title: str,
    scale_type: str = "linear",
    scheme: str = "viridis",
    selected: tuple[int, int] | None = None,
    size: int = 420,
):
    """Block-reduced detector image; all underlying counts remain unmodified."""
    if scale_type not in {"linear", "log"}:
        raise ValueError("scale_type must be 'linear' or 'log'")
    fig = go.Figure()
    if frame.empty:
        return fig
    block = int(frame["block"].iloc[0])
    starts_x = sorted(frame["column0"].unique().tolist())
    starts_y = sorted(frame["row0"].unique().tolist())
    raw = np.full((len(starts_y), len(starts_x)), np.nan, dtype=float)
    hover = np.full(raw.shape, "unavailable", dtype=object)
    xidx = {v: j for j, v in enumerate(starts_x)}
    yidx = {v: i for i, v in enumerate(starts_y)}
    for rec in frame.itertuples(index=False):
        i, j = yidx[rec.row0], xidx[rec.column0]
        raw[i, j] = rec.value
        hover[i, j] = (
            f"Rows {rec.row0}–{rec.row1}<br>Columns {rec.column0}–{rec.column1}"
            f"<br>{value_title}: {rec.value:.6g}"
        )
    plotted = _signed_log(raw) if scale_type == "log" else raw
    colorbar = dict(title=value_title)
    if scale_type == "log":
        vals, labels = _signed_ticks(raw)
        colorbar.update(tickvals=vals, ticktext=labels)
    # Heatmap cells are block-wide; clipping the axes to the true detector
    # boundaries removes overshoot when the last block is truncated.
    nx = int(frame["column1"].max()) + 1
    ny = int(frame["row1"].max()) + 1
    fig.add_trace(go.Heatmap(
        z=plotted, x0=block / 2, dx=block, y0=block / 2, dy=block,
        text=hover, hovertemplate="%{text}<extra></extra>",
        colorscale=scheme, colorbar=colorbar,
    ))
    if selected is not None:
        row, column = selected
        hit = frame.loc[
            (frame["row0"] <= row) & (row <= frame["row1"])
            & (frame["column0"] <= column) & (column <= frame["column1"])
        ]
        if not hit.empty:
            cell = hit.iloc[0]
            fig.add_shape(
                type="rect", x0=float(cell.column0), x1=float(cell.column1 + 1),
                y0=float(cell.row0), y1=float(cell.row1 + 1),
                line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)",
            )
    subtitle = "full resolution" if block == 1 else f"display blocks of {block} × {block} pixels"
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        width=size + 150, height=size + 85,
        xaxis=dict(title="column", range=[0, nx], constrain="domain"),
        yaxis=dict(title="row", range=[ny, 0], scaleanchor="x", scaleratio=1),
    )
    return fig


def pixel_spectrum_chart(
    frame, *, title: str, y_type: str = "linear", width: int = 560, height: int = 300
):
    """True accepted pre-response spectra for one pixel, colored by component."""
    if y_type not in {"linear", "log"}:
        raise ValueError("y_type must be 'linear' or 'log'")
    fig = go.Figure()
    for i, (component, grp) in enumerate(frame.groupby("component", sort=False)):
        grp = grp.sort_values("energy_eV")
        values = grp["value"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=grp["energy_eV"].to_numpy(dtype=float),
            y=np.where(values > 0, values, np.nan) if y_type == "log" else values,
            mode="lines", name=str(component),
            line=dict(color=qualitative.Plotly[i % len(qualitative.Plotly)]),
            hovertemplate=f"{component}<br>Energy: %{{x:.6g}} eV"
                          "<br>Flux: %{y:.6g} photons/e⁻/eV<extra></extra>",
        ))
    fig.update_layout(title=title, width=width, height=height,
                      xaxis_title="Photon energy (eV)",
                      yaxis=dict(title="Photons / electron / eV", type=y_type))
    return fig


def histogram_chart(
    frame,
    *,
    title: str,
    count_title: str,
    y_type: str = "linear",
    width: int = 560,
    height: int = 300,
):
    """Measured spectrum as bars with the original half-open bin edges."""
    if y_type not in {"linear", "log"}:
        raise ValueError("y_type must be 'linear' or 'log'")
    left = frame["low_eV"].to_numpy(dtype=float)
    right = frame["high_eV"].to_numpy(dtype=float)
    values = frame["counts"].to_numpy(dtype=float)
    plot_values = _signed_log(values) if y_type == "log" else values
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=(left + right) / 2,
        width=right - left,
        y=plot_values,
        customdata=np.column_stack((left, right, values)),
        hovertemplate="Bin [%{customdata[0]:.6g}, %{customdata[1]:.6g}) eV"
                      "<br>Counts: %{customdata[2]:.6g}<extra></extra>",
        marker_line_width=0,
    ))
    yaxis = dict(title=count_title)
    if y_type == "log":
        vals, labels = _signed_ticks(values)
        yaxis.update(tickmode="array", tickvals=vals, ticktext=labels)
    fig.update_layout(
        title=title, width=width, height=height,
        xaxis_title="Measured energy (eV)", yaxis=yaxis, bargap=0,
    )
    return fig


__all__ = ["detector_image_chart", "histogram_chart", "pixel_spectrum_chart"]
