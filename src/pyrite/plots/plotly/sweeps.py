"""Plotly metric scans and regular-grid sweep heatmaps (#622).

All physics and geometry reductions come from :mod:`pyrite.plots._frames`;
heatmap cells contain only plotted summaries, not full Monte Carlo trajectories.
The selectable map widget continues to use Altair until Plotly click-selection
is proven equivalent in marimo.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

_COLORS = qualitative.Plotly
from plotly.subplots import make_subplots

from ...results import records_for_cases
from .._common import _metrics_map
from .._frames import (
    _AXIS_SPECS,
    _HEATMAP_QUANTITIES,
    _METRIC_LABELS,
    _axis_label,
    _effective_x,
    _resolve_quantity,
    heatmap_frame,
    metric_vs_frame,
    pick_hue,
    scan_mode,
)


def metric_vs_chart(
    results,
    settings,
    *,
    x="thickness_ang",
    metric="line_flux",
    hue="E0_keV",
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    cxr_metric="sharpness",
    logx=False,
    logy=False,
    width=720,
    height=420,
    metrics=None,
):
    """Renderer-independent one-dimensional metric-vs-knob frame as Plotly lines."""
    recs = records_for_cases(results, cases)
    if not recs:
        return None
    eff_x = _effective_x(recs, x, hue)
    frame = metric_vs_frame(
        results, settings, x=x, metric=metric, hue=hue,
        select=select, cases=cases, rel_prominence=rel_prominence,
        cxr_metric=cxr_metric, metrics=metrics,
    )
    if frame.empty:
        return None
    fig = go.Figure()
    for i, (key, group) in enumerate(frame.groupby("hue", sort=False)):
        points = group.sort_values("x")
        fig.add_trace(go.Scatter(
            x=points["x"].to_numpy(dtype=float),
            y=points["metric"].to_numpy(dtype=float),
            mode="lines+markers", name=str(key),
            marker=dict(size=5), line=dict(width=1.8, color=_COLORS[i % len(_COLORS)]),
            hovertemplate=f"{hue}: {key}<br>x: %{{x:.6g}}<br>value: %{{y:.6g}}<extra></extra>",
        ))
    metric_label = _METRIC_LABELS.get(metric, metric)
    fig.update_layout(
        title=f"{metric_label} vs {_axis_label(eff_x)}  (best per point: {select})",
        width=width, height=height,
        xaxis=dict(title=_axis_label(eff_x), type="log" if logx else "linear"),
        yaxis=dict(title=metric_label, type="log" if logy else "linear"),
        legend=dict(title=_AXIS_SPECS.get(hue, (hue,))[0]),
    )
    return fig


def heatmap_chart(
    results,
    settings,
    *,
    quantity="peak_flux",
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    cxr_metric="sharpness",
    min_flux_frac=0.02,
    min_cxr_quality=0.2,
    width=360,
    height=300,
    metrics=None,
    color_domain=None,
):
    """Dense sweep grids with NaN gaps, per-cell metadata, and shared color range."""
    key, label, cmap = _resolve_quantity(quantity)
    frame = heatmap_frame(
        results, settings, quantity=key, x=x, y=y, panel=panel,
        select=select, cases=cases, rel_prominence=rel_prominence,
        cxr_metric=cxr_metric, min_flux_frac=min_flux_frac,
        min_cxr_quality=min_cxr_quality, metrics=metrics,
    )
    if frame.empty:
        return None
    panels = list(dict.fromkeys(frame["panel"].tolist()))
    fig = make_subplots(
        rows=1, cols=len(panels), subplot_titles=panels,
        horizontal_spacing=min(0.08, 0.2 / len(panels)),
    )
    all_values = frame["value"].to_numpy(dtype=float)
    finite = all_values[np.isfinite(all_values)]
    zmin, zmax = color_domain if color_domain is not None else (
        (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    )
    if zmin == zmax:
        zmax = zmin + 1.0
    # A missing sample in the input frame deliberately produces a NaN, not a
    # zero-count pixel (zero and unavailable data are scientifically distinct).
    for i, p in enumerate(panels):
        part = frame[frame["panel"] == p]
        xs = sorted(part["x"].unique().tolist())
        ys = sorted(part["y"].unique().tolist())
        z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        hover = np.full(z.shape, "unavailable", dtype=object)
        xix = {v: j for j, v in enumerate(xs)}
        yix = {v: j for j, v in enumerate(ys)}
        for r in part.itertuples(index=False):
            row, col = yix[r.y], xix[r.x]
            z[row, col] = float(r.value)
            hover[row, col] = f"{r.name}<br>{_axis_label(x)}: {r.x:.6g}<br>{_axis_label(y)}: {r.y:.6g}"
        fig.add_trace(
            go.Heatmap(
                z=z, x=xs, y=ys, text=hover,
                hovertemplate="%{text}<br>value: %{z:.6g}<extra></extra>",
                coloraxis="coloraxis", xgap=1, ygap=1,
            ),
            row=1, col=i + 1,
        )
        fig.update_xaxes(title_text=_axis_label(x), row=1, col=i + 1)
        fig.update_yaxes(title_text=_axis_label(y), row=1, col=i + 1)
    fig.update_layout(
        title=label, width=max(520, width * len(panels)), height=height + 110,
        coloraxis=dict(colorscale=str(cmap), cmin=zmin, cmax=zmax, colorbar=dict(title=label)),
        showlegend=False,
    )
    return fig


def scan_charts(
    results,
    settings,
    *,
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    hue=None,
    quantities=None,
    heatmap_min=4,
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    cxr_metric="sharpness",
    min_flux_frac=0.02,
    min_cxr_quality=0.2,
    logx=False,
    logy=False,
    force=None,
):
    """Use shared auto-scan policy, evaluating physics metrics only once."""
    recs = records_for_cases(results, cases)
    if not recs:
        return []
    quantities = [_resolve_quantity(q) for q in (quantities or _HEATMAP_QUANTITIES)]
    mode = scan_mode(recs, x, y, heatmap_min, force)
    metrics = _metrics_map(recs, settings, rel_prominence, cxr_metric)
    if mode == "heatmap":
        charts = [
            heatmap_chart(
                results, settings, quantity=(key, label, cmap), x=x, y=y, panel=panel,
                select=select, cases=cases, rel_prominence=rel_prominence,
                cxr_metric=cxr_metric, min_flux_frac=min_flux_frac,
                min_cxr_quality=min_cxr_quality, metrics=metrics,
            )
            for key, label, cmap in quantities
        ]
    else:
        cxr_x, hue = pick_hue(recs, x, y, panel, hue)
        charts = [
            metric_vs_chart(
                results, settings, x=cxr_x, metric=key, hue=hue,
                select=select, cases=cases, rel_prominence=rel_prominence,
                cxr_metric=cxr_metric, logx=logx, logy=logy, metrics=metrics,
            )
            for key, _, _ in quantities
        ]
    return [c for c in charts if c is not None]
