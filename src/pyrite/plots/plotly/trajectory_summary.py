"""Plotly views of renderer-free saved-trajectory summary tables.

The tables come from :func:`pyrite.montecarlo.trajectory_summary.summarize_trajectories`;
this module only draws them. Histogram bars span their explicit ``low_*`` /
``high_*`` edges; non-finite overflow bins stay in the source tables.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .penetration import _dark_layout


def _span_bars(fig, table, low, high, y, *, name=None, row=None, col=None):
    finite = table[np.isfinite(table[low]) & np.isfinite(table[high])]
    lo = finite[low].to_numpy(dtype=float)
    hi = finite[high].to_numpy(dtype=float)
    fig.add_trace(
        go.Bar(
            x=0.5 * (lo + hi),
            width=hi - lo,
            y=finite[y].to_numpy(dtype=float),
            name=name,
            showlegend=name is not None,
            customdata=np.column_stack([lo, hi]),
            hovertemplate="[%{customdata[0]:.4g}, %{customdata[1]:.4g}): %{y:.4g}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _summary_figure(name, table):
    if name == "fates":
        fig = go.Figure(go.Bar(x=table["fate"].astype(str), y=table["fraction"]))
        return fig, "fate", "fraction"
    if name == "generations":
        fig = go.Figure(go.Bar(x=table["generation"].astype(str), y=table["tracks"]))
        return fig, "generation", "tracks"
    if name == "depth_loss":
        fig = go.Figure()
        for layer, part in table.groupby("layer", sort=False):
            _span_bars(
                fig, part, "depth_low_ang", "depth_high_ang", "continuous_loss_keV", name=str(layer)
            )
        return fig, "depth (Å)", "continuous loss (keV/bin)"
    low = next(key for key in table if key.startswith("low_"))
    high = low.replace("low_", "high_")
    if name == "hard_events":
        channels = list(dict.fromkeys(table["hard_channel"].astype(str)))
        fig = make_subplots(rows=1, cols=len(channels), subplot_titles=channels, shared_yaxes=True)
        kinds = list(dict.fromkeys(table["event_kind"].astype(str)))
        for col, channel in enumerate(channels, start=1):
            for kind in kinds:
                part = table[
                    (table["hard_channel"].astype(str) == channel)
                    & (table["event_kind"].astype(str) == kind)
                ]
                if not part.empty:
                    _span_bars(
                        fig,
                        part,
                        low,
                        high,
                        "count",
                        name=kind if col == 1 else None,
                        row=1,
                        col=col,
                    )
        fig.update_layout(barmode="stack")
        return fig, low.removeprefix("low_"), "count"
    fig = go.Figure()
    _span_bars(fig, table, low, high, "count")
    return fig, low.removeprefix("low_"), "count"


def summary_charts(tables: dict, *, width: int = 420, height: int = 220) -> dict:
    """One Plotly figure per non-empty summary table, with explicit units."""
    charts = {}
    for name, table in tables.items():
        if table.empty:
            continue
        fig, x_title, y_title = _summary_figure(name, table)
        charts[name] = _dark_layout(
            fig,
            title=name.replace("_", " "),
            width=width,
            height=height,
            x_title=x_title,
            y_title=y_title,
        )
    return charts
