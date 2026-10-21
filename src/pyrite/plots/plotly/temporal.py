"""Interactive temporal intensity curves from renderer-neutral profile frames."""

import plotly.graph_objects as go
from plotly.colors import qualitative

_COLORS = qualitative.Plotly

from .._temporal_frames import temporal_frame


def temporal_chart(
    results, *, hue="E0_keV", include_coherent=True, max_points=5000,
    width=720, height=300,
):
    """Line intensity I(t), preserving the original incoherent/coherent components."""
    frame = temporal_frame(results, hue=hue, include_coherent=include_coherent, max_points=max_points)
    if frame.empty:
        return None
    fig = go.Figure()
    for i, (label, group) in enumerate(frame.groupby(hue, sort=False, dropna=False)):
        for component, part in group.groupby("component", sort=False):
            part = part.sort_values("time_fs")
            fig.add_trace(
                go.Scatter(
                    x=part["time_fs"].to_numpy(dtype=float),
                    y=part["intensity"].to_numpy(dtype=float),
                    mode="lines",
                    line=dict(color=_COLORS[i % len(_COLORS)],
                              dash="dash" if component == "coherent" else "solid"),
                    name=f"{label} · {component}",
                    legendgroup=str(i),
                    hovertemplate=f"{hue}: {label}<br>{component}<br>Time: %{{x:.5g}} fs"
                                  "<br>Intensity: %{y:.5g} photons/sr/e⁻/fs<extra></extra>",
                )
            )
    fig.update_layout(
        title="Line temporal intensity profile", width=width, height=height,
        xaxis_title="Arrival time − mean (fs)",
        yaxis_title="I(t) (photons/sr/e⁻/fs)",
        legend=dict(title=hue),
    )
    return fig
