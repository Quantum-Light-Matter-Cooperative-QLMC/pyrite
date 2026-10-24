"""Shared Plotly angle × energy maps and profile cuts for the radiation apps.

Figures only redraw already computed arrays; nothing here scores or solves.
Every panel stores values with shape (angle, energy). NaN marks unsupported
samples and is drawn as the gray plot background, never as zero.
"""

from dataclasses import dataclass
from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: Gray behind NaN (unsupported) cells, distinct from the colormap floor.
UNSUPPORTED_GRAY = "#d8d8d8"


@dataclass(frozen=True, eq=False)
class AngularPanel:
    """One computed map: ``values[angle, energy]`` in the figure's value unit."""

    title: str
    angle_deg: np.ndarray
    energy_eV: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        shape = (len(self.angle_deg), len(self.energy_eV))
        if np.shape(self.values) != shape:
            raise ValueError(f"panel values must have shape (angle, energy) = {shape}")


def geometric(grid) -> bool:
    """Whether a grid is geometric (log-spaced) rather than linear."""
    grid = np.asarray(grid, dtype=float)
    if len(grid) <= 2 or grid[0] <= 0:
        return False
    ratio = grid[1:] / grid[:-1]
    return bool(np.allclose(ratio, ratio[0])) and ratio[0] > 1.01


def color_range(panels, *, logarithmic=True, dynamic_range=1e8):
    """Shared (lo, hi) over every panel; log ranges are in decades."""
    finite = np.concatenate([np.ravel(p.values)[np.isfinite(np.ravel(p.values))] for p in panels])
    positive = finite[finite > 0]
    if logarithmic:
        if not len(positive):
            return (-1.0, 0.0)
        hi = float(np.log10(positive.max()))
        return (max(float(np.log10(positive.min())), hi - np.log10(dynamic_range)), hi)
    hi = float(finite.max()) if len(finite) else 1.0
    lo = min(0.0, float(finite.min())) if len(finite) else 0.0
    return (lo, hi if hi > lo else lo + 1.0)


def _closed(panel: AngularPanel, periodic: bool):
    """Append the first periodic angle at +360° so the circle closes visually."""
    if not periodic:
        return panel.angle_deg, panel.values
    angle = np.append(panel.angle_deg, panel.angle_deg[0] + 360)
    return angle, np.concatenate((panel.values, panel.values[:1]), axis=0)


def heatmap_figure(
    panels,
    *,
    value_label: str,
    angle_label: str = "Signed observation angle (deg)",
    logarithmic: bool = True,
    dynamic_range: float = 1e8,
    periodic: bool = False,
    log_energy: bool | None = None,
    height: int = 460,
):
    """Side-by-side heatmaps sharing both axes and one color scale.

    Only the leftmost panel labels its energy axis and one colorbar carries
    the value label. Hover reports the true (unlogged) value. Log color
    clips at ``dynamic_range`` below the maximum; zeros take the floor color.
    """
    panels = tuple(panels)
    if not panels:
        raise ValueError("supply at least one panel")
    lo, hi = color_range(panels, logarithmic=logarithmic, dynamic_range=dynamic_range)
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[p.title for p in panels],
    )
    for col, panel in enumerate(panels, start=1):
        angle, values = _closed(panel, periodic)
        shown = values.T
        if logarithmic:
            with np.errstate(divide="ignore", invalid="ignore"):
                shown = np.where(np.isnan(shown), np.nan, np.log10(np.clip(shown, 10.0**lo, None)))
        fig.add_trace(
            go.Heatmap(
                x=angle,
                y=panel.energy_eV,
                z=shown,
                customdata=values.T,
                coloraxis="coloraxis",
                hovertemplate=(
                    "angle %{x:.4g}°<br>E %{y:.5g} eV<br>%{customdata:.4g}<extra>"
                    + panel.title
                    + "</extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(range=[angle[0], angle[-1]], row=1, col=col)
    log_y = geometric(panels[0].energy_eV) if log_energy is None else log_energy
    fig.update_yaxes(type="log" if log_y else "linear")
    fig.update_yaxes(title_text="Photon energy (eV)", row=1, col=1)
    colorbar: dict[str, object] = {"title": {"text": value_label, "side": "right"}}
    if logarithmic:
        tick = np.arange(np.ceil(lo), np.floor(hi) + 1)
        colorbar.update(tickvals=tick, ticktext=[f"1e{int(t)}" for t in tick])
    fig.update_layout(
        coloraxis=dict(colorscale="Inferno", cmin=lo, cmax=hi, colorbar=colorbar),
        plot_bgcolor=UNSUPPORTED_GRAY,
        height=height,
        margin=dict(l=70, r=20, t=50, b=60),
    )
    fig.add_annotation(
        text=angle_label,
        x=0.5,
        y=-0.14,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13),
    )
    return fig


def nearest_index(grid, value) -> int:
    return int(np.argmin(np.abs(np.asarray(grid, dtype=float) - value)))


def profile_figure(
    panels,
    *,
    energy_eV: float,
    angle_deg: float,
    value_label: str,
    angle_label: str = "Signed observation angle (deg)",
    logarithmic: bool = True,
    height: int = 340,
):
    """Angular cut at the nearest energy and spectral cut at the nearest angle.

    One trace per panel in each subplot, so tilts or components overlay.
    The plotted sample coordinates are reported in the subplot titles.
    """
    panels = tuple(panels)
    first = panels[0]
    e_index = nearest_index(first.energy_eV, energy_eV)
    a_index = nearest_index(first.angle_deg, angle_deg)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=(
            f"Angular cut at {first.energy_eV[e_index]:.5g} eV",
            f"Spectrum at {first.angle_deg[a_index]:.4g}°",
        ),
    )
    palette = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b")
    for number, panel in enumerate(panels):
        color = palette[number % len(palette)]
        e_i = nearest_index(panel.energy_eV, energy_eV)
        a_i = nearest_index(panel.angle_deg, angle_deg)
        style = dict(name=panel.title, legendgroup=panel.title, line=dict(color=color))
        fig.add_trace(go.Scatter(x=panel.angle_deg, y=panel.values[:, e_i], **style), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=panel.energy_eV, y=panel.values[a_i], showlegend=False, **style),
            row=1,
            col=2,
        )
    kind = "log" if logarithmic else "linear"
    fig.update_yaxes(type=kind, title_text=value_label, row=1, col=1)
    fig.update_yaxes(type=kind, row=1, col=2)
    fig.update_xaxes(title_text=angle_label, row=1, col=1)
    fig.update_xaxes(
        title_text="Photon energy (eV)",
        type="log" if geometric(first.energy_eV) else "linear",
        row=1,
        col=2,
    )
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=50, b=50))
    return fig


def material_options() -> dict[str, str]:
    """Catalog material labels mapped to keys; target lowering validates each."""
    from ..materials import CATALOG

    return {CATALOG.material(key).label: key for key in CATALOG.material_keys}


def transport_cores(backend: str) -> list[str]:
    """Transport cores runnable on the process array backend.

    The CUDA core needs the CUDA array backend, which is fixed at launch
    (``pyrite app <name> launch --backend cuda``).
    """
    return ["auto", "lockstep", "per-electron", *(["cuda"] if backend == "cuda" else [])]


def form_template(groups: dict[str, tuple[str, ...]]) -> str:
    """HTML batch template: one labeled column per control group.

    Values name the batch keys rendered, in order, inside each group; the grid
    wraps to fewer columns on narrow screens.
    """
    columns = "".join(
        '<div class="pyrite-group" style="display:flex;flex-direction:column;gap:.55rem">'
        f'<div class="pyrite-shell pyrite-group__label">{escape(label)}</div>'
        + "".join(f"<div>{{{key}}}</div>" for key in keys)
        + "</div>"
        for label, keys in groups.items()
    )
    return (
        '<div style="display:grid;gap:1.25rem 1.5rem;'
        'grid-template-columns:repeat(auto-fit,minmax(19rem,1fr))">'
        f"{columns}</div>"
    )
