from pyrite.apps._design import apply_altair_theme, apply_plotly_theme
from pyrite.plots._frames import heatmap_frame
from pyrite.plots.altair.sweeps import heatmap_select_chart
from pyrite.plots.plotly.map_select import selectable_map_figure, single_panel_frame
from pyrite.results import records

from .controls import MAP_QUANTITIES
from .map_selection import selection_for_scope


def is_heatmap_sweep(results) -> bool:
    """A Map heatmap needs an azimuth × polar-tilt sweep (at least 4 × 4)."""
    tilts = {record["case"]["tilt_deg"] for record in records(results)}
    azimuths = {record["case"]["tilt_azim_deg"] for record in records(results)}
    return len(tilts) >= 4 and len(azimuths) >= 4


def make_map_widget(mo, alt, results, settings, cases, *, quantity, energy, theme):
    """One click-selectable heatmap of ``quantity`` at beam ``energy``, or ``None``."""
    if not records(results):
        return None
    specification = next(
        (entry for entry in MAP_QUANTITIES if entry[0] == quantity), MAP_QUANTITIES[0]
    )
    chart = heatmap_select_chart(
        results,
        settings,
        quantity=specification,
        panel_value=energy,
        cases=cases,
        cxr_metric="prominence",
        color_domain=(0.0, 1.0) if specification[0] == "hit_frac" else None,
    )
    if chart is None:
        return None
    # marimo needs the Vega-Lite selection params, which vegafusion compilation removes.
    with alt.data_transformers.enable("default"):
        return mo.ui.altair_chart(
            apply_altair_theme(chart, theme),
            chart_selection=False,
            legend_selection=False,
        )


def make_plotly_map_widget(
    mo, results, settings, cases, *, quantity, energy, theme, scope, on_selection
):
    """Opt-in map pilot; return ``(widget, valid_frame)`` or ``(None, None)``.

    The frame is built once per slice, not on a reactive selection update.
    Keep this path separate from the default Altair widget until browser
    click, keyboard accessibility and selection persistence are verified.
    """
    if not records(results):
        return None, None
    spec = next((entry for entry in MAP_QUANTITIES if entry[0] == quantity), MAP_QUANTITIES[0])
    frame = heatmap_frame(
        results, settings, quantity=spec[0], cases=cases,
        cxr_metric="prominence",
    )
    frame = single_panel_frame(frame, energy)
    chart = selectable_map_figure(
        frame, label=spec[1], cmap=spec[2],
        color_domain=(0.0, 1.0) if spec[0] == "hit_frac" else None,
    )
    if chart is None:
        return None, frame
    return (
        mo.ui.plotly(
            apply_plotly_theme(chart, theme),
            on_change=lambda points: on_selection(selection_for_scope(points, frame, scope)),
        ),
        frame,
    )
