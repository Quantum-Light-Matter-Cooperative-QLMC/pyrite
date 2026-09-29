from pyrite.apps._design import apply_altair_theme
from pyrite.plots.altair.sweeps import heatmap_select_chart
from pyrite.results import records

from .controls import MAP_QUANTITIES


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
        line_metric="prominence",
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
