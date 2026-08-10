from __future__ import annotations

from pyrite.plots.altair.sweeps import heatmap_select_chart
from pyrite.plots.mpl.sweeps import _HEATMAP_QUANTITIES as HEATMAP_QUANTITIES
from pyrite.results import records


def make_heatmap_widget(mo, alt, results, settings, *, energy):
    if not records(results):
        return None
    chart = heatmap_select_chart(
        results,
        settings,
        panel_value=energy,
        line_metric="prominence",
    )
    if chart is None:
        return None
    # marimo needs the Vega-Lite selection params, which vegafusion compilation removes.
    with alt.data_transformers.enable("default"):
        return mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)


def make_scan_heatmap_widgets(mo, alt, results, settings, cases, *, energy):
    specifications = list(HEATMAP_QUANTITIES) + [
        ("hit_frac", "electron footprint-hit fraction", "magma")
    ]
    widgets = {}
    if not records(results):
        return {key: None for key, _label, _color_map in specifications}
    with alt.data_transformers.enable("default"):
        for key, label, color_map in specifications:
            chart = heatmap_select_chart(
                results,
                settings,
                quantity=(key, label, color_map),
                panel_value=energy,
                cases=cases,
                line_metric="prominence",
                color_domain=(0.0, 1.0) if key == "hit_frac" else None,
            )
            widgets[key] = (
                mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
                if chart is not None
                else None
            )
    return widgets
