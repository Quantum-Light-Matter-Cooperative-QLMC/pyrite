"""Map view: one clickable azimuth × polar-tilt heatmap, metric trends, and rankings."""

from pyrite.apps._design import static_altair_chart
from pyrite.plots.altair.spectra import spectrum_chart
from pyrite.plots.altair.sweeps import metric_vs_chart, scan_charts
from pyrite.results import select_results, top_geometries

from .common import themed_chart


def render_rankings(mo, *, context):
    frame = top_geometries(context.results, context.settings, top_n=20, select="quality_line")
    if frame.empty:
        return mo.md("*No ranked geometries for this checkpoint.*")
    return mo.vstack(
        [
            mo.md(
                "**Top 20 geometries** ranked by *quality × integrated line flux* "
                f"({context.settings.beam_current_na:g} nA beam; quality score in [0, 1])."
            ),
            mo.ui.table(
                frame,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                style_cell=lambda col, row_id, val: {"white-space": "nowrap"},
            ),
        ]
    )


def _selected_cell_spectra(mo, *, context, map_results, selection, energy, theme):
    if selection is None or len(selection) == 0:
        return mo.md("*Click a heatmap cell to plot that geometry's spectrum here.*")
    row = selection.iloc[0]
    tilt = float(row["y"])
    selected = select_results(
        map_results,
        tilt_azim_deg=float(row["x"]),
        tilt_deg=tilt,
        E0_keV=energy,
    )
    charts = [
        spectrum_chart(
            selected,
            context.settings,
            tilt_deg=tilt,
            include_coherent=context.show_both_emissions,
            band=band,
        )
        for band in ("narrow", "broad")
    ]
    charts = [
        static_altair_chart(mo, context.title_for_face(themed_chart(chart, theme)))
        for chart in charts
        if chart is not None
    ]
    heading = mo.md(f"**Selected cell** — polar tilt {tilt:g} deg, azimuth {float(row['x']):g} deg")
    return mo.vstack([heading, *charts]) if charts else mo.md("*No spectrum for that cell.*")


def render_map_trends(mo, *, context, map_results, theme):
    """Line/coherent flux versus polar tilt plus the inline geometry ranking."""
    metric_charts = [
        metric_vs_chart(map_results, context.settings, x="tilt_deg", metric=metric, hue="E0_keV")
        for metric in ("line_flux", "coherent_flux")
    ]
    metric_charts = [themed_chart(chart, theme) for chart in metric_charts if chart is not None]
    return mo.vstack(
        [
            mo.md("**Integrated line and coherent flux versus polar tilt**"),
            mo.vstack([static_altair_chart(mo, chart) for chart in metric_charts])
            if metric_charts
            else mo.md("*No metric results.*"),
            render_rankings(mo, context=context),
        ]
    )


def render_map(
    mo,
    *,
    context,
    map_results,
    heatmap_mode,
    quantity_ui,
    map_widget,
    selection,
    energy,
    trends,
    theme,
):
    """The Map view: quantity dropdown + heatmap (or 1-D fallback), then ``trends``.

    ``trends`` comes from :func:`render_map_trends` in its own cell, so clicking
    a heatmap cell does not recompute the metric charts or rankings.
    """
    parts = []
    if heatmap_mode:
        parts.extend(
            [
                mo.md(
                    "Azimuth × polar tilt at the sidebar beam energy and thickness. "
                    "Click a cell to plot its spectrum."
                ),
                quantity_ui,
                map_widget
                if map_widget is not None
                else mo.md("*No heatmap for this quantity at this slice.*"),
                _selected_cell_spectra(
                    mo,
                    context=context,
                    map_results=map_results,
                    selection=selection,
                    energy=energy,
                    theme=theme,
                ),
            ]
        )
    else:
        charts = scan_charts(
            map_results,
            context.settings,
            cases=context.cases,
            line_metric="prominence",
        )
        charts = [themed_chart(chart, theme) for chart in charts if chart is not None]
        parts.extend(
            [
                mo.md(
                    "*No heatmap — needs an azimuth × polar-tilt sweep. Scan plots at the "
                    "sidebar thickness follow.*"
                ),
                mo.vstack([static_altair_chart(mo, chart) for chart in charts])
                if charts
                else mo.md("*No scan results.*"),
            ]
        )

    parts.append(trends)
    return mo.vstack(parts)
