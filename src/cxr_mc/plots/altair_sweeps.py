"""altair_sweeps

Altair / Vega-Lite renderers for the parametric-sweep figures -- the interactive
counterparts of the matplotlib :mod:`cxr_mc.plots.sweeps` heatmaps and metric
scans. Same contract as :mod:`cxr_mc.plots.altair_spectra`: these reuse the exact
per-record metric prep (``results.line_metrics`` + ``results.selection_score``)
and the shared axis/metric registries from :mod:`cxr_mc.plots.sweeps`, so the
numbers are identical -- only the renderer differs. Not re-exported from
``cxr_mc.plots`` (frozen export-set guard); import from the submodule:

    from cxr_mc.plots.altair_sweeps import metric_vs_chart, heatmap_chart, scan_charts

``metric_vs_chart`` / ``heatmap_chart`` return a single :class:`altair.Chart`
(``None`` when empty); ``scan_charts`` auto-picks heatmap-vs-lines per the swept
axis counts (mirroring :func:`cxr_mc.plots.plot_scan`) and returns a LIST of
charts, one per quantity. See :mod:`cxr_mc.plots.altair_spectra` for the
Vega-Lite 5000-row cap note (sweep frames are small -- one row per cell -- so this
is rarely an issue here).
"""

import altair as alt

from ..results import records_for_cases
from ._frames import (
    _effective_x,
    heatmap_frame,
    metric_vs_frame,
    pick_hue,
    scan_mode,
)
from .sweeps import (
    _AXIS_SPECS,
    _HEATMAP_QUANTITIES,
    _METRIC_LABELS,
    _axis_label,
    _resolve_quantity,
)

# matplotlib colormap name -> Vega-Lite colour scheme
_VEGA_SCHEME = {
    "viridis": "viridis",
    "plasma": "plasma",
    "magma": "magma",
    "cividis": "cividis",
    "inferno": "inferno",
    "Greens": "greens",
}


def _scheme(cmap):
    return _VEGA_SCHEME.get(cmap, str(cmap).lower())


# ---- 1-D metric scan ---------------------------------------------------------
def metric_vs_chart(
    results,
    settings,
    *,
    x="thickness_ang",
    metric="line_flux",
    hue="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    logx=False,
    logy=False,
    width=720,
    height=420,
):
    """Interactive 1-D parameter scan: ``metric`` vs swept ``x``, one line per
    ``hue`` value (best geometry per point). The Altair counterpart of
    :func:`cxr_mc.plots.plot_metric_vs`. Returns an :class:`altair.Chart`, or
    ``None`` when there are no records."""
    recs = records_for_cases(results, cases)
    if not recs:
        return None
    eff_x = _effective_x(recs, x, hue)
    df = metric_vs_frame(
        results,
        settings,
        x=x,
        metric=metric,
        hue=hue,
        select=select,
        cases=cases,
        rel_prominence=rel_prominence,
        line_metric=line_metric,
    )
    if df.empty:
        return None
    metric_label = _METRIC_LABELS.get(metric, metric)
    x_scale = alt.Scale(type="log") if logx else alt.Scale()
    y_scale = alt.Scale(type="log") if logy else alt.Scale()
    base = alt.Chart(df).encode(
        x=alt.X("x:Q", title=_axis_label(eff_x), scale=x_scale),
        y=alt.Y("metric:Q", title=metric_label, scale=y_scale),
        color=alt.Color("hue:N", title=_AXIS_SPECS.get(hue, (hue,))[0]),
        tooltip=["hue:N", "x:Q", "metric:Q"],
    )
    chart = base.mark_line(point=True, strokeWidth=1.8)
    title = f"{metric_label} vs {_axis_label(eff_x)}  (best per point: {select})"
    return chart.properties(width=width, height=height, title=title).interactive()


# ---- 2-D parametric heatmap --------------------------------------------------
def heatmap_chart(
    results,
    settings,
    *,
    quantity: "str | tuple" = "peak_flux",
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    cell=44,
):
    """Interactive parametric heatmap of one quantity over ``x`` x ``y``, faceted
    into one panel per ``panel`` value, best record per cell. The Altair
    counterpart of :func:`cxr_mc.plots.plot_heatmaps` (one quantity). ``quantity``
    is a bare metric key or a ``(key, label, cmap)`` triple. Returns an
    :class:`altair.Chart`, or ``None`` when there are no records."""
    key, label, cmap = _resolve_quantity(quantity)
    df = heatmap_frame(
        results,
        settings,
        quantity=key,
        x=x,
        y=y,
        panel=panel,
        select=select,
        cases=cases,
        rel_prominence=rel_prominence,
        line_metric=line_metric,
        min_flux_frac=min_flux_frac,
        min_line_quality=min_line_quality,
    )
    if df.empty:
        return None
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("x:O", title=_axis_label(x), sort="ascending"),
            y=alt.Y("y:O", title=_axis_label(y), sort="ascending"),
            color=alt.Color("value:Q", title=label, scale=alt.Scale(scheme=_scheme(cmap))),  # type: ignore[arg-type]
            tooltip=["panel:N", "x:O", "y:O", "value:Q"],
        )
        .properties(width=cell * max(df["x"].nunique(), 1), height=cell * max(df["y"].nunique(), 1))
        .facet(column=alt.Column("panel:N", title=_AXIS_SPECS.get(panel, (panel,))[0]))
        .properties(title=f"{label}    (best per cell: {select})")
    )
    return chart


# ---- auto-picking scan (heatmap vs lines) ------------------------------------
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
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    logx=False,
    logy=False,
    force=None,
):
    """Auto-pick a HEATMAP or LINE scan from how many values each axis sweeps, then
    render one chart per quantity -- the Altair counterpart of
    :func:`cxr_mc.plots.plot_scan`. Heatmap when BOTH ``x`` and ``y`` sweep
    >= ``heatmap_min`` values; otherwise lines (denser axis on x, sparser as hue).
    ``force`` overrides ("heatmap" | "lines"). Returns a LIST of
    :class:`altair.Chart` (empty when there are no records)."""
    recs = records_for_cases(results, cases)
    if not recs:
        return []
    quantities = [_resolve_quantity(q) for q in (quantities or _HEATMAP_QUANTITIES)]

    mode = scan_mode(recs, x, y, heatmap_min, force)

    if mode == "heatmap":
        charts = [
            heatmap_chart(
                results,
                settings,
                quantity=(key, label, cmap),
                x=x,
                y=y,
                panel=panel,
                select=select,
                cases=cases,
                rel_prominence=rel_prominence,
                line_metric=line_metric,
                min_flux_frac=min_flux_frac,
                min_line_quality=min_line_quality,
            )
            for key, label, cmap in quantities
        ]
        return [c for c in charts if c is not None]

    # line mode: denser axis -> x, sparser -> hue (unless hue is given)
    line_x, hue = pick_hue(recs, x, y, panel, hue)
    charts = [
        metric_vs_chart(
            results,
            settings,
            x=line_x,
            metric=key,
            hue=hue,
            select=select,
            cases=cases,
            rel_prominence=rel_prominence,
            line_metric=line_metric,
            logx=logx,
            logy=logy,
        )
        for key, _, _ in quantities
    ]
    return [c for c in charts if c is not None]
