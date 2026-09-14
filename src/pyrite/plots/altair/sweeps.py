"""Altair sweeps

Altair / Vega-Lite renderers for the parametric-sweep figures -- the interactive
counterparts of the matplotlib :mod:`pyrite.plots.mpl.sweeps` heatmaps and metric
scans. Same contract as :mod:`pyrite.plots.altair.spectra`: these reuse the exact
per-record metric prep (``results.line_metrics`` + ``results.selection_score``)
and the shared axis/metric registries from :mod:`pyrite.plots._frames`, so the
numbers are identical -- only the renderer differs. Not re-exported from
``pyrite.plots`` (frozen export-set guard); import from the submodule:

    from pyrite.plots.altair.sweeps import metric_vs_chart, heatmap_chart, scan_charts

``metric_vs_chart`` / ``heatmap_chart`` return a single :class:`altair.Chart`
(``None`` when empty); ``scan_charts`` auto-picks heatmap-vs-lines per the swept
axis counts (mirroring :func:`pyrite.plots.plot_scan`) and returns a LIST of
charts, one per quantity. See :mod:`pyrite.plots.altair.spectra` for the
Vega-Lite 5000-row cap note (sweep frames are small -- one row per cell -- so this
is rarely an issue here).
"""

import altair as alt

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
from ._typing import _mark_chart

# Tick-label format for numeric axes (d3-format): 3 significant digits, no
# trailing zeros -- keeps a 25-tilt sweep (e.g. np.linspace(0.1, 85, 25),
# which produces values like 33.300000000000004) from spamming full-precision
# floats down the heatmap axes.
_TICK_FMT = ".3~g"

# Fixed OVERALL chart footprint for heatmap_chart -- independent of how many
# distinct x/y values are swept. Vega-Lite auto-divides an ordinal scale's
# fixed range into equal bands, so a 40-tilt sweep gets thinner cells rather
# than a wider chart (the "heatmaps expand with axis size" bug).
_HEATMAP_WIDTH = 360
_HEATMAP_HEIGHT = 300


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
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    logx=False,
    logy=False,
    width=720,
    height=420,
    metrics=None,
):
    """Interactive 1-D parameter scan: ``metric`` vs swept ``x``, one line per
    ``hue`` value (best geometry per point). The Altair counterpart of
    :func:`pyrite.plots.plot_metric_vs`. ``metrics`` is a precomputed
    ``_common._metrics_map`` (computed in the frame builder when omitted);
    :func:`scan_charts` passes one shared map across its quantities. Returns an
    :class:`altair.Chart`, or ``None`` when there are no records."""
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
        metrics=metrics,
    )
    if df.empty:
        return None
    metric_label = _METRIC_LABELS.get(metric, metric)
    x_scale = alt.Scale(type="log") if logx else alt.Scale()
    y_scale = alt.Scale(type="log") if logy else alt.Scale()
    base = alt.Chart(df).encode(
        x=alt.X("x:Q", title=_axis_label(eff_x), scale=x_scale, axis=alt.Axis(format=_TICK_FMT)),
        y=alt.Y("metric:Q", title=metric_label, scale=y_scale, axis=alt.Axis(format=_TICK_FMT)),
        color=alt.Color("hue:N", title=_AXIS_SPECS.get(hue, (hue,))[0]),
        tooltip=["hue:N", "x:Q", "metric:Q"],
    )
    chart = _mark_chart(base.mark_line(point=True, strokeWidth=1.8))
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
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    width=_HEATMAP_WIDTH,
    height=_HEATMAP_HEIGHT,
    metrics=None,
    color_domain=None,
):
    """Interactive parametric heatmap of one quantity over ``x`` x ``y``, faceted
    into one panel per ``panel`` value, best record per cell. The Altair
    counterpart of :func:`pyrite.plots.plot_heatmaps` (one quantity). ``quantity``
    is a bare metric key or a ``(key, label, cmap)`` triple. ``width``/``height``
    are the FIXED per-panel footprint (independent of how many x/y values are
    swept -- Vega-Lite auto-divides the ordinal scale's fixed range into equal
    bands, so a dense sweep gets thinner cells rather than a wider chart).
    ``metrics`` is a precomputed ``_common._metrics_map`` (computed in the frame
    builder when omitted); :func:`scan_charts` passes one shared map across its
    quantities. ``color_domain`` pins the colour scale to a fixed ``[lo, hi]``
    (default None -> auto-fit to the data) -- e.g. ``(0.0, 1.0)`` for a fraction
    so full colour always means 1.0, not just this frame's max.
    Returns an :class:`altair.Chart`, or ``None`` when there are no records."""
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
        metrics=metrics,
    )
    if df.empty:
        return None
    _scale = (
        alt.Scale(scheme=_scheme(cmap))  # type: ignore[arg-type]
        if color_domain is None
        else alt.Scale(scheme=_scheme(cmap), domain=list(color_domain))  # type: ignore[arg-type]
    )
    chart = (
        _mark_chart(alt.Chart(df).mark_rect())
        .encode(
            x=alt.X("x:O", title=_axis_label(x), sort="ascending", axis=alt.Axis(format=_TICK_FMT)),
            y=alt.Y("y:O", title=_axis_label(y), sort="ascending", axis=alt.Axis(format=_TICK_FMT)),
            color=alt.Color("value:Q", title=label, scale=_scale),  # type: ignore[arg-type]
            tooltip=["panel:N", "x:O", "y:O", "value:Q"],
        )
        .properties(width=width, height=height)
        .facet(
            column=alt.Column(
                "panel:N",
                title=_AXIS_SPECS.get(panel, (panel,))[0],
                # Nominal fields default to alphabetical sort ("100 keV" before
                # "30 keV") -- order panels by the underlying numeric panel_raw
                # instead, so energy facets read low -> high.
                sort=alt.EncodingSortField(field="panel_raw", op="min", order="ascending"),
            )
        )
        .properties(title=f"{label}    (best per cell: {select})")
    )
    return chart


# ---- click-selectable single-panel heatmap ----------------------------------
def heatmap_select_chart(
    results,
    settings,
    *,
    quantity: "str | tuple" = "peak_flux",
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    panel_value=None,
    select="quality_line",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    width=_HEATMAP_WIDTH,
    height=_HEATMAP_HEIGHT,
    metrics=None,
    color_domain=None,
):
    """A SINGLE-panel, click-selectable variant of :func:`heatmap_chart`, for
    picking one cell interactively. Unlike ``heatmap_chart`` this does NOT facet
    (faceted charts don't compose with a marimo point selection); instead it
    renders exactly one ``panel`` -- ``panel_value`` (a raw ``panel`` value,
    e.g. an E0 in keV) selects which, defaulting to the first present. An Altair
    ``selection_point`` on the (x, y) encodings is attached via ``add_params``
    so a marimo ``mo.ui.altair_chart(chart, chart_selection=False)`` wrapper
    exposes the clicked row -- carrying ``name`` / ``panel_raw`` from
    :func:`heatmap_frame` -- through ``chart.value``. A second, purely visual
    hover selection dims cells other than the one under the pointer (full color
    at rest, and again once the pointer leaves the chart); the click selection
    itself carries no opacity/dimming so it doesn't wash out the whole grid
    between clicks. The clicked cell instead gets a stroke outline. ``quantity``
    is a bare key or ``(key, label, cmap)`` triple. ``color_domain`` pins the
    colour scale to a fixed ``[lo, hi]`` (default None -> auto-fit to the data),
    same as :func:`heatmap_chart`. Returns an :class:`altair.Chart`, or ``None``
    when the (filtered) frame is empty."""
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
        metrics=metrics,
    )
    if df.empty:
        return None
    # Reduce to a single panel so one grid <-> one (x, y) cell (no facet).
    keep = panel_value if panel_value is not None else sorted(df["panel_raw"].unique())[0]
    df = df.loc[df["panel_raw"] == keep]
    if df.empty:
        return None
    sel = alt.selection_point(on="click", encodings=["x", "y"], empty=False)
    # Name matches marimo's "bin_coloring" convention (frontend/src/plugins/impl/vega/params.ts):
    # params so-named are excluded from the signal listeners that feed `mo.ui.altair_chart(...).value`,
    # so this purely-visual hover selection can never perturb the click selection read downstream.
    # Leave nearest=False/omitted: nearest selects from rect anchor points, offsetting
    # the hover hitbox by half a heatmap cell instead of using the rect under the pointer.
    hover = alt.selection_point(
        name="bin_coloring",
        on="pointerover",
        empty=True,
        clear="mouseout",
        encodings=["x", "y"],
    )
    panel_label = str(df["panel"].to_numpy()[0])
    _scale = (
        alt.Scale(scheme=_scheme(cmap))  # type: ignore[arg-type]
        if color_domain is None
        else alt.Scale(scheme=_scheme(cmap), domain=list(color_domain))  # type: ignore[arg-type]
    )
    chart = (
        _mark_chart(alt.Chart(df).mark_rect())
        .encode(
            x=alt.X("x:O", title=_axis_label(x), sort="ascending", axis=alt.Axis(format=_TICK_FMT)),
            y=alt.Y("y:O", title=_axis_label(y), sort="ascending", axis=alt.Axis(format=_TICK_FMT)),
            color=alt.Color("value:Q", title=label, scale=_scale),  # type: ignore[arg-type]
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.35)),
            stroke=alt.condition(sel, alt.value("black"), alt.value(None)),
            strokeWidth=alt.condition(sel, alt.value(2.0), alt.value(0.0)),
            tooltip=["panel:N", "x:O", "y:O", "value:Q", "name:N"],
        )
        .add_params(sel, hover)
        .properties(
            width=width,
            height=height,
            title=f"{label}  @ {panel_label}    (click a cell)",
        )
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
    select="quality_line",
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
    :func:`pyrite.plots.plot_scan`. Heatmap when BOTH ``x`` and ``y`` sweep
    >= ``heatmap_min`` values; otherwise lines (denser axis on x, sparser as hue).
    ``force`` overrides ("heatmap" | "lines"). Returns a LIST of
    :class:`altair.Chart` (empty when there are no records)."""
    recs = records_for_cases(results, cases)
    if not recs:
        return []
    quantities = [_resolve_quantity(q) for q in (quantities or _HEATMAP_QUANTITIES)]

    mode = scan_mode(recs, x, y, heatmap_min, force)
    # one metrics map shared across every quantity's chart (quantity-independent)
    metrics = _metrics_map(recs, settings, rel_prominence, line_metric)

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
                metrics=metrics,
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
            metrics=metrics,
        )
        for key, _, _ in quantities
    ]
    return [c for c in charts if c is not None]
