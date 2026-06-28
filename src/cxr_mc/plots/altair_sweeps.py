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
import pandas as pd

from ..results import (
    line_metrics,
    records,
    selection_score,
)
from .sweeps import (
    _AXIS_SPECS,
    _FLUX_GATED,
    _HEATMAP_QUANTITIES,
    _METRIC_LABELS,
    _axis_disp,
    _axis_label,
    _resolve_quantity,
    _value_label,
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

_FALLBACK_X = ("tilt_deg", "thickness_ang", "E0_keV", "tilt_azim_deg", "B_ang2")


def _scheme(cmap):
    return _VEGA_SCHEME.get(cmap, str(cmap).lower())


def _ndistinct(recs, field):
    return len({r["case"][field] for r in recs if field in r["case"]})


def _metrics_map(recs, settings, rel_prominence, line_metric):
    """``id(rec) -> line_metrics(rec)`` for every record -- the same per-record
    metric dict the matplotlib sweeps build, computed once."""
    return {id(r): line_metrics(r, settings, rel_prominence, metric=line_metric) for r in recs}


# ---- 1-D metric scan ---------------------------------------------------------
def _effective_x(recs, x, hue):
    """Mirror :func:`cxr_mc.plots.plot_metric_vs`'s single-valued-x guard: if ``x``
    sweeps <2 values, substitute the first fallback knob that actually sweeps (and
    isn't the hue), so the curve is meaningful instead of a vertical stack."""
    if _ndistinct(recs, x) >= 2:
        return x
    return next(
        (f for f in _FALLBACK_X if f != x and f != hue and _ndistinct(recs, f) >= 2),
        x,
    )


def metric_vs_frame(
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
):
    """Tidy long-form table for a 1-D metric scan: one row per (hue value, x value),
    ``metric`` reduced over every OTHER swept dimension to its best geometry
    (``results.selection_score`` ``select``). Mirrors the data behind
    :func:`cxr_mc.plots.plot_metric_vs`, including the single-valued-x fallback.
    ``x`` is in DISPLAY units (e.g. thickness in microns). Columns:
    ``x, metric, hue`` (``hue`` is the ``"30 keV"``-style value label)."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        return pd.DataFrame(columns=["x", "metric", "hue"])
    x = _effective_x(recs, x, hue)
    metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
    div_x = _AXIS_SPECS.get(x, (None, 1.0))[1]
    rows = []
    for hv in sorted({r["case"][hue] for r in recs}):
        hr = [r for r in recs if r["case"][hue] == hv]
        for xv in sorted({r["case"][x] for r in hr}):
            cell = [r for r in hr if r["case"][x] == xv]
            best = max(cell, key=lambda r: selection_score(metrics[id(r)], select))
            rows.append(
                {
                    "x": float(xv) / div_x,
                    "metric": float(metrics[id(best)][metric]),
                    "hue": _value_label(hue, hv),
                }
            )
    return pd.DataFrame(rows, columns=["x", "metric", "hue"])


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
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
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
def heatmap_frame(
    results,
    settings,
    *,
    quantity="peak_flux",
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
):
    """Tidy long-form table for one heatmap quantity over ``x`` x ``y``, one block
    per ``panel`` value. Each (x, y) cell is reduced to its best record
    (``selection_score`` ``select``); flux-gated quantities blank out cells with
    near-zero emission or an ill-defined line (dropped rows -> gaps), mirroring
    :func:`cxr_mc.plots.plot_heatmaps`. ``x`` / ``y`` are in DISPLAY units.
    Columns: ``x, y, panel, value``."""
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        return pd.DataFrame(columns=["x", "y", "panel", "value"])
    metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
    gated = quantity in _FLUX_GATED
    rows = []
    for pv in sorted({r["case"][panel] for r in recs}):
        er = [r for r in recs if r["case"][panel] == pv]
        fmax = max((metrics[id(r)]["peak_flux"] for r in er), default=0.0)
        floor = min_flux_frac * fmax
        best = {}  # (xv, yv) -> (score, rec)
        for r in er:
            ck = (r["case"][x], r["case"][y])
            s = selection_score(metrics[id(r)], select)
            if ck not in best or s > best[ck][0]:
                best[ck] = (s, r)
        for (xv, yv), (_, r) in best.items():
            m = metrics[id(r)]
            if gated and (m["peak_flux"] < floor or m["line_quality"] < min_line_quality):
                continue  # near-zero emission / ill-defined line -> blank
            rows.append(
                {
                    "x": _axis_disp(x, [xv])[0],
                    "y": _axis_disp(y, [yv])[0],
                    "panel": _value_label(panel, pv),
                    "value": float(m[quantity]),
                }
            )
    return pd.DataFrame(rows, columns=["x", "y", "panel", "value"])


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
    names = None if cases is None else {c["name"] for c in cases}
    recs = records(results, names)
    if not recs:
        return []
    quantities = [_resolve_quantity(q) for q in (quantities or _HEATMAP_QUANTITIES)]

    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    if force in ("heatmap", "lines"):
        mode = force
    elif nx >= heatmap_min and ny >= heatmap_min:
        mode = "heatmap"
    else:
        mode = "lines"

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
    line_x, other = (x, y) if nx >= ny else (y, x)
    if hue is None:
        if _ndistinct(recs, other) >= 2:
            hue = other
        elif _ndistinct(recs, panel) >= 2:
            hue = panel
        else:
            hue = other
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
