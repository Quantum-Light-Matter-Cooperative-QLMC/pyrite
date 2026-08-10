"""_frames

Renderer-neutral tidy-data builders for the parametric-sweep figures. Both
:mod:`pyrite.plots.mpl.sweeps` (matplotlib) and :mod:`pyrite.plots.altair.sweeps`
(Altair) render FROM these builders -- the per-cell/per-point "reduce every
swept case to its best record" reduction lives here exactly once. No
matplotlib or Altair imports; renderers turn the returned ``pandas.DataFrame``
/ guard values into figures.
"""

import pandas as pd

from ..results import records_for_cases, selection_score
from ..sweep import fmt_thickness
from ._common import _metrics_map


def _ndistinct(recs, field):
    return len({r["case"][field] for r in recs if field in r["case"]})


_FALLBACK_X = ("tilt_deg", "thickness_ang", "E0_keV", "tilt_azim_deg", "B_ang2")


def _effective_x(recs, x, hue):
    """If ``x`` sweeps <2 values, substitute the first fallback knob that
    actually sweeps (and isn't the hue), so a 1-D scan/scan-frame is a
    meaningful curve instead of a vertical stack. Shared by
    :func:`pyrite.plots.mpl.sweeps.plot_metric_vs` and :func:`metric_vs_frame`."""
    if _ndistinct(recs, x) >= 2:
        return x
    return next(
        (f for f in _FALLBACK_X if f != x and f != hue and _ndistinct(recs, f) >= 2),
        x,
    )


def scan_mode(recs, x, y, heatmap_min, force=None):
    """Heatmap-vs-lines auto-pick shared by ``plot_scan`` (matplotlib) and
    ``scan_charts`` (Altair): heatmap when BOTH ``x`` and ``y`` sweep at least
    ``heatmap_min`` values, else lines. ``force`` ("heatmap"|"lines")
    overrides the pick."""
    if force in ("heatmap", "lines"):
        return force
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    return "heatmap" if nx >= heatmap_min and ny >= heatmap_min else "lines"


def pick_hue(recs, x, y, panel, hue=None):
    """Line-mode axis/hue pick shared by ``plot_scan`` and ``scan_charts``:
    the denser of ``x``/``y`` becomes the line x-axis, the sparser becomes
    ``hue`` (unless ``hue`` is given explicitly), falling back to ``panel``
    if the sparser axis doesn't vary, and finally to the sparser axis itself
    if nothing else varies (a single line). Returns ``(line_x, hue)``."""
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    line_x, other = (x, y) if nx >= ny else (y, x)
    if hue is not None:
        return line_x, hue
    if _ndistinct(recs, other) >= 2:
        return line_x, other
    if _ndistinct(recs, panel) >= 2:
        return line_x, panel
    return line_x, other


# ---- axis/metric display registries (shared with pyrite.plots.mpl.sweeps) --------
# Per case field: (axis label, divide-to-display, display unit, value format).
# Lets ANY swept knob be a heatmap/scan axis with sensible labels and units.
_AXIS_SPECS = {
    "tilt_deg": ("polar tilt", 1.0, "deg", "{:g}"),
    "tilt_azim_deg": ("azimuthal tilt", 1.0, "deg", "{:g}"),
    "E0_keV": ("beam energy", 1.0, "keV", "{:g}"),
    "thickness_ang": ("thickness", 1e4, r"μm", "{:g}"),
    "B_ang2": ("B-factor", 1.0, r"$\AA^2$", "{:g}"),
}

# Line-characterization maps are meaningless where the line is ill-defined --
# either near-zero emission OR a broad ramp / a cluster of comparable peaks
# (low line_quality, see results.line_quality). Gate these by BOTH peak flux
# and line_quality. The always-well-defined maps (peak_flux, coherent_flux,
# total_flux) and the diagnostic line_quality map itself are never gated.
_FLUX_GATED = {"line_eV", "fwhm_eV", "line_frac", "line_flux"}


def _axis_disp(key, vals):
    """Swept raw values -> display units (e.g. thickness Angstrom -> microns)."""
    div = _AXIS_SPECS.get(key, (None, 1.0))[1]
    return [float(v) / div for v in vals]


def _value_label(key, v):
    """'30 keV' / '17 um' style label for one swept value."""
    if key == "thickness_ang":
        return fmt_thickness(float(v))
    _lbl, div, unit, fmt = _AXIS_SPECS.get(key, (key, 1.0, "", "{:g}"))
    return f"{fmt.format(float(v) / div)} {unit}".strip()


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
    metrics=None,
):
    """Tidy long-form table for a 1-D metric scan: one row per (hue value, x value),
    ``metric`` reduced over every OTHER swept dimension to its best geometry
    (``results.selection_score`` ``select``). Mirrors the data behind
    :func:`pyrite.plots.plot_metric_vs`, including the single-valued-x fallback.
    ``x`` is in DISPLAY units (e.g. thickness in microns). Columns:
    ``x, metric, hue`` (``hue`` is the ``"30 keV"``-style value label).
    ``metrics`` is a precomputed ``_common._metrics_map`` for THESE records
    (computed here when omitted) -- multi-quantity drivers pass one shared map
    instead of re-deriving it per quantity."""
    recs = records_for_cases(results, cases)
    if not recs:
        return pd.DataFrame(columns=["x", "metric", "hue"])
    x = _effective_x(recs, x, hue)
    if metrics is None:
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
    metrics=None,
):
    """Tidy long-form table for one heatmap quantity over ``x`` x ``y``, one block
    per ``panel`` value. Each (x, y) cell is reduced to its best record
    (``selection_score`` ``select``); flux-gated quantities blank out cells with
    near-zero emission or an ill-defined line (dropped rows -> gaps), mirroring
    :func:`pyrite.plots.plot_heatmaps`. ``x`` / ``y`` are in DISPLAY units.
    Columns: ``x, y, panel, value, name, panel_raw``. ``name`` is the config
    name of the cell's best record and ``panel_raw`` its raw ``panel`` value --
    both carried so an interactive click on a cell (see
    :func:`pyrite.plots.altair.sweeps.heatmap_select_chart`) maps back to an
    exact geometry / parameter set. ``metrics`` is a precomputed
    ``_common._metrics_map`` for THESE records (computed here when omitted) --
    multi-quantity drivers pass one shared map instead of re-deriving it per
    quantity."""
    cols = ["x", "y", "panel", "value", "name", "panel_raw"]
    recs = records_for_cases(results, cases)
    if not recs:
        return pd.DataFrame(columns=cols)
    if metrics is None:
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
                    "name": r["case"]["name"],
                    "panel_raw": float(pv),
                }
            )
    return pd.DataFrame(rows, columns=cols)
