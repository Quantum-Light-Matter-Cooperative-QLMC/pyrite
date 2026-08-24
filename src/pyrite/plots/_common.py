"""_common

Shared figure plumbing: per-record line/brem split and the per-tilt figure loop.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..detectors import Detector, LegacyEDS
from ..results import (
    beam_current_na,
    detected_background,
    line_metrics,
)

# cache of the (expensive) Timepix efficiency-curve response, keyed by hardware +
# MC settings, so re-running the detector cell with unchanged settings doesn't
# rebuild it. (The per-grid detected-spectra response is already cached inside
# timepix_response.get_response.)
_EFF_CACHE = {}


def _mode(settings):
    return (
        "EDS-convolved" if getattr(settings, "convolve_with_det", False) else "response-free source"
    )


def _peak_line(r):
    """Response-free source line peak of one record -- the azimuth-selection key
    (strongest line wins)."""
    return float(np.max(r["spec"]))


def _best_azimuth(grp, collapse_azimuth):
    """Collapse a same-energy record group to its single strongest-line azimuth
    when ``collapse_azimuth`` (and there's more than one to choose from); else
    return ``grp`` unchanged. The per-energy azimuth selection shared by every
    spectra/detector draw."""
    if collapse_azimuth and len(grp) > 1:
        return [max(grp, key=_peak_line)]
    return grp


def _case_title(case, tail="", *, latex=True, e0_keV=None, tilt_fmt="0.1f"):
    """Case-identifying title prefix -- material, thickness (um), polar tilt --
    shared by every spectra/detector title, with an optional inline beam energy
    (``e0_keV``) and a trailing renderer-specific clause (``tail``). ``latex=True``
    (matplotlib) renders `$\\mu$m` / `\\degree` math; ``latex=False`` (Vega-Lite)
    renders plain ASCII (`um` / `deg`) and joins ``tail`` with `--` instead of an
    em dash."""
    mat = case["name"].split()[0]
    thick = case["thickness_ang"] / 1e4
    tilt = format(case["tilt_deg"], tilt_fmt)
    if latex:
        e0 = f", {e0_keV:g} keV" if e0_keV is not None else ""
        head = rf"{mat}, {thick:.1f} $\mu$m{e0}, $\theta_\mathrm{{tilt}}={tilt}\degree$"
        return f"{head} — {tail}" if tail else head
    e0 = f", {e0_keV:g} keV" if e0_keV is not None else ""
    head = f"{mat}, {thick:.1f} um{e0}, theta_tilt={tilt} deg"
    return f"{head} -- {tail}" if tail else head


# Cross-call cache for the (expensive, per-record scipy peak-finding)
# `line_metrics` result -- `_metrics_map` already dedupes within one call via
# `id(rec)`, but every fresh call (e.g. a marimo tab re-rendering because an
# unrelated widget elsewhere changed) redid the whole O(records) pass from
# scratch. Measured on the densest checkpoint (mose2, 3720 records): this pass
# is why `heatmap_select_chart`/`scan_charts` cost ~0.9-1.1s per call.
#
# Keyed on CONTENT (the record's (name, E0_keV) pair), not `id(r)`/`id(settings)`:
# an identity key would be unsafe for a process-lifetime cache, since CPython
# reuses a freed object's address for the next allocation -- two unrelated
# records built at different times could collide on `id()` alone and silently
# return each other's metrics. `(case["name"], case["E0_keV"])` is already the
# results store's own primary key (`results[name][E0] = record`, see
# `results.store.store_result`), so it's guaranteed unique per record and --
# unlike the rest of `case` (which can carry unhashable `composition`/
# `hkl_list`/`abs_layers` entries) is always a plain hashable (str, float)
# pair. Derived source current is the only record/settings value
# `line_metrics` reads today -- extend this key if that grows.
_LINE_METRICS_CACHE = {}
_LINE_METRICS_CACHE_MAX = 100_000


def _cached_line_metrics(r, settings, rel_prominence, line_metric):
    case = r["case"]
    key = (
        case["name"],
        case["E0_keV"],
        beam_current_na(r, settings),
        rel_prominence,
        line_metric,
    )
    cached = _LINE_METRICS_CACHE.get(key)
    if cached is not None:
        return cached
    value = line_metrics(r, settings, rel_prominence, metric=line_metric)
    if len(_LINE_METRICS_CACHE) >= _LINE_METRICS_CACHE_MAX:
        _LINE_METRICS_CACHE.clear()
    _LINE_METRICS_CACHE[key] = value
    return value


def _metrics_map(recs, settings, rel_prominence, line_metric):
    """``id(rec) -> line_metrics(rec)`` for every record -- the same per-record
    metric dict every sweep draw (matplotlib and altair) builds. Computed once
    per record and cached ACROSS calls (see ``_cached_line_metrics``), so
    re-rendering after a change that doesn't touch these records (e.g. a
    marimo tab re-rendering because an unrelated widget changed) reuses the
    prior peak-finding results instead of redoing them."""
    return {id(r): _cached_line_metrics(r, settings, rel_prominence, line_metric) for r in recs}


def _line_brem(r, settings, convolve=None):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / brem-source flags. ``convolve`` overrides
    settings.convolve_with_det when given (True/False), so a caller can draw the
    response-free source (convolve=False) and detector-convolved (convolve=True) spectra
    side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    detector = Detector(response=LegacyEDS(apply_qe=settings.apply_detector_qe, convolve=do_conv))
    line_det = detector.score(r["E_grid"], r["spec"], fwhm_eV=r["fwhm"])
    brem_det = detected_background(r, settings, convolve=do_conv) / r["scale"]
    return line_det, brem_det


# ---- per-tilt figure helper --------------------------------------------------
def _per_tilt_figs(recs, settings, draw, figsize, *, empty_msg="no results yet", **kw):
    """Shared body of every ``plot_*`` wrapper: one freshly-drawn figure PER POLAR
    TILT. ``draw(fig, tilt_recs, settings, **kw)`` renders a single tilt onto a
    cleared figure (the same ``_draw_*`` the interactive ``browse`` uses), so the
    wrappers and the slider stay in lockstep. Handles the empty-records check, the
    per-tilt grouping, and collecting the figure list."""
    if not recs:
        print(empty_msg)
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    figs = []
    for t in tilts:
        fig = plt.figure(figsize=figsize)
        draw(fig, [r for r in recs if r["case"]["tilt_deg"] == t], settings, **kw)
        figs.append(fig)
    return figs
