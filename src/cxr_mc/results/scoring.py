"""
results.scoring
================

Ranking records by their :func:`~cxr_mc.results.metrics.line_metrics`:
:func:`selection_score` is the shared "how good is this geometry" scalar
behind the heatmap cell reduction, ``plot_metric_vs``, and the top-N browser
here (:func:`top_geometries` / :func:`show_top`).
"""

import numpy as np
import pandas as pd

from .metrics import line_metrics
from .selection import records

# ---- "best geometry" selection -----------------------------------------------
# Modes for ranking a record by its line_metrics, shared by every "pick the best
# case" path (the heatmap cell reduction, plot_metric_vs, the top-N browser), so
# they all agree on what "best" means.
SELECTION_MODES = ("peak", "line_flux", "coherent_flux", "quality_peak", "quality_line")


def selection_score(m, mode="quality_peak"):
    """Score a line_metrics dict ``m`` for "best geometry" selection. Higher wins.

      "peak"          : peak spectral flux (the legacy best_azimuth criterion --
                        favours the tallest spike, spurious lines included).
      "line_flux"     : integrated flux under the dominant found line.
      "coherent_flux" : integrated flux of ALL coherent lines (no peak finding).
      "quality_peak"  : peak_flux * line_quality (DEFAULT) -- favours geometries
                        that are both bright AND have a well-defined line, so a
                        tall-but-messy spike loses to a clean line.
      "quality_line"  : line_flux * line_quality.

    ``m`` is the dict from line_metrics. Non-finite scores sort to the bottom."""
    q = m.get("line_quality", 1.0)
    val = {
        "peak": m["peak_flux"],
        "line_flux": m["line_flux"],
        "coherent_flux": m["coherent_flux"],
        "quality_peak": m["peak_flux"] * q,
        "quality_line": m["line_flux"] * q,
    }.get(mode)
    if val is None:
        raise ValueError(f"unknown select mode {mode!r}; have {list(SELECTION_MODES)}")
    return val if np.isfinite(val) else -np.inf


# ---- compact ranked table ----------------------------------------------------
def top_geometries(
    results,
    settings,
    top_n=15,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
    names=None,
):
    """A compact, ranked table of the BEST geometries across a results store --
    the readable alternative to dumping every (tilt, azimuth, energy) row. Ranks
    by results.selection_score(``select``) and returns the top ``top_n`` as a
    best-first DataFrame: polar/azimuth tilt, beam energy, dominant line
    energy, line-definition quality, peak spectral flux, integrated coherent flux,
    and the dominant line's share of the total. ``names`` restricts to those
    configs (e.g. one material)."""
    recs = records(results, names)
    if not recs:
        return pd.DataFrame()
    scored = []
    for r in recs:
        m = line_metrics(r, settings, rel_prominence, metric=line_metric)
        scored.append((selection_score(m, select), r, m))
    scored.sort(key=lambda t: -t[0])
    rows = []
    for rank, (_, r, m) in enumerate(scored[:top_n], 1):
        c = r["case"]
        rows.append(
            {
                "rank": rank,
                "polar": round(c["tilt_deg"], 1),
                "azim": round(c["tilt_azim_deg"], 1),
                "E [keV]": c["E0_keV"],
                "line [eV]": round(m["line_eV"]),
                "quality": round(m["line_quality"], 2),
                "peak [Phs/eV/s]": float(f"{m['peak_flux']:.3g}"),
                "coherent [Phs/s]": float(f"{m['coherent_flux']:.3g}"),
                "line/tot": (round(m["line_frac"], 2) if np.isfinite(m["line_frac"]) else np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("rank")


def show_top(results, settings, top_n=15, select="quality_peak", **kw):
    """Print + render the compact :func:`top_geometries` table -- a short, sorted
    'here are the best N geometries' view instead of the full per-row dump."""
    from IPython.display import display

    df = top_geometries(results, settings, top_n=top_n, select=select, **kw)
    if df.empty:
        print("no results yet")
        return
    print(
        f"top {len(df)} geometries by '{select}'  (beam {settings.beam_current_na:g} nA; "
        f"peak = intrinsic coherent line density; quality in [0,1])"
    )
    display(df)
