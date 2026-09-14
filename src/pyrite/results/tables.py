"""
results.tables
==============

DataFrame views over a ``results`` store: the tidy long-format
:func:`results_dataframe` (one row per config/energy, every swept case knob +
line metrics) and the photon-counting :func:`summary_table` /
:func:`show_summary`.
"""

from collections.abc import Hashable, Sequence
from typing import cast

import numpy as np
import pandas as pd

from .._formatting import fmt_thickness
from ..detectors import Detector, LegacyEDS
from ..materials import CATALOG
from .metrics import line_metrics
from .store import Settings, beam_current_na, detected_background


def results_dataframe(
    results,
    settings=None,
    *,
    metrics=True,
    rel_prominence=0.03,
    n_fwhm=3.0,
    metric="sharpness",
):
    """Tidy long-format ``DataFrame``: one row per (config, beam energy), columns =
    every swept ``case`` knob + the derived :func:`line_metrics`. This is the
    first-class primitive for slicing/faceting many simultaneously-swept knobs
    (TODO P3 #9) -- ``df.groupby`` / ``df.pivot_table`` / seaborn ``relplot`` /
    plotly over ANY subset of knobs, instead of the heatmap/line auto-pick. See
    :func:`plots.facet_metric` for a small-multiples view built on it.

    metrics : include the line metrics (``peak_flux``, ``coherent_flux``,
        ``line_eV``, ``fwhm_eV``, ``line_flux``, ``line_frac``, ``total_flux``,
        ``coherent_brem_ratio``, ``line_quality``); needs a :class:`Settings`
        (defaults to ``Settings()``). False -> just the case knobs + the cheap
        stored scalars (``E_pk``, ``eta``), skipping the peak-finding.
    """
    if settings is None:
        settings = Settings()
    rows = []
    for name, by_E in results.items():
        for E0, r in by_E.items():
            row = {"name": name, **r["case"]}
            row.setdefault("E0_keV", E0)
            row["E_pk"] = r.get("E_pk")
            row["eta"] = r.get("eta")
            if metrics:
                row.update(line_metrics(r, settings, rel_prominence, n_fwhm, metric))
            rows.append(row)
    return pd.DataFrame(rows)


# ---- statistics table --------------------------------------------------------
_ROUND = {
    "polar [deg]": 1,
    "azimuth [deg]": 1,
    "line [eV]": 0,
    "peak [Phs/eV/s]": 2,
    "peak [Phs/eV/s/nA]": 2,
    "peak/bg": 2,
    "line [cts/s]": 1,
    "line [cts/s/nA]": 1,
    "brem [cts/s]": 1,
    "brem [cts/s/nA]": 1,
    "total [cts/s]": 1,
    "total [cts/s/nA]": 1,
}
_CONFIG_COLS = {"material", "thickness", "polar [deg]", "azimuth [deg]"}


def summary_table(recs, settings):
    """Photon-counting stats for a list of records. The geometry (material,
    thickness, polar/azimuthal tilt) is broken out of the config name into its
    own columns under a 'config' super-header; the rest are the line peak, the
    EDS-convolved peak height, peak-over-background, and the integrated line /
    brem / total count rates [counts/s] at the case's bunch charge and
    repetition rate, alongside current-normalized per-nA values. Old checkpoints fall
    back to ``settings.beam_current_na``. Returns a DataFrame with a 2-level
    column index (empty if ``recs`` is empty).

    Parameters
    ----------
    recs
        Iterable of result records.
    settings
        Read-time detector and legacy current controls.

    Returns
    -------
    pandas.DataFrame
        One row per record. Non-empty results use a two-level column index with
        geometry under ``"config"``; empty input returns an empty frame.
    """
    rows = []
    for r in sorted(
        recs,
        key=lambda r: (
            r["case"]["tilt_deg"],
            r["case"]["tilt_azim_deg"],
            r["case"]["E0_keV"],
        ),
    ):
        c = r["case"]
        cur = beam_current_na(r, settings)
        detector = Detector(response=LegacyEDS(apply_qe=settings.apply_detector_qe, convolve=True))
        line_det = detector.score(r["E_grid"], r["spec"], fwhm_eV=r["fwhm"])
        brem_det = detected_background(r, settings) / r["scale"]
        i_pk = np.argmax(line_det)
        line_cts_per_na = np.trapezoid(r["spec"], r["E_grid"]) * r["scale"]
        line_cts = line_cts_per_na * cur
        # brem over the FULL measured range (the wide grid) when available, so the
        # total rate reflects the real measurement out to the beam energy; fall
        # back to the line-grid brem if a (stale) wide brem is non-finite
        brem_cts_per_na = np.trapezoid(r["brem"], r["E_grid"]) * r["scale"]
        if r.get("brem_wide") is not None:
            wide_per_na = np.trapezoid(r["brem_wide"], r["E_grid_brem"]) * r["scale"]
            if np.isfinite(wide_per_na):
                brem_cts_per_na = wide_per_na
        brem_cts = brem_cts_per_na * cur
        rows.append(
            {
                "material": (
                    CATALOG.materials[c["crystal"]].label
                    if c["crystal"] in CATALOG.materials
                    else c["crystal"]
                ),
                "thickness": fmt_thickness(c["thickness_ang"]),
                "polar [deg]": c["tilt_deg"],
                "azimuth [deg]": c["tilt_azim_deg"],
                "Ee [keV]": c["E0_keV"],
                "line [eV]": r["E_grid"][i_pk],
                "peak [Phs/eV/s/nA]": line_det[i_pk] * r["scale"],
                "peak [Phs/eV/s]": line_det[i_pk] * r["scale"] * cur,
                "peak/bg": (line_det[i_pk] / brem_det[i_pk]) if brem_det[i_pk] else np.inf,
                "line [cts/s/nA]": line_cts_per_na,
                "line [cts/s]": line_cts,
                "brem [cts/s/nA]": brem_cts_per_na,
                "brem [cts/s]": brem_cts,
                "total [cts/s/nA]": line_cts_per_na + brem_cts_per_na,
                "total [cts/s]": line_cts + brem_cts,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.round(cast(dict[Hashable | Sequence[Hashable], int], _ROUND))
    df.columns = pd.MultiIndex.from_tuples(
        [("config" if c in _CONFIG_COLS else "", c) for c in df.columns]
    )
    return df


def show_summary(recs, settings):
    """Print + render the stats table for a list of records (e.g. one chunk, or
    best_azimuth(records(results)))."""
    from IPython.display import display

    df = summary_table(recs, settings)
    if df.empty:
        return
    print(
        ("window-QE applied, " if settings.apply_detector_qe else "unity QE, ")
        + "per-nA columns are current-normalized; rate columns use each case's "
        "bunch charge x repetition rate  |  "
        "peak/bg = EDS-convolved peak height / background at the peak"
    )
    display(df)
