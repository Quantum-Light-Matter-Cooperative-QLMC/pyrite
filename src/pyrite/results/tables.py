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
from .._spectral_components import cxr_spectrum
from ..detectors import Detector, LegacyEDS
from ..materials import CATALOG
from .metrics import cxr_metrics
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
    every swept ``case`` knob + the derived :func:`cxr_metrics`. This is the
    first-class primitive for slicing/faceting many simultaneously-swept knobs
    (TODO P3 #9) -- ``df.groupby`` / ``df.pivot_table`` / seaborn ``relplot`` /
    plotly over ANY subset of knobs, instead of the heatmap/line auto-pick. See
    :func:`plots.facet_metric` for a small-multiples view built on it.

    metrics : include the line metrics (``peak_flux``, ``coherent_flux``,
        ``line_eV``, ``fwhm_eV``, ``line_flux``, ``line_frac``, ``total_flux``,
        ``coherent_brem_ratio``, ``cxr_quality``); needs a :class:`Settings`
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
                row.update(cxr_metrics(r, settings, rel_prominence, n_fwhm, metric))
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
    "line [Phs/s]": 1,
    "line [Phs/s/nA]": 1,
    "brem [Phs/s]": 1,
    "brem [Phs/s/nA]": 1,
    "total [Phs/s]": 1,
    "total [Phs/s/nA]": 1,
}
_CONFIG_COLS = {"material", "thickness", "polar [deg]", "azimuth [deg]"}


def summary_table(recs, settings):
    """Line/background stats for a list of records. The geometry (material,
    thickness, polar/azimuthal tilt) is broken out of the config name into its
    own columns under a 'config' super-header; the rest are the line peak, the
    ``LegacyEDS``-scored (EDS-convolved, optionally window-QE) peak spectral
    density ``peak [Phs/eV/s]``, peak-over-background, and the integrated
    line / brem / total photon rates ``[Phs/s]`` at the case's bunch charge
    and repetition rate, alongside current-normalized ``[Phs/s/nA]`` values.
    The integrals are **source-side**: photons emitted into the detector
    aperture, with no detector QE, convolution or counting applied, so they
    are not detected counts. Old checkpoints fall
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
        cxr = cxr_spectrum(r)
        cxr_det = detector.score(r["E_grid"], cxr, fwhm_eV=r["fwhm"])
        brem_det = detected_background(r, settings) / r["scale"]
        i_pk = np.argmax(cxr_det)
        cxr_phs_per_na = np.trapezoid(cxr, r["E_grid"]) * r["scale"]
        cxr_phs = cxr_phs_per_na * cur
        # brem over the FULL measured range (the wide grid) when available, so the
        # total rate reflects the real measurement out to the beam energy; fall
        # back to the line-grid brem if a (stale) wide brem is non-finite
        brem_phs_per_na = np.trapezoid(r["brem"], r["E_grid"]) * r["scale"]
        if r.get("brem_wide") is not None:
            wide_per_na = np.trapezoid(r["brem_wide"], r["E_grid_brem"]) * r["scale"]
            if np.isfinite(wide_per_na):
                brem_phs_per_na = wide_per_na
        brem_phs = brem_phs_per_na * cur
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
                "peak [Phs/eV/s/nA]": cxr_det[i_pk] * r["scale"],
                "peak [Phs/eV/s]": cxr_det[i_pk] * r["scale"] * cur,
                "peak/bg": (cxr_det[i_pk] / brem_det[i_pk]) if brem_det[i_pk] else np.inf,
                "line [Phs/s/nA]": cxr_phs_per_na,
                "line [Phs/s]": cxr_phs,
                "brem [Phs/s/nA]": brem_phs_per_na,
                "brem [Phs/s]": brem_phs,
                "total [Phs/s/nA]": cxr_phs_per_na + brem_phs_per_na,
                "total [Phs/s]": cxr_phs + brem_phs,
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
        "peak columns are LegacyEDS-scored densities; [Phs/s] integrals are "
        "source-side photons into the aperture (no QE/convolution)  |  "
        "peak/bg = EDS-convolved peak height / background at the peak"
    )
    display(df)
