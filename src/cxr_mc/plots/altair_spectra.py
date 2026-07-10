"""altair_spectra

Altair / Vega-Lite renderer for the intrinsic CXR spectra -- a fast, interactive
alternative to the matplotlib :mod:`cxr_mc.plots.spectra` figures. The headline
motivation for the jupyter -> marimo migration is that the matplotlib spectra are
sluggish over the fine energy grids; Vega-Lite renders them in the browser and
pans/zooms interactively at a fraction of the redraw cost.

Non-destructive: this module reuses the exact per-record data prep
(:func:`cxr_mc.plots._common._line_brem`) that the matplotlib path uses, so the
physics and units are identical -- only the renderer differs. The matplotlib
``plots/`` package is left untouched, and these names are intentionally NOT
re-exported from ``cxr_mc.plots`` (that package has a frozen export-set guard);
import them from the submodule:

    from cxr_mc.plots.altair_spectra import spectrum_chart, compare_spectrum_chart

Functions return :class:`altair.Chart` objects, which render directly in marimo
and Jupyter.

Note on size: Vega-Lite caps a spec at 5000 data rows by default. A dense spectrum
(several thousand grid points x several beam energies) can exceed that; enable a
larger transport once at the top of a notebook with
``altair.data_transformers.enable("vegafusion")`` (shipped with ``marimo[recommended]``)
or ``altair.data_transformers.disable_max_rows()``. This module does not mutate that
global state itself.
"""

import altair as alt
import numpy as np
import pandas as pd

from ..results import records
from ._common import _best_azimuth, _case_title, _line_brem

_FRAME_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]

# Case fields a `compare_spectrum_chart` caller may color by, and the axis/legend
# metadata for each -- readable title + unit suffix used both in the legend and
# the frame-building loop below.
_COMPARE_HUE_FIELDS = {
    "E0_keV": "beam energy (keV)",
    "tilt_deg": "polar tilt (deg)",
    "tilt_azim_deg": "azimuth (deg)",
}


def _tilt_records(results, tilt_deg=None):
    """Records for ONE polar tilt: the one nearest ``tilt_deg`` (default: the
    first/lowest tilt present). Returns ``[]`` when ``results`` is empty."""
    recs = records(results)
    if not recs:
        return []
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    t = tilts[0] if tilt_deg is None else min(tilts, key=lambda x: abs(x - tilt_deg))
    return [r for r in recs if r["case"]["tilt_deg"] == t]


def spectrum_frame(recs, settings, *, include_brem=True, collapse_azimuth=True):
    """Tidy long-form spectrum table for ``recs`` (already restricted to one polar
    tilt): one row per (beam energy, energy-grid point, component). Mirrors the
    intrinsic per-energy view of :func:`cxr_mc.plots.spectra._draw_by_energy`
    -- ``component`` is ``"total"`` (line + brem) or ``"brem"`` (the dashed
    underlay), both already multiplied by the per-record ``scale``. Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, component``."""
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    frames = []
    for E0 in energies:
        grp = _best_azimuth([r for r in recs if r["case"]["E0_keV"] == E0], collapse_azimuth)
        for r in grp:
            E = np.asarray(r["E_grid"], dtype=float)
            line_det, brem_det = _line_brem(r, settings, convolve=False)
            line_det = np.asarray(line_det, dtype=float)
            brem_det = np.asarray(brem_det, dtype=float)
            total = (line_det + brem_det) if include_brem else line_det
            az = float(r["case"]["tilt_azim_deg"])
            frames.append(
                pd.DataFrame(
                    {
                        "energy_eV": E,
                        "intensity": total * r["scale"],
                        "E0_keV": float(E0),
                        "azimuth_deg": az,
                        "component": "total",
                    }
                )
            )
            if include_brem:
                frames.append(
                    pd.DataFrame(
                        {
                            "energy_eV": E,
                            "intensity": brem_det * r["scale"],
                            "E0_keV": float(E0),
                            "azimuth_deg": az,
                            "component": "brem",
                        }
                    )
                )
    if not frames:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def spectrum_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    include_brem=True,
    collapse_azimuth=True,
    x_domain=None,
    y_type="linear",
    width=720,
    height=360,
):
    """Interactive Altair line chart of the INTRINSIC spectra at ONE polar tilt,
    one line per beam energy (brem drawn as a faint dashed underlay, omitted
    entirely when ``include_brem=False`` -- the CXR-only view with no
    incoherent bremsstrahlung background). The Altair counterpart of
    :func:`cxr_mc.plots.plot_by_energy` / ``browse(kind="by_energy")`` -- same
    data prep, Vega-Lite renderer with pan/zoom. Pass ``tilt_deg`` to pick a
    tilt (default: the lowest present); ``x_domain=(lo, hi)`` fixes the photon
    -energy axis limits (default: autoscale to the data); ``y_type`` is
    ``"linear"`` or ``"log"``. Returns an :class:`altair.Chart`, or ``None``
    when there are no records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = spectrum_frame(
        recs, settings, include_brem=include_brem, collapse_azimuth=collapse_azimuth
    )
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "intrinsic", latex=False)
    x_scale = alt.Scale(domain=list(x_domain)) if x_domain is not None else alt.Undefined
    y_scale = alt.Scale(type=y_type)  # type: ignore[arg-type]

    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale),
        y=alt.Y("intensity:Q", title="Intensity (Phs/eV/s/nA)", scale=y_scale),
        color=alt.Color("E0_keV:N", title="beam energy (keV)"),
        tooltip=["E0_keV:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = [base.transform_filter(alt.datum.component == "total").mark_line(strokeWidth=1.4)]
    if include_brem:
        layers.append(
            base.transform_filter(alt.datum.component == "brem").mark_line(
                strokeWidth=0.7, strokeDash=[4, 3], opacity=0.7
            )
        )
    return alt.layer(*layers).properties(width=width, height=height, title=title).interactive()


def _compare_frame(recs, settings, *, hue, include_brem=True):
    """Tidy long-form spectrum table for ``recs``, ONE row per (hue value, energy
    -grid point, component) -- the generalization of :func:`spectrum_frame` to an
    arbitrary case field (``hue``). Duplicate records sharing a ``hue`` value
    (e.g. several azimuths at the same polar tilt) collapse to the single
    strongest-peak record, exactly like :func:`_best_azimuth`'s per-energy
    selection (``max(np.max(r["spec"]))`` wins) -- so `hue="E0_keV"` on a
    single-tilt slice reduces to the SAME per-energy selection
    ``spectrum_frame``/``_best_azimuth`` make, and reproduces
    :func:`spectrum_chart`'s data exactly (see the correctness-oracle test in
    ``tests/test_altair_spectra_compare.py``). Columns: ``energy_eV, intensity,
    E0_keV, tilt_deg, tilt_azim_deg, component`` -- every hue field is carried on
    every row (not just the active ``hue``), so callers can facet/tooltip on any
    of them without rebuilding the frame."""
    groups: dict[float, list] = {}
    for r in recs:
        key = float(r["case"][hue])
        groups.setdefault(key, []).append(r)

    frames = []
    for grp in groups.values():
        r = max(grp, key=lambda rr: float(np.max(rr["spec"])))
        E = np.asarray(r["E_grid"], dtype=float)
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        line_det = np.asarray(line_det, dtype=float)
        brem_det = np.asarray(brem_det, dtype=float)
        total = (line_det + brem_det) if include_brem else line_det
        row_meta = {
            "E0_keV": float(r["case"]["E0_keV"]),
            "tilt_deg": float(r["case"]["tilt_deg"]),
            "tilt_azim_deg": float(r["case"]["tilt_azim_deg"]),
        }
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": E,
                    "intensity": total * r["scale"],
                    **row_meta,
                    "component": "total",
                }
            )
        )
        if include_brem:
            frames.append(
                pd.DataFrame(
                    {
                        "energy_eV": E,
                        "intensity": brem_det * r["scale"],
                        **row_meta,
                        "component": "brem",
                    }
                )
            )
    _columns = ["energy_eV", "intensity", "E0_keV", "tilt_deg", "tilt_azim_deg", "component"]
    if not frames:
        return pd.DataFrame(columns=_columns)
    return pd.concat(frames, ignore_index=True)


def compare_spectrum_chart(
    results,
    settings,
    *,
    hue,
    include_brem=True,
    x_domain=None,
    y_type="linear",
    width=720,
    height=360,
):
    """Interactive Altair line chart overlaying spectra ONE LINE PER DISTINCT
    VALUE of ``hue`` (one of ``"E0_keV"``, ``"tilt_deg"``, ``"tilt_azim_deg"``),
    for whatever ``results`` the caller already sliced down (e.g. via
    ``results.select_results`` pinning the OTHER swept knobs). The generalized
    counterpart of :func:`spectrum_chart`, which fixes the hue to ``E0_keV`` at
    ONE polar tilt; this instead lets the CALLER pick which case field varies
    across lines, so it also drives the "Polar-angle comparison" (``hue=
    "tilt_deg"``) and "Azimuthal comparison" (``hue="tilt_azim_deg"``) notebook
    tabs. Same physics/units as ``spectrum_chart`` (:func:`_line_brem`, the per
    -eV detected line + brem densities times the record ``scale``) -- this is a
    viz generalization, not new physics.

    Records sharing a ``hue`` value collapse to the single strongest-peak one
    (see :func:`_compare_frame`), so accidental duplicates (e.g. several
    azimuths at the same polar tilt when comparing by tilt) don't overplot.
    When ``hue`` has only one distinct value in ``results`` this still renders
    -- a single line -- callers should add their own "nothing to compare" note
    in that case (the chart itself has no way to know the FULL sweep, only what
    it was given). Returns an :class:`altair.Chart`, or ``None`` when there are
    no records."""
    if hue not in _COMPARE_HUE_FIELDS:
        raise ValueError(f"hue must be one of {sorted(_COMPARE_HUE_FIELDS)}, got {hue!r}")
    recs = records(results)
    if not recs:
        return None
    df = _compare_frame(recs, settings, hue=hue, include_brem=include_brem)
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "comparison", latex=False)
    x_scale = alt.Scale(domain=list(x_domain)) if x_domain is not None else alt.Undefined
    y_scale = alt.Scale(type=y_type)  # type: ignore[arg-type]
    hue_title = _COMPARE_HUE_FIELDS[hue]

    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale),
        y=alt.Y("intensity:Q", title="Intensity (Phs/eV/s/nA)", scale=y_scale),
        color=alt.Color(f"{hue}:N", title=hue_title),
        tooltip=[f"{hue}:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = [base.transform_filter(alt.datum.component == "total").mark_line(strokeWidth=1.4)]
    if include_brem:
        layers.append(
            base.transform_filter(alt.datum.component == "brem").mark_line(
                strokeWidth=0.7, strokeDash=[4, 3], opacity=0.7
            )
        )
    return alt.layer(*layers).properties(width=width, height=height, title=title).interactive()
