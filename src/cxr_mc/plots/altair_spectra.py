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
from ..results.store import _detected_background_wide
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


def _validate_band(band):
    if band not in {"narrow", "broad"}:
        raise ValueError(f"band must be 'narrow' or 'broad', got {band!r}")


def _scale(scale_type, domain=None):
    if scale_type not in {"linear", "log"}:
        raise ValueError(f"scale type must be 'linear' or 'log', got {scale_type!r}")
    kwargs = {}
    if scale_type != "linear":
        kwargs["type"] = scale_type
    if domain is not None:
        lo, hi = domain
        if lo is not None and hi is not None:
            kwargs["domain"] = [lo, hi]
        else:
            if lo is not None:
                kwargs["domainMin"] = lo
            if hi is not None:
                kwargs["domainMax"] = hi
    return alt.Scale(**kwargs) if kwargs else alt.Undefined


def _record_frame(r, settings, *, include_brem, band, meta):
    E = np.asarray(r["E_grid"], dtype=float)
    line_det, brem_det = _line_brem(r, settings, convolve=False)
    line_det = np.asarray(line_det, dtype=float)
    brem_det = np.asarray(brem_det, dtype=float)

    total_E = E
    total = (line_det + brem_det) if include_brem else line_det
    brem_E = E
    brem = brem_det

    if band == "broad" and include_brem:
        E_wide, brem_wide = _detected_background_wide(r, settings, convolve=False)
        E_wide = np.asarray(E_wide, dtype=float)
        brem_wide = np.asarray(brem_wide, dtype=float) / r["scale"]
        beam_max_eV = float(r["case"].get("E0_keV", np.inf)) * 1000.0
        if E.size:
            wide_mask = (E_wide > float(np.nanmax(E))) & (E_wide <= beam_max_eV)
        else:
            wide_mask = E_wide <= beam_max_eV
        if np.any(wide_mask):
            E_tail = E_wide[wide_mask]
            brem_tail = brem_wide[wide_mask]
            total_E = np.concatenate([E, E_tail])
            total = np.concatenate([total, brem_tail])
            brem_E = np.concatenate([E, E_tail])
            brem = np.concatenate([brem, brem_tail])

    frames = [
        pd.DataFrame(
            {
                "energy_eV": total_E,
                "intensity": total * r["scale"],
                **meta,
                "component": "total",
            }
        )
    ]
    if include_brem:
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": brem_E,
                    "intensity": brem * r["scale"],
                    **meta,
                    "component": "brem",
                }
            )
        )
    return frames


def _peak_preserving_indices(y, max_points):
    y = np.asarray(y, dtype=float)
    n = y.size
    if max_points is None or n <= max_points:
        return np.arange(n)
    max_points = int(max_points)
    if max_points <= 0:
        return np.arange(n)
    if max_points == 1:
        return np.array([int(np.nanargmax(np.nan_to_num(y, nan=-np.inf)))])
    if max_points == 2:
        return np.array([0, n - 1])

    keep = {0, n - 1}
    for bucket in np.array_split(np.arange(1, n - 1), max_points - 2):
        if bucket.size == 0:
            continue
        vals = np.nan_to_num(y[bucket], nan=-np.inf)
        keep.add(int(bucket[int(np.argmax(vals))]))
    return np.fromiter(sorted(keep), dtype=int)


def _decimate_frame(df, max_points):
    if max_points is None or df.empty or len(df) <= max_points:
        return df
    group_cols = [c for c in df.columns if c not in {"energy_eV", "intensity"}]
    groups = list(df.groupby(group_cols, sort=False, dropna=False))
    if not groups:
        return df
    per_group = max(3, int(max_points) // len(groups))
    frames = []
    for _, grp in groups:
        grp = grp.sort_values("energy_eV")
        idx = _peak_preserving_indices(grp["intensity"].to_numpy(), per_group)
        frames.append(grp.iloc[idx])
    return pd.concat(frames, ignore_index=True)


def spectrum_frame(
    recs,
    settings,
    *,
    include_brem=True,
    collapse_azimuth=True,
    band="narrow",
    max_points=None,
):
    """Tidy long-form spectrum table for ``recs`` (already restricted to one polar
    tilt): one row per (beam energy, energy-grid point, component). Mirrors the
    intrinsic per-energy view of :func:`cxr_mc.plots.spectra._draw_by_energy`.

    ``band="narrow"`` uses the line grid exactly as before. ``band="broad"``
    keeps that fine grid through the line-grid cutoff and appends the stored
    ``brem_wide`` continuum above it when available. ``max_points`` applies a
    peak-preserving plot-time decimation across rendered traces.
    """
    _validate_band(band)
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    frames = []
    for E0 in energies:
        grp = _best_azimuth([r for r in recs if r["case"]["E0_keV"] == E0], collapse_azimuth)
        for r in grp:
            az = float(r["case"]["tilt_azim_deg"])
            meta = {
                "E0_keV": float(E0),
                "azimuth_deg": az,
            }
            frames.extend(
                _record_frame(r, settings, include_brem=include_brem, band=band, meta=meta)
            )
    if not frames:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)


def spectrum_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    include_brem=True,
    collapse_azimuth=True,
    x_domain=None,
    x_type="linear",
    y_type="linear",
    band="narrow",
    max_points=5000,
    width=720,
    height=360,
):
    """Interactive Altair line chart of the intrinsic spectra at ONE polar tilt.

    ``band="narrow"`` draws the current line-grid view. ``band="broad"`` appends
    the stored wide bremsstrahlung tail above the line-grid cutoff, so broadband
    spectra can extend to the beam-energy grid without recomputing CXR lines.
    ``x_type``/``y_type`` are ``"linear"`` or ``"log"``; ``max_points`` applies
    peak-preserving plot-time decimation before the Vega-Lite spec is built.
    Returns an :class:`altair.Chart`, or ``None`` when there are no records.
    """
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = spectrum_frame(
        recs,
        settings,
        include_brem=include_brem,
        collapse_azimuth=collapse_azimuth,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "intrinsic", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = _scale(y_type)

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


def _compare_frame(recs, settings, *, hue, include_brem=True, band="narrow", max_points=None):
    """Tidy long-form spectrum table for ``recs``, ONE row per (hue value, energy
    -grid point, component) -- the generalization of :func:`spectrum_frame` to an
    arbitrary case field (``hue``). Duplicate records sharing a ``hue`` value
    (e.g. several azimuths at the same polar tilt) collapse to the single
    strongest-peak record, exactly like :func:`_best_azimuth`'s per-energy
    selection (``max(np.max(r["spec"]))`` wins). ``band`` and ``max_points`` have
    the same meaning as in :func:`spectrum_frame`.
    """
    _validate_band(band)
    groups: dict[float, list] = {}
    for r in recs:
        key = float(r["case"][hue])
        groups.setdefault(key, []).append(r)

    frames = []
    for grp in groups.values():
        r = max(grp, key=lambda rr: float(np.max(rr["spec"])))
        row_meta = {
            "E0_keV": float(r["case"]["E0_keV"]),
            "tilt_deg": float(r["case"]["tilt_deg"]),
            "tilt_azim_deg": float(r["case"]["tilt_azim_deg"]),
        }
        frames.extend(
            _record_frame(r, settings, include_brem=include_brem, band=band, meta=row_meta)
        )
    _columns = ["energy_eV", "intensity", "E0_keV", "tilt_deg", "tilt_azim_deg", "component"]
    if not frames:
        return pd.DataFrame(columns=_columns)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)


def compare_spectrum_chart(
    results,
    settings,
    *,
    hue,
    include_brem=True,
    x_domain=None,
    x_type="linear",
    y_type="linear",
    band="narrow",
    max_points=5000,
    width=720,
    height=360,
):
    """Interactive Altair line chart overlaying spectra ONE LINE PER DISTINCT
    VALUE of ``hue`` (one of ``"E0_keV"``, ``"tilt_deg"``, ``"tilt_azim_deg"``),
    for whatever ``results`` the caller already sliced down. Same physics/units
    as ``spectrum_chart``; ``band``, ``x_type``, and ``max_points`` match that
    function. Returns an :class:`altair.Chart`, or ``None`` when there are no
    records.
    """
    if hue not in _COMPARE_HUE_FIELDS:
        raise ValueError(f"hue must be one of {sorted(_COMPARE_HUE_FIELDS)}, got {hue!r}")
    recs = records(results)
    if not recs:
        return None
    df = _compare_frame(
        recs,
        settings,
        hue=hue,
        include_brem=include_brem,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "comparison", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = _scale(y_type)
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
