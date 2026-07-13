"""altair_detectors

Altair / Vega-Lite renderers for the detector-view spectra (Timepix3 and Eagle
XO) -- the interactive counterparts of the matplotlib
:mod:`cxr_mc.plots.detectors` figures. Same contract as
:mod:`cxr_mc.plots.altair_spectra`: these reuse the exact per-record detector
data prep (``detectors._tpx_detected`` / ``detectors._eag_detected`` and the
Eagle response's ``charge_density``), so the physics and units are identical --
only the renderer differs. The matplotlib ``detectors`` module is left untouched,
and these names are intentionally NOT re-exported from ``cxr_mc.plots`` (that
package has a frozen export-set guard); import them from the submodule:

    from cxr_mc.plots.altair_detectors import eaglexo_detected_chart

Functions return :class:`altair.Chart` objects (``None`` when there are no
records), which render directly in marimo and Jupyter.

Scope: the three SPECTRAL detector views, where Vega-Lite's cheap pan/zoom is
the win -- Timepix3 detected-vs-incident, Eagle XO detected-vs-incident, and the
Eagle XO recorded-charge density. The static efficiency curves
(``plot_timepix_efficiency`` / ``plot_eaglexo_efficiency``) and the geometry
heatmap (``plot_eaglexo_charge_map``) stay on matplotlib for now.

See :mod:`cxr_mc.plots.altair_spectra` for the Vega-Lite 5000-row cap note: a
dense detector overlay (fine line grid + wide brem x several beam energies) can
exceed it; enable ``altair.data_transformers.enable("vegafusion")`` or
``altair.data_transformers.disable_max_rows()`` once at the top of a notebook.
"""

import altair as alt
import numpy as np
import pandas as pd
from altair.utils.schemapi import UndefinedType

from ..detectors import eaglexo_response as eag
from ..detectors import timepix_response as tpx
from ._common import _best_azimuth, _case_title
from .altair_spectra import _scale as _axis_scale
from .altair_spectra import _tilt_records, _validate_band, _windowed_frame
from .detectors import (
    SI_K_EDGE_EV,
    _eag_detected,
    _eag_wide_brem,
    _eag_wide_charge,
    _thr_keV,
    _tpx_detected,
)

# incident-vs-detected long form (Timepix + Eagle photon density)
_DET_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "kind", "band"]
# Eagle recorded-charge density long form
_CHARGE_COLUMNS = ["energy_eV", "charge_density", "E0_keV", "azimuth_deg", "band"]


def _collapsed(recs, *, collapse_azimuth):
    """Group ``recs`` by beam energy, optionally collapsing each group to its
    best (max intrinsic line) azimuth -- the per-energy record selection shared by
    every detector draw. Yields ``(E0_keV, [records])`` in ascending energy."""
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    for E0 in energies:
        grp = _best_azimuth([r for r in recs if r["case"]["E0_keV"] == E0], collapse_azimuth)
        yield E0, grp


def _title(recs, tail):
    return _case_title(recs[0]["case"], tail, latex=False)


def _y_scale(df, y_field, y_type, x_domain=None, y_domain=None, floor_frac=1e-5):
    if y_type not in {"linear", "log"}:
        raise ValueError(f"scale type must be 'linear' or 'log', got {y_type!r}")
    if y_domain is not None:
        return _axis_scale(y_type, y_domain)
    if y_type == "linear":
        if x_domain is None:
            return _axis_scale("linear")
        windowed = _windowed_frame(df.rename(columns={y_field: "intensity"}), x_domain)
        hi = float(windowed["intensity"].max()) if not windowed.empty else 0.0
        if hi <= 0:
            return _axis_scale("linear")
        return alt.Scale(domainMin=0.0, domainMax=hi * 1.05)

    windowed = _windowed_frame(df.rename(columns={y_field: "intensity"}), x_domain)
    positive = windowed.loc[windowed["intensity"] > 0, "intensity"]
    if positive.empty:
        return _axis_scale("log")
    floor = max(float(positive.min()), float(positive.max()) * floor_frac)
    return alt.Scale(type="log", domainMin=floor, clamp=True)


def _detector_x_scale(x_type, x_domain):
    if x_type == "log" and x_domain is None:
        return _logx()
    return _axis_scale(x_type, x_domain)


def _broad_incident(r):
    """One uniform broadband incident spectrum for a detector response.

    The coherent spectrum lives on the fine line grid while bremsstrahlung is
    stored on a coarser grid out to the beam energy.  Timepix3 redistributes
    photon energy, so it must see their sum on one grid: applying it to the two
    pieces independently discards events that cross the line-grid boundary and
    produces a visible step at that boundary.

    The wide brem grid is already uniform and matches Timepix3's
    recorded-energy bin scale, so interpolate only the coherent contribution onto it.
    Outside the line grid the coherent term is zero by construction.
    """
    E_line = np.asarray(r["E_grid"], dtype=float)
    E_brem = r.get("E_grid_brem")
    brem_wide = r.get("brem_wide")
    if E_brem is None or brem_wide is None:
        incident = np.asarray(r["spec"], dtype=float) + np.asarray(r["brem"], dtype=float)
        return E_line, incident * r["scale"]

    E = np.asarray(E_brem, dtype=float)
    brem = np.asarray(brem_wide, dtype=float)
    # The 0-eV brem bin cannot contribute to a photon-energy spectrum and is
    # below the line grid's modeled band.  Starting at the line-grid floor also
    # keeps the response grid physically meaningful.
    mask = E >= float(np.nanmin(E_line))
    E = E[mask]
    brem = brem[mask]
    coherent = np.interp(E, E_line, np.asarray(r["spec"], dtype=float), left=0.0, right=0.0)
    return E, (coherent + brem) * r["scale"]


# ---- Timepix3 detected vs incident -------------------------------------------
def timepix_detected_frame(
    recs,
    settings,
    *,
    thickness_um=300.0,
    bias_v=100.0,
    n_mc=80000,
    seed=0,
    collapse_azimuth=True,
    band="broad",
):
    """Tidy long-form incident-vs-Timepix3-detected table for ``recs`` (already
    restricted to one polar tilt): one row per (beam energy, grid point, kind),
    ``kind`` in ``{"incident", "detected"}``, both already scaled to Phs/eV/s/nA.
    Mirrors :func:`cxr_mc.plots.detectors._draw_timepix_detected`. ``band`` is
    ``"line"`` (fine line grid) or ``"broad"`` (the wide brem grid with the
    coherent contribution interpolated onto it).  The broad view applies one
    Timepix response to the entire incident spectrum. Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, kind, band``."""
    _validate_band(band)
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            az = float(r["case"]["tilt_azim_deg"])
            if band == "broad" and r.get("brem_wide") is not None:
                E, inc = _broad_incident(r)
                resp = tpx.get_response(
                    E,
                    n_mc=n_mc,
                    seed=seed,
                    thickness_um=thickness_um,
                    bias_v=bias_v,
                )
                det = resp.apply(inc)
                rows = [("broad", E, inc, det)]
            else:
                E = np.asarray(r["E_grid"], dtype=float)
                inc, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
                rows = [("line", E, inc, det)]

            for band_name, Ex, yi, yd in rows:
                for kind, y in (("incident", yi), ("detected", yd)):
                    frames.append(
                        pd.DataFrame(
                            {
                                "energy_eV": Ex,
                                "intensity": np.asarray(y, dtype=float),
                                "E0_keV": float(E0),
                                "azimuth_deg": az,
                                "kind": kind,
                                "band": band_name,
                            }
                        )
                    )
    if not frames:
        return pd.DataFrame(columns=_DET_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _detected_layers(
    df,
    *,
    x_scale: alt.Scale | UndefinedType = alt.Undefined,
    y_scale: alt.Scale | UndefinedType = alt.Undefined,
    color_field="E0_keV",
    color_title="beam energy (keV)",
):
    """The shared incident (dashed)/detected (solid) line layers for a
    detected-vs-incident frame. ``detail`` splits the line/brem bands so the two
    grids never join across their gap."""
    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale),
        y=alt.Y("intensity:Q", title="Phs/eV/s/nA", scale=y_scale),
        color=alt.Color(f"{color_field}:N", title=color_title),
        detail="band:N",
        tooltip=["E0_keV:N", "energy_eV:Q", "intensity:Q", "kind:N", "band:N"],
    )
    detected = base.transform_filter(alt.datum.kind == "detected").mark_line(strokeWidth=1.3)
    incident = base.transform_filter(alt.datum.kind == "incident").mark_line(
        strokeWidth=0.9, strokeDash=[2, 2], opacity=0.6
    )
    return incident, detected


def timepix_detected_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    thickness_um=300.0,
    bias_v=100.0,
    n_mc=80000,
    seed=0,
    collapse_azimuth=True,
    x_domain=None,
    y_domain=None,
    x_type="linear",
    y_type="log",
    band="broad",
    width=720,
    height=360,
):
    """Interactive log-y overlay of incident (dotted) vs Timepix3-detected (solid)
    spectra at ONE polar tilt, one colour per beam energy, with the counting
    threshold marked. The Altair counterpart of
    :func:`cxr_mc.plots.plot_timepix_detected` / ``browse(kind="timepix")``.
    ``x_domain=(lo, hi)`` fixes the photon-energy axis limits (default:
    autoscale). Returns an :class:`altair.Chart`, or ``None`` when there are no
    records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = timepix_detected_frame(
        recs,
        settings,
        thickness_um=thickness_um,
        bias_v=bias_v,
        n_mc=n_mc,
        seed=seed,
        collapse_azimuth=collapse_azimuth,
        band=band,
    )
    if df.empty:
        return None
    incident, detected = _detected_layers(
        df,
        x_scale=_detector_x_scale(x_type, x_domain),
        y_scale=_y_scale(df, "intensity", y_type, x_domain, y_domain),
    )
    thr = (
        alt.Chart(pd.DataFrame({"E": [_thr_keV() * 1e3]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="E:Q")
    )
    title = _title(recs, "Timepix3 detected (solid) vs incident (dotted)")
    return (
        alt.layer(incident, detected, thr)
        .properties(width=width, height=height, title=title)
        .interactive()
    )


# ---- Eagle XO detected vs incident -------------------------------------------
def eaglexo_detected_frame(
    recs,
    settings,
    *,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    band="broad",
):
    """Tidy long-form incident-vs-Eagle-XO-detected table for ``recs`` (one polar
    tilt): one row per (beam energy, grid point, kind, band). ``kind`` in
    ``{"incident", "detected"}``; ``band`` is ``"line"`` (fine line grid) or
    ``"brem"`` (the wide brem grid, present only when the sweep stored
    ``brem_wide``). Mirrors :func:`cxr_mc.plots.detectors._draw_eaglexo_detected`,
    including the thin-sensor QE roll-off on the wide brem. Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, kind, band``."""
    _validate_band(band)
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            E = np.asarray(r["E_grid"], dtype=float)
            inc, det = _eag_detected(r, settings, coating, resolve_energy)
            az = float(r["case"]["tilt_azim_deg"])
            rows = [("line", E, inc, det)]
            if band == "broad" and r.get("brem_wide") is not None:
                Eb, inc_b, det_b = _eag_wide_brem(r, coating)
                rows.append(("brem", Eb, inc_b, det_b))
            for band, Ex, yi, yd in rows:
                for kind, y in (("incident", yi), ("detected", yd)):
                    frames.append(
                        pd.DataFrame(
                            {
                                "energy_eV": Ex,
                                "intensity": np.asarray(y, dtype=float),
                                "E0_keV": float(E0),
                                "azimuth_deg": az,
                                "kind": kind,
                                "band": band,
                            }
                        )
                    )
    if not frames:
        return pd.DataFrame(columns=_DET_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def eaglexo_detected_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    coating="BN",
    resolve_energy=False,
    collapse_azimuth=True,
    show_qe=True,
    x_domain=None,
    y_domain=None,
    x_type="log",
    y_type="log",
    band="broad",
    width=720,
    height=360,
):
    """Interactive log-log overlay of incident (dotted) vs Eagle-XO-detected
    (solid) spectra at ONE polar tilt, with the Si-K edge marked and (when
    ``show_qe``) a faint QE envelope on an independent right axis -- the camera's
    signature, soft PXR lines passing at ~90% QE while the hard brem is crushed by
    the thin sensor. The Altair counterpart of
    :func:`cxr_mc.plots.plot_eaglexo_detected` / ``browse(kind="eaglexo")``.
    ``x_domain=(lo, hi)`` fixes the photon-energy axis limits (default:
    autoscale). Returns an :class:`altair.Chart`, or ``None`` when there are no
    records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = eaglexo_detected_frame(
        recs,
        settings,
        coating=coating,
        resolve_energy=resolve_energy,
        collapse_azimuth=collapse_azimuth,
        band=band,
    )
    if df.empty:
        return None
    xsc = _detector_x_scale(x_type, x_domain)
    ysc = _y_scale(df, "intensity", y_type, x_domain, y_domain)
    incident, detected = _detected_layers(df, x_scale=xsc, y_scale=ysc)
    layers = [
        incident,
        detected,
        alt.Chart(pd.DataFrame({"E": [SI_K_EDGE_EV]}))
        .mark_rule(color="gray", strokeDash=[6, 3])
        .encode(x="E:Q"),
    ]
    if show_qe:
        E_all = df["energy_eV"].to_numpy()
        lo = max(float(np.min(E_all)), 1.0)
        hi = float(np.max(E_all))
        Eqe = np.geomspace(lo, hi, 400)
        qe_df = pd.DataFrame({"energy_eV": Eqe, "QE": eag.qe(Eqe, coating)})
        layers.append(
            alt.Chart(qe_df)
            .mark_line(color="gray", opacity=0.5)
            .encode(
                x=alt.X("energy_eV:Q", scale=xsc),
                y=alt.Y("QE:Q", title="QE", scale=alt.Scale(domain=[0, 1.05])),
            )
        )
    blur = ", energy-resolved" if resolve_energy else ""
    title = _title(recs, f"Eagle XO detected (solid) vs incident (dotted), {coating}{blur}")
    chart = alt.layer(*layers).resolve_scale(y="independent" if show_qe else "shared")
    # x-only pan/zoom -- an interval bound to the x scale keeps the QE envelope's
    # independent y axis stable (full .interactive() fights the dual y resolve).
    pan = alt.selection_interval(bind="scales", encodings=["x"])
    return chart.add_params(pan).properties(width=width, height=height, title=title)


# ---- Eagle XO recorded-charge density ----------------------------------------
def eaglexo_charge_frame(recs, settings, *, coating="BN", collapse_azimuth=True, band="broad"):
    """Tidy long-form Eagle XO recorded-charge density [e-/eV/s] for ``recs`` (one
    polar tilt): one row per (beam energy, grid point, band). ``band`` is
    ``"line"`` (fine grid, lines + line-grid brem) or ``"brem"`` (the wide brem
    grid). Every photon is weighted by E/W_Si, so the hard brem carries far more
    charge than its photon count. Mirrors
    :func:`cxr_mc.plots.detectors._draw_eaglexo_charge`. Columns:
    ``energy_eV, charge_density, E0_keV, azimuth_deg, band``."""
    _validate_band(band)
    cur = settings.beam_current_na
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            resp = eag.get_response(r["E_grid"], coating=coating)
            E = np.asarray(r["E_grid"], dtype=float)
            cd_line = resp.charge_density((r["spec"] + r["brem"]) * r["scale"]) * cur
            az = float(r["case"]["tilt_azim_deg"])
            frames.append(
                pd.DataFrame(
                    {
                        "energy_eV": E,
                        "charge_density": np.asarray(cd_line, dtype=float),
                        "E0_keV": float(E0),
                        "azimuth_deg": az,
                        "band": "line",
                    }
                )
            )
            if band == "broad" and r.get("brem_wide") is not None:
                Eb, cd_b = _eag_wide_charge(r, coating, cur)
                frames.append(
                    pd.DataFrame(
                        {
                            "energy_eV": Eb,
                            "charge_density": np.asarray(cd_b, dtype=float),
                            "E0_keV": float(E0),
                            "azimuth_deg": az,
                            "band": "brem",
                        }
                    )
                )
    if not frames:
        return pd.DataFrame(columns=_CHARGE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def eaglexo_charge_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    coating="BN",
    collapse_azimuth=True,
    x_domain=None,
    y_domain=None,
    x_type="log",
    y_type="log",
    band="broad",
    width=720,
    height=360,
):
    """Interactive log-log Eagle XO recorded-charge density [e-/eV/s] at ONE polar
    tilt (lines solid, wide brem dashed), one colour per beam energy, Si-K edge
    marked. A CCD integrates charge rather than counting photons, so this shows
    where the recorded signal actually comes from. The Altair counterpart of
    :func:`cxr_mc.plots.plot_eaglexo_charge` / ``browse(kind="eaglexo_charge")``.
    ``x_domain=(lo, hi)`` fixes the photon-energy axis limits (default:
    autoscale). Returns an :class:`altair.Chart`, or ``None`` when there are no
    records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = eaglexo_charge_frame(
        recs, settings, coating=coating, collapse_azimuth=collapse_azimuth, band=band
    )
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        x=alt.X(
            "energy_eV:Q",
            title="Photon energy (eV)",
            scale=_detector_x_scale(x_type, x_domain),
        ),
        y=alt.Y(
            "charge_density:Q",
            title="charge density (e-/eV/s)",
            scale=_y_scale(df, "charge_density", y_type, x_domain, y_domain),
        ),
        color=alt.Color("E0_keV:N", title="beam energy (keV)"),
        tooltip=["E0_keV:N", "energy_eV:Q", "charge_density:Q", "band:N"],
    )
    line = base.transform_filter(alt.datum.band == "line").mark_line(strokeWidth=1.3)
    brem = base.transform_filter(alt.datum.band == "brem").mark_line(
        strokeWidth=0.8, strokeDash=[5, 3], opacity=0.85
    )
    sik = (
        alt.Chart(pd.DataFrame({"E": [SI_K_EDGE_EV]}))
        .mark_rule(color="gray", strokeDash=[6, 3])
        .encode(x="E:Q")
    )
    title = _title(recs, f"Eagle XO recorded charge density ({coating}, dashed = brem)")
    return (
        alt.layer(line, brem, sik).properties(width=width, height=height, title=title).interactive()
    )


def _logx(domain=None):
    """A log x-scale that tolerates the brem grid's E=0 row (Vega-Lite drops
    non-positive values on a log axis). ``domain=(lo, hi)`` fixes the axis
    limits (default: autoscale to the data)."""
    return (
        alt.Scale(type="log", domain=list(domain)) if domain is not None else alt.Scale(type="log")
    )
