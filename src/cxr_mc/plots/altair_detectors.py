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

from .. import eaglexo_response as eag
from ._common import _best_azimuth
from .altair_spectra import _tilt_records
from .detectors import (
    SI_K_EDGE_EV,
    _eag_detected,
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
    case = recs[0]["case"]
    return (
        f"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} um, "
        f"theta_tilt={case['tilt_deg']:.1f} deg -- {tail}"
    )


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
):
    """Tidy long-form incident-vs-Timepix3-detected table for ``recs`` (already
    restricted to one polar tilt): one row per (beam energy, grid point, kind),
    ``kind`` in ``{"incident", "detected"}``, both already scaled to Phs/eV/s/nA.
    Mirrors :func:`cxr_mc.plots.detectors._draw_timepix_detected`. The ``band``
    column is always ``"line"`` here (Timepix uses the line grid only). Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, kind, band``."""
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            E = np.asarray(r["E_grid"], dtype=float)
            inc, det = _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed)
            az = float(r["case"]["tilt_azim_deg"])
            for kind, y in (("incident", inc), ("detected", det)):
                frames.append(
                    pd.DataFrame(
                        {
                            "energy_eV": E,
                            "intensity": np.asarray(y, dtype=float),
                            "E0_keV": float(E0),
                            "azimuth_deg": az,
                            "kind": kind,
                            "band": "line",
                        }
                    )
                )
    if not frames:
        return pd.DataFrame(columns=_DET_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _detected_layers(df):
    """The shared incident (dashed)/detected (solid) line layers for a
    detected-vs-incident frame. ``detail`` splits the line/brem bands so the two
    grids never join across their gap."""
    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)"),
        y=alt.Y("intensity:Q", title="Phs/eV/s/nA", scale=alt.Scale(type="log")),
        color=alt.Color("E0_keV:N", title="beam energy (keV)"),
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
    width=720,
    height=360,
):
    """Interactive log-y overlay of incident (dotted) vs Timepix3-detected (solid)
    spectra at ONE polar tilt, one colour per beam energy, with the counting
    threshold marked. The Altair counterpart of
    :func:`cxr_mc.plots.plot_timepix_detected` / ``browse(kind="timepix")``.
    Returns an :class:`altair.Chart`, or ``None`` when there are no records."""
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
    )
    if df.empty:
        return None
    incident, detected = _detected_layers(df)
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
):
    """Tidy long-form incident-vs-Eagle-XO-detected table for ``recs`` (one polar
    tilt): one row per (beam energy, grid point, kind, band). ``kind`` in
    ``{"incident", "detected"}``; ``band`` is ``"line"`` (fine line grid) or
    ``"brem"`` (the wide brem grid, present only when the sweep stored
    ``brem_wide``). Mirrors :func:`cxr_mc.plots.detectors._draw_eaglexo_detected`,
    including the thin-sensor QE roll-off on the wide brem. Columns:
    ``energy_eV, intensity, E0_keV, azimuth_deg, kind, band``."""
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            E = np.asarray(r["E_grid"], dtype=float)
            inc, det = _eag_detected(r, settings, coating, resolve_energy)
            az = float(r["case"]["tilt_azim_deg"])
            rows = [("line", E, inc, det)]
            if r.get("brem_wide") is not None:
                Eb = np.asarray(r["E_grid_brem"], dtype=float)
                inc_b = np.asarray(r["brem_wide"], dtype=float) * r["scale"]
                det_b = inc_b * eag.qe(Eb, coating)
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
    width=720,
    height=360,
):
    """Interactive log-log overlay of incident (dotted) vs Eagle-XO-detected
    (solid) spectra at ONE polar tilt, with the Si-K edge marked and (when
    ``show_qe``) a faint QE envelope on an independent right axis -- the camera's
    signature, soft PXR lines passing at ~90% QE while the hard brem is crushed by
    the thin sensor. The Altair counterpart of
    :func:`cxr_mc.plots.plot_eaglexo_detected` / ``browse(kind="eaglexo")``.
    Returns an :class:`altair.Chart`, or ``None`` when there are no records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = eaglexo_detected_frame(
        recs,
        settings,
        coating=coating,
        resolve_energy=resolve_energy,
        collapse_azimuth=collapse_azimuth,
    )
    if df.empty:
        return None
    incident, detected = _detected_layers(df)
    incident = incident.encode(x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=_logx()))
    detected = detected.encode(x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=_logx()))
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
                x=alt.X("energy_eV:Q", scale=_logx()),
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
def eaglexo_charge_frame(recs, settings, *, coating="BN", collapse_azimuth=True):
    """Tidy long-form Eagle XO recorded-charge density [e-/eV/s] for ``recs`` (one
    polar tilt): one row per (beam energy, grid point, band). ``band`` is
    ``"line"`` (fine grid, lines + line-grid brem) or ``"brem"`` (the wide brem
    grid). Every photon is weighted by E/W_Si, so the hard brem carries far more
    charge than its photon count. Mirrors
    :func:`cxr_mc.plots.detectors._draw_eaglexo_charge`. Columns:
    ``energy_eV, charge_density, E0_keV, azimuth_deg, band``."""
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
            if r.get("brem_wide") is not None:
                Eb = np.asarray(r["E_grid_brem"], dtype=float)
                inc_b = np.nan_to_num(np.asarray(r["brem_wide"], dtype=float) * r["scale"])
                cd_b = inc_b * eag.qe(Eb, coating) * (Eb / eag.W_EHP_EV) * cur
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
    width=720,
    height=360,
):
    """Interactive log-log Eagle XO recorded-charge density [e-/eV/s] at ONE polar
    tilt (lines solid, wide brem dashed), one colour per beam energy, Si-K edge
    marked. A CCD integrates charge rather than counting photons, so this shows
    where the recorded signal actually comes from. The Altair counterpart of
    :func:`cxr_mc.plots.plot_eaglexo_charge` / ``browse(kind="eaglexo_charge")``.
    Returns an :class:`altair.Chart`, or ``None`` when there are no records."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = eaglexo_charge_frame(recs, settings, coating=coating, collapse_azimuth=collapse_azimuth)
    if df.empty:
        return None
    base = alt.Chart(df).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=_logx()),
        y=alt.Y(
            "charge_density:Q",
            title="charge density (e-/eV/s)",
            scale=alt.Scale(type="log"),
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


def _logx():
    """A log x-scale that tolerates the brem grid's E=0 row (Vega-Lite drops
    non-positive values on a log axis)."""
    return alt.Scale(type="log")
