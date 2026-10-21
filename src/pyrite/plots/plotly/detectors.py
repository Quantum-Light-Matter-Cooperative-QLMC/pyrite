"""Plotly detector spectra using backend-neutral response frames (#622).

Functions mirror the established Altair detector builders, including the
beam-current conversion, one-pass broadband Timepix scoring, separate line/brem
grids, charge-density units and Eagle quantum-efficiency secondary axis.  Frame
sampling is peak-aware, shared with Altair, and never modifies detector scoring.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from ...detectors import eaglexo_response as eag
from .._common import SI_K_EDGE_EV, _thr_keV
from .._detector_frames import (
    _INTENSITY_TITLE,
    _chart_frame,
    _title,
    eaglexo_charge_frame,
    eaglexo_detected_frame,
    timepix_detected_frame,
)
from .._spectrum_frames import _tilt_records, _windowed_frame

_MAX_ROWS = 5000
_COLORS = qualitative.Plotly


def _ranges(frame, field, *, x_type, y_type, x_domain, y_domain):
    """Display-only logarithmic domains, preserving original frame numbers."""
    for axis, kind in (("x", x_type), ("y", y_type)):
        if kind not in {"linear", "log"}:
            raise ValueError(f"{axis} scale type must be 'linear' or 'log', got {kind!r}")
    xr = None
    if x_domain is not None:
        lo, hi = map(float, x_domain)
        if not np.isfinite((lo, hi)).all() or not lo < hi:
            raise ValueError("x_domain must contain increasing finite bounds")
        if x_type == "log":
            if lo <= 0:
                raise ValueError("log x_domain bounds must be positive")
            xr = [float(np.log10(lo)), float(np.log10(hi))]
        else:
            xr = [lo, hi]

    yr = None
    if y_domain is not None:
        lo, hi = map(float, y_domain)
        if not np.isfinite((lo, hi)).all() or not lo < hi:
            raise ValueError("y_domain must contain increasing finite bounds")
        if y_type == "log":
            if lo <= 0:
                raise ValueError("log y_domain bounds must be positive")
            yr = [float(np.log10(lo)), float(np.log10(hi))]
        else:
            yr = [lo, hi]
    elif y_type == "log":
        visible = _windowed_frame(frame, x_domain)
        values = visible[field].to_numpy(dtype=float)
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            floor = max(float(positive.min()), float(positive.max()) * 1e-5)
            yr = [float(np.log10(floor)), float(np.log10(positive.max() * 1.05))]
    elif x_domain is not None:
        values = _windowed_frame(frame, x_domain)[field].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and finite.max() > 0:
            yr = [0.0, float(finite.max() * 1.05)]
    return xr, yr


def _prepare(frame, field, *, x_type, y_type, x_domain, y_domain):
    # Limit serialized rows, not model evaluation; weak-response peaks preserved.
    frame = _chart_frame(frame, value_column=field, max_rows=_MAX_ROWS)
    xr, yr = _ranges(
        frame,
        field,
        x_type=x_type,
        y_type=y_type,
        x_domain=x_domain,
        y_domain=y_domain,
    )
    return frame, xr, yr


def _draw_curves(
    frame,
    *,
    field,
    title,
    ylabel,
    x_type,
    y_type,
    x_domain,
    y_domain,
    width,
    height,
    detected=False,
):
    df, xr, yr = _prepare(
        frame,
        field,
        x_type=x_type,
        y_type=y_type,
        x_domain=x_domain,
        y_domain=y_domain,
    )
    fig = go.Figure()
    # The original Altair legend is keyed by beam energy; all line/brem and
    # incident/detected traces for that energy share one Plotly legendgroup.
    energies = sorted(float(x) for x in df["E0_keV"].unique())
    for i, energy in enumerate(energies):
        groups = df[df["E0_keV"] == energy]
        shown = False
        for (azimuth, band, kind), part in _groups(groups, detected):
            part = part.sort_values("energy_eV")
            x = part["energy_eV"].to_numpy(dtype=float)
            y = part[field].to_numpy(dtype=float)
            xdisplay = np.where(x > 0, x, np.nan) if x_type == "log" else x
            ydisplay = np.where(y > 0, y, np.nan) if y_type == "log" else y
            is_incident = kind == "incident"
            is_brem = band == "brem"
            dash = "dot" if is_incident else ("dash" if is_brem else "solid")
            linewidth = 0.9 if is_incident or is_brem else 1.3
            label = f"{energy:g} keV"
            fig.add_trace(
                go.Scatter(
                    x=xdisplay,
                    y=ydisplay,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=not shown,
                    line=dict(color=_COLORS[i % len(_COLORS)], dash=dash, width=linewidth),
                    opacity=0.65 if is_incident else 1.0,
                    # Preserve true source values in hover; masked log-zeros stay in frame.
                    customdata=np.column_stack((x, y)),
                    hovertemplate=(
                        f"E₀: {energy:g} keV<br>azimuth: {azimuth:g}°"
                        f"<br>{band} / {kind}<br>energy: %{{customdata[0]:.6g}} eV"
                        f"<br>value: %{{customdata[1]:.6g}}<extra></extra>"
                    ),
                )
            )
            shown = True
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(title="Photon energy (eV)", type=x_type, range=xr),
        yaxis=dict(title=ylabel, type=y_type, range=yr),
        legend=dict(title="Beam energy (keV)", groupclick="togglegroup"),
        hovermode="closest",
    )
    return fig


def _groups(frame, detected):
    # Avoid joining physically distinct line/brem grids and individual azimuths.
    cols = ["azimuth_deg", "band"] + (["kind"] if detected else [])
    for key, part in frame.groupby(cols, sort=False, dropna=False):
        if detected:
            az, band, kind = key
        else:
            az, band = key
            kind = "charge"
        yield (float(az), str(band), str(kind)), part


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
    """Timepix incident/detected spectra with one broadband response evaluation."""
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
    fig = _draw_curves(
        df,
        field="intensity",
        title=_title(recs, "Timepix3 detected (solid) vs incident (dotted)"),
        ylabel=_INTENSITY_TITLE,
        x_type=x_type,
        y_type=y_type,
        x_domain=x_domain,
        y_domain=y_domain,
        width=width,
        height=height,
        detected=True,
    )
    fig.add_vline(x=_thr_keV() * 1e3, line=dict(color="gray", dash="dash", width=1))
    return fig


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
    """Eagle XO incident/detected photon spectrum, optionally overlaid with QE."""
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
    suffix = ", energy-resolved" if resolve_energy else ""
    fig = _draw_curves(
        df,
        field="intensity",
        title=_title(recs, f"Eagle XO detected (solid) vs incident (dotted), {coating}{suffix}"),
        ylabel=_INTENSITY_TITLE,
        x_type=x_type,
        y_type=y_type,
        x_domain=x_domain,
        y_domain=y_domain,
        width=width,
        height=height,
        detected=True,
    )
    fig.add_vline(x=SI_K_EDGE_EV, line=dict(color="gray", dash="dash", width=1))
    if show_qe:
        E_all = df["energy_eV"].to_numpy(dtype=float)
        lo, hi = max(float(np.nanmin(E_all)), 1.0), float(np.nanmax(E_all))
        if lo < hi:
            Eqe = np.geomspace(lo, hi, 400)
            fig.add_trace(
                go.Scatter(
                    x=Eqe,
                    y=np.asarray(eag.qe(Eqe, coating), dtype=float),
                    mode="lines",
                    name="Quantum efficiency",
                    yaxis="y2",
                    line=dict(color="gray", width=1),
                    opacity=0.5,
                    hovertemplate="energy: %{x:.6g} eV<br>QE: %{y:.4g}<extra></extra>",
                )
            )
            fig.update_layout(
                yaxis2=dict(
                    title="QE",
                    overlaying="y",
                    side="right",
                    range=[0, 1.05],
                    showgrid=False,
                )
            )
    return fig


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
    """Eagle XO recorded charge density, separating line and broadband grids."""
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = eaglexo_charge_frame(
        recs,
        settings,
        coating=coating,
        collapse_azimuth=collapse_azimuth,
        band=band,
    )
    if df.empty:
        return None
    fig = _draw_curves(
        df,
        field="charge_density",
        title=_title(recs, f"Eagle XO recorded charge density ({coating}, dashed = brem)"),
        ylabel="charge density (e-/eV/s)",
        x_type=x_type,
        y_type=y_type,
        x_domain=x_domain,
        y_domain=y_domain,
        width=width,
        height=height,
    )
    fig.add_vline(x=SI_K_EDGE_EV, line=dict(color="gray", dash="dash", width=1))
    return fig
