"""Renderer-neutral detector-response spectrum frames.

Plotly and legacy Altair share these exact model evaluations, unit conversions,
and peak-aware sampling.  No plotting backend is imported here.
"""

import numpy as np
import pandas as pd

from .._spectral_components import cxr_spectrum, incident_spectrum
from ..detectors import Detector, Timepix3
from ..detectors import eaglexo_response as eag
from ..results import beam_current_na
from ._common import (
    _best_azimuth,
    _case_title,
    _eag_detected,
    _eag_wide_brem,
    _eag_wide_charge,
    _tpx_detected,
)
from ._spectrum_frames import _peak_preserving_indices, _tilt_records, _validate_band

# incident-vs-detected long form (Timepix + Eagle photon density)
_DET_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "kind", "band"]
# Records carry per-nA flux; these charts report per-uA, matching
# :mod:`pyrite.plots.altair.spectra`. Display convention only.
_NA_PER_UA = 1000.0
_INTENSITY_TITLE = "Phs/s/eV/uA"
# Eagle recorded-charge density long form
_CHARGE_COLUMNS = ["energy_eV", "charge_density", "E0_keV", "azimuth_deg", "band"]
_CHART_MAX_ROWS = 5000


def _chart_frame(df, *, value_column, max_rows=_CHART_MAX_ROWS):
    """Bound serialized detector data while retaining peaks on each curve.

    Incident and detected rows share sampled energies, so their overlays stay
    aligned. Normalize each kind before selecting peaks: the weaker detected
    response must not lose its narrow features to the incident curve.
    """
    if len(df) <= max_rows:
        return df
    group_columns = ["E0_keV", "azimuth_deg", "band"]
    groups = list(df.groupby(group_columns, sort=False, dropna=False))
    kinds_per_group = 2 if "kind" in df else 1
    points_per_group = max(2, max_rows // (len(groups) * kinds_per_group))
    sampled = []
    for _, group in groups:
        if "kind" in group:
            curves = [
                part.sort_values("energy_eV") for _, part in group.groupby("kind", sort=False)
            ]
            energy = curves[0]["energy_eV"].to_numpy()
            if any(
                not np.array_equal(curve["energy_eV"].to_numpy(), energy) for curve in curves[1:]
            ):
                raise ValueError("detector curves must share one energy grid")
            normalized = [
                np.abs(curve[value_column].to_numpy())
                / max(np.nanmax(np.abs(curve[value_column])), 1e-300)
                for curve in curves
            ]
            basis = np.maximum.reduce(normalized)
        else:
            curves = [group.sort_values("energy_eV")]
            basis = np.abs(curves[0][value_column].to_numpy())
        indices = _peak_preserving_indices(basis, points_per_group)
        sampled.extend(curve.iloc[indices] for curve in curves)
    return pd.concat(sampled, ignore_index=True)


def _collapsed(recs, *, collapse_azimuth):
    """Group ``recs`` by beam energy, optionally collapsing each group to its
    best (maximum response-free source line) azimuth -- the per-energy record selection shared by
    every detector draw. Yields ``(E0_keV, [records])`` in ascending energy."""
    energies = sorted({r["case"]["E0_keV"] for r in recs})
    for E0 in energies:
        grp = _best_azimuth([r for r in recs if r["case"]["E0_keV"] == E0], collapse_azimuth)
        yield E0, grp


def _title(recs, tail):
    return _case_title(recs[0]["case"], tail, latex=False)


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
    E_cxr = np.asarray(r["E_grid"], dtype=float)
    E_brem = r.get("E_grid_brem")
    brem_wide = r.get("brem_wide")
    if E_brem is None or brem_wide is None:
        incident = np.asarray(incident_spectrum(r), dtype=float)
        return E_cxr, incident * r["scale"]

    E = np.asarray(E_brem, dtype=float)
    brem = np.asarray(brem_wide, dtype=float)
    # The 0-eV brem bin cannot contribute to a photon-energy spectrum and is
    # below the line grid's modeled band.  Starting at the line-grid floor also
    # keeps the response grid physically meaningful.
    mask = E >= float(np.nanmin(E_cxr))
    E = E[mask]
    brem = brem[mask]
    coherent = np.interp(E, E_cxr, np.asarray(cxr_spectrum(r), dtype=float), left=0.0, right=0.0)
    return E, (coherent + brem) * r["scale"]


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
    ``kind`` in ``{"incident", "detected"}``, both already scaled to Phs/s/eV/uA.
    Mirrors :func:`pyrite.plots.mpl.detectors._draw_timepix_detected`. ``band`` is
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
                detector = Detector(
                    response=Timepix3(
                        n_mc=n_mc,
                        seed=seed,
                        thickness_um=thickness_um,
                        bias_v=bias_v,
                    )
                )
                det = detector.score(E, inc)
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
                                "intensity": np.asarray(y, dtype=float) * _NA_PER_UA,
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
    ``{"incident", "detected"}``, both scaled to Phs/s/eV/uA; ``band`` is
    ``"line"`` (fine line grid) or ``"brem"`` (the wide brem grid, present only
    when the sweep stored ``brem_wide``). Mirrors :func:`pyrite.plots.mpl.detectors._draw_eaglexo_detected`,
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
                                "intensity": np.asarray(y, dtype=float) * _NA_PER_UA,
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


def eaglexo_charge_frame(recs, settings, *, coating="BN", collapse_azimuth=True, band="broad"):
    """Tidy long-form Eagle XO recorded-charge density [e-/eV/s] for ``recs`` (one
    polar tilt): one row per (beam energy, grid point, band). ``band`` is
    ``"line"`` (fine grid, lines + line-grid brem) or ``"brem"`` (the wide brem
    grid). Every photon is weighted by E/W_Si, so the hard brem carries far more
    charge than its photon count. Mirrors
    :func:`pyrite.plots.mpl.detectors._draw_eaglexo_charge`. Columns:
    ``energy_eV, charge_density, E0_keV, azimuth_deg, band``."""
    _validate_band(band)
    frames = []
    for E0, grp in _collapsed(recs, collapse_azimuth=collapse_azimuth):
        for r in grp:
            cur = beam_current_na(r, settings)
            resp = eag.get_response(r["E_grid"], coating=coating)
            E = np.asarray(r["E_grid"], dtype=float)
            cd_cxr = resp.charge_density(incident_spectrum(r) * r["scale"]) * cur
            az = float(r["case"]["tilt_azim_deg"])
            frames.append(
                pd.DataFrame(
                    {
                        "energy_eV": E,
                        "charge_density": np.asarray(cd_cxr, dtype=float),
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
