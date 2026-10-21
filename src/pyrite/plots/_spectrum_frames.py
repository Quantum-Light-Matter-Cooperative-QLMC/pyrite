"""Renderer-neutral spectral frames shared by Plotly and legacy Altair views.

Only numerical preparation lives here; no Altair/Plotly/Matplotlib dependency.
Spectra retain raw energy coordinates, component normalization and peak-aware
sampling from the established Altair implementation (#622).
"""

import numpy as np
import pandas as pd

from .._spectral_components import cxr_spectrum
from ..results import records
from ..results.store import _detected_background_wide
from ._common import _best_azimuth, _cxr_brem

_FRAME_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
_TAIL_BUDGET_DIVISOR = 10
_NA_PER_UA = 1000.0
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


def _windowed_frame(df, x_domain):
    """Sub-frame of ``df`` whose ``energy_eV`` falls inside ``x_domain`` -- used
    ONLY to size a y-scale's domain, never to restrict what's drawn. Vega-Lite
    always computes a scale's domain from that encoding's whole dataset,
    regardless of another encoding's domain, so narrowing the x-window (e.g.
    broad x-max) wouldn't otherwise shrink the y-axis to the data actually
    visible in that window. Falls back to the full frame when the window
    excludes everything, so an out-of-range ``x_domain`` can't blank the axis.
    """
    if x_domain is None:
        return df
    lo, hi = x_domain
    mask = pd.Series(True, index=df.index)
    if lo is not None:
        mask &= df["energy_eV"] >= lo
    if hi is not None:
        mask &= df["energy_eV"] <= hi
    windowed = df.loc[mask]
    return windowed if not windowed.empty else df


def _narrow_line_domain(records, band, x_domain):
    """Resolve an automatic narrowband domain from the widest line grid."""
    if band != "narrow" or x_domain is not None:
        return x_domain

    domains = []
    for record in records:
        energy = np.asarray(record["E_grid"], dtype=float)
        finite = energy[np.isfinite(energy)]
        if finite.size:
            domains.append((float(finite.min()), float(finite.max())))
    return max(domains, key=lambda domain: domain[1] - domain[0], default=None)


def _characteristic_view(
    record, *, include_characteristic, include_channeling=True, include_transition=True
):
    """Drop the line-grid components that are not selected."""
    dropped = {
        key
        for key, selected in (
            ("spec_characteristic", include_characteristic),
            ("spec_channeling", include_channeling),
            ("spec_transition", include_transition),
        )
        if not selected and key in record
    }
    if not dropped:
        return record
    return {key: value for key, value in record.items() if key not in dropped}


def _selected_cxr_record(
    record,
    line_source,
    *,
    include_cxr,
    include_characteristic,
    include_channeling=True,
    include_transition=True,
):
    """Score selected line-grid terms without adding a component twice."""
    selected = np.zeros_like(np.asarray(line_source, dtype=float))
    if include_cxr:
        selected += np.asarray(line_source, dtype=float)
    components = (
        ("spec_characteristic", include_characteristic),
        ("spec_channeling", include_channeling),
        ("spec_transition", include_transition),
    )
    for key, include in components:
        if include and record.get(key) is not None:
            selected += np.asarray(record[key], dtype=float)
    scored_record = {**record, "spec": selected}
    for key, _ in components:
        scored_record.pop(key, None)
    return scored_record


def _record_frame(
    r,
    settings,
    *,
    include_brem,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    sum_components=False,
    meta,
):
    if sum_components:
        scored_record = _selected_cxr_record(
            r,
            r["spec"],
            include_cxr=include_cxr,
            include_characteristic=include_characteristic,
            include_channeling=include_channeling,
        )
    else:
        scored_record = _characteristic_view(
            r,
            include_characteristic=include_characteristic,
            include_channeling=include_channeling,
        )
    E = np.asarray(r["E_grid"], dtype=float)
    scale = r["scale"] * _NA_PER_UA
    cxr_det, brem_det = _cxr_brem(scored_record, settings, convolve=False)
    cxr_det = np.asarray(cxr_det, dtype=float)
    brem_det = np.asarray(brem_det, dtype=float)

    total_E = E
    total = (cxr_det + brem_det) if include_brem else cxr_det
    cxr_basis = total if sum_components else cxr_det
    cxr_grid = np.ones(E.shape, dtype=bool)
    brem_E = E
    brem = brem_det

    if include_brem:
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
            cxr_basis = np.concatenate([cxr_det, np.zeros(E_tail.shape, dtype=float)])
            cxr_grid = np.concatenate([cxr_grid, np.zeros(E_tail.shape, dtype=bool)])
            brem_E = np.concatenate([E, E_tail])
            brem = np.concatenate([brem, brem_tail])

    frames = [
        pd.DataFrame(
            {
                "energy_eV": total_E,
                "intensity": total * scale,
                "_line_intensity": cxr_basis * scale,
                "_line_grid": cxr_grid,
                **meta,
                "component": "total",
            }
        )
    ]
    if include_brem and not sum_components:
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": brem_E,
                    "intensity": brem * scale,
                    "_line_intensity": cxr_basis * scale,
                    "_line_grid": cxr_grid,
                    **meta,
                    "component": "brem",
                }
            )
        )
    if include_cxr and not sum_components:
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": total_E,
                    "intensity": cxr_basis * scale,
                    "_line_intensity": cxr_basis * scale,
                    "_line_grid": cxr_grid,
                    **meta,
                    "component": "line",
                }
            )
        )
    if include_coherent and r.get("spec_coherent") is not None:
        if sum_components:
            coherent_record = _selected_cxr_record(
                r,
                r["spec_coherent"],
                include_cxr=include_cxr,
                include_characteristic=include_characteristic,
                include_channeling=include_channeling,
            )
        else:
            coherent_record = {**scored_record, "spec": r["spec_coherent"]}
        coherent_cxr, _ = _cxr_brem(coherent_record, settings, convolve=False)
        coherent_cxr = np.asarray(coherent_cxr, dtype=float)
        # The coherent curve shares the total's grid. Its wide tail contains
        # brem in summed mode and zeros in decomposition mode.
        tail_len = total_E.size - E.size
        coherent_total = (
            coherent_cxr + brem_det if sum_components and include_brem else coherent_cxr
        )
        coherent_tail = (
            brem_tail if tail_len and sum_components and include_brem else np.zeros(tail_len)
        )
        coherent_basis = (
            np.concatenate([coherent_total, coherent_tail]) if tail_len > 0 else coherent_total
        )
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": total_E,
                    "intensity": coherent_basis * scale,
                    "_line_intensity": coherent_basis * scale,
                    "_line_grid": cxr_grid,
                    **meta,
                    "component": "coherent",
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
    """Decimate logical traces without charging paired components twice.

    The coherent-line grid owns the budget. When a wide background tail is
    present, retain the complete line grid if it fits and spend the remainder
    on the tail. Otherwise reserve ten percent for the tail and use the rest
    for peak-preserving line sampling. Total/background components then reuse
    identical selected coordinates.
    """
    if max_points is None or df.empty or len(df) <= max_points:
        return df.drop(columns=["_line_intensity", "_line_grid"], errors="ignore")
    trace_cols = [
        c
        for c in df.columns
        if c not in {"energy_eV", "intensity", "_line_intensity", "_line_grid", "component"}
    ]
    traces = list(df.groupby(trace_cols, sort=False, dropna=False))
    if not traces:
        return df.drop(columns=["_line_intensity", "_line_grid"], errors="ignore")
    per_trace = max(3, int(max_points) // len(traces))
    frames = []
    for _, trace in traces:
        components = {
            component: grp.sort_values("energy_eV")
            for component, grp in trace.groupby("component", sort=False, dropna=False)
        }
        basis = components.get("total", next(iter(components.values())))
        basis_energy = basis["energy_eV"].to_numpy()
        for component in components.values():
            component_energy = component["energy_eV"].to_numpy()
            if not np.array_equal(component_energy, basis_energy):
                raise ValueError("spectrum components must share one energy grid")

        basis_intensity = basis["_line_intensity"].to_numpy()
        line_positions = np.flatnonzero(basis["_line_grid"].to_numpy())
        tail_positions = np.flatnonzero(~basis["_line_grid"].to_numpy())
        if line_positions.size and tail_positions.size:
            minimum_tail = min(tail_positions.size, max(1, per_trace // _TAIL_BUDGET_DIVISOR))
            cxr_budget = min(line_positions.size, per_trace - minimum_tail)
            tail_budget = min(tail_positions.size, per_trace - cxr_budget)
        elif line_positions.size:
            cxr_budget, tail_budget = min(line_positions.size, per_trace), 0
        else:
            cxr_budget, tail_budget = 0, min(tail_positions.size, per_trace)
        line_idx = (
            line_positions[_peak_preserving_indices(basis_intensity[line_positions], cxr_budget)]
            if cxr_budget
            else np.array([], dtype=int)
        )
        tail_idx = (
            tail_positions[_peak_preserving_indices(basis_intensity[tail_positions], tail_budget)]
            if tail_budget
            else np.array([], dtype=int)
        )
        idx = np.sort(np.concatenate([line_idx, tail_idx]))
        for component in components.values():
            frames.append(component.iloc[idx])
    return pd.concat(frames, ignore_index=True).drop(columns=["_line_intensity", "_line_grid"])


def spectrum_frame(
    recs,
    settings,
    *,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    collapse_azimuth=True,
    band="narrow",
    max_points=None,
):
    """Tidy long-form spectrum table for ``recs`` (already restricted to one polar
    tilt): one row per (beam energy, energy-grid point, component). Mirrors the
    response-free per-energy source view of :func:`pyrite.plots.mpl.spectra._draw_by_energy`.

    Whenever ``include_brem`` and a wide continuum (``brem_wide`` /
    ``E_grid_brem``) are available, the ``brem``/``total`` traces extend past
    the line grid's own cutoff with that continuum (line contribution taken as
    zero there) for BOTH ``band`` values, so neither view cuts off abruptly at
    the line grid's edge. ``band="narrow"``/``"broad"`` no longer changes this
    data prep -- callers use it only to pick which default ``x_domain`` a chart
    renders (see :func:`spectrum_chart`). ``max_points`` bounds distinct energy
    coordinates across logical traces. Paired total/background components share
    those coordinates, so showing the background cannot consume the coherent
    line's sampling budget.
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
                _record_frame(
                    r,
                    settings,
                    include_brem=include_brem,
                    include_cxr=include_cxr,
                    include_characteristic=include_characteristic,
                    include_channeling=include_channeling,
                    include_coherent=include_coherent,
                    meta=meta,
                )
            )
    if not frames:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)


def _compare_frame(
    recs,
    settings,
    *,
    hue,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    sum_components=False,
    band="narrow",
    max_points=None,
):
    """Tidy long-form spectrum table for ``recs``, ONE row per (hue value, energy
    -grid point, component) -- the generalization of :func:`spectrum_frame` to an
    arbitrary case field (``hue``). Duplicate records sharing a ``hue`` value
    (e.g. several azimuths at the same polar tilt) collapse to the single
    strongest-peak record, exactly like :func:`_best_azimuth`'s per-energy
    selection (the largest displayed line peak wins). ``band`` and
    ``max_points`` have the same meaning as in :func:`spectrum_frame`.
    """
    _validate_band(band)
    groups: dict[float, list] = {}
    for r in recs:
        key = float(r["case"][hue])
        groups.setdefault(key, []).append(r)

    frames = []
    for grp in groups.values():
        r = max(
            grp,
            key=lambda rr: float(
                np.max(
                    cxr_spectrum(
                        rr, characteristic=include_characteristic, channeling=include_channeling
                    )
                )
            ),
        )
        row_meta = {
            "E0_keV": float(r["case"]["E0_keV"]),
            "tilt_deg": float(r["case"]["tilt_deg"]),
            "tilt_azim_deg": float(r["case"]["tilt_azim_deg"]),
        }
        frames.extend(
            _record_frame(
                r,
                settings,
                include_brem=include_brem,
                include_cxr=include_cxr,
                include_characteristic=include_characteristic,
                include_channeling=include_channeling,
                include_coherent=include_coherent,
                sum_components=sum_components,
                meta=row_meta,
            )
        )
    _columns = ["energy_eV", "intensity", "E0_keV", "tilt_deg", "tilt_azim_deg", "component"]
    if not frames:
        return pd.DataFrame(columns=_columns)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)


def _multi_case_frame(
    cases,
    settings,
    *,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    sum_components=False,
    band="narrow",
    max_points=None,
):
    """Tidy long-form spectrum table, ONE row per (case-basket entry, energy-grid
    point, component) -- the case-basket counterpart of :func:`_compare_frame`
    with its best-peak collapse removed: every ``(record, label)`` pair in
    ``cases`` keeps its own line, even when two entries share every case field
    :func:`_compare_frame` would hue by (the whole point of a hand-picked
    basket, where the user already chose both on purpose)."""
    _validate_band(band)
    frames = []
    for r, label in cases:
        frames.extend(
            _record_frame(
                r,
                settings,
                include_brem=include_brem,
                include_cxr=include_cxr,
                include_characteristic=include_characteristic,
                include_channeling=include_channeling,
                include_coherent=include_coherent,
                sum_components=sum_components,
                meta={"label": str(label)},
            )
        )
    columns = ["energy_eV", "intensity", "label", "component"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)
