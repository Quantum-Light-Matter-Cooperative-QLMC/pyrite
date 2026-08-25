"""Altair spectra

Altair / Vega-Lite renderer for response-free CXR source spectra -- a fast, interactive
alternative to the matplotlib :mod:`pyrite.plots.mpl.spectra` figures. The headline
motivation for the jupyter -> marimo migration is that the matplotlib spectra are
sluggish over the fine energy grids; Vega-Lite renders them in the browser and
pans/zooms interactively at a fraction of the redraw cost.

Non-destructive: this module reuses the exact per-record data prep
(:func:`pyrite.plots._common._line_brem`) that the matplotlib path uses, so the
physics and units are identical -- only the renderer differs. The matplotlib
``plots/`` package is left untouched, and these names are intentionally NOT
re-exported from ``pyrite.plots`` (that package has a frozen export-set guard);
import them from the submodule:

    from pyrite.plots.altair.spectra import (
        spectrum_chart,
        compare_spectrum_chart,
        multi_case_spectrum_chart,
        material_comparison_chart,
    )

Functions return :class:`altair.Chart` objects, which render directly in marimo
and Jupyter.

Note on size: Vega-Lite caps a spec at 5000 data rows by default. A dense spectrum
(several thousand grid points x several beam energies) can exceed that; enable a
larger transport once at the top of a notebook with
``altair.data_transformers.enable("vegafusion")`` (shipped with ``marimo[recommended]``)
or ``altair.data_transformers.disable_max_rows()``. This module does not mutate that
global state itself.
"""

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from ...results import records
from ...results.store import _detected_background_wide
from .._common import _best_azimuth, _case_title, _comparison_drop_message, _line_brem
from ._typing import _mark_chart

_FRAME_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
_TAIL_BUDGET_DIVISOR = 10

# Case fields a `compare_spectrum_chart` caller may color by, and the axis/legend
# metadata for each -- readable title + unit suffix used both in the legend and
# the frame-building loop below.
_COMPARE_HUE_FIELDS = {
    "E0_keV": "beam energy (keV)",
    "tilt_deg": "polar tilt (deg)",
    "tilt_azim_deg": "azimuth (deg)",
}

_TITLE_FONT_SIZE = 18
_AXIS_LABEL_FONT_SIZE = 14
_AXIS_TITLE_FONT_SIZE = 16
_LEGEND_LABEL_FONT_SIZE = 14
_LEGEND_TITLE_FONT_SIZE = 16


def _spectrum_axis() -> alt.Axis:
    return alt.Axis(
        labelFontSize=_AXIS_LABEL_FONT_SIZE,
        titleFontSize=_AXIS_TITLE_FONT_SIZE,
    )


def _spectrum_legend() -> alt.Legend:
    return alt.Legend(
        labelFontSize=_LEGEND_LABEL_FONT_SIZE,
        titleFontSize=_LEGEND_TITLE_FONT_SIZE,
    )


def _spectrum_title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, fontSize=_TITLE_FONT_SIZE)


def _style_chart(chart):
    """Apply spectral typography at the top level of a composed chart."""
    return (
        chart.configure_axis(
            labelFontSize=_AXIS_LABEL_FONT_SIZE,
            titleFontSize=_AXIS_TITLE_FONT_SIZE,
        )
        .configure_legend(
            labelFontSize=_LEGEND_LABEL_FONT_SIZE,
            titleFontSize=_LEGEND_TITLE_FONT_SIZE,
        )
        .configure_title(fontSize=_TITLE_FONT_SIZE)
    )


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


def _log_y_scale(df, floor_frac=1e-3):
    """Log-scale ``y`` with an explicit positive ``domainMin``, clamped.

    Vega-Lite auto-computes the y domain from the data's raw (min, max) when
    no domain is given, and a log scale whose domain touches 0 collapses
    every point onto one pixel position instead of raising. The wide
    bremsstrahlung tail (``band="broad"``) tapers to exactly 0 at its
    radiative endpoint (photon energy == beam energy), so that 0 always
    reaches the auto domain unless excluded here. Mirrors the matplotlib
    renderer's ``ax.set_ylim(floor, ...)`` guard in ``plots/mpl/spectra.py``.

    That endpoint value still sits below ``domainMin`` once excluded, and
    Vega-Lite does not clamp out-of-domain values by default: it extrapolates
    ``log(0) == -Infinity`` to an invalid pixel position, which renders as a
    spurious spike pinned to the axis extreme instead of just disappearing
    below the floor (matplotlib clips silently at the axis limits; Vega-Lite
    needs ``clamp`` told explicitly to do the same).

    ``df`` is whatever frame the caller wants the domain fit to -- pass a
    window from :func:`_windowed_frame` to size the domain to a visible x-range
    instead of the full dataset.
    """
    positive = df.loc[df["intensity"] > 0, "intensity"]
    if positive.empty:
        return _scale("log")
    floor = max(float(positive.min()), float(positive.max()) * floor_frac)
    return alt.Scale(type="log", domainMin=floor, clamp=True)


def _linear_y_scale(df, x_domain):
    """Explicit zero-anchored linear y-domain sized to the visible x-window.

    Mirrors :func:`_log_y_scale`'s reason for existing: Vega-Lite computes a
    scale's domain from the encoding's whole dataset, so restricting
    ``x_domain`` (e.g. broad x-max) wouldn't otherwise autoscale y to the data
    actually visible in that window. Falls back to Vega-Lite's own auto
    domain (``_scale("linear")``, unbounded) when no ``x_domain`` is set --
    the common case -- to leave that default behavior unchanged.
    """
    if x_domain is None:
        return _scale("linear")
    windowed = _windowed_frame(df, x_domain)
    hi = float(windowed["intensity"].max()) if not windowed.empty else 0.0
    if hi <= 0:
        return _scale("linear")
    return alt.Scale(domainMin=0.0, domainMax=hi * 1.05)


def _scale(scale_type, domain=None):
    if scale_type not in {"linear", "log"}:
        raise ValueError(f"scale type must be 'linear' or 'log', got {scale_type!r}")
    kwargs: dict[str, Any] = {}
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


def _record_frame(r, settings, *, include_brem, include_line=False, include_coherent=False, meta):
    E = np.asarray(r["E_grid"], dtype=float)
    line_det, brem_det = _line_brem(r, settings, convolve=False)
    line_det = np.asarray(line_det, dtype=float)
    brem_det = np.asarray(brem_det, dtype=float)

    total_E = E
    total = (line_det + brem_det) if include_brem else line_det
    line_basis = line_det
    line_grid = np.ones(E.shape, dtype=bool)
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
            line_basis = np.concatenate([line_det, np.zeros(E_tail.shape, dtype=float)])
            line_grid = np.concatenate([line_grid, np.zeros(E_tail.shape, dtype=bool)])
            brem_E = np.concatenate([E, E_tail])
            brem = np.concatenate([brem, brem_tail])

    frames = [
        pd.DataFrame(
            {
                "energy_eV": total_E,
                "intensity": total * r["scale"],
                "_line_intensity": line_basis * r["scale"],
                "_line_grid": line_grid,
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
                    "_line_intensity": line_basis * r["scale"],
                    "_line_grid": line_grid,
                    **meta,
                    "component": "brem",
                }
            )
        )
    if include_line:
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": total_E,
                    "intensity": line_basis * r["scale"],
                    "_line_intensity": line_basis * r["scale"],
                    "_line_grid": line_grid,
                    **meta,
                    "component": "line",
                }
            )
        )
    if include_coherent and r.get("spec_coherent") is not None:
        coherent_line, _ = _line_brem({**r, "spec": r["spec_coherent"]}, settings, convolve=False)
        coherent_line = np.asarray(coherent_line, dtype=float)
        # `total_E`'s tail (past the line grid) is brem-only; the coherent overlay
        # has no signal there, so it zero-pads to share the SAME energy coordinates
        # as `total`/`line` -- required by `_decimate_frame`'s per-trace grid check.
        tail_len = total_E.size - E.size
        coherent_basis = (
            np.concatenate([coherent_line, np.zeros(tail_len, dtype=float)])
            if tail_len > 0
            else coherent_line
        )
        frames.append(
            pd.DataFrame(
                {
                    "energy_eV": total_E,
                    "intensity": coherent_basis * r["scale"],
                    "_line_intensity": coherent_basis * r["scale"],
                    "_line_grid": line_grid,
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
            line_budget = min(line_positions.size, per_trace - minimum_tail)
            tail_budget = min(tail_positions.size, per_trace - line_budget)
        elif line_positions.size:
            line_budget, tail_budget = min(line_positions.size, per_trace), 0
        else:
            line_budget, tail_budget = 0, min(tail_positions.size, per_trace)
        line_idx = (
            line_positions[_peak_preserving_indices(basis_intensity[line_positions], line_budget)]
            if line_budget
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


def _compact_component_frame(df):
    """Store paired total/brem values once per rendered energy coordinate."""
    index_cols = [c for c in df.columns if c not in {"intensity", "component"}]
    compact = df.pivot(index=index_cols, columns="component", values="intensity").reset_index()
    compact.columns.name = None
    return compact


def _fold_components(chart, include_brem, include_line=False, include_coherent=False):
    components = ["total"]
    if include_brem:
        components.append("brem")
    if include_line:
        components.append("line")
    if include_coherent:
        components.append("coherent")
    return chart.transform_fold(components, as_=["component", "intensity"])


def spectrum_frame(
    recs,
    settings,
    *,
    include_brem=True,
    include_line=False,
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
                    include_line=include_line,
                    include_coherent=include_coherent,
                    meta=meta,
                )
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
    include_line=False,
    include_coherent=False,
    collapse_azimuth=True,
    x_domain=None,
    x_type="linear",
    y_type="linear",
    band="narrow",
    max_points=5000,
    width=720,
    height=360,
):
    """Interactive Altair line chart of the response-free source spectra at ONE polar tilt.

    Both ``band`` values now draw the wide bremsstrahlung continuum past the
    line grid's cutoff whenever it's stored (see :func:`spectrum_frame`);
    ``band`` only tags which default ``x_domain`` this call renders, letting a
    caller pair a narrow line-grid-scale view with a broad beam-energy-scale
    view of the SAME underlying data. ``x_type``/``y_type`` are ``"linear"``
    or ``"log"``; ``max_points`` bounds serialized energy-coordinate rows
    across logical traces before the Vega-Lite spec is built (with a minimum
    of three rows per trace). Total/background values share each row, and line
    points take priority over the wide tail. Returns an :class:`altair.Chart`,
    or ``None`` when there are no records.
    """
    recs = _tilt_records(results, tilt_deg)
    if not recs:
        return None
    df = spectrum_frame(
        recs,
        settings,
        include_brem=include_brem,
        include_line=include_line,
        include_coherent=include_coherent,
        collapse_azimuth=collapse_azimuth,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "response-free source", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, x_domain))
        if y_type == "log"
        else _linear_y_scale(df, x_domain)
    )

    compact = _compact_component_frame(df)
    base = _fold_components(
        alt.Chart(compact), include_brem, include_line, include_coherent
    ).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title="Intensity (Phs/eV/s/nA)",
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color("E0_keV:N", title="beam energy (keV)", legend=_spectrum_legend()),
        tooltip=["E0_keV:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, include_brem, include_line, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title(title))
        .interactive()
    )


def _component_layers(base, include_brem, include_line=False, include_coherent=False):
    """Shared component -> mark-style layering for the three spectrum-chart
    builders below: ``total`` is the incoherent (or checkpoint-default) line,
    solid; ``brem`` and ``line`` are decomposition overlays of that SAME trace;
    ``coherent`` overlays the checkpoint's ``spec_coherent`` line for a
    ``both``-emission view, dashed so it stays visually distinct from ``total``
    while sharing its hue color."""
    layers = [base.transform_filter(alt.datum.component == "total").mark_line(strokeWidth=1.4)]
    if include_brem:
        layers.append(
            base.transform_filter(alt.datum.component == "brem").mark_line(
                strokeWidth=0.7, strokeDash=[4, 3], opacity=0.7
            )
        )
    if include_line:
        layers.append(
            base.transform_filter(alt.datum.component == "line").mark_line(
                strokeWidth=1.0, strokeDash=[1, 1], opacity=0.85
            )
        )
    if include_coherent:
        layers.append(
            base.transform_filter(alt.datum.component == "coherent").mark_line(
                strokeWidth=1.4, strokeDash=[6, 2]
            )
        )
    return layers


def _compare_frame(
    recs,
    settings,
    *,
    hue,
    include_brem=True,
    include_line=False,
    include_coherent=False,
    band="narrow",
    max_points=None,
):
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
            _record_frame(
                r,
                settings,
                include_brem=include_brem,
                include_line=include_line,
                include_coherent=include_coherent,
                meta=row_meta,
            )
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
    include_line=False,
    include_coherent=False,
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
        include_line=include_line,
        include_coherent=include_coherent,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    title = _case_title(recs[0]["case"], "comparison", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, x_domain))
        if y_type == "log"
        else _linear_y_scale(df, x_domain)
    )
    hue_title = _COMPARE_HUE_FIELDS[hue]

    compact = _compact_component_frame(df)
    base = _fold_components(
        alt.Chart(compact), include_brem, include_line, include_coherent
    ).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title="Intensity (Phs/eV/s/nA)",
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color(f"{hue}:N", title=hue_title, legend=_spectrum_legend()),
        tooltip=[f"{hue}:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, include_brem, include_line, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title(title))
        .interactive()
    )


def _multi_case_frame(
    cases,
    settings,
    *,
    include_brem=True,
    include_line=False,
    include_coherent=False,
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
                include_line=include_line,
                include_coherent=include_coherent,
                meta={"label": str(label)},
            )
        )
    columns = ["energy_eV", "intensity", "label", "component"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _decimate_frame(pd.concat(frames, ignore_index=True), max_points)


def multi_case_spectrum_chart(
    cases,
    settings,
    *,
    include_brem=True,
    include_line=False,
    include_coherent=False,
    x_domain=None,
    x_type="linear",
    y_type="linear",
    band="narrow",
    max_points=5000,
    width=720,
    height=360,
):
    """Interactive Altair line chart overlaying an explicit, hand-picked list of
    ``(record, label)`` pairs -- the cross-material "case basket" comparison
    tab. Unlike :func:`compare_spectrum_chart`, this never collapses entries
    that share a hue-able case field to a single strongest-peak line: every
    entry in ``cases`` draws, colored by its own ``label`` (see
    :func:`pyrite.results.selection.case_label`), so two basket entries that
    happen to share every case field both survive. ``include_brem``,
    ``x_domain``, ``x_type``/``y_type``, ``band``, and ``max_points`` match
    :func:`compare_spectrum_chart`. Returns ``None`` when ``cases`` is empty.
    """
    if not cases:
        return None
    df = _multi_case_frame(
        cases,
        settings,
        include_brem=include_brem,
        include_line=include_line,
        include_coherent=include_coherent,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, x_domain))
        if y_type == "log"
        else _linear_y_scale(df, x_domain)
    )

    compact = _compact_component_frame(df)
    base = _fold_components(
        alt.Chart(compact), include_brem, include_line, include_coherent
    ).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title="Intensity (Phs/eV/s/nA)",
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color("label:N", title="case", legend=_spectrum_legend()),
        tooltip=["label:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, include_brem, include_line, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title("Case comparison"))
        .interactive()
    )


_COMPARISON_SELECTION_TITLES = {
    "quality_peak": "highest line-definition quality",
    "peak": "highest peak flux",
    "line_brem_ratio": "highest local line-to-bremsstrahlung ratio",
}


def material_comparison_chart(
    points,
    dropped,
    select="quality_peak",
    beam_energy_keV=None,
    min_line_quality: float | None = 0.5,
    width=760,
    height=460,
):
    """Interactive counterpart of :func:`pyrite.plots.draw_material_comparison` --
    same precomputed ``(label, line_eV, line_flux, quality, case)`` points and the
    same dropped-material print, only the renderer differs. Point labels shorten
    to just the material name; the selected geometry's beam energy/theta/phi
    (baked into the matplotlib annotation text) move into the hover tooltip
    instead, so Vega-Lite's own overlap handling replaces the matplotlib
    bounding-box separation pass. Returns an :class:`altair.Chart`, or ``None``
    when ``points`` is empty.
    """
    if not points:
        print("no results in any material")
        return None
    df = pd.DataFrame(
        {
            "label": [p[0] for p in points],
            "line_keV": [p[1] / 1e3 for p in points],
            "line_flux": [p[2] for p in points],
            "quality": [p[3] for p in points],
            "E0_keV": [p[4]["E0_keV"] for p in points],
            "tilt_deg": [p[4]["tilt_deg"] for p in points],
            "tilt_azim_deg": [p[4]["tilt_azim_deg"] for p in points],
        }
    )
    selection_title = _COMPARISON_SELECTION_TITLES.get(select, select.replace("_", " "))
    energy_scope = (
        "all beam energies" if beam_energy_keV is None else f"{beam_energy_keV:g} keV beam energy"
    )
    quality_scope = "" if min_line_quality is None else f", line quality >= {min_line_quality:g}"
    title = f"Cross-material comparison — {selection_title} ({energy_scope}{quality_scope})"

    base = alt.Chart(df).encode(
        x=alt.X(
            "line_keV:Q",
            title="dominant coherent line energy (keV)",
            axis=_spectrum_axis(),
        ),
        y=alt.Y(
            "line_flux:Q",
            title="integrated line flux at best geometry (Phs/s)",
            scale=alt.Scale(type="log"),
            axis=_spectrum_axis(),
        ),
    )
    points_mark = _mark_chart(
        base.mark_point(filled=True, size=110, stroke="black", strokeWidth=0.6)
    ).encode(
        color=alt.Color(
            "quality:Q",
            title="line-definition quality",
            scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
            legend=_spectrum_legend(),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="material"),
            alt.Tooltip("line_keV:Q", title="line energy (keV)", format=".3~g"),
            alt.Tooltip("line_flux:Q", title="line flux (Phs/s)", format=".3~g"),
            alt.Tooltip("quality:Q", title="line-definition quality", format=".2f"),
            alt.Tooltip("E0_keV:Q", title="beam energy (keV)"),
            alt.Tooltip("tilt_deg:Q", title="theta (deg)"),
            alt.Tooltip("tilt_azim_deg:Q", title="phi (deg)"),
        ],
    )
    labels_mark = _mark_chart(
        base.mark_text(align="left", dx=7, fontSize=_AXIS_LABEL_FONT_SIZE)
    ).encode(text="label:N")
    chart = (points_mark + labels_mark).properties(
        width=width, height=height, title=_spectrum_title(title)
    )
    if dropped:
        print(
            f"Dropped from cross-material comparison (select={select!r}{quality_scope}): "
            f"{_comparison_drop_message(dropped)}"
        )
    return _style_chart(chart.interactive())
