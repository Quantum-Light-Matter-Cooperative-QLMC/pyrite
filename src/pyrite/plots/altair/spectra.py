"""Altair spectra

Altair / Vega-Lite renderer for response-free CXR source spectra -- a fast, interactive
alternative to the matplotlib :mod:`pyrite.plots.mpl.spectra` figures. The headline
motivation for the jupyter -> marimo migration is that the matplotlib spectra are
sluggish over the fine energy grids; Vega-Lite renders them in the browser and
pans/zooms interactively at a fraction of the redraw cost.

Non-destructive: this module reuses the exact per-record data prep
(:func:`pyrite.plots._common._cxr_brem`) that the matplotlib path uses, so the
physics is identical -- only the renderer and the reported beam-current unit
differ (these charts report Phs/s/eV/uA; the matplotlib path reports the stored
per-nA values). The matplotlib ``plots/`` package is left untouched, and these
names are intentionally NOT re-exported from ``pyrite.plots`` (that package has
a frozen export-set guard); import them from the submodule:

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
import pandas as pd

from ...results import records
from .._common import _case_title, _comparison_drop_message
from ._typing import _mark_chart

_FRAME_COLUMNS = ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
_TAIL_BUDGET_DIVISOR = 10

# ``r["scale"]`` carries per-nA units (see results.store: `scale=... * PER_NA`);
# these charts report per-uA instead, so every intensity column folds in the
# 1000 nA/uA factor. Purely a display convention -- the stored records and the
# matplotlib renderers are untouched.
_NA_PER_UA = 1000.0
_INTENSITY_TITLE = "Intensity (Phs/s/eV/uA)"

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


def _compact_component_frame(df):
    """Store paired total/brem values once per rendered energy coordinate."""
    index_cols = [c for c in df.columns if c not in {"intensity", "component"}]
    compact = df.pivot(index=index_cols, columns="component", values="intensity").reset_index()
    compact.columns.name = None
    return compact


def _fold_components(chart, include_brem, include_cxr=False, include_coherent=False):
    components = ["total"]
    if include_brem:
        components.append("brem")
    if include_cxr:
        components.append("line")
    if include_coherent:
        components.append("coherent")
    return chart.transform_fold(components, as_=["component", "intensity"])


def spectrum_chart(
    results,
    settings,
    *,
    tilt_deg=None,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
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
        include_cxr=include_cxr,
        include_characteristic=include_characteristic,
        include_channeling=include_channeling,
        include_coherent=include_coherent,
        collapse_azimuth=collapse_azimuth,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    automatic_x_domain = x_domain is None
    x_domain = _narrow_line_domain(recs, band, x_domain)
    title = _case_title(recs[0]["case"], "response-free source", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, None if automatic_x_domain else x_domain))
        if y_type == "log"
        else _linear_y_scale(df, None if automatic_x_domain else x_domain)
    )

    compact = _compact_component_frame(df)
    base = _fold_components(alt.Chart(compact), include_brem, include_cxr, include_coherent).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title=_INTENSITY_TITLE,
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color("E0_keV:N", title="beam energy (keV)", legend=_spectrum_legend()),
        tooltip=["E0_keV:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, include_brem, include_cxr, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title(title))
        .interactive(bind_y=False)
    )


def _component_layers(base, include_brem, include_cxr=False, include_coherent=False):
    """Shared component -> mark-style layering for the three spectrum-chart
    builders below: ``total`` is the incoherent (or checkpoint-default) line,
    solid; ``brem`` and ``line`` are decomposition overlays of that SAME trace;
    ``coherent`` overlays the checkpoint's ``spec_coherent`` line for a
    ``both``-emission view, dashed so it stays visually distinct from ``total``
    while sharing its hue color."""
    layers = [base.transform_filter(alt.datum.component == "total").mark_line(strokeWidth=2.2)]
    if include_brem:
        layers.append(
            base.transform_filter(alt.datum.component == "brem").mark_line(
                strokeWidth=1.1, strokeDash=[4, 3], opacity=0.7
            )
        )
    if include_cxr:
        layers.append(
            base.transform_filter(alt.datum.component == "line").mark_line(
                strokeWidth=1.6, strokeDash=[1, 1], opacity=0.85
            )
        )
    if include_coherent:
        layers.append(
            base.transform_filter(alt.datum.component == "coherent").mark_line(
                strokeWidth=2.2, strokeDash=[6, 2]
            )
        )
    return layers


def compare_spectrum_chart(
    results,
    settings,
    *,
    hue,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    sum_components=False,
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
    function. With ``sum_components=True``, component flags select terms in one
    total curve; the default retains separate decomposition overlays. Returns
    an :class:`altair.Chart`, or ``None`` when there are no records.
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
        include_cxr=include_cxr,
        include_characteristic=include_characteristic,
        include_channeling=include_channeling,
        include_coherent=include_coherent,
        sum_components=sum_components,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    automatic_x_domain = x_domain is None
    x_domain = _narrow_line_domain(recs, band, x_domain)
    title = _case_title(recs[0]["case"], "comparison", latex=False)
    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, None if automatic_x_domain else x_domain))
        if y_type == "log"
        else _linear_y_scale(df, None if automatic_x_domain else x_domain)
    )
    hue_title = _COMPARE_HUE_FIELDS[hue]

    compact = _compact_component_frame(df)
    show_brem_layer = include_brem and not sum_components
    show_cxr_layer = include_cxr and not sum_components
    base = _fold_components(
        alt.Chart(compact), show_brem_layer, show_cxr_layer, include_coherent
    ).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title=_INTENSITY_TITLE,
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color(f"{hue}:N", title=hue_title, legend=_spectrum_legend()),
        tooltip=[f"{hue}:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, show_brem_layer, show_cxr_layer, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title(title))
        .interactive(bind_y=False)
    )


def multi_case_spectrum_chart(
    cases,
    settings,
    *,
    include_brem=True,
    include_cxr=False,
    include_characteristic=True,
    include_channeling=True,
    include_coherent=False,
    sum_components=False,
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
    :func:`compare_spectrum_chart`. ``sum_components=True`` selects terms in
    one total curve. Returns ``None`` when ``cases`` is empty.
    """
    if not cases:
        return None
    df = _multi_case_frame(
        cases,
        settings,
        include_brem=include_brem,
        include_cxr=include_cxr,
        include_characteristic=include_characteristic,
        include_channeling=include_channeling,
        include_coherent=include_coherent,
        sum_components=sum_components,
        band=band,
        max_points=max_points,
    )
    if df.empty:
        return None

    automatic_x_domain = x_domain is None
    x_domain = _narrow_line_domain((record for record, _ in cases), band, x_domain)
    x_scale = _scale(x_type, x_domain)
    y_scale = (
        _log_y_scale(_windowed_frame(df, None if automatic_x_domain else x_domain))
        if y_type == "log"
        else _linear_y_scale(df, None if automatic_x_domain else x_domain)
    )

    compact = _compact_component_frame(df)
    show_brem_layer = include_brem and not sum_components
    show_cxr_layer = include_cxr and not sum_components
    base = _fold_components(
        alt.Chart(compact), show_brem_layer, show_cxr_layer, include_coherent
    ).encode(
        x=alt.X("energy_eV:Q", title="Photon energy (eV)", scale=x_scale, axis=_spectrum_axis()),
        y=alt.Y(
            "intensity:Q",
            title=_INTENSITY_TITLE,
            scale=y_scale,
            axis=_spectrum_axis(),
        ),
        color=alt.Color("label:N", title="case", legend=_spectrum_legend()),
        tooltip=["label:N", "energy_eV:Q", "intensity:Q", "component:N"],
    )
    layers = _component_layers(base, show_brem_layer, show_cxr_layer, include_coherent)
    return _style_chart(
        alt.layer(*layers)
        .properties(width=width, height=height, title=_spectrum_title("Case comparison"))
        .interactive(bind_y=False)
    )


_COMPARISON_SELECTION_TITLES = {
    "quality_line": "highest quality-weighted integrated line flux",
    "line_flux": "highest integrated line flux",
    "quality_peak": "highest quality-weighted sampled peak density",
    "peak": "highest sampled peak density (spacing-dependent)",
    "line_brem_ratio": "highest local line-to-bremsstrahlung ratio",
}


def material_comparison_chart(
    points,
    dropped,
    select="quality_line",
    beam_energy_keV=None,
    min_cxr_quality: float | None = 0.5,
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
    quality_scope = "" if min_cxr_quality is None else f", line quality >= {min_cxr_quality:g}"
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
    return _style_chart(chart.interactive(bind_y=False))


# #622: renderer-neutral implementations take precedence over legacy local
# definitions. Public helpers remain available to existing Altair callers.
from .._spectrum_frames import (  # noqa: E402
    _compare_frame,
    _multi_case_frame,
    _narrow_line_domain,
    _tilt_records,
    _windowed_frame,
    spectrum_frame,
)
