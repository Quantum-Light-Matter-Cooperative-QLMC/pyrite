import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    # A dense spectrum (fine grid x several beam energies) can exceed Vega-Lite's
    # default 5000-row cap; vegafusion (shipped with marimo[recommended]) lifts it,
    # falling back to disabling the cap outright.
    try:
        alt.data_transformers.enable("vegafusion")
    except Exception:
        alt.data_transformers.disable_max_rows()

    from cxr_mc.config import PENETRATION_TILT_DEG, default_settings, trajectory_sweep
    from cxr_mc.plots import (
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_material_comparison,
        plot_timepix_efficiency,
        plot_trajectory_grid,
    )
    from cxr_mc.plots.altair_detectors import (
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        timepix_detected_chart,
    )
    from cxr_mc.plots.altair_spectra import compare_spectrum_chart, spectrum_chart
    from cxr_mc.plots.altair_sweeps import (
        heatmap_select_chart,
        metric_vs_chart,
        scan_charts,
    )
    from cxr_mc.plots.altair_trajectories import (
        penetration_survival_chart,
        trajectory_chart,
    )
    from cxr_mc.results import (
        filter_results,
        records,
        select_results,
        sweep_values,
        top_geometries,
    )
    from cxr_mc.run import cases_from_results, load_checkpoint
    from cxr_mc.sweep import MATERIAL_LABELS, build_cases

    return (
        MATERIAL_LABELS,
        PENETRATION_TILT_DEG,
        build_cases,
        cases_from_results,
        compare_spectrum_chart,
        default_settings,
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        filter_results,
        heatmap_select_chart,
        load_checkpoint,
        metric_vs_chart,
        mo,
        penetration_survival_chart,
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_material_comparison,
        plot_timepix_efficiency,
        plot_trajectory_grid,
        records,
        scan_charts,
        select_results,
        spectrum_chart,
        sweep_values,
        timepix_detected_chart,
        top_geometries,
        trajectory_chart,
        trajectory_sweep,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bulk-crystal Coherent X-ray Radiation — analysis & visualization
    Loads the checkpoint written by **`scan_app.py`** and draws every figure — no
    sweep runs here.
    """)
    return


@app.cell
def _(mo):
    from cxr_mc.analyze import get_default_material, initial_material

    initial_material_value = initial_material(mo.cli_args(), get_default_material())
    return (initial_material_value,)


@app.cell
def _(initial_material_value, mo):
    material_ui = mo.ui.dropdown(
        [
            "hopg",
            "hbn",
            "diamond",
            "silicon",
            "mose2",
            "wse2",
            "ptse2",
            "hfse2",
            "zrse2",
            "ws2",
            "mos2",
        ],
        value=initial_material_value,
        label="Material (match the scan you ran)",
    )
    material_ui
    return (material_ui,)


@app.cell
def _(cases_from_results, default_settings, filter_results, load_checkpoint, material_ui):
    MATERIAL = material_ui.value
    settings = default_settings()
    _results = load_checkpoint(MATERIAL)  # {name: {E0: record}}
    cases = cases_from_results(_results)  # rebuild the case list from the records
    res = filter_results(_results, cases)  # all loaded cases for this material
    return MATERIAL, cases, res, settings


@app.cell
def _(mo, records, res, sweep_values):
    # What's actually in this checkpoint -- swept knobs and their values (a pickle
    # accumulates every case ever run). Slice with results.select_results if needed.
    # Tucked in a collapsible accordion: secondary detail, open it when you need it.
    mo.accordion(
        {
            "Checkpoint contents (swept knobs & values)": (
                sweep_values(res) if records(res) else mo.md("*(load a checkpoint above)*")
            )
        }
    )
    return


@app.cell
def _(mo, records, res):
    # Tilt selector -- drives every per-tilt chart below (replaces the browse slider).
    _tilts = sorted({r["case"]["tilt_deg"] for r in records(res)})
    _opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    tilt_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="polar tilt")
    tilt_ui
    return (tilt_ui,)


@app.cell
def _(mo, records, res, sweep_values):
    # Thickness selector -- a crystal-thickness sweep is otherwise silently collapsed:
    # best_azimuth keeps, per beam energy, only the record with the strongest peak
    # line across ALL thicknesses, so every other thickness vanishes from the spectra,
    # detector, scan and ranking views without a trace. Pin ONE thickness here (like
    # the polar-tilt selector) so those views show it honestly; step through the
    # dropdown to compare thicknesses. Shown only when >1 thickness was swept; a
    # single-thickness checkpoint leaves the slice a no-op.
    _thk = sweep_values(res).get("thickness_ang", []) if records(res) else []
    _opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
    thickness_ui = mo.ui.dropdown(_opts, value=list(_opts)[-1], label="crystal thickness")
    thickness_ui if len(_thk) > 1 else mo.md("")
    return (thickness_ui,)


@app.cell
def _(res, select_results, thickness_ui):
    # The pinned-thickness view of the checkpoint. Every per-geometry chart below
    # draws from this instead of the raw `res`, so a thickness sweep no longer
    # collapses to one hidden record. None (no data / single option) -> pass through.
    res_view = (
        select_results(res, thickness_ang=thickness_ui.value)
        if thickness_ui.value is not None
        else res
    )
    return (res_view,)


@app.cell
def _(mo, records, res, sweep_values):
    # Beam energy shown in the click-to-select heatmap below. That heatmap can't
    # facet by energy -- a faceted Altair chart doesn't compose with a marimo point
    # selection -- so we pick ONE energy here and render a single panel instead.
    _e0 = sweep_values(res).get("E0_keV", []) if records(res) else []
    _opts = {f"{e:g} keV": e for e in _e0} or {"— no data —": None}
    heatmap_E0_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="heatmap beam energy")
    return (heatmap_E0_ui,)


@app.cell
def _(heatmap_E0_ui, heatmap_select_chart, mo, res_view, settings):
    # The interactive pixel-select heatmap: click a (azimuth, polar-tilt) cell to
    # drive the spectrum in the Intrinsic-spectra tab. Built here as a TOP-LEVEL
    # reactive element (a ui made inside a lazy tab builder wouldn't be in the DAG,
    # so its clicks couldn't trigger re-renders). chart_selection=False: the chart
    # carries its own point-selection param (see heatmap_select_chart), so let that
    # drive rather than marimo's default interval brush. None when the pinned view
    # has no heatmap to draw.
    _chart = heatmap_select_chart(
        res_view, settings, panel_value=heatmap_E0_ui.value, line_metric="prominence"
    )
    heatmap_select = (
        mo.ui.altair_chart(_chart, chart_selection=False, legend_selection=False)
        if _chart is not None
        else None
    )
    return (heatmap_select,)


@app.cell
def _(mo, records, res, sweep_values):
    # Selectors for the "Polar-angle comparison" tab -- overlays one spectrum
    # line per POLAR TILT, so the OTHER swept knobs (beam energy, azimuth,
    # thickness) must each be pinned to a single value here. Built as TOP-LEVEL
    # reactive elements for the same reason as heatmap_E0_ui above: a mo.ui
    # element created INSIDE a lazy tab builder isn't in marimo's reactive DAG,
    # so changing it wouldn't re-render the tab. A dim with only one swept value
    # still gets its dropdown here (so `.value` exists for the tab to read) --
    # the tab body shows a static label instead of the widget for that dim.
    _sv = sweep_values(res) if records(res) else {}
    _e0 = _sv.get("E0_keV", [])
    _azim = _sv.get("tilt_azim_deg", [])
    _thk = _sv.get("thickness_ang", [])
    _tilts = _sv.get("tilt_deg", [])

    _e0_opts = {f"{e:g} keV": e for e in _e0} or {"— no data —": None}
    polar_E0_ui = mo.ui.dropdown(_e0_opts, value=next(iter(_e0_opts)), label="beam energy")

    _azim_opts = {f"{a:g} deg": a for a in _azim} or {"— no data —": None}
    polar_azim_ui = mo.ui.dropdown(_azim_opts, value=next(iter(_azim_opts)), label="azimuth")

    _thk_opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
    polar_thk_ui = mo.ui.dropdown(_thk_opts, value=list(_thk_opts)[-1], label="thickness")

    # Default multiselect picks up to ~4 tilts spread across the available
    # range (first, last, and evenly-spaced in between) -- a useful comparison
    # out of the box without forcing the user to hand-pick values first.
    if len(_tilts) <= 4:
        _default_tilts = list(_tilts)
    else:
        _n = len(_tilts) - 1
        _idx = sorted({round(_n * k / 3) for k in range(4)})
        _default_tilts = [_tilts[i] for i in _idx]
    polar_tilts_ui = mo.ui.multiselect(
        {f"{t:g} deg": t for t in _tilts},
        value=[f"{t:g} deg" for t in _default_tilts],
        label="polar tilts to compare",
    )
    return polar_E0_ui, polar_azim_ui, polar_thk_ui, polar_tilts_ui


@app.cell
def _(mo, records, res, sweep_values):
    # Selectors for the "Azimuthal comparison" tab -- overlays one spectrum
    # line per AZIMUTH, pinning beam energy / polar tilt / thickness. Same
    # top-level-DAG requirement as the polar-tab selectors above.
    _sv = sweep_values(res) if records(res) else {}
    _e0 = _sv.get("E0_keV", [])
    _tilts = _sv.get("tilt_deg", [])
    _thk = _sv.get("thickness_ang", [])
    _azim = _sv.get("tilt_azim_deg", [])

    _e0_opts = {f"{e:g} keV": e for e in _e0} or {"— no data —": None}
    azim_E0_ui = mo.ui.dropdown(_e0_opts, value=next(iter(_e0_opts)), label="beam energy")

    _tilt_opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    azim_tilt_ui = mo.ui.dropdown(_tilt_opts, value=next(iter(_tilt_opts)), label="polar tilt")

    _thk_opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
    azim_thk_ui = mo.ui.dropdown(_thk_opts, value=list(_thk_opts)[-1], label="thickness")

    if len(_azim) <= 4:
        _default_azims = list(_azim)
    else:
        _n = len(_azim) - 1
        _idx = sorted({round(_n * k / 3) for k in range(4)})
        _default_azims = [_azim[i] for i in _idx]
    azim_azims_ui = mo.ui.multiselect(
        {f"{a:g} deg": a for a in _azim},
        value=[f"{a:g} deg" for a in _default_azims],
        label="azimuths to compare",
    )
    return azim_E0_ui, azim_azims_ui, azim_thk_ui, azim_tilt_ui


@app.cell
def _(mo):
    # Spectral-plot controls: CXR-only (no brem background) toggle, adjustable
    # x-axis (photon energy) limits, and a lin/log y switch -- shared by the
    # intrinsic-spectra chart and the Timepix3 detected-vs-incident chart (the
    # two line-per-energy spectral views). Leave x-limits at 0 for autoscale.
    brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    xmin_ui = mo.ui.number(value=0.0, label="x-min (eV, 0 = auto)")
    xmax_ui = mo.ui.number(value=0.0, label="x-max (eV, 0 = auto)")
    ylog_ui = mo.ui.switch(value=False, label="log y")
    mo.hstack([brem_ui, xmin_ui, xmax_ui, ylog_ui])
    return brem_ui, xmax_ui, xmin_ui, ylog_ui


@app.cell
def _(xmax_ui, xmin_ui):
    # (0, 0) -> None (autoscale); otherwise an explicit (lo, hi) domain.
    x_domain = (xmin_ui.value, xmax_ui.value) if (xmin_ui.value or xmax_ui.value) else None
    return (x_domain,)


@app.cell
def _(PENETRATION_TILT_DEG, mo):
    _angle_opts = {f"{t:g} deg": t for t in PENETRATION_TILT_DEG}
    penetration_angle_ui = mo.ui.dropdown(
        _angle_opts, value="-15 deg", label="polar tilt (penetration)"
    )
    return (penetration_angle_ui,)


@app.cell
def _(
    MATERIAL,
    MATERIAL_LABELS,
    azim_E0_ui,
    azim_azims_ui,
    azim_thk_ui,
    azim_tilt_ui,
    brem_ui,
    build_cases,
    cases,
    compare_spectrum_chart,
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    heatmap_E0_ui,
    heatmap_select,
    load_checkpoint,
    metric_vs_chart,
    mo,
    penetration_angle_ui,
    penetration_survival_chart,
    plot_best_spectra,
    plot_eaglexo_charge_map,
    plot_eaglexo_efficiency,
    plot_material_comparison,
    plot_timepix_efficiency,
    plot_trajectory_grid,
    polar_E0_ui,
    polar_azim_ui,
    polar_thk_ui,
    polar_tilts_ui,
    records,
    res,
    res_view,
    scan_charts,
    select_results,
    settings,
    spectrum_chart,
    sweep_values,
    tilt_ui,
    timepix_detected_chart,
    top_geometries,
    trajectory_chart,
    trajectory_sweep,
    x_domain,
    ylog_ui,
):
    # All figures live in one lazy, tabbed layout. Each tab value is a zero-arg
    # builder closure (not a pre-built object), so `lazy=True` defers the *compute*
    # — only the open tab runs. The expensive matplotlib panels are nested in
    # `lazy` accordions; the builders return a Figure, which marimo renders on
    # expand. Closures capture `tilt_ui.value`, so switching tilt rebuilds the
    # tabs and the next view recomputes with the new tilt.
    _tilt = tilt_ui.value

    def _rankings_tab():
        # Rank across ALL thicknesses (raw `res`, not the pinned `res_view`): the
        # thickness column disambiguates rows, so a thickness sweep shows every
        # thickness competing head-to-head rather than being collapsed to the pin.
        df = top_geometries(res, settings, top_n=20, select="quality_peak")
        if df.empty:
            return mo.md(f"**No checkpoint for `{MATERIAL}`** — run `scan_app.py` first.")
        return mo.vstack(
            [
                mo.md(
                    f"Top 20 geometries ranked by *quality × peak flux* "
                    f"({settings.beam_current_na:g} nA beam; quality score in [0, 1])."
                ),
                mo.ui.table(
                    df,
                    pagination=False,
                    selection=None,
                    show_column_summaries=False,
                    show_data_types=False,
                    style_cell=lambda col, row_id, val: {"white-space": "nowrap"},
                ),
            ]
        )

    def _spectra_tab():
        _md = mo.md(
            "The INTRINSIC coherent CXR line spectrum at the selected polar tilt, "
            "one line per beam energy; toggle the brem background off above for "
            "the CXR-only view. Pan/zoom in the chart, or set explicit x-limits "
            "and switch to log y with the controls above."
        )
        _chart = spectrum_chart(
            res_view,
            settings,
            tilt_deg=_tilt,
            include_brem=brem_ui.value,
            x_domain=x_domain,
            y_type="log" if ylog_ui.value else "linear",
        )

        # Pixel-select: click a heatmap cell -> spectrum for THAT (azimuth, tilt)
        # geometry at the heatmap's beam energy, on the pinned-thickness view.
        # `x`/`y` are the encoded (tilt_azim_deg, tilt_deg) of the clicked cell.
        def _pixel_spectrum():
            if heatmap_select is None:
                return mo.md("*No heatmap for this view — needs an azimuth × polar-tilt sweep.*")
            _sel = heatmap_select.value
            if _sel is None or len(_sel) == 0:
                return mo.md("*Click a heatmap cell above to plot its spectrum here.*")
            _row = _sel.iloc[0]
            _sub = select_results(
                res_view,
                tilt_azim_deg=float(_row["x"]),
                tilt_deg=float(_row["y"]),
                E0_keV=heatmap_E0_ui.value,
            )
            _sp = spectrum_chart(
                _sub,
                settings,
                tilt_deg=float(_row["y"]),
                include_brem=brem_ui.value,
                x_domain=x_domain,
                y_type="log" if ylog_ui.value else "linear",
            )
            return _sp if _sp is not None else mo.md("*No spectrum for that cell.*")

        _hm = (
            heatmap_select
            if heatmap_select is not None
            else mo.md("*No 2-D heatmap for this checkpoint / thickness.*")
        )
        return mo.vstack(
            [
                _md,
                _chart if _chart is not None else mo.md("*No spectra — run the scan first.*"),
                mo.md("---"),
                mo.md(
                    "**Pixel-select** — pick a beam energy, then click a heatmap cell to "
                    "plot that geometry's spectrum below."
                ),
                heatmap_E0_ui,
                _hm,
                _pixel_spectrum(),
            ]
        )

    def _polar_compare_tab():
        # Overlays one spectrum line per POLAR TILT (the polar_tilts_ui
        # multiselect), at a PINNED beam energy / azimuth / thickness -- the
        # complement of _spectra_tab, which fixes tilt and varies energy. Uses
        # the RAW `res` (not the pinned `res_view`): thickness is pinned by
        # THIS tab's own polar_thk_ui dropdown instead, so starting from
        # res_view would apply the (unrelated) top-of-notebook thickness pin
        # underneath it.
        _md = mo.md(
            "The INTRINSIC coherent CXR line spectrum, one line per POLAR TILT, at "
            "a pinned beam energy / azimuth / thickness. Pick a few tilts below to "
            "compare; the brem/x-limit/log-y controls above still apply."
        )
        _sv = sweep_values(res) if records(res) else {}
        _e0_widget = (
            polar_E0_ui
            if len(_sv.get("E0_keV", [])) > 1
            else mo.md(
                f"beam energy: {polar_E0_ui.value:g} keV" if polar_E0_ui.value is not None else ""
            )
        )
        _azim_widget = (
            polar_azim_ui
            if len(_sv.get("tilt_azim_deg", [])) > 1
            else mo.md(
                f"azimuth: {polar_azim_ui.value:g} deg" if polar_azim_ui.value is not None else ""
            )
        )
        _thk_widget = (
            polar_thk_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"thickness: {polar_thk_ui.value:g} Å" if polar_thk_ui.value is not None else ""
            )
        )
        _note = (
            mo.md("*Only one polar tilt in this checkpoint — nothing to compare.*")
            if len(_sv.get("tilt_deg", [])) <= 1
            else None
        )
        if not polar_tilts_ui.value:
            return mo.vstack(
                [
                    _md,
                    _e0_widget,
                    _azim_widget,
                    _thk_widget,
                    polar_tilts_ui,
                    mo.md("*Select at least one polar tilt above to plot.*"),
                ]
            )
        _constraints = {"tilt_deg": list(polar_tilts_ui.value)}
        if polar_E0_ui.value is not None:
            _constraints["E0_keV"] = polar_E0_ui.value
        if polar_azim_ui.value is not None:
            _constraints["tilt_azim_deg"] = polar_azim_ui.value
        if polar_thk_ui.value is not None:
            _constraints["thickness_ang"] = polar_thk_ui.value
        _sub = select_results(res, **_constraints)
        _chart = compare_spectrum_chart(
            _sub,
            settings,
            hue="tilt_deg",
            include_brem=brem_ui.value,
            x_domain=x_domain,
            y_type="log" if ylog_ui.value else "linear",
        )
        _parts = [_md, _e0_widget, _azim_widget, _thk_widget, polar_tilts_ui]
        if _note is not None:
            _parts.append(_note)
        _parts.append(_chart if _chart is not None else mo.md("*No spectra for this slice.*"))
        return mo.vstack(_parts)

    def _azim_compare_tab():
        # Overlays one spectrum line per AZIMUTH (the azim_azims_ui
        # multiselect), at a pinned beam energy / polar tilt / thickness --
        # mirrors _polar_compare_tab with the swept dimension swapped. Also
        # uses the raw `res`; thickness is pinned by azim_thk_ui here.
        _md = mo.md(
            "The INTRINSIC coherent CXR line spectrum, one line per AZIMUTH, at "
            "a pinned beam energy / polar tilt / thickness. Pick a few azimuths "
            "below to compare; the brem/x-limit/log-y controls above still apply."
        )
        _sv = sweep_values(res) if records(res) else {}
        _e0_widget = (
            azim_E0_ui
            if len(_sv.get("E0_keV", [])) > 1
            else mo.md(
                f"beam energy: {azim_E0_ui.value:g} keV" if azim_E0_ui.value is not None else ""
            )
        )
        _tilt_widget = (
            azim_tilt_ui
            if len(_sv.get("tilt_deg", [])) > 1
            else mo.md(
                f"polar tilt: {azim_tilt_ui.value:g} deg" if azim_tilt_ui.value is not None else ""
            )
        )
        _thk_widget = (
            azim_thk_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"thickness: {azim_thk_ui.value:g} Å" if azim_thk_ui.value is not None else ""
            )
        )
        _note = (
            mo.md("*Only one azimuth in this checkpoint — nothing to compare.*")
            if len(_sv.get("tilt_azim_deg", [])) <= 1
            else None
        )
        if not azim_azims_ui.value:
            return mo.vstack(
                [
                    _md,
                    _e0_widget,
                    _tilt_widget,
                    _thk_widget,
                    azim_azims_ui,
                    mo.md("*Select at least one azimuth above to plot.*"),
                ]
            )
        _constraints = {"tilt_azim_deg": list(azim_azims_ui.value)}
        if azim_E0_ui.value is not None:
            _constraints["E0_keV"] = azim_E0_ui.value
        if azim_tilt_ui.value is not None:
            _constraints["tilt_deg"] = azim_tilt_ui.value
        if azim_thk_ui.value is not None:
            _constraints["thickness_ang"] = azim_thk_ui.value
        _sub = select_results(res, **_constraints)
        _chart = compare_spectrum_chart(
            _sub,
            settings,
            hue="tilt_azim_deg",
            include_brem=brem_ui.value,
            x_domain=x_domain,
            y_type="log" if ylog_ui.value else "linear",
        )
        _parts = [_md, _e0_widget, _tilt_widget, _thk_widget, azim_azims_ui]
        if _note is not None:
            _parts.append(_note)
        _parts.append(_chart if _chart is not None else mo.md("*No spectra for this slice.*"))
        return mo.vstack(_parts)

    def _scans_tab():
        # Each plot is a separate mo.lazy(fn) so the tab returns instantly with
        # placeholders; scans, metric lines, and the best-spectra accordion each
        # compute and render independently as they become visible.
        def _scan_charts_item():
            charts = scan_charts(res_view, settings, cases=cases, line_metric="prominence")
            return mo.vstack(charts) if charts else mo.md("*No scan results.*")

        def _metric_lines_item():
            line = metric_vs_chart(
                res_view, settings, x="tilt_deg", metric="line_flux", hue="E0_keV"
            )
            peak = metric_vs_chart(
                res_view, settings, x="tilt_deg", metric="peak_flux", hue="E0_keV"
            )
            parts = [c for c in (line, peak) if c is not None]
            return mo.vstack(parts) if parts else mo.md("*No metric results.*")

        return mo.vstack(
            [
                mo.md(
                    "`scan_charts` auto-picks a heatmap or line scans (one per swept quantity); "
                    "then 1-D metric scans vs polar tilt. Each section loads as it becomes "
                    "visible. The best-spectra panel loads on expand."
                ),
                mo.lazy(_scan_charts_item, show_loading_indicator=True),
                mo.lazy(_metric_lines_item, show_loading_indicator=True),
                mo.accordion(
                    {
                        "Best spectra (matplotlib)": lambda: plot_best_spectra(
                            res_view, settings, top_n=12, select="quality_peak"
                        )
                    },
                    lazy=True,
                ),
            ]
        )

    def _detectors_tab():
        # Eagle XO and Timepix3 render into a nested mo.accordion (lazy=True)
        # rather than a dropdown or nested mo.ui.tabs: a mo.ui.dropdown created
        # inside a lazy builder isn't in marimo's reactive DAG, so it can't drive
        # re-renders, and nested mo.ui.tabs-in-tabs is a known marimo rendering
        # bug (charts inside the inner tab silently render blank -- this is what
        # produced the "blank detector-tab plots" bug: marimo-team/marimo#6919).
        # An accordion lazily defers each section's compute exactly like the
        # inner tabs did, without the nesting problem.
        def _eaglexo_inner():
            _md = mo.md(
                "Raptor Eagle XO direct-detection CCD (`solid_angle x QE(E)`): soft PXR "
                "lines pass at ~90% QE, hard brem is crushed by the thin sensor. "
                "Photon density (detected vs incident), then recorded-charge density."
            )
            _detected = eaglexo_detected_chart(
                res_view, settings, tilt_deg=_tilt, x_domain=x_domain
            )
            _charge = eaglexo_charge_chart(res_view, settings, tilt_deg=_tilt, x_domain=x_domain)
            _parts = [_md, *(c for c in (_detected, _charge) if c is not None)]
            _parts.append(
                mo.accordion(
                    {
                        "Efficiency: QE + solid angle + resolution (matplotlib)": lambda: (
                            plot_eaglexo_efficiency(sensor="4240")
                        ),
                        "Charge geometry map (matplotlib)": lambda: (
                            plot_eaglexo_charge_map(res_view, settings, cases=cases)
                            if records(res_view)
                            else mo.md("*No results.*")
                        ),
                    },
                    lazy=True,
                )
            )
            return mo.vstack(_parts)

        def _timepix_inner():
            _md = mo.md(
                "Si quad forward model: photoabsorption → charge sharing → per-pixel "
                "threshold counting. Detected vs incident at the selected tilt."
            )
            _detected = timepix_detected_chart(
                res_view, settings, tilt_deg=_tilt, x_domain=x_domain
            )
            _parts = [_md, *([_detected] if _detected is not None else [])]
            _parts.append(
                mo.accordion(
                    {
                        "Efficiency curve (matplotlib)": lambda: plot_timepix_efficiency(
                            thickness_um=300.0, bias_v=100.0
                        )
                    },
                    lazy=True,
                )
            )
            return mo.vstack(_parts)

        return mo.accordion({"Eagle XO": _eaglexo_inner, "Timepix3": _timepix_inner}, lazy=True)

    def _penetration_tab():
        _angle = penetration_angle_ui.value
        _md = mo.md(
            "Surviving-electron fraction vs depth (one curve per beam energy) and "
            "an interactive low-Ne track cross-section, at the polar tilt selected "
            "in this tab (default -15 deg, a low nonzero angle -- avoids both the fully-"
            "normal and grazing-incidence edge cases). These run the cheap CPU-only "
            "transport directly — no checkpoint needed. For a stacked/multilayer "
            "material (e.g. mos2 on sapphire) the cascade is transported through "
            "the FULL stack, not just the top film. Dense grid loads on expand."
        )
        _sweep = trajectory_sweep(MATERIAL, energies=(30, 60))
        _traj = build_cases(_sweep, settings.n_electrons, settings.n_electrons_brem)
        if not _traj:
            return mo.vstack([_md, mo.md("*No trajectory cases.*")])
        _survival = penetration_survival_chart(_traj, Ne=500, tilt=_angle)
        # Pick the lowest energy at the selected tilt for the single-track view.
        _nc = min(_traj, key=lambda c: (abs(c["tilt_deg"] - _angle), c["E0_keV"]))
        _track = trajectory_chart(_nc, Ne=40)
        _parts = [_md, penetration_angle_ui, *(p for p in (_survival, _track) if p is not None)]
        _parts.append(
            mo.accordion(
                {
                    "Dense penetration grid (datashader, matplotlib)": lambda: plot_trajectory_grid(
                        _traj, energy=30, Ne=120
                    )
                },
                lazy=True,
            )
        )
        return mo.vstack(_parts)

    def _cross_material_tab():
        _md = mo.md(
            "For every material whose checkpoint exists, the single best geometry's "
            "dominant line: energy vs flux, coloured by line quality."
        )
        _by_material = {}
        for _m in MATERIAL_LABELS:
            _r = load_checkpoint(_m)
            if _r:
                _by_material[MATERIAL_LABELS[_m]] = _r
        if len(_by_material) >= 2:
            return mo.vstack(
                [_md, plot_material_comparison(_by_material, settings, select="quality_peak")]
            )
        return mo.vstack(
            [_md, mo.md("*Run `scan_app.py` for more materials to populate this comparison.*")]
        )

    mo.ui.tabs(
        {
            "Energy comparison": _spectra_tab,
            "Polar-angle comparison": _polar_compare_tab,
            "Azimuthal comparison": _azim_compare_tab,
            "Top geometries": _rankings_tab,
            "Geometry & scans": _scans_tab,
            "Detectors": _detectors_tab,
            "Penetration": _penetration_tab,
            "Cross-material": _cross_material_tab,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
