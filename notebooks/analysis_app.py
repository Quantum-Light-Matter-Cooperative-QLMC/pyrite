# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    from _design import page_title, style_sheet
    from _widgets import MaterialSelect

    # A dense spectrum (fine grid x several beam energies) can exceed Vega-Lite's
    # default 5000-row cap; vegafusion (shipped with marimo[recommended]) lifts it,
    # falling back to disabling the cap outright.
    try:
        alt.data_transformers.enable("vegafusion")
    except Exception:
        alt.data_transformers.disable_max_rows()

    from cxr_mc.analyze import load_analysis_checkpoint
    from cxr_mc.config import default_settings
    from cxr_mc.materials import CATALOG
    from cxr_mc.plots import (
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_timepix_efficiency,
    )
    from cxr_mc.plots.altair_detectors import (
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        timepix_detected_chart,
    )
    from cxr_mc.plots.altair_spectra import (
        compare_spectrum_chart,
        multi_case_spectrum_chart,
        spectrum_chart,
    )
    from cxr_mc.plots.altair_sweeps import (
        heatmap_select_chart,
        metric_vs_chart,
        scan_charts,
    )
    from cxr_mc.plots.sweeps import _HEATMAP_QUANTITIES as HEATMAP_QUANTITIES
    from cxr_mc.results import (
        case_label,
        case_table_rows,
        filter_results,
        records,
        select_results,
        select_thickness,
        slim_case_record,
        sweep_values,
        thicknesses_by_energy,
        top_geometries,
    )
    from cxr_mc.run import cases_from_results
    from cxr_mc.sweep import fmt_thickness

    return (
        CATALOG,
        HEATMAP_QUANTITIES,
        MaterialSelect,
        case_label,
        case_table_rows,
        cases_from_results,
        compare_spectrum_chart,
        default_settings,
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        filter_results,
        fmt_thickness,
        heatmap_select_chart,
        load_analysis_checkpoint,
        metric_vs_chart,
        mo,
        multi_case_spectrum_chart,
        page_title,
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_timepix_efficiency,
        records,
        scan_charts,
        select_results,
        select_thickness,
        slim_case_record,
        spectrum_chart,
        style_sheet,
        sweep_values,
        thicknesses_by_energy,
        timepix_detected_chart,
        top_geometries,
    )


@app.cell(hide_code=True)
def _(mo, page_title, style_sheet):
    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Coherent X-ray radiation analysis",
                "Explore spectra, optimize geometry, inspect instrument response, and compare materials.",
                eyebrow="Electron transport and radiation from crystalline materials",
            ),
        ]
    )
    return


@app.cell
def _(MaterialSelect, mo):
    from cxr_mc.analyze import (
        get_default_material,
        initial_material,
        material_menu,
        select_initial_material,
    )
    from cxr_mc.run import _DEFAULT_CHECKPOINT_DIR

    requested_material = initial_material(mo.cli_args(), get_default_material())
    material_options = material_menu(_DEFAULT_CHECKPOINT_DIR)
    initial_selection = select_initial_material(requested_material, material_options)
    material_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(material_options),
            value=initial_selection,
            label="Material",
            disabled=initial_selection is None,
        )
    )
    material_ui
    return (material_ui,)


@app.cell
def _(MaterialSelect, material_ui, mo):
    from cxr_mc.analyze import face_menu as _face_menu
    from cxr_mc.analyze import select_initial_material as _select_initial_face
    from cxr_mc.run import _DEFAULT_CHECKPOINT_DIR as _FACE_CHECKPOINT_DIR

    # Rebuilds whenever the material changes, so the flat/blazed disabled flags
    # track that material's available checkpoints. Default face = first
    # non-disabled row (flat preferred; blazed only when it's the sole one).
    _face_material = material_ui.value["value"]
    face_options = _face_menu(_face_material, _FACE_CHECKPOINT_DIR) if _face_material else ()
    initial_face = _select_initial_face(None, face_options)
    face_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(face_options),
            value=initial_face,
            label="Face",
            disabled=initial_face is None,
        )
    )
    face_ui
    return (face_ui,)


@app.cell
def _(MaterialSelect, material_ui, mo):
    from cxr_mc.analyze import profile_menu as _profile_menu
    from cxr_mc.analyze import select_initial_material as _select_initial_profile
    from cxr_mc.run import _DEFAULT_CHECKPOINT_DIR as _PROFILE_CHECKPOINT_DIR

    # Rebuilds whenever the material changes. "Standard" is the canonical
    # <material> checkpoint; other rows are named catalog_profile / --quick
    # checkpoints for the same material, resolved by reading each stem's
    # meta.json sidecar (never re-derived by re-hashing -- see profile_menu's
    # docstring). Default = Standard when it exists, else the newest variant.
    _profile_material = material_ui.value["value"]
    profile_options = (
        _profile_menu(_profile_material, _PROFILE_CHECKPOINT_DIR) if _profile_material else ()
    )
    initial_profile = _select_initial_profile(None, profile_options)
    profile_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(profile_options),
            value=initial_profile,
            label="Profile",
            disabled=initial_profile is None,
        )
    )
    profile_ui
    return (profile_ui,)


@app.cell
def _(mo, res_all):
    # Emission view selector, composed BESIDE the material/face selectors. Its
    # options are gated on the loaded checkpoint's STORED spectra: `Coherent`
    # appears only when a record carries a `spec_coherent` (a `coherent`/`both`
    # run), so an incoherent checkpoint offers just `Incoherent`. Every spectrum
    # read below routes through `apply_emission` keyed on this value, so a `both`
    # checkpoint toggles incoherent-vs-coherent everywhere without per-plot code.
    from cxr_mc.analyze import emission_menu as _emission_menu

    _rows = _emission_menu(res_all)
    _opts = {row["label"]: row["value"] for row in _rows if not row["disabled"]} or {
        "Incoherent": "incoherent"
    }
    emission_ui = mo.ui.radio(
        options=_opts,
        value=next(iter(_opts)),
        label="Emission",
        inline=True,
    )
    emission_ui
    return (emission_ui,)


@app.cell
def _(emission_ui, res_all):
    # The emission-picked view of the loaded checkpoint: for `Coherent` each
    # record's `spec` is pointed at its `spec_coherent` (shallow copy, arrays
    # shared, checkpoint untouched); `Incoherent` passes `res_all` through. Every
    # downstream `res`/`res_view`/scan/detector reader inherits the selection.
    from cxr_mc.analyze import apply_emission as _apply_emission

    _emission = emission_ui.value if emission_ui.value is not None else "incoherent"
    res = _apply_emission(res_all, _emission)
    return (res,)


@app.cell
def _(
    cases_from_results,
    default_settings,
    face_ui,
    filter_results,
    load_analysis_checkpoint,
    material_ui,
    profile_ui,
):
    from cxr_mc.analyze import checkpoint_stem

    MATERIAL = material_ui.value["value"]
    FACE = face_ui.value["value"]
    PROFILE = profile_ui.value["value"] if profile_ui.value is not None else None
    settings = default_settings()
    # A non-Standard profile selection names its own checkpoint stem directly
    # (it already commits to a specific grid/beam/profile combination, so face
    # doesn't apply); Standard falls back to the flat/blazed face stem as
    # before. MATERIAL stays the catalog key everywhere else (labels,
    # CATALOG.material(MATERIAL), manifest lookups).
    if PROFILE is not None and PROFILE != MATERIAL:
        _stem = PROFILE
    elif MATERIAL is not None and FACE is not None:
        _stem = checkpoint_stem(MATERIAL, FACE)
    else:
        _stem = None
    _loaded = load_analysis_checkpoint(_stem) if _stem is not None else None
    if _loaded is None:
        MATERIAL = None
        FACE = None
    _results = _loaded or {}
    cases = cases_from_results(_results)  # rebuild the case list from the records
    res_all = filter_results(_results, cases)  # all loaded cases for this material

    def face_title(chart):
        """Append a ` (blazed)` suffix to a chart's title on the blazed face, so a
        blazed spectrum is never mistaken for the flat one. No-op for flat / None."""
        if chart is None or FACE != "blazed" or not hasattr(chart, "title"):
            return chart
        return chart.properties(title=f"{chart.title} (blazed)")

    return FACE, MATERIAL, cases, face_title, res_all, settings


@app.cell
def _(mo, records, res):
    # Energy-comparison tilt selector. Created top-level for reactivity, rendered
    # only inside the Energy comparison tab.
    _tilts = sorted({r["case"]["tilt_deg"] for r in records(res)})
    _opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    tilt_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="polar tilt")
    return (tilt_ui,)


@app.cell
def _(fmt_thickness, mo, records, res, sweep_values, thicknesses_by_energy):
    # Energy-comparison thickness selector. Created top-level for reactivity,
    # rendered only inside the Energy comparison tab.
    #
    # A crystal-thickness sweep is otherwise silently collapsed:
    # best_azimuth keeps, per beam energy, only the record with the strongest peak
    # line across ALL thicknesses, so every other thickness vanishes from the spectra,
    # detector, scan and ranking views without a trace. Pin ONE thickness here (like
    # the polar-tilt selector) so those views show it honestly; step through the
    # dropdown to compare thicknesses. Shown only when >1 thickness was swept; a
    # single-thickness checkpoint leaves the slice a no-op.
    _thk = sweep_values(res).get("thickness_ang", []) if records(res) else []
    _opts = {fmt_thickness(t): t for t in _thk} or {"— no data —": None}
    # Default to the THICKEST slab at which the LOWEST beam energy was still
    # computed. The penetration watchdog stops computing an energy past the depth
    # where its beam is dead, so the bulk-thickness end of the sweep silently
    # loses 30/50 keV. Anchoring the default here keeps every beam energy visible
    # out of the box; stepping to a thicker slab hits select_thickness's
    # per-energy fallback (with a note) rather than dropping the low energies.
    _inv = thicknesses_by_energy(res) if records(res) else {}
    _default_key = fmt_thickness(_inv[min(_inv)][-1]) if _inv else list(_opts)[-1]
    thickness_ui = mo.ui.dropdown(_opts, value=_default_key, label="crystal thickness")
    return (thickness_ui,)


@app.cell
def _(res, select_thickness, thickness_ui):
    # The pinned-thickness view of the checkpoint. Every per-geometry chart below
    # draws from this instead of the raw `res`, so a thickness sweep no longer
    # collapses to one hidden record. None (no data / single option) -> pass through.
    #
    # select_thickness (not select_results) so a bulk thickness never DROPS a beam
    # energy the penetration watchdog stopped computing early: those energies fall
    # back to their thickest computed slab (spectrum saturated past that depth),
    # carrying a case["thickness_fallback"] marker the spectra tab surfaces.
    res_view = select_thickness(res, thickness_ui.value) if thickness_ui.value is not None else res
    return (res_view,)


@app.cell
def _(fmt_thickness, mo, records, res, sweep_values):
    # Geometry & scans thickness selector.
    _thk = sweep_values(res).get("thickness_ang", []) if records(res) else []
    _opts = {fmt_thickness(t): t for t in _thk} or {"— no data —": None}
    scan_thickness_ui = mo.ui.dropdown(_opts, value=list(_opts)[-1], label="crystal thickness")
    return (scan_thickness_ui,)


@app.cell
def _(res, scan_thickness_ui, select_results):
    scan_res_view = (
        select_results(res, thickness_ang=scan_thickness_ui.value)
        if scan_thickness_ui.value is not None
        else res
    )
    return (scan_res_view,)


@app.cell
def _(fmt_thickness, mo, records, res, sweep_values):
    # Detector tab selectors.
    _sv = sweep_values(res) if records(res) else {}
    _tilts = _sv.get("tilt_deg", [])
    _azim = _sv.get("tilt_azim_deg", [])
    _thk = _sv.get("thickness_ang", [])
    _tilt_opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    detector_tilt_ui = mo.ui.dropdown(_tilt_opts, value=next(iter(_tilt_opts)), label="polar tilt")

    _azim_opts = {f"{a:g} °": a for a in _azim} or {"— no data —": None}
    detector_azim_ui = mo.ui.dropdown(_azim_opts, value=next(iter(_azim_opts)), label="azimuth")

    _thk_opts = {fmt_thickness(t): t for t in _thk} or {"— no data —": None}
    detector_thickness_ui = mo.ui.dropdown(
        _thk_opts, value=list(_thk_opts)[-1], label="crystal thickness"
    )

    detector_auto_ui = mo.ui.switch(value=True, label="Auto narrow domains")
    detector_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV)")
    detector_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV)")
    detector_ymin_ui = mo.ui.number(value=0.0, label="narrow y-min")
    detector_ymax_ui = mo.ui.number(value=0.0, label="narrow y-max")
    detector_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    detector_ylog_ui = mo.ui.switch(value=True, label="narrow log y")

    detector_broad_auto_ui = mo.ui.switch(value=True, label="Auto broad domains")
    detector_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV)")
    detector_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV)")
    detector_broad_ymin_ui = mo.ui.number(value=0.0, label="broad y-min")
    detector_broad_ymax_ui = mo.ui.number(value=0.0, label="broad y-max")
    detector_broad_xlog_ui = mo.ui.switch(value=True, label="broad log x")
    detector_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        detector_auto_ui,
        detector_azim_ui,
        detector_broad_auto_ui,
        detector_broad_xlog_ui,
        detector_broad_xmax_ui,
        detector_broad_xmin_ui,
        detector_broad_ylog_ui,
        detector_broad_ymax_ui,
        detector_broad_ymin_ui,
        detector_thickness_ui,
        detector_tilt_ui,
        detector_xlog_ui,
        detector_xmax_ui,
        detector_xmin_ui,
        detector_ylog_ui,
        detector_ymax_ui,
        detector_ymin_ui,
    )


@app.cell
def _(detector_azim_ui, detector_thickness_ui, res, select_results):
    _constraints = {}
    if detector_thickness_ui.value is not None:
        _constraints["thickness_ang"] = detector_thickness_ui.value
    if detector_azim_ui.value is not None:
        _constraints["tilt_azim_deg"] = detector_azim_ui.value
    detector_res_view = select_results(res, **_constraints) if _constraints else res
    return (detector_res_view,)


@app.cell
def _(
    detector_auto_ui,
    detector_broad_auto_ui,
    detector_broad_xmax_ui,
    detector_broad_xmin_ui,
    detector_broad_ymax_ui,
    detector_broad_ymin_ui,
    detector_xmax_ui,
    detector_xmin_ui,
    detector_ymax_ui,
    detector_ymin_ui,
):
    def _domain(auto, xmin, xmax):
        if auto:
            return None
        return (xmin, xmax)

    detector_x_domain = _domain(
        detector_auto_ui.value, detector_xmin_ui.value, detector_xmax_ui.value
    )
    detector_y_domain = _domain(
        detector_auto_ui.value, detector_ymin_ui.value, detector_ymax_ui.value
    )
    detector_broad_x_domain = _domain(
        detector_broad_auto_ui.value,
        detector_broad_xmin_ui.value,
        detector_broad_xmax_ui.value,
    )
    detector_broad_y_domain = _domain(
        detector_broad_auto_ui.value,
        detector_broad_ymin_ui.value,
        detector_broad_ymax_ui.value,
    )
    return (
        detector_broad_x_domain,
        detector_broad_y_domain,
        detector_x_domain,
        detector_y_domain,
    )


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
    # reactive element (a ui made inside a tab body is easier to lose from the DAG,
    # so its clicks may not trigger re-renders). chart_selection=False: the chart
    # carries its own point-selection param (see heatmap_select_chart), so let that
    # drive rather than marimo's default interval brush. None when the pinned view
    # has no heatmap to draw.
    #
    # mo.ui.altair_chart's click round-trip depends on the serialized spec's
    # top-level Vega-Lite `params` array to know which named selection to
    # listen for and report back as `.value`. The notebook enables vegafusion
    # globally (to lift Vega-Lite's 5000-row cap for the dense spectra charts),
    # but vegafusion serializes an already-COMPILED Vega spec (`signals`, no
    # `params`) -- the click still highlights visually (that's baked into the
    # compiled signal graph) but the selection can never reach the kernel. This
    # heatmap grid is tiny, so build it under the default transformer instead.
    import altair as _alt

    _chart = heatmap_select_chart(
        res_view, settings, panel_value=heatmap_E0_ui.value, line_metric="prominence"
    )
    if _chart is None:
        heatmap_select = None
    else:
        with _alt.data_transformers.enable("default"):
            heatmap_select = mo.ui.altair_chart(
                _chart, chart_selection=False, legend_selection=False
            )
    return (heatmap_select,)


@app.cell
def _(mo, records, scan_res_view, sweep_values):
    # Beam energy for the click-to-select heatmap in the Optimize tab, same
    # reasoning as heatmap_E0_ui: a faceted Altair chart can't carry a marimo
    # point selection, so pick ONE energy and render a single panel. Sourced from
    # scan_res_view (not res_view) so it matches the thickness pinned by
    # scan_thickness_ui rather than the Explore tab's thickness_ui.
    _e0 = sweep_values(scan_res_view).get("E0_keV", []) if records(scan_res_view) else []
    _opts = {f"{e:g} keV": e for e in _e0} or {"— no data —": None}
    scan_heatmap_E0_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="heatmap beam energy")
    return (scan_heatmap_E0_ui,)


@app.cell
def _(
    HEATMAP_QUANTITIES,
    cases,
    heatmap_select_chart,
    mo,
    scan_heatmap_E0_ui,
    scan_res_view,
    settings,
):
    # Click-selectable heatmap for EVERY quantity the Optimize tab's scan maps
    # already show (scan_charts' default quantity set) plus hit_frac, all pinned
    # to scan_heatmap_E0_ui's energy -- so clicking ANY of the existing heatmaps
    # (not a separate one) drives its own spectrum panel below it. Built top-level
    # for the same DAG-reactivity reason as heatmap_select above: a ui made inside
    # a tab body can lose its click round-trip. Faceted charts don't compose with
    # a marimo point selection (see heatmap_select_chart), so each quantity gets
    # its own SINGLE-panel chart at the picked energy rather than the usual
    # energy-faceted view. Keyed dict quantity -> mo.ui.altair_chart, or None when
    # that quantity's frame is empty for this view/thickness.
    import altair as _alt

    _specs = list(HEATMAP_QUANTITIES) + [("hit_frac", "electron footprint-hit fraction", "magma")]
    scan_pixel_heatmaps = {}
    with _alt.data_transformers.enable("default"):
        for _key, _label, _cmap in _specs:
            _domain = (0.0, 1.0) if _key == "hit_frac" else None
            _chart = heatmap_select_chart(
                scan_res_view,
                settings,
                quantity=(_key, _label, _cmap),
                panel_value=scan_heatmap_E0_ui.value,
                cases=cases,
                line_metric="prominence",
                color_domain=_domain,
            )
            scan_pixel_heatmaps[_key] = (
                mo.ui.altair_chart(_chart, chart_selection=False, legend_selection=False)
                if _chart is not None
                else None
            )
    return (scan_pixel_heatmaps,)


@app.cell
def _(fmt_thickness, mo, records, res, sweep_values):
    # Selectors for the "Polar-angle comparison" tab -- overlays one spectrum
    # line per POLAR TILT, so the OTHER swept knobs (beam energy, azimuth,
    # thickness) must each be pinned to a single value here. Built as TOP-LEVEL
    # reactive elements for the same reason as heatmap_E0_ui above: keep mo.ui
    # elements outside tab-body composition so changing them re-renders the tab.
    # A dim with only one swept value
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

    _thk_opts = {fmt_thickness(t): t for t in _thk} or {"— no data —": None}
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
def _(fmt_thickness, mo, records, res, sweep_values):
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

    _thk_opts = {fmt_thickness(t): t for t in _thk} or {"— no data —": None}
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
    # Polar-angle comparison spectral controls.
    polar_brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    polar_auto_ui = mo.ui.switch(value=True, label="Auto narrow domain")
    polar_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV)")
    polar_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV)")
    polar_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    polar_ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    polar_broad_auto_ui = mo.ui.switch(value=True, label="Auto broad domain")
    polar_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV)")
    polar_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV)")
    polar_broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    polar_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        polar_auto_ui,
        polar_brem_ui,
        polar_broad_auto_ui,
        polar_broad_xlog_ui,
        polar_broad_xmax_ui,
        polar_broad_xmin_ui,
        polar_broad_ylog_ui,
        polar_xlog_ui,
        polar_xmax_ui,
        polar_xmin_ui,
        polar_ylog_ui,
    )


@app.cell
def _(
    polar_auto_ui,
    polar_broad_auto_ui,
    polar_broad_xmax_ui,
    polar_broad_xmin_ui,
    polar_xmax_ui,
    polar_xmin_ui,
):
    def _domain(auto, xmin, xmax):
        if auto:
            return None
        return (xmin, xmax)

    polar_x_domain = _domain(polar_auto_ui.value, polar_xmin_ui.value, polar_xmax_ui.value)
    polar_broad_x_domain = _domain(
        polar_broad_auto_ui.value, polar_broad_xmin_ui.value, polar_broad_xmax_ui.value
    )
    return polar_broad_x_domain, polar_x_domain


@app.cell
def _(mo):
    # Azimuthal comparison spectral controls.
    azim_brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    azim_auto_ui = mo.ui.switch(value=True, label="Auto narrow domain")
    azim_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV)")
    azim_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV)")
    azim_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    azim_ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    azim_broad_auto_ui = mo.ui.switch(value=True, label="Auto broad domain")
    azim_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV)")
    azim_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV)")
    azim_broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    azim_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        azim_auto_ui,
        azim_brem_ui,
        azim_broad_auto_ui,
        azim_broad_xlog_ui,
        azim_broad_xmax_ui,
        azim_broad_xmin_ui,
        azim_broad_ylog_ui,
        azim_xlog_ui,
        azim_xmax_ui,
        azim_xmin_ui,
        azim_ylog_ui,
    )


@app.cell
def _(
    azim_auto_ui,
    azim_broad_auto_ui,
    azim_broad_xmax_ui,
    azim_broad_xmin_ui,
    azim_xmax_ui,
    azim_xmin_ui,
):
    def _domain(auto, xmin, xmax):
        if auto:
            return None
        return (xmin, xmax)

    azim_x_domain = _domain(azim_auto_ui.value, azim_xmin_ui.value, azim_xmax_ui.value)
    azim_broad_x_domain = _domain(
        azim_broad_auto_ui.value, azim_broad_xmin_ui.value, azim_broad_xmax_ui.value
    )
    return azim_broad_x_domain, azim_x_domain


@app.cell
def _(mo):
    # Energy-comparison spectral controls. Created top-level for reactivity,
    # rendered only inside the Energy comparison tab.
    brem_ui = mo.ui.checkbox(value=True, label="show brem background")

    narrow_auto_ui = mo.ui.switch(value=True, label="Auto narrow domain")
    narrow_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV)")
    narrow_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV)")
    narrow_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    broad_auto_ui = mo.ui.switch(value=True, label="Auto broad domain")
    broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV)")
    broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV)")
    broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        brem_ui,
        broad_auto_ui,
        broad_xlog_ui,
        broad_xmax_ui,
        broad_xmin_ui,
        broad_ylog_ui,
        narrow_auto_ui,
        narrow_xlog_ui,
        narrow_xmax_ui,
        narrow_xmin_ui,
        ylog_ui,
    )


@app.cell
def _(
    broad_auto_ui,
    broad_xmax_ui,
    broad_xmin_ui,
    narrow_auto_ui,
    narrow_xmax_ui,
    narrow_xmin_ui,
):
    def _domain(auto, xmin, xmax):
        if auto:
            return None
        return (xmin, xmax)

    x_domain = _domain(narrow_auto_ui.value, narrow_xmin_ui.value, narrow_xmax_ui.value)
    broad_x_domain = _domain(broad_auto_ui.value, broad_xmin_ui.value, broad_xmax_ui.value)
    return broad_x_domain, x_domain


@app.cell
def _(FACE, MATERIAL, mo, res, settings, top_geometries):
    def rankings_tab():
        # Rank across ALL thicknesses (raw `res`, not the pinned `res_view`): the
        # thickness column disambiguates rows, so a thickness sweep shows every
        # thickness competing head-to-head rather than being collapsed to the pin.
        df = top_geometries(res, settings, top_n=20, select="quality_peak")
        if df.empty:
            _how = "cxr blaze" if FACE == "blazed" else "scan_app.py"
            return mo.md(f"**No {FACE} checkpoint for `{MATERIAL}`** — run `{_how}` first.")
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

    return (rankings_tab,)


@app.cell
def _(
    brem_ui,
    broad_auto_ui,
    broad_x_domain,
    broad_xlog_ui,
    broad_xmax_ui,
    broad_xmin_ui,
    broad_ylog_ui,
    face_title,
    fmt_thickness,
    heatmap_E0_ui,
    heatmap_select,
    mo,
    narrow_auto_ui,
    narrow_xlog_ui,
    narrow_xmax_ui,
    narrow_xmin_ui,
    records,
    res,
    res_view,
    select_results,
    settings,
    spectrum_chart,
    sweep_values,
    thickness_ui,
    tilt_ui,
    x_domain,
    ylog_ui,
):
    def spectra_tab():
        _md = mo.md(
            "Coherent CXR line spectra at pinned polar tilt and thickness. "
            "Beam energy varies across curves."
        )
        _sv = sweep_values(res) if records(res) else {}
        # Beam energies whose beam died before the pinned thickness: select_thickness
        # substituted their thickest computed slab (spectrum saturated past that
        # depth). Surface it rather than silently dropping them.
        _fallback_energies = sorted(
            {_r["case"]["E0_keV"] for _r in records(res_view) if "thickness_fallback" in _r["case"]}
        )
        _fallback_note = (
            mo.md(
                "*Beam died before pinned thickness for "
                + ", ".join(f"{_e:g}" for _e in _fallback_energies)
                + " keV — shown at max computed slab (watchdog-saturated).*"
            )
            if _fallback_energies
            else None
        )
        _thk_widget = (
            thickness_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"crystal thickness: {fmt_thickness(thickness_ui.value)}"
                if thickness_ui.value is not None
                else ""
            )
        )
        _controls = mo.vstack(
            [
                mo.hstack([tilt_ui, _thk_widget, brem_ui], wrap=True),
                mo.accordion(
                    {
                        "Axes and scaling": mo.vstack(
                            [
                                mo.hstack(
                                    [
                                        narrow_auto_ui,
                                        narrow_xmin_ui,
                                        narrow_xmax_ui,
                                        narrow_xlog_ui,
                                        ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                                mo.hstack(
                                    [
                                        broad_auto_ui,
                                        broad_xmin_ui,
                                        broad_xmax_ui,
                                        broad_xlog_ui,
                                        broad_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                            ]
                        )
                    }
                ),
            ]
        )

        def _narrow_spectrum():
            _chart = spectrum_chart(
                res_view,
                settings,
                tilt_deg=tilt_ui.value,
                include_brem=brem_ui.value,
                x_domain=x_domain,
                x_type="log" if narrow_xlog_ui.value else "linear",
                y_type="log" if ylog_ui.value else "linear",
                band="narrow",
            )
            return (
                face_title(_chart)
                if _chart is not None
                else mo.md("*No narrowband spectra -- run the scan first.*")
            )

        def _broad_spectrum():
            _chart = spectrum_chart(
                res_view,
                settings,
                tilt_deg=tilt_ui.value,
                include_brem=brem_ui.value,
                x_domain=broad_x_domain,
                x_type="log" if broad_xlog_ui.value else "linear",
                y_type="log" if broad_ylog_ui.value else "linear",
                band="broad",
            )
            return (
                face_title(_chart)
                if _chart is not None
                else mo.md("*No broadband spectra -- run the scan first.*")
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
            _sp_narrow = spectrum_chart(
                _sub,
                settings,
                tilt_deg=float(_row["y"]),
                include_brem=brem_ui.value,
                x_domain=x_domain,
                x_type="log" if narrow_xlog_ui.value else "linear",
                y_type="log" if ylog_ui.value else "linear",
                band="narrow",
            )
            _sp_broad = spectrum_chart(
                _sub,
                settings,
                tilt_deg=float(_row["y"]),
                include_brem=brem_ui.value,
                x_domain=broad_x_domain,
                x_type="log" if broad_xlog_ui.value else "linear",
                y_type="log" if broad_ylog_ui.value else "linear",
                band="broad",
            )
            _charts = [c for c in (_sp_narrow, _sp_broad) if c is not None]
            return mo.vstack(_charts) if _charts else mo.md("*No spectrum for that cell.*")

        _hm = (
            heatmap_select
            if heatmap_select is not None
            else mo.md("*No 2-D heatmap for this checkpoint / thickness.*")
        )
        _parts = [_md]
        if _fallback_note is not None:
            _parts.append(_fallback_note)
        _parts.extend(
            [
                _controls,
                mo.md("**Narrowband**"),
                mo.lazy(_narrow_spectrum, show_loading_indicator=True),
                mo.md("**Broadband**"),
                mo.lazy(_broad_spectrum, show_loading_indicator=True),
                mo.md("---"),
                mo.md(
                    "**Pixel-select** — pick a beam energy, then click a heatmap cell to "
                    "plot that geometry's spectrum below."
                ),
                heatmap_E0_ui,
                _hm,
                mo.lazy(_pixel_spectrum, show_loading_indicator=True),
            ]
        )
        return mo.vstack(_parts)

    return (spectra_tab,)


@app.cell
def _(
    compare_spectrum_chart,
    fmt_thickness,
    mo,
    polar_E0_ui,
    polar_auto_ui,
    polar_azim_ui,
    polar_brem_ui,
    polar_broad_auto_ui,
    polar_broad_x_domain,
    polar_broad_xlog_ui,
    polar_broad_xmax_ui,
    polar_broad_xmin_ui,
    polar_broad_ylog_ui,
    polar_thk_ui,
    polar_tilts_ui,
    polar_x_domain,
    polar_xlog_ui,
    polar_xmax_ui,
    polar_xmin_ui,
    polar_ylog_ui,
    records,
    res,
    select_results,
    settings,
    sweep_values,
):
    def polar_compare_tab():
        # Overlays one spectrum line per POLAR TILT (the polar_tilts_ui
        # multiselect), at a PINNED beam energy / azimuth / thickness -- the
        # complement of _spectra_tab, which fixes tilt and varies energy. Uses
        # the RAW `res` (not the pinned `res_view`): thickness is pinned by
        # THIS tab's own polar_thk_ui dropdown instead, so starting from
        # res_view would apply the (unrelated) top-of-notebook thickness pin
        # underneath it.
        _md = mo.md(
            "Coherent CXR line spectra at pinned beam energy, azimuth, and thickness. "
            "Polar tilt varies across selected curves."
        )
        _spectral_controls = mo.vstack(
            [
                polar_brem_ui,
                mo.accordion(
                    {
                        "Axes and scaling": mo.vstack(
                            [
                                mo.hstack(
                                    [
                                        polar_auto_ui,
                                        polar_xmin_ui,
                                        polar_xmax_ui,
                                        polar_xlog_ui,
                                        polar_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                                mo.hstack(
                                    [
                                        polar_broad_auto_ui,
                                        polar_broad_xmin_ui,
                                        polar_broad_xmax_ui,
                                        polar_broad_xlog_ui,
                                        polar_broad_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                            ]
                        )
                    }
                ),
            ]
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
                f"thickness: {fmt_thickness(polar_thk_ui.value)}"
                if polar_thk_ui.value is not None
                else ""
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
                    _spectral_controls,
                    polar_tilts_ui,
                    mo.md("*Select at least one polar tilt above to plot.*"),
                ]
            )

        def _narrow_chart_item():
            _constraints = {"tilt_deg": list(polar_tilts_ui.value)}
            if polar_E0_ui.value is not None:
                _constraints["E0_keV"] = polar_E0_ui.value
            if polar_azim_ui.value is not None:
                _constraints["tilt_azim_deg"] = polar_azim_ui.value
            if polar_thk_ui.value is not None:
                _constraints["thickness_ang"] = polar_thk_ui.value
            _chart = compare_spectrum_chart(
                select_results(res, **_constraints),
                settings,
                hue="tilt_deg",
                include_brem=polar_brem_ui.value,
                x_domain=polar_x_domain,
                x_type="log" if polar_xlog_ui.value else "linear",
                y_type="log" if polar_ylog_ui.value else "linear",
                band="narrow",
            )
            return (
                _chart if _chart is not None else mo.md("*No narrowband spectra for this slice.*")
            )

        def _broad_chart_item():
            _constraints = {"tilt_deg": list(polar_tilts_ui.value)}
            if polar_E0_ui.value is not None:
                _constraints["E0_keV"] = polar_E0_ui.value
            if polar_azim_ui.value is not None:
                _constraints["tilt_azim_deg"] = polar_azim_ui.value
            if polar_thk_ui.value is not None:
                _constraints["thickness_ang"] = polar_thk_ui.value
            _chart = compare_spectrum_chart(
                select_results(res, **_constraints),
                settings,
                hue="tilt_deg",
                include_brem=polar_brem_ui.value,
                x_domain=polar_broad_x_domain,
                x_type="log" if polar_broad_xlog_ui.value else "linear",
                y_type="log" if polar_broad_ylog_ui.value else "linear",
                band="broad",
            )
            return _chart if _chart is not None else mo.md("*No broadband spectra for this slice.*")

        _parts = [
            _md,
            _e0_widget,
            _azim_widget,
            _thk_widget,
            _spectral_controls,
            polar_tilts_ui,
        ]
        if _note is not None:
            _parts.append(_note)
        _parts.extend(
            [
                mo.md("**Narrowband**"),
                mo.lazy(_narrow_chart_item, show_loading_indicator=True),
                mo.md("**Broadband**"),
                mo.lazy(_broad_chart_item, show_loading_indicator=True),
            ]
        )
        return mo.vstack(_parts)

    return (polar_compare_tab,)


@app.cell
def _(
    azim_E0_ui,
    azim_auto_ui,
    azim_azims_ui,
    azim_brem_ui,
    azim_broad_auto_ui,
    azim_broad_x_domain,
    azim_broad_xlog_ui,
    azim_broad_xmax_ui,
    azim_broad_xmin_ui,
    azim_broad_ylog_ui,
    azim_thk_ui,
    azim_tilt_ui,
    azim_x_domain,
    azim_xlog_ui,
    azim_xmax_ui,
    azim_xmin_ui,
    azim_ylog_ui,
    compare_spectrum_chart,
    fmt_thickness,
    mo,
    records,
    res,
    select_results,
    settings,
    sweep_values,
):
    def azim_compare_tab():
        # Overlays one spectrum line per AZIMUTH (the azim_azims_ui
        # multiselect), at a pinned beam energy / polar tilt / thickness --
        # mirrors _polar_compare_tab with the swept dimension swapped. Also
        # uses the raw `res`; thickness is pinned by azim_thk_ui here.
        _md = mo.md(
            "Coherent CXR line spectra at pinned beam energy, polar tilt, and thickness. "
            "Azimuth varies across selected curves."
        )
        _spectral_controls = mo.vstack(
            [
                azim_brem_ui,
                mo.accordion(
                    {
                        "Axes and scaling": mo.vstack(
                            [
                                mo.hstack(
                                    [
                                        azim_auto_ui,
                                        azim_xmin_ui,
                                        azim_xmax_ui,
                                        azim_xlog_ui,
                                        azim_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                                mo.hstack(
                                    [
                                        azim_broad_auto_ui,
                                        azim_broad_xmin_ui,
                                        azim_broad_xmax_ui,
                                        azim_broad_xlog_ui,
                                        azim_broad_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                            ]
                        )
                    }
                ),
            ]
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
                f"thickness: {fmt_thickness(azim_thk_ui.value)}"
                if azim_thk_ui.value is not None
                else ""
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
                    _spectral_controls,
                    azim_azims_ui,
                    mo.md("*Select at least one azimuth above to plot.*"),
                ]
            )

        def _narrow_chart_item():
            _constraints = {"tilt_azim_deg": list(azim_azims_ui.value)}
            if azim_E0_ui.value is not None:
                _constraints["E0_keV"] = azim_E0_ui.value
            if azim_tilt_ui.value is not None:
                _constraints["tilt_deg"] = azim_tilt_ui.value
            if azim_thk_ui.value is not None:
                _constraints["thickness_ang"] = azim_thk_ui.value
            _chart = compare_spectrum_chart(
                select_results(res, **_constraints),
                settings,
                hue="tilt_azim_deg",
                include_brem=azim_brem_ui.value,
                x_domain=azim_x_domain,
                x_type="log" if azim_xlog_ui.value else "linear",
                y_type="log" if azim_ylog_ui.value else "linear",
                band="narrow",
            )
            return (
                _chart if _chart is not None else mo.md("*No narrowband spectra for this slice.*")
            )

        def _broad_chart_item():
            _constraints = {"tilt_azim_deg": list(azim_azims_ui.value)}
            if azim_E0_ui.value is not None:
                _constraints["E0_keV"] = azim_E0_ui.value
            if azim_tilt_ui.value is not None:
                _constraints["tilt_deg"] = azim_tilt_ui.value
            if azim_thk_ui.value is not None:
                _constraints["thickness_ang"] = azim_thk_ui.value
            _chart = compare_spectrum_chart(
                select_results(res, **_constraints),
                settings,
                hue="tilt_azim_deg",
                include_brem=azim_brem_ui.value,
                x_domain=azim_broad_x_domain,
                x_type="log" if azim_broad_xlog_ui.value else "linear",
                y_type="log" if azim_broad_ylog_ui.value else "linear",
                band="broad",
            )
            return _chart if _chart is not None else mo.md("*No broadband spectra for this slice.*")

        _parts = [
            _md,
            _e0_widget,
            _tilt_widget,
            _thk_widget,
            _spectral_controls,
            azim_azims_ui,
        ]
        if _note is not None:
            _parts.append(_note)
        _parts.extend(
            [
                mo.md("**Narrowband**"),
                mo.lazy(_narrow_chart_item, show_loading_indicator=True),
                mo.md("**Broadband**"),
                mo.lazy(_broad_chart_item, show_loading_indicator=True),
            ]
        )
        return mo.vstack(_parts)

    return (azim_compare_tab,)


@app.cell
def _(mo):
    # Case-comparison basket -- lives outside the `res`-derived reactivity
    # graph the other compare tabs sit in, so it survives switching
    # material/face above: picking cases from several checkpoints in a row
    # accumulates them instead of resetting.
    get_case_basket, set_case_basket = mo.state([])
    return get_case_basket, set_case_basket


@app.cell
def _(case_table_rows, mo, res):
    # Case picker over the RAW `res` (mirrors the polar/azim tabs above):
    # thickness is a selectable dimension here, so the top-of-notebook
    # thickness pin must not apply underneath it.
    case_picker_ui = mo.ui.table(
        case_table_rows(res), selection="multi", label="cases in this checkpoint"
    )
    return (case_picker_ui,)


@app.cell
def _(
    FACE,
    MATERIAL,
    case_label,
    case_picker_ui,
    mo,
    res,
    set_case_basket,
    slim_case_record,
    sweep_values,
):
    _CASE_BASKET_CAP = 12

    def _add_selected_cases(n):
        _selected = case_picker_ui.value
        if _selected:
            _varying = set(sweep_values(res))
            _new = []
            for _row in _selected:
                _record = res[_row["name"]][_row["E0_keV"]]
                _label = case_label(
                    _record["case"], material_label=MATERIAL, face=FACE, varying=_varying
                )
                _new.append(slim_case_record(_record, material=MATERIAL, label=_label, face=FACE))
            set_case_basket(lambda old: (old + _new)[-_CASE_BASKET_CAP:])
        return n + 1

    case_basket_add_ui = mo.ui.button(
        value=0, on_click=_add_selected_cases, label="Add selected cases"
    )
    return (case_basket_add_ui,)


@app.cell
def _(get_case_basket, mo):
    # Index-keyed (not label-keyed) so two basket entries that happen to
    # render the same label stay individually removable. Own cell, separate
    # from the buttons that read `.value` below -- a cell may not both create
    # a mo.ui widget and read its own `.value` (see
    # test_all_ui_values_are_read_downstream_of_creation).
    _basket_options = {
        f"{i}: {entry['case']['label']}": i for i, entry in enumerate(get_case_basket())
    }
    case_basket_remove_select_ui = mo.ui.multiselect(_basket_options, label="remove from basket")
    return (case_basket_remove_select_ui,)


@app.cell
def _(case_basket_remove_select_ui, mo, set_case_basket):
    def _remove_selected(n):
        _drop = set(case_basket_remove_select_ui.value)
        if _drop:
            set_case_basket(lambda old: [e for i, e in enumerate(old) if i not in _drop])
        return n + 1

    case_basket_remove_ui = mo.ui.button(
        value=0, on_click=_remove_selected, label="Remove selected"
    )

    def _clear_basket(n):
        set_case_basket(lambda old: [])
        return n + 1

    case_basket_clear_ui = mo.ui.button(value=0, on_click=_clear_basket, label="Clear basket")
    return case_basket_clear_ui, case_basket_remove_ui


@app.cell
def _(mo):
    # Case-comparison spectral controls -- own widget instances, independent
    # of the polar/azim tabs' equivalents above.
    case_brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    case_auto_ui = mo.ui.switch(value=True, label="Auto narrow domain")
    case_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV)")
    case_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV)")
    case_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    case_ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    case_broad_auto_ui = mo.ui.switch(value=True, label="Auto broad domain")
    case_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV)")
    case_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV)")
    case_broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    case_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        case_auto_ui,
        case_brem_ui,
        case_broad_auto_ui,
        case_broad_xlog_ui,
        case_broad_xmax_ui,
        case_broad_xmin_ui,
        case_broad_ylog_ui,
        case_xlog_ui,
        case_xmax_ui,
        case_xmin_ui,
        case_ylog_ui,
    )


@app.cell
def _(
    case_auto_ui,
    case_broad_auto_ui,
    case_broad_xmax_ui,
    case_broad_xmin_ui,
    case_xmax_ui,
    case_xmin_ui,
):
    def _domain(auto, xmin, xmax):
        if auto:
            return None
        return (xmin, xmax)

    case_x_domain = _domain(case_auto_ui.value, case_xmin_ui.value, case_xmax_ui.value)
    case_broad_x_domain = _domain(
        case_broad_auto_ui.value, case_broad_xmin_ui.value, case_broad_xmax_ui.value
    )
    return case_broad_x_domain, case_x_domain


@app.cell
def _(
    case_auto_ui,
    case_basket_add_ui,
    case_basket_clear_ui,
    case_basket_remove_select_ui,
    case_basket_remove_ui,
    case_brem_ui,
    case_broad_auto_ui,
    case_broad_x_domain,
    case_broad_xlog_ui,
    case_broad_xmax_ui,
    case_broad_xmin_ui,
    case_broad_ylog_ui,
    case_picker_ui,
    case_x_domain,
    case_xlog_ui,
    case_xmax_ui,
    case_xmin_ui,
    case_ylog_ui,
    get_case_basket,
    mo,
    multi_case_spectrum_chart,
    settings,
):
    _CASE_BASKET_CAP = 12

    def case_compare_tab():
        # Cross-material "case basket": user hand-picks individual cases from
        # whichever checkpoint is loaded above and overlays them, unlike the
        # single-swept-dimension polar/azim tabs. See
        # docs/case-compare-tab-plan.md.
        _basket = get_case_basket()
        _picker_block = mo.vstack(
            [
                mo.md(
                    "Pick cases from the checkpoint currently loaded above, then add them "
                    "to the comparison basket. Switch material/face and add more to compare "
                    "across materials — the basket persists."
                ),
                case_picker_ui,
                case_basket_add_ui,
                mo.hstack(
                    [case_basket_remove_select_ui, case_basket_remove_ui, case_basket_clear_ui],
                    wrap=True,
                ),
            ]
        )
        if not _basket:
            return mo.vstack(
                [_picker_block, mo.md("*Add 2+ cases from the picker above to compare.*")]
            )

        _cases = [(entry, entry["case"]["label"]) for entry in _basket]
        _spectral_controls = mo.vstack(
            [
                case_brem_ui,
                mo.accordion(
                    {
                        "Axes and scaling": mo.vstack(
                            [
                                mo.hstack(
                                    [
                                        case_auto_ui,
                                        case_xmin_ui,
                                        case_xmax_ui,
                                        case_xlog_ui,
                                        case_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                                mo.hstack(
                                    [
                                        case_broad_auto_ui,
                                        case_broad_xmin_ui,
                                        case_broad_xmax_ui,
                                        case_broad_xlog_ui,
                                        case_broad_ylog_ui,
                                    ],
                                    wrap=True,
                                ),
                            ]
                        )
                    }
                ),
            ]
        )

        def _narrow_chart_item():
            _chart = multi_case_spectrum_chart(
                _cases,
                settings,
                include_brem=case_brem_ui.value,
                x_domain=case_x_domain,
                x_type="log" if case_xlog_ui.value else "linear",
                y_type="log" if case_ylog_ui.value else "linear",
                band="narrow",
            )
            return _chart if _chart is not None else mo.md("*No narrowband spectra in the basket.*")

        def _broad_chart_item():
            _chart = multi_case_spectrum_chart(
                _cases,
                settings,
                include_brem=case_brem_ui.value,
                x_domain=case_broad_x_domain,
                x_type="log" if case_broad_xlog_ui.value else "linear",
                y_type="log" if case_broad_ylog_ui.value else "linear",
                band="broad",
            )
            return _chart if _chart is not None else mo.md("*No broadband spectra in the basket.*")

        _basket_rows = [
            {"label": entry["case"]["label"], "material": entry["case"]["material"]}
            for entry in _basket
        ]
        _cap_note = (
            mo.md(f"*Basket capped at {_CASE_BASKET_CAP} entries — oldest drop as you add more.*")
            if len(_basket) >= _CASE_BASKET_CAP
            else None
        )
        _parts = [_picker_block]
        if _cap_note is not None:
            _parts.append(_cap_note)
        _parts.extend(
            [
                mo.md("**Basket contents**"),
                mo.ui.table(_basket_rows, selection=None),
                _spectral_controls,
                mo.md("**Narrowband**"),
                mo.lazy(_narrow_chart_item, show_loading_indicator=True),
                mo.md("**Broadband**"),
                mo.lazy(_broad_chart_item, show_loading_indicator=True),
            ]
        )
        return mo.vstack(_parts)

    return (case_compare_tab,)


@app.cell
def _(
    HEATMAP_QUANTITIES,
    cases,
    fmt_thickness,
    metric_vs_chart,
    mo,
    plot_best_spectra,
    records,
    res,
    scan_charts,
    scan_heatmap_E0_ui,
    scan_pixel_heatmaps,
    scan_res_view,
    scan_thickness_ui,
    select_results,
    settings,
    spectrum_chart,
    sweep_values,
):
    _tilts = {r["case"]["tilt_deg"] for r in records(scan_res_view)}
    _azims = {r["case"]["tilt_azim_deg"] for r in records(scan_res_view)}
    # Mirrors scan_charts' own heatmap_min=4 auto-pick: a heatmap is only
    # meaningful once both axes sweep >= 4 values, otherwise scan_charts falls
    # back to line plots (nothing to click there). Computed at CELL top level
    # (not inside scans_tab) so the mo.lazy-deferred nested closures below can
    # resolve it as a cell global instead of a true function-local closure var
    # -- marimo's lazy re-render can't reconstruct genuine nested-function
    # closures over locals, only free vars resolvable at cell/module scope.
    _is_heatmap_mode = len(_tilts) >= 4 and len(_azims) >= 4

    def scans_tab():
        # Click a heatmap cell -> spectrum for THAT (azimuth, tilt) geometry at the
        # heatmap's beam energy, on the scan-pinned-thickness view. `x`/`y` are the
        # encoded (tilt_azim_deg, tilt_deg) of the clicked cell. Shown directly
        # under the SAME heatmap that was clicked, so each of the (already
        # on-screen) quantity heatmaps drives its own spectrum -- no ambiguity
        # about which of several independent click widgets fired last.
        def _pixel_heatmap_block(key):
            widget = scan_pixel_heatmaps.get(key)
            if widget is None:
                return mo.md(f"*No {key} heatmap — needs an azimuth × polar-tilt sweep.*")

            def _spectrum():
                _sel = widget.value
                if _sel is None or len(_sel) == 0:
                    return mo.md("*Click a cell above to plot its spectrum here.*")
                _row = _sel.iloc[0]
                _sub = select_results(
                    scan_res_view,
                    tilt_azim_deg=float(_row["x"]),
                    tilt_deg=float(_row["y"]),
                    E0_keV=scan_heatmap_E0_ui.value,
                )
                _sp_narrow = spectrum_chart(
                    _sub, settings, tilt_deg=float(_row["y"]), band="narrow"
                )
                _sp_broad = spectrum_chart(_sub, settings, tilt_deg=float(_row["y"]), band="broad")
                _charts = [c for c in (_sp_narrow, _sp_broad) if c is not None]
                return mo.vstack(_charts) if _charts else mo.md("*No spectrum for that cell.*")

            return mo.vstack([widget, mo.lazy(_spectrum, show_loading_indicator=True)])

        def _scan_charts_item():
            if not _is_heatmap_mode:
                charts = scan_charts(scan_res_view, settings, cases=cases, line_metric="prominence")
                return mo.vstack(charts) if charts else mo.md("*No scan results.*")
            _blocks = [_pixel_heatmap_block(_key) for _key, _label, _cmap in HEATMAP_QUANTITIES]
            return mo.vstack(_blocks) if _blocks else mo.md("*No scan results.*")

        def _hit_frac_item():
            if not _is_heatmap_mode:
                return mo.md("*No electron-hit map — needs an azimuth × polar-tilt sweep.*")
            return _pixel_heatmap_block("hit_frac")

        def _metric_lines_item():
            line = metric_vs_chart(
                scan_res_view, settings, x="tilt_deg", metric="line_flux", hue="E0_keV"
            )
            peak = metric_vs_chart(
                scan_res_view, settings, x="tilt_deg", metric="peak_flux", hue="E0_keV"
            )
            parts = [c for c in (line, peak) if c is not None]
            return mo.vstack(parts) if parts else mo.md("*No metric results.*")

        _sv = sweep_values(res) if records(res) else {}
        _thk_widget = (
            scan_thickness_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"crystal thickness: {fmt_thickness(scan_thickness_ui.value)}"
                if scan_thickness_ui.value is not None
                else ""
            )
        )
        _parts = [
            mo.md(
                "`scan_charts` auto-picks a heatmap or line scans (one per swept quantity); "
                "then 1-D metric scans vs polar tilt. Each section loads as it becomes "
                "visible. The best-spectra panel loads on expand."
            ),
            _thk_widget,
        ]
        if _is_heatmap_mode:
            _parts.extend(
                [
                    mo.md(
                        "**Click any heatmap cell below to plot its spectrum** — one "
                        "beam energy shown at a time (a faceted chart can't carry a "
                        "click selection)."
                    ),
                    scan_heatmap_E0_ui,
                ]
            )
        _parts.append(mo.lazy(_scan_charts_item, show_loading_indicator=True))
        _parts.extend(
            [
                mo.md(
                    "**Electron footprint-hit fraction** — of the electrons launched at "
                    "the crystal, the share that landed on its finite footprint "
                    "(bright = all hit, dark = all missed); 1.0 everywhere for a "
                    "laterally infinite crystal."
                ),
                mo.lazy(_hit_frac_item, show_loading_indicator=True),
                mo.lazy(_metric_lines_item, show_loading_indicator=True),
                mo.accordion(
                    {
                        "Best spectra (matplotlib)": lambda: plot_best_spectra(
                            scan_res_view, settings, top_n=12, select="quality_peak"
                        )
                    },
                    lazy=True,
                ),
            ]
        )
        return mo.vstack(_parts)

    return (scans_tab,)


@app.cell
def _(
    cases,
    detector_auto_ui,
    detector_azim_ui,
    detector_broad_auto_ui,
    detector_broad_x_domain,
    detector_broad_xlog_ui,
    detector_broad_xmax_ui,
    detector_broad_xmin_ui,
    detector_broad_y_domain,
    detector_broad_ylog_ui,
    detector_broad_ymax_ui,
    detector_broad_ymin_ui,
    detector_res_view,
    detector_thickness_ui,
    detector_tilt_ui,
    detector_x_domain,
    detector_xlog_ui,
    detector_xmax_ui,
    detector_xmin_ui,
    detector_y_domain,
    detector_ylog_ui,
    detector_ymax_ui,
    detector_ymin_ui,
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    fmt_thickness,
    mo,
    plot_eaglexo_charge_map,
    plot_eaglexo_efficiency,
    plot_timepix_efficiency,
    records,
    res,
    settings,
    sweep_values,
    timepix_detected_chart,
):
    def detectors_tab():
        # Eagle XO and Timepix3 render into a nested mo.accordion
        # rather than a dropdown or nested mo.ui.tabs: keeping the controls in
        # top-level cells makes their reactivity explicit, and nested
        # mo.ui.tabs-in-tabs is a known marimo rendering
        # bug (charts inside the inner tab silently render blank -- this is what
        # produced the "blank detector-tab plots" bug: marimo-team/marimo#6919).
        # The accordion lazily defers each section's compute exactly like the
        # inner tabs did, without the nesting problem.
        _sv = sweep_values(res) if records(res) else {}
        _thk_widget = (
            detector_thickness_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"crystal thickness: {fmt_thickness(detector_thickness_ui.value)}"
                if detector_thickness_ui.value is not None
                else ""
            )
        )
        _azim_widget = (
            detector_azim_ui
            if len(_sv.get("tilt_azim_deg", [])) > 1
            else mo.md(
                f"azimuth: {detector_azim_ui.value:g} deg"
                if detector_azim_ui.value is not None
                else ""
            )
        )
        _controls = mo.vstack(
            [
                mo.hstack([detector_tilt_ui, _azim_widget, _thk_widget], wrap=True),
                mo.vstack(
                    [
                        mo.hstack(
                            [
                                detector_auto_ui,
                                detector_xmin_ui,
                                detector_xmax_ui,
                                detector_ymin_ui,
                                detector_ymax_ui,
                                detector_xlog_ui,
                                detector_ylog_ui,
                            ],
                            wrap=True,
                        ),
                        mo.hstack(
                            [
                                detector_broad_auto_ui,
                                detector_broad_xmin_ui,
                                detector_broad_xmax_ui,
                                detector_broad_ymin_ui,
                                detector_broad_ymax_ui,
                                detector_broad_xlog_ui,
                                detector_broad_ylog_ui,
                            ],
                            wrap=True,
                        ),
                    ]
                ),
            ]
        )

        # Each detector chart is rendered twice, mirroring the spectra tab: a
        # narrow "line" band (PXR/CXR lines only, its own domain + log toggles)
        # stacked above a "broad" band (lines + brem, separate domain + logs).
        # `_band_pair` fans one chart function across both bands so the narrow-
        # and broad-axis controls both drive a chart instead of sitting orphaned.
        def _band_pair(chart_fn):
            _line = chart_fn(
                detector_res_view,
                settings,
                tilt_deg=detector_tilt_ui.value,
                x_domain=detector_x_domain,
                y_domain=detector_y_domain,
                x_type="log" if detector_xlog_ui.value else "linear",
                y_type="log" if detector_ylog_ui.value else "linear",
                band="narrow",
            )
            _broad = chart_fn(
                detector_res_view,
                settings,
                tilt_deg=detector_tilt_ui.value,
                x_domain=detector_broad_x_domain,
                y_domain=detector_broad_y_domain,
                x_type="log" if detector_broad_xlog_ui.value else "linear",
                y_type="log" if detector_broad_ylog_ui.value else "linear",
                band="broad",
            )
            return [c for c in (_line, _broad) if c is not None]

        def _eaglexo_inner():
            _md = mo.md(
                "Raptor Eagle XO direct-detection CCD (`solid_angle x QE(E)`): soft PXR "
                "lines pass at ~90% QE, hard brem is crushed by the thin sensor. "
                "Photon density (detected vs incident), then recorded-charge density."
            )
            _parts = [_md, *_band_pair(eaglexo_detected_chart), *_band_pair(eaglexo_charge_chart)]
            _parts.append(
                mo.accordion(
                    {
                        "Efficiency: QE + solid angle + resolution (matplotlib)": lambda: (
                            plot_eaglexo_efficiency(sensor="4240")
                        ),
                        "Charge geometry map (matplotlib)": lambda: (
                            plot_eaglexo_charge_map(detector_res_view, settings, cases=cases)
                            if records(detector_res_view)
                            else mo.md("*No results.*")
                        ),
                    },
                )
            )
            return mo.vstack(_parts)

        def _timepix_inner():
            _md = mo.md(
                "Si quad forward model: photoabsorption → charge sharing → per-pixel "
                "threshold counting. Detected vs incident at the selected tilt."
            )
            _parts = [_md, *_band_pair(timepix_detected_chart)]
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

        return mo.vstack(
            [
                _controls,
                mo.accordion({"Eagle XO": _eaglexo_inner, "Timepix3": _timepix_inner}, lazy=True),
            ]
        )

    return (detectors_tab,)


@app.cell
def _(CATALOG, mo):
    # Cross-material controls are top-level so switching them reruns the lazy
    # tab body. The selector contains only energies present in loaded data --
    # read from each material's checkpoint manifest, not load_checkpoint,
    # which would unpickle every material's checkpoint (140-225 MB gzip each)
    # just to enumerate beam energies.
    from cxr_mc.analyze import analysis_checkpoint_manifest

    _energies = set()
    for _material_key in CATALOG.material_keys:
        _manifest = analysis_checkpoint_manifest(_material_key)
        if _manifest:
            _energies.update(_manifest["energies_keV"])
    cross_material_energy_options = {f"{energy:g} keV": energy for energy in sorted(_energies)} or {
        "— no data —": None
    }
    compare_all_beam_energies_ui = mo.ui.checkbox(value=True, label="Compare all beam energies")
    return (
        analysis_checkpoint_manifest,
        compare_all_beam_energies_ui,
        cross_material_energy_options,
    )


@app.cell
def _(compare_all_beam_energies_ui, cross_material_energy_options, mo):
    cross_material_energy_ui = mo.ui.dropdown(
        cross_material_energy_options,
        value=next(iter(cross_material_energy_options)),
        label="beam energy",
        disabled=compare_all_beam_energies_ui.value,
    )
    return (cross_material_energy_ui,)


@app.cell
def _(
    CATALOG,
    analysis_checkpoint_manifest,
    compare_all_beam_energies_ui,
    cross_material_energy_ui,
    mo,
    settings,
):
    from cxr_mc.analyze import cached_analysis
    from cxr_mc.plots import (
        MATERIAL_COMPARISON_SUMMARY_VERSION,
        material_comparison_summary,
        select_material_comparison,
    )
    from cxr_mc.plots.altair_spectra import material_comparison_chart

    def cross_material_tab():
        _md = mo.md(
            "For every material whose checkpoint exists, the single best geometry's "
            "dominant line: energy vs flux. The three comparisons select by line "
            "quality, peak flux, and local line-to-bremsstrahlung ratio. Each "
            "point is labeled with its material; hover a point for the selected "
            "geometry's beam energy, \u03b8, and \u03c6. "
            "Every comparison rejects candidate lines with quality below 0.5 "
            "before selecting the best point; a material with no line clearing "
            "that bar is dropped from the plot and reported in a printed "
            "statement below it. Each material's checkpoint is analyzed once and "
            "cached, so reopening this tab re-analyzes only checkpoints that "
            "changed since."
        )
        _beam_energy = (
            None if compare_all_beam_energies_ui.value else cross_material_energy_ui.value
        )
        _n_with_data = sum(
            1
            for _material_key in CATALOG.material_keys
            if (analysis_checkpoint_manifest(_material_key) or {}).get("n_records", 0) > 0
        )

        _summaries = {}
        for _material_key in CATALOG.material_keys:
            _summary = cached_analysis(
                _material_key,
                lambda _results: material_comparison_summary(_results, settings),
                (
                    "material_comparison_summary",
                    MATERIAL_COMPARISON_SUMMARY_VERSION,
                    0.03,
                    "sharpness",
                    settings.beam_current_na,
                ),
            )
            if _summary:
                _summaries[_material_key] = _summary

        def _comparison(select):
            _pts = []
            _dropped = {}
            for _material_key, _summary in _summaries.items():
                _point, _reason = select_material_comparison(
                    _summary,
                    select=select,
                    beam_energy_keV=_beam_energy,
                    min_line_quality=0.5,
                )
                if _point is None:
                    if _reason is not None:
                        _dropped[CATALOG.material(_material_key).label] = _reason
                    continue
                _label = CATALOG.material(_material_key).label
                _pts.append((_label, *_point))
            return material_comparison_chart(
                _pts, _dropped, select=select, beam_energy_keV=_beam_energy, min_line_quality=0.5
            )

        if _n_with_data >= 2:
            return mo.vstack(
                [
                    _md,
                    mo.hstack([compare_all_beam_energies_ui, cross_material_energy_ui], wrap=True),
                    _comparison("quality_peak"),
                    _comparison("peak"),
                    _comparison("line_brem_ratio"),
                ]
            )
        return mo.vstack(
            [
                _md,
                mo.md("*Run `scan_app.py` for more materials to populate this comparison.*"),
            ]
        )

    return (cross_material_tab,)


@app.cell
def _(
    MATERIAL,
    azim_compare_tab,
    case_compare_tab,
    cross_material_tab,
    detectors_tab,
    mo,
    polar_compare_tab,
    rankings_tab,
    scans_tab,
    spectra_tab,
):
    # Every top-level tab holds exactly one view, so each is a bare tab body
    # rather than a nested action-accordion group.
    if MATERIAL is None:
        view = mo.callout(
            mo.md(
                "**No checkpoint data available.** "
                "Run `cxr run standard -m <material>` to create one."
            ),
            kind="info",
        )
    else:
        view = mo.ui.tabs(
            {
                "Explore": lambda: mo.accordion(
                    {
                        "Compare beam energies": spectra_tab,
                        "Compare polar angles": polar_compare_tab,
                        "Compare azimuths": azim_compare_tab,
                        "Compare any cases": case_compare_tab,
                    },
                    lazy=True,
                    multiple=True,
                ),
                "Optimize": lambda: mo.vstack(
                    [
                        scans_tab(),
                        mo.accordion(
                            {"Rank geometries": rankings_tab},
                            lazy=True,
                            multiple=True,
                        ),
                    ]
                ),
                "Instruments": detectors_tab,
                "Compare": cross_material_tab,
            },
            lazy=True,
        )
    mo.vstack(
        [
            view,
            mo.md(
                "*3D trajectory and crystal-structure views live in `trace_app.py` "
                "(`marimo run notebooks/trace_app.py`).*"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
