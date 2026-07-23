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
    import traitlets
    from _design import page_title, style_sheet
    from anywidget import AnyWidget

    class MaterialSelect(AnyWidget):
        _esm = r"""
        function render({ model, el }) {
          const label = document.createElement("label");
          label.textContent = model.get("label");
          const select = document.createElement("select");
          select.setAttribute("aria-label", model.get("label"));
          for (const row of model.get("options")) {
            const option = document.createElement("option");
            option.value = row.value;
            option.textContent = row.label;
            option.disabled = row.disabled;
            select.appendChild(option);
          }
          select.value = model.get("value") ?? "";
          select.disabled = model.get("disabled");
          select.addEventListener("change", () => {
            model.set("value", select.value);
            model.save_changes();
          });
          label.appendChild(select);
          el.replaceChildren(label);
        }
        export default { render };
        """
        options = traitlets.List().tag(sync=True)
        value = traitlets.Unicode(allow_none=True, default_value=None).tag(sync=True)
        label = traitlets.Unicode().tag(sync=True)
        disabled = traitlets.Bool().tag(sync=True)

    # A dense spectrum (fine grid x several beam energies) can exceed Vega-Lite's
    # default 5000-row cap; vegafusion (shipped with marimo[recommended]) lifts it,
    # falling back to disabling the cap outright.
    try:
        alt.data_transformers.enable("vegafusion")
    except Exception:
        alt.data_transformers.disable_max_rows()

    from cxr_mc.config import default_settings, trajectory_sweep
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
    from cxr_mc.plots.altair_spectra import compare_spectrum_chart, spectrum_chart
    from cxr_mc.plots.altair_sweeps import (
        heatmap_chart,
        heatmap_select_chart,
        metric_vs_chart,
        scan_charts,
    )
    from cxr_mc.plots.altair_trajectories import (
        penetration_survival_chart,
        trajectory_chart,
    )
    from cxr_mc.plots.plotly_trajectories import (
        trajectory_volume_animation,
        trajectory_volume_data,
    )
    from cxr_mc.results import (
        filter_results,
        records,
        select_results,
        select_thickness,
        sweep_values,
        thicknesses_by_energy,
        top_geometries,
    )
    from cxr_mc.run import cases_from_results, load_checkpoint
    from cxr_mc.sweep import build_cases, fmt_thickness

    return (
        CATALOG,
        MaterialSelect,
        build_cases,
        cases_from_results,
        compare_spectrum_chart,
        default_settings,
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        filter_results,
        fmt_thickness,
        heatmap_chart,
        heatmap_select_chart,
        load_checkpoint,
        metric_vs_chart,
        mo,
        page_title,
        penetration_survival_chart,
        plot_best_spectra,
        plot_eaglexo_charge_map,
        plot_eaglexo_efficiency,
        plot_timepix_efficiency,
        records,
        scan_charts,
        select_results,
        select_thickness,
        spectrum_chart,
        style_sheet,
        sweep_values,
        thicknesses_by_energy,
        timepix_detected_chart,
        top_geometries,
        trajectory_chart,
        trajectory_sweep,
        trajectory_volume_animation,
        trajectory_volume_data,
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
            label="Material ",
            disabled=initial_selection is None,
        )
    )
    material_ui
    return (material_ui,)


@app.cell
def _(
    cases_from_results,
    default_settings,
    filter_results,
    load_checkpoint,
    material_ui,
):
    MATERIAL = material_ui.value["value"]
    settings = default_settings()
    _results = load_checkpoint(MATERIAL) if MATERIAL is not None else {}
    cases = cases_from_results(_results)  # rebuild the case list from the records
    res = filter_results(_results, cases)  # all loaded cases for this material
    return MATERIAL, cases, res, settings


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
def _(CATALOG, MATERIAL, fmt_thickness, mo):
    # The penetration figures run transport directly, so use the selected
    # material's configured scan grids instead of requiring a checkpoint.
    _scan = CATALOG.material(MATERIAL).scan
    _energy_values = tuple(float(value) for value in _scan.energy_keV)
    _thickness_values = tuple(float(value) for value in _scan.thickness_ang)
    _tilt_values = tuple(float(value) for value in _scan.tilt_deg)

    _source_options = {"Presets": "grid", "Custom": "manual"}
    penetration_energy_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    penetration_energy_grid_ui = mo.ui.dropdown(
        {f"{value:g} keV": value for value in _energy_values},
        value=f"{_energy_values[0]:g} keV",
        label="",
    )
    penetration_energy_manual_ui = mo.ui.number(
        start=1.0, stop=300.0, step=1.0, value=_energy_values[0], label="(keV)"
    )

    penetration_thickness_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    penetration_thickness_grid_ui = mo.ui.dropdown(
        {fmt_thickness(value): value for value in _thickness_values},
        value=fmt_thickness(_thickness_values[0]),
        label="",
    )
    penetration_thickness_manual_ui = mo.ui.number(
        start=0.001,
        stop=10000.0,
        step=0.001,
        value=min(max(_thickness_values[0] / 1e4, 0.001), 10000.0),
        label="(µm)",
    )

    penetration_tilt_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    if len(_tilt_values) > 1:
        default_tilt_ind = len(_tilt_values) // 2
        default_tilt_value = _tilt_values[default_tilt_ind]
    else:
        default_tilt_value = _tilt_values[0]

    penetration_tilt_grid_ui = mo.ui.dropdown(
        {f"{value:g} deg": value for value in _tilt_values},
        value=f"{default_tilt_value:g} deg",
        label="",
    )
    penetration_tilt_manual_ui = mo.ui.number(
        start=0.0, stop=89.9, step=0.1, value=_tilt_values[0], label="(deg)"
    )
    return (
        penetration_energy_grid_ui,
        penetration_energy_manual_ui,
        penetration_energy_source_ui,
        penetration_thickness_grid_ui,
        penetration_thickness_manual_ui,
        penetration_thickness_source_ui,
        penetration_tilt_grid_ui,
        penetration_tilt_manual_ui,
        penetration_tilt_source_ui,
    )


@app.cell
def _(
    penetration_energy_grid_ui,
    penetration_energy_manual_ui,
    penetration_energy_source_ui,
    penetration_thickness_grid_ui,
    penetration_thickness_manual_ui,
    penetration_thickness_source_ui,
    penetration_tilt_grid_ui,
    penetration_tilt_manual_ui,
    penetration_tilt_source_ui,
):
    penetration_energy_keV = (
        penetration_energy_grid_ui.value
        if penetration_energy_source_ui.value == "grid"
        else penetration_energy_manual_ui.value
    )
    penetration_thickness_ang = (
        penetration_thickness_grid_ui.value
        if penetration_thickness_source_ui.value == "grid"
        else penetration_thickness_manual_ui.value * 1e4
    )
    penetration_tilt_deg = (
        penetration_tilt_grid_ui.value
        if penetration_tilt_source_ui.value == "grid"
        else penetration_tilt_manual_ui.value
    )
    return (
        penetration_energy_keV,
        penetration_thickness_ang,
        penetration_tilt_deg,
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
def _(MATERIAL, mo, res, settings, top_geometries):
    def rankings_tab():
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
                _chart
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
                _chart
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
def _(
    cases,
    fmt_thickness,
    heatmap_chart,
    metric_vs_chart,
    mo,
    plot_best_spectra,
    records,
    res,
    scan_charts,
    scan_res_view,
    scan_thickness_ui,
    settings,
    sweep_values,
):
    def scans_tab():
        def _scan_charts_item():
            charts = scan_charts(scan_res_view, settings, cases=cases, line_metric="prominence")
            return mo.vstack(charts) if charts else mo.md("*No scan results.*")

        def _hit_frac_item():
            # Finite-crystal electron-hit map over the SAME polar x azimuth axes as
            # the scan heatmaps: fraction of launched electrons that landed on the
            # crystal footprint (bright = every electron hit, dark = all missed).
            # Colour pinned to [0, 1] so brightness reads as an absolute hit rate,
            # not this frame's max. NaN cells (pre-feature checkpoints that never
            # recorded it) drop out as gaps; the laterally infinite slab reads a
            # uniform 1.0 (nothing can miss an infinite crystal).
            chart = heatmap_chart(
                scan_res_view,
                settings,
                quantity="hit_frac",
                cases=cases,
                line_metric="prominence",
                color_domain=(0.0, 1.0),
            )
            return (
                chart
                if chart is not None
                else mo.md("*No electron-hit map — needs an azimuth × polar-tilt sweep.*")
            )

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
        return mo.vstack(
            [
                mo.md(
                    "`scan_charts` auto-picks a heatmap or line scans (one per swept quantity); "
                    "then 1-D metric scans vs polar tilt. Each section loads as it becomes "
                    "visible. The best-spectra panel loads on expand."
                ),
                _thk_widget,
                mo.lazy(_scan_charts_item, show_loading_indicator=True),
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
def _(mo):
    # Off by default: the fitted-window 3D view exaggerates lateral scale so the
    # cascade is legible. Checking this redraws the box at the true 5x5 mm crystal
    # footprint and overlays the beam entry footprint, which outgrows the crystal
    # at grazing tilt (1/cos stretch). The ~micron cascade then collapses toward
    # the origin -- expected at true scale.
    penetration_realistic_ui = mo.ui.checkbox(value=False, label="")
    return (penetration_realistic_ui,)


@app.cell
def _(mo):
    # Transported-dataset cache. Playback is now client-side (Plotly's own
    # frame animation inside the figure -- see trajectory_volume_animation),
    # so nothing here needs to WRITE this back reactively the way the old
    # frame counter did; this cache exists purely to avoid re-running Monte
    # Carlo transport when an UNRELATED control (e.g. a different tab) reruns
    # this cell but the transport parameters (case, Ne, seed, realistic,
    # beam_fwhm) haven't changed.
    get_penetration_data, set_penetration_data = mo.state(None)
    return get_penetration_data, set_penetration_data


@app.cell
def _(mo):
    # Survival-chart cache, same pattern as get_penetration_data above.
    # penetration_survival_chart(...) runs its OWN fresh Ne=500-electron
    # transport, so without caching it would redo that work every time the
    # tab body reruns for ANY reason -- e.g. toggling Realistic mode, which
    # the survival curve does not even depend on. Keyed on the smaller set of
    # knobs the chart itself actually depends on (material, beam energy,
    # polar tilt, thickness) in the tab body below. Altair charts are plain
    # objects, so caching/reusing one is safe.
    get_penetration_survival, set_penetration_survival = mo.state(None)
    return get_penetration_survival, set_penetration_survival


@app.cell
def _(mo):
    # Regenerate's click COUNT feeds the trajectory `seed`: every click draws a
    # fresh transport, and everything else re-renders with the SAME seed until
    # Regenerate is clicked again.
    penetration_regen_ui = mo.ui.button(
        value=0, on_click=lambda value: value + 1, label="🎲 Regenerate"
    )
    penetration_ne_ui = mo.ui.slider(
        start=25, stop=300, value=75, step=5, label="Electrons (Ne)", show_value=True
    )
    penetration_beam_fwhm_ui = mo.ui.number(
        value=1.0, start=0.001, step=0.1, label="Beam FWHM (mm)"
    )
    return penetration_beam_fwhm_ui, penetration_ne_ui, penetration_regen_ui


@app.cell
def _(mo):
    # Speed now drives the animation FIGURE's own frame duration
    # (trajectory_volume_animation's base_ms / speed), not a server-side
    # mo.ui.refresh tick -- Play/Pause/scrub all live inside the Plotly figure
    # itself now, so the old Play switch, Repeat switch, and refresh-ticker
    # cell are gone; see the animation builder's docstring for the duration
    # mapping and the loop-doesn't-exist caveat that replaces Repeat.
    penetration_speed_ui = mo.ui.slider(
        start=0.25, stop=4, value=1, step=0.25, label="Speed (×)", show_value=True
    )
    return (penetration_speed_ui,)


@app.cell
def _(
    MATERIAL,
    build_cases,
    get_penetration_data,
    get_penetration_survival,
    mo,
    penetration_beam_fwhm_ui,
    penetration_energy_grid_ui,
    penetration_energy_keV,
    penetration_energy_manual_ui,
    penetration_energy_source_ui,
    penetration_ne_ui,
    penetration_realistic_ui,
    penetration_regen_ui,
    penetration_speed_ui,
    penetration_survival_chart,
    penetration_thickness_ang,
    penetration_thickness_grid_ui,
    penetration_thickness_manual_ui,
    penetration_thickness_source_ui,
    penetration_tilt_deg,
    penetration_tilt_grid_ui,
    penetration_tilt_manual_ui,
    penetration_tilt_source_ui,
    set_penetration_data,
    set_penetration_survival,
    settings,
    trajectory_chart,
    trajectory_sweep,
    trajectory_volume_animation,
    trajectory_volume_data,
):
    def penetration_tab():
        _angle = penetration_tilt_deg
        _md = mo.md(
            "An interactive 3D electron track cutaway and a surviving-electron fraction vs depth, at the polar tilt selected "
            "in this tab. The translucent crystal's lateral extent is fitted to the tracks; depth "
            "and layer interfaces retain true scale. "
        )
        _sweep = trajectory_sweep(
            MATERIAL,
            energies=(penetration_energy_keV,),
            tilts=(penetration_tilt_deg,),
            thickness_ang=penetration_thickness_ang,
        )
        _traj = build_cases(_sweep, settings.n_electrons, settings.n_electrons_brem)
        if not _traj:
            return mo.vstack([_md, mo.md("*No trajectory cases.*")])
        # The sweep has one selected energy and tilt; keep the nearest-case guard
        # in case a future sweep adds a surrounding grid.
        _nc = min(_traj, key=lambda c: (abs(c["tilt_deg"] - _angle), c["E0_keV"]))

        # Survival-chart cache: penetration_survival_chart runs its OWN fresh
        # Ne=500-electron transport, so without caching it would redo that work
        # every time the tab body reruns for ANY reason -- e.g. toggling
        # Realistic mode, which the survival curve does not even depend on.
        # Keyed on just the knobs the chart itself depends on.
        _survival_key = (
            MATERIAL,
            penetration_energy_keV,
            penetration_tilt_deg,
            penetration_thickness_ang,
        )
        _cached_survival = get_penetration_survival()
        if _cached_survival is not None and _cached_survival[0] == _survival_key:
            _survival = _cached_survival[1]
        else:
            _survival = penetration_survival_chart(_traj, Ne=500, tilt=_angle)
            set_penetration_survival((_survival_key, _survival))

        _seed = int(penetration_regen_ui.value)
        _Ne = int(penetration_ne_ui.value)
        _beam_fwhm = float(penetration_beam_fwhm_ui.value)
        _realistic = penetration_realistic_ui.value

        # The transported dataset depends only on (case, Ne, seed, realistic,
        # beam_fwhm) -- NOT on playback state (playback is now client-side, see
        # trajectory_volume_animation). Cache the whole `data` dict in mo.state
        # keyed on that tuple so Monte Carlo transport runs once per parameter
        # set, not once per rerun of an unrelated control.
        _data_key = (
            _nc["name"],
            _nc["E0_keV"],
            _nc.get("tilt_deg"),
            _Ne,
            _seed,
            _realistic,
            _beam_fwhm,
        )
        _cached = get_penetration_data()
        if _cached is not None and _cached[0] == _data_key:
            _data = _cached[1]
        else:
            _data = trajectory_volume_data(
                _nc, Ne=_Ne, seed=_seed, realistic=_realistic, beam_fwhm_mm=_beam_fwhm
            )
            set_penetration_data((_data_key, _data))

        # Playback (Play/Pause, frame scrubbing) is native Plotly animation
        # running entirely in the browser -- see trajectory_volume_animation's
        # docstring. The Speed slider maps to the figure's own frame duration;
        # changing it rebuilds the figure once, server-side, same as any other
        # control here.
        _volume = trajectory_volume_animation(
            _nc,
            _data,
            realistic=_realistic,
            beam_fwhm_mm=_beam_fwhm,
            speed=float(penetration_speed_ui.value),
        )
        # Narrower than the old full-row default so the 2D cross-section fits
        # beside it instead of stacked below.
        if _volume is not None:
            _volume.update_layout(width=700)

        def _row_label(text):
            return mo.md(text).style({"min-width": "9rem", "display": "inline-block"})

        # Beam FWHM only matters in true-beamsize mode; gray it out (dim +
        # non-interactive) rather than hide it, so its value stays visible.
        _beam_fwhm_display = (
            penetration_beam_fwhm_ui
            if _realistic
            else penetration_beam_fwhm_ui.style({"opacity": "0.4", "pointer-events": "none"})
        )

        # Built eagerly (cheap at Ne=40); no longer behind a lazy accordion.
        # Narrower than its 480 default now that it sits beside the 3D plot
        # rather than filling the row on its own.
        _cross_section_chart = trajectory_chart(_nc, Ne=40, width=420)
        _cross_section_block = _cross_section_chart

        _trace_cols = [p for p in (_volume, _cross_section_block) if p is not None]
        _trace_row = (
            mo.hstack(_trace_cols, justify="start", align="start", gap=2, wrap=True)
            if _trace_cols
            else None
        )

        _parts = [
            _md,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Beam energy**"),
                                    penetration_energy_source_ui,
                                    (
                                        penetration_energy_grid_ui
                                        if penetration_energy_source_ui.value == "grid"
                                        else penetration_energy_manual_ui
                                    ),
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            mo.hstack(
                                [
                                    _row_label("True beamsize"),
                                    penetration_realistic_ui,
                                    _beam_fwhm_display,
                                ],
                                justify="end",
                                align="end",
                                gap=1,
                                wrap=True,
                            ),
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Crystal thickness**"),
                                    penetration_thickness_source_ui,
                                    (
                                        penetration_thickness_grid_ui
                                        if penetration_thickness_source_ui.value == "grid"
                                        else penetration_thickness_manual_ui
                                    ),
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            penetration_ne_ui,
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Polar tilt**"),
                                    penetration_tilt_source_ui,
                                    (
                                        penetration_tilt_grid_ui
                                        if penetration_tilt_source_ui.value == "grid"
                                        else penetration_tilt_manual_ui
                                    ),
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            penetration_speed_ui,
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    mo.hstack([penetration_regen_ui], justify="end", wrap=True),
                ]
            ),
            *(p for p in (_trace_row, _survival) if p is not None),
        ]
        return mo.vstack(_parts)

    return (penetration_tab,)


@app.cell
def _(CATALOG, mo):
    # Cross-material controls are top-level so switching them reruns the lazy
    # tab body. The selector contains only energies present in loaded data --
    # read from each material's checkpoint manifest, not load_checkpoint,
    # which would unpickle every material's checkpoint (140-225 MB gzip each)
    # just to enumerate beam energies.
    from cxr_mc.run import checkpoint_manifest

    _energies = set()
    for _material_key in CATALOG.material_keys:
        _manifest = checkpoint_manifest(_material_key)
        if _manifest:
            _energies.update(_manifest["energies_keV"])
    cross_material_energy_options = {f"{energy:g} keV": energy for energy in sorted(_energies)} or {
        "— no data —": None
    }
    compare_all_beam_energies_ui = mo.ui.checkbox(value=True, label="Compare all beam energies")
    return (
        checkpoint_manifest,
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
    checkpoint_manifest,
    compare_all_beam_energies_ui,
    cross_material_energy_ui,
    mo,
    settings,
):
    from cxr_mc.plots import draw_material_comparison, material_comparison_point
    from cxr_mc.run import cached_material_analysis

    def cross_material_tab():
        _md = mo.md(
            "For every material whose checkpoint exists, the single best geometry's "
            "dominant line: energy vs flux. The three comparisons select by line "
            "quality, peak flux, and local line-to-bremsstrahlung ratio. Each "
            "label reports the selected geometry's beam energy, \u03b8, and \u03c6. "
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
            if (checkpoint_manifest(_material_key) or {}).get("n_records", 0) > 0
        )

        def _comparison(select):
            _pts = []
            _dropped = []
            for _material_key in CATALOG.material_keys:
                _point = cached_material_analysis(
                    _material_key,
                    lambda _results: material_comparison_point(
                        _results,
                        settings,
                        select=select,
                        beam_energy_keV=_beam_energy,
                        min_line_quality=0.5,
                    ),
                    (select, _beam_energy, 0.5, settings.beam_current_na),
                )
                if _point is None:
                    continue
                _label = CATALOG.material(_material_key).label
                if _point == "dropped":
                    _dropped.append(_label)
                    continue
                _pts.append((_label, *_point))
            return draw_material_comparison(
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
    azim_compare_tab,
    cross_material_tab,
    detectors_tab,
    mo,
    penetration_tab,
    polar_compare_tab,
    rankings_tab,
    scans_tab,
    spectra_tab,
):
    # Four scientific tasks form the only top-level navigation. Accordions keep
    # individual view builders lazy without nesting tab widgets.
    def task_surface(views):
        return mo.accordion(views, lazy=True, multiple=True)

    mo.ui.tabs(
        {
            "Explore": lambda: task_surface(
                {
                    "Compare beam energies": spectra_tab,
                    "Compare polar angles": polar_compare_tab,
                    "Compare azimuths": azim_compare_tab,
                },
            ),
            "Optimize": lambda: task_surface(
                {
                    "Rank geometries": rankings_tab,
                    "Inspect scan maps": scans_tab,
                },
            ),
            "Instruments": detectors_tab,
            "Trace": penetration_tab,
            "Compare": lambda: task_surface({"Compare materials": cross_material_tab}),
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
