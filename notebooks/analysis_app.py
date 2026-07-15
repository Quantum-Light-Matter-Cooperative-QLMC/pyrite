# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import traitlets
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
    from cxr_mc.sweep import build_cases

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
            label="Material (match the scan you ran)",
            disabled=initial_selection is None,
        )
    )
    material_ui
    return (material_ui,)


@app.cell
def _(cases_from_results, default_settings, filter_results, load_checkpoint, material_ui):
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
def _(mo, records, res, sweep_values):
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
    _opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
    thickness_ui = mo.ui.dropdown(_opts, value=list(_opts)[-1], label="crystal thickness")
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
    # Geometry & scans thickness selector.
    _thk = sweep_values(res).get("thickness_ang", []) if records(res) else []
    _opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
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
def _(mo, records, res, sweep_values):
    # Detector tab selectors.
    _sv = sweep_values(res) if records(res) else {}
    _tilts = _sv.get("tilt_deg", [])
    _azim = _sv.get("tilt_azim_deg", [])
    _thk = _sv.get("thickness_ang", [])
    _tilt_opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    detector_tilt_ui = mo.ui.dropdown(_tilt_opts, value=next(iter(_tilt_opts)), label="polar tilt")

    _azim_opts = {f"{a:g} deg": a for a in _azim} or {"â€” no data â€”": None}
    detector_azim_ui = mo.ui.dropdown(_azim_opts, value=next(iter(_azim_opts)), label="azimuth")

    _thk_opts = {f"{t:g} Å ({t / 1e4:g} µm)": t for t in _thk} or {"— no data —": None}
    detector_thickness_ui = mo.ui.dropdown(
        _thk_opts, value=list(_thk_opts)[-1], label="crystal thickness"
    )

    detector_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV, 0 = auto)")
    detector_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV, 0 = auto)")
    detector_ymin_ui = mo.ui.number(value=0.0, label="narrow y-min (0 = auto)")
    detector_ymax_ui = mo.ui.number(value=0.0, label="narrow y-max (0 = auto)")
    detector_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    detector_ylog_ui = mo.ui.switch(value=True, label="narrow log y")

    detector_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV, 0 = auto)")
    detector_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV, 0 = auto)")
    detector_broad_ymin_ui = mo.ui.number(value=0.0, label="broad y-min (0 = auto)")
    detector_broad_ymax_ui = mo.ui.number(value=0.0, label="broad y-max (0 = auto)")
    detector_broad_xlog_ui = mo.ui.switch(value=True, label="broad log x")
    detector_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")
    return (
        detector_azim_ui,
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
    detector_broad_xmax_ui,
    detector_broad_xmin_ui,
    detector_broad_ymax_ui,
    detector_broad_ymin_ui,
    detector_xmax_ui,
    detector_xmin_ui,
    detector_ymax_ui,
    detector_ymin_ui,
):
    def _domain(xmin, xmax):
        if not xmin and not xmax:
            return None
        return (xmin if xmin else None, xmax if xmax else None)

    detector_x_domain = _domain(detector_xmin_ui.value, detector_xmax_ui.value)
    detector_y_domain = _domain(detector_ymin_ui.value, detector_ymax_ui.value)
    detector_broad_x_domain = _domain(detector_broad_xmin_ui.value, detector_broad_xmax_ui.value)
    detector_broad_y_domain = _domain(detector_broad_ymin_ui.value, detector_broad_ymax_ui.value)
    return (
        detector_broad_x_domain,
        detector_broad_y_domain,
        detector_x_domain,
        detector_y_domain,
    )


@app.cell
def _(CATALOG, MATERIAL, mo):
    # The penetration figures run transport directly, so use the selected
    # material's configured scan grids instead of requiring a checkpoint.
    _scan = CATALOG.material(MATERIAL).scan
    _energy_values = tuple(float(value) for value in _scan.energy_keV)
    _thickness_values = tuple(float(value) for value in _scan.thickness_ang)
    _tilt_values = tuple(float(value) for value in _scan.tilt_deg)

    _source_options = {"material scan grid": "grid", "manual entry": "manual"}
    penetration_energy_source_ui = mo.ui.dropdown(
        _source_options, value="grid", label="beam energy source"
    )
    penetration_energy_grid_ui = mo.ui.dropdown(
        {f"{value:g} keV": value for value in _energy_values},
        value=f"{_energy_values[0]:g} keV",
        label="beam energy",
    )
    penetration_energy_manual_ui = mo.ui.number(
        start=1.0, stop=300.0, step=1.0, value=_energy_values[0], label="beam energy (keV)"
    )

    penetration_thickness_source_ui = mo.ui.dropdown(
        _source_options, value="grid", label="crystal thickness source"
    )
    penetration_thickness_grid_ui = mo.ui.dropdown(
        {f"{value:g} Å ({value / 1e4:g} µm)": value for value in _thickness_values},
        value=f"{_thickness_values[0]:g} Å ({_thickness_values[0] / 1e4:g} µm)",
        label="crystal thickness",
    )
    penetration_thickness_manual_ui = mo.ui.number(
        start=0.001,
        stop=10.0,
        step=0.001,
        value=min(max(_thickness_values[0] / 1e7, 0.001), 10.0),
        label="crystal thickness (mm)",
    )

    penetration_tilt_source_ui = mo.ui.dropdown(
        _source_options, value="grid", label="polar tilt source"
    )
    penetration_tilt_grid_ui = mo.ui.dropdown(
        {f"{value:g} deg": value for value in _tilt_values},
        value=f"{_tilt_values[0]:g} deg",
        label="polar tilt",
    )
    penetration_tilt_manual_ui = mo.ui.number(
        start=0.0, stop=89.9, step=0.1, value=_tilt_values[0], label="polar tilt (deg)"
    )

    penetration_energy_keV = (
        penetration_energy_grid_ui.value
        if penetration_energy_source_ui.value == "grid"
        else penetration_energy_manual_ui.value
    )
    penetration_thickness_ang = (
        penetration_thickness_grid_ui.value
        if penetration_thickness_source_ui.value == "grid"
        else penetration_thickness_manual_ui.value * 1e7
    )
    penetration_tilt_deg = (
        penetration_tilt_grid_ui.value
        if penetration_tilt_source_ui.value == "grid"
        else penetration_tilt_manual_ui.value
    )
    return (
        penetration_energy_grid_ui,
        penetration_energy_keV,
        penetration_energy_manual_ui,
        penetration_energy_source_ui,
        penetration_thickness_ang,
        penetration_thickness_grid_ui,
        penetration_thickness_manual_ui,
        penetration_thickness_source_ui,
        penetration_tilt_deg,
        penetration_tilt_grid_ui,
        penetration_tilt_manual_ui,
        penetration_tilt_source_ui,
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
def _(mo, records, res, sweep_values):
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
    # Polar-angle comparison spectral controls.
    polar_brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    polar_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV, 0 = auto)")
    polar_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV, 0 = auto)")
    polar_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    polar_ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    polar_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV, 0 = auto)")
    polar_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV, 0 = auto)")
    polar_broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    polar_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")

    return (
        polar_brem_ui,
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
def _(polar_broad_xmax_ui, polar_broad_xmin_ui, polar_xmax_ui, polar_xmin_ui):
    def _domain(xmin, xmax):
        if not xmin and not xmax:
            return None
        return (xmin if xmin else None, xmax if xmax else None)

    polar_x_domain = _domain(polar_xmin_ui.value, polar_xmax_ui.value)
    polar_broad_x_domain = _domain(polar_broad_xmin_ui.value, polar_broad_xmax_ui.value)
    return polar_broad_x_domain, polar_x_domain


@app.cell
def _(mo):
    # Azimuthal comparison spectral controls.
    azim_brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    azim_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV, 0 = auto)")
    azim_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV, 0 = auto)")
    azim_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    azim_ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    azim_broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV, 0 = auto)")
    azim_broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV, 0 = auto)")
    azim_broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    azim_broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")

    return (
        azim_brem_ui,
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
def _(azim_broad_xmax_ui, azim_broad_xmin_ui, azim_xmax_ui, azim_xmin_ui):
    def _domain(xmin, xmax):
        if not xmin and not xmax:
            return None
        return (xmin if xmin else None, xmax if xmax else None)

    azim_x_domain = _domain(azim_xmin_ui.value, azim_xmax_ui.value)
    azim_broad_x_domain = _domain(azim_broad_xmin_ui.value, azim_broad_xmax_ui.value)
    return azim_broad_x_domain, azim_x_domain


@app.cell
def _(mo):
    # Energy-comparison spectral controls. Created top-level for reactivity,
    # rendered only inside the Energy comparison tab.
    brem_ui = mo.ui.checkbox(value=True, label="show brem background")

    narrow_xmin_ui = mo.ui.number(value=0.0, label="narrow x-min (eV, 0 = auto)")
    narrow_xmax_ui = mo.ui.number(value=0.0, label="narrow x-max (eV, 0 = auto)")
    narrow_xlog_ui = mo.ui.switch(value=False, label="narrow log x")
    ylog_ui = mo.ui.switch(value=False, label="narrow log y")

    broad_xmin_ui = mo.ui.number(value=0.0, label="broad x-min (eV, 0 = auto)")
    broad_xmax_ui = mo.ui.number(value=0.0, label="broad x-max (eV, 0 = auto)")
    broad_xlog_ui = mo.ui.switch(value=False, label="broad log x")
    broad_ylog_ui = mo.ui.switch(value=True, label="broad log y")

    return (
        brem_ui,
        broad_xlog_ui,
        broad_xmax_ui,
        broad_xmin_ui,
        broad_ylog_ui,
        narrow_xlog_ui,
        narrow_xmax_ui,
        narrow_xmin_ui,
        ylog_ui,
    )


@app.cell
def _(broad_xmax_ui, broad_xmin_ui, narrow_xmax_ui, narrow_xmin_ui):
    # (0, 0) -> None (autoscale); otherwise an explicit (lo, hi) domain.
    def _domain(xmin, xmax):
        if not xmin and not xmax:
            return None
        return (xmin if xmin else None, xmax if xmax else None)

    x_domain = _domain(narrow_xmin_ui.value, narrow_xmax_ui.value)
    broad_x_domain = _domain(broad_xmin_ui.value, broad_xmax_ui.value)
    return broad_x_domain, x_domain


@app.cell
def _(
    MATERIAL,
    mo,
    res,
    settings,
    top_geometries,
):
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
    broad_x_domain,
    broad_xlog_ui,
    broad_xmax_ui,
    broad_xmin_ui,
    broad_ylog_ui,
    heatmap_E0_ui,
    heatmap_select,
    mo,
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
            "The INTRINSIC coherent CXR line spectrum at the selected polar tilt, "
            "one line per beam energy."
        )
        _sv = sweep_values(res) if records(res) else {}
        _thk_widget = (
            thickness_ui
            if len(_sv.get("thickness_ang", [])) > 1
            else mo.md(
                f"crystal thickness: {thickness_ui.value:g} Å"
                if thickness_ui.value is not None
                else ""
            )
        )
        _controls = mo.vstack(
            [
                mo.hstack([tilt_ui, _thk_widget, brem_ui]),
                mo.hstack([narrow_xmin_ui, narrow_xmax_ui, narrow_xlog_ui, ylog_ui]),
                mo.hstack([broad_xmin_ui, broad_xmax_ui, broad_xlog_ui, broad_ylog_ui]),
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
        return mo.vstack(
            [
                _md,
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

    return (spectra_tab,)


@app.cell
def _(
    compare_spectrum_chart,
    mo,
    polar_E0_ui,
    polar_azim_ui,
    polar_brem_ui,
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
            "The INTRINSIC coherent CXR line spectrum, one line per POLAR TILT, at "
            "a pinned beam energy / azimuth / thickness. Pick a few tilts below to "
            "compare."
        )
        _spectral_controls = mo.vstack(
            [
                polar_brem_ui,
                mo.hstack([polar_xmin_ui, polar_xmax_ui, polar_xlog_ui, polar_ylog_ui]),
                mo.hstack(
                    [
                        polar_broad_xmin_ui,
                        polar_broad_xmax_ui,
                        polar_broad_xlog_ui,
                        polar_broad_ylog_ui,
                    ]
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

        _parts = [_md, _e0_widget, _azim_widget, _thk_widget, _spectral_controls, polar_tilts_ui]
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
    azim_azims_ui,
    azim_brem_ui,
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
            "The INTRINSIC coherent CXR line spectrum, one line per AZIMUTH, at "
            "a pinned beam energy / polar tilt / thickness. Pick a few azimuths "
            "below to compare."
        )
        _spectral_controls = mo.vstack(
            [
                azim_brem_ui,
                mo.hstack([azim_xmin_ui, azim_xmax_ui, azim_xlog_ui, azim_ylog_ui]),
                mo.hstack(
                    [
                        azim_broad_xmin_ui,
                        azim_broad_xmax_ui,
                        azim_broad_xlog_ui,
                        azim_broad_ylog_ui,
                    ]
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

        _parts = [_md, _e0_widget, _tilt_widget, _thk_widget, _spectral_controls, azim_azims_ui]
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
                f"crystal thickness: {scan_thickness_ui.value:g} Å"
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
    detector_azim_ui,
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
                f"crystal thickness: {detector_thickness_ui.value:g} Å"
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
                mo.hstack([detector_tilt_ui, _azim_widget, _thk_widget]),
                mo.hstack(
                    [
                        detector_xmin_ui,
                        detector_xmax_ui,
                        detector_ymin_ui,
                        detector_ymax_ui,
                        detector_xlog_ui,
                        detector_ylog_ui,
                    ]
                ),
                mo.hstack(
                    [
                        detector_broad_xmin_ui,
                        detector_broad_xmax_ui,
                        detector_broad_ymin_ui,
                        detector_broad_ymax_ui,
                        detector_broad_xlog_ui,
                        detector_broad_ylog_ui,
                    ]
                ),
            ]
        )

        def _eaglexo_inner():
            _md = mo.md(
                "Raptor Eagle XO direct-detection CCD (`solid_angle x QE(E)`): soft PXR "
                "lines pass at ~90% QE, hard brem is crushed by the thin sensor. "
                "Photon density (detected vs incident), then recorded-charge density."
            )
            _detected = eaglexo_detected_chart(
                detector_res_view,
                settings,
                tilt_deg=detector_tilt_ui.value,
                x_domain=detector_x_domain,
            )
            _charge = eaglexo_charge_chart(
                detector_res_view,
                settings,
                tilt_deg=detector_tilt_ui.value,
                x_domain=detector_x_domain,
            )
            _parts = [_md, *(c for c in (_detected, _charge) if c is not None)]
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
                detector_res_view,
                settings,
                tilt_deg=detector_tilt_ui.value,
                x_domain=detector_x_domain,
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

        return mo.vstack(
            [
                _controls,
                mo.accordion({"Eagle XO": _eaglexo_inner, "Timepix3": _timepix_inner}, lazy=True),
            ]
        )

    return (detectors_tab,)


@app.cell
def _(
    MATERIAL,
    build_cases,
    mo,
    penetration_energy_grid_ui,
    penetration_energy_keV,
    penetration_energy_manual_ui,
    penetration_energy_source_ui,
    penetration_thickness_ang,
    penetration_thickness_grid_ui,
    penetration_thickness_manual_ui,
    penetration_thickness_source_ui,
    penetration_tilt_deg,
    penetration_tilt_grid_ui,
    penetration_tilt_manual_ui,
    penetration_tilt_source_ui,
    penetration_survival_chart,
    plot_trajectory_grid,
    settings,
    trajectory_chart,
    trajectory_sweep,
):
    def penetration_tab():
        _angle = penetration_tilt_deg
        _md = mo.md(
            "Surviving-electron fraction vs depth (one curve per beam energy) and "
            "an interactive low-Ne track cross-section, at the polar tilt selected "
            "in this tab. These run the cheap CPU-only "
            "transport directly — no checkpoint needed. For a stacked/multilayer "
            "material (e.g. mos2 on sapphire) the cascade is transported through "
            "the FULL stack, not just the top film. Manual beam energies above 30 keV "
            "are exploratory because the free-path fit is documented through 30 keV. "
            "Dense grid loads on expand."
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
        _survival = penetration_survival_chart(_traj, Ne=500, tilt=_angle)
        # The sweep has one selected energy and tilt; keep the nearest-case guard
        # in case a future sweep adds a surrounding grid.
        _nc = min(_traj, key=lambda c: (abs(c["tilt_deg"] - _angle), c["E0_keV"]))
        _track = trajectory_chart(_nc, Ne=40)
        _parts = [
            _md,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            penetration_energy_source_ui,
                            (
                                penetration_energy_grid_ui
                                if penetration_energy_source_ui.value == "grid"
                                else penetration_energy_manual_ui
                            ),
                        ]
                    ),
                    mo.hstack(
                        [
                            penetration_thickness_source_ui,
                            (
                                penetration_thickness_grid_ui
                                if penetration_thickness_source_ui.value == "grid"
                                else penetration_thickness_manual_ui
                            ),
                        ]
                    ),
                    mo.hstack(
                        [
                            penetration_tilt_source_ui,
                            (
                                penetration_tilt_grid_ui
                                if penetration_tilt_source_ui.value == "grid"
                                else penetration_tilt_manual_ui
                            ),
                        ]
                    ),
                ]
            ),
            *(p for p in (_survival, _track) if p is not None),
        ]

        def _dense_grid():
            _dense_sweep = trajectory_sweep(
                MATERIAL,
                energies=(penetration_energy_keV,),
                tilts=(penetration_tilt_deg,),
                thickness_ang=penetration_thickness_ang,
            )
            _dense_traj = build_cases(_dense_sweep, settings.n_electrons, settings.n_electrons_brem)
            return plot_trajectory_grid(_dense_traj, energy=penetration_energy_keV, Ne=120)

        _parts.append(
            mo.accordion(
                {"Dense penetration grid (datashader, matplotlib)": _dense_grid},
                lazy=True,
            )
        )
        return mo.vstack(_parts)

    return (penetration_tab,)


@app.cell
def _(
    CATALOG,
    load_checkpoint,
    mo,
    plot_material_comparison,
    settings,
):
    def cross_material_tab():
        _md = mo.md(
            "For every material whose checkpoint exists, the single best geometry's "
            "dominant line: energy vs flux. The three comparisons select by line "
            "quality, peak flux, and local line-to-bremsstrahlung ratio; the two "
            "diagnostic selections apply a 100 eV line-energy floor."
        )
        _by_material = {}
        for _material_key in CATALOG.material_keys:
            _r = load_checkpoint(_material_key)
            if _r:
                _by_material[CATALOG.material(_material_key).label] = _r
        if len(_by_material) >= 2:
            return mo.vstack(
                [
                    _md,
                    plot_material_comparison(_by_material, settings, select="quality_peak"),
                    plot_material_comparison(
                        _by_material, settings, select="peak", min_line_eV=100.0
                    ),
                    plot_material_comparison(
                        _by_material, settings, select="line_brem_ratio", min_line_eV=100.0
                    ),
                ]
            )
        return mo.vstack(
            [_md, mo.md("*Run `scan_app.py` for more materials to populate this comparison.*")]
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
    # Each tab value is a zero-arg builder closure from its own cell, so
    # `lazy=True` still defers chart work while each builder closes over
    # only the widgets and data it actually reads.
    mo.ui.tabs(
        {
            "Energy comparison": spectra_tab,
            "Polar-angle comparison": polar_compare_tab,
            "Azimuthal comparison": azim_compare_tab,
            "Top geometries": rankings_tab,
            "Geometry & scans": scans_tab,
            "Detectors": detectors_tab,
            "Penetration": penetration_tab,
            "Cross-material": cross_material_tab,
        },
        lazy=True,
    )
    return


@app.cell
def _(mo, records, res, sweep_values):
    # What's actually in this checkpoint -- swept knobs and their values.
    mo.accordion(
        {
            "Checkpoint contents (swept knobs & values)": (
                sweep_values(res) if records(res) else mo.md("*(load a checkpoint above)*")
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
