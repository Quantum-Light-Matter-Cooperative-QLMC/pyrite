# /// script
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")

with app.setup:
    import altair as alt
    import marimo as mo

    from pyrite.apps._design import (
        configure_matplotlib_theme,
        page_title,
        resolved_theme,
        style_sheet,
        theme_switch,
    )
    from pyrite.apps._widgets import MaterialSelect
    from pyrite.apps.analysis_ui import (
        DimensionComparisonSpec,
        load_context,
        resolve_axis_pair,
        select_emission,
    )
    from pyrite.apps.analysis_ui.controls import (
        make_detector_controls,
        make_dimension_controls,
        make_energy_controls,
        make_heatmap_energy_control,
        make_scan_thickness_control,
    )
    from pyrite.apps.analysis_ui.interactive import make_heatmap_widget, make_scan_heatmap_widgets
    from pyrite.apps.analysis_ui.views import (
        render_detectors,
        render_dimension_comparison,
        render_energy_comparison,
        render_rankings,
        render_scans,
    )
    from pyrite.apps.analyze import (
        emission_menu,
        face_menu,
        get_default_material,
        initial_material,
        material_menu,
        profile_menu,
        select_initial_material,
    )
    from pyrite.results import select_results, select_thickness
    from pyrite.runs.run import DEFAULT_CHECKPOINT_DIR

    try:
        alt.data_transformers.enable("vegafusion")
        vegafusion_error = None
    except Exception as exc:
        alt.data_transformers.disable_max_rows()
        vegafusion_error = str(exc)


@app.cell(hide_code=True)
def _():
    theme_ui = theme_switch(mo)
    header = [
        style_sheet(mo),
        mo.hstack(
            [
                page_title(
                    mo,
                    "Spectral Analysis",
                    "Explore spectra, optimize geometry, and inspect instrument response.",
                    eyebrow="PyRITE",
                ),
                theme_ui,
            ],
            justify="space-between",
            align="start",
            wrap=True,
        ),
    ]
    if vegafusion_error is not None:
        header.append(
            mo.callout(
                mo.md(
                    "Vegafusion could not be enabled; Altair's row limit was disabled instead. "
                    f"`{vegafusion_error}`"
                ),
                kind="warn",
            )
        )
    mo.vstack(header)
    return (theme_ui,)


@app.cell
def _(theme_ui):
    app_theme = resolved_theme(theme_ui)
    _ = configure_matplotlib_theme(app_theme)
    return (app_theme,)


@app.cell
def _():
    requested_material = initial_material(mo.cli_args(), get_default_material())
    _options = material_menu(DEFAULT_CHECKPOINT_DIR)
    _initial_selection = select_initial_material(requested_material, _options)
    material_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(_options),
            value=_initial_selection,
            label="Material",
            disabled=_initial_selection is None,
        )
    )
    return (material_ui,)


@app.cell
def _(material_ui):
    _material = material_ui.value["value"]
    _options = face_menu(_material, DEFAULT_CHECKPOINT_DIR) if _material else ()
    _initial_selection = select_initial_material(None, _options)
    face_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(_options),
            value=_initial_selection,
            label="Face",
            disabled=_initial_selection is None,
        )
    )
    return (face_ui,)


@app.cell
def _(material_ui):
    _material = material_ui.value["value"]
    _options = profile_menu(_material, DEFAULT_CHECKPOINT_DIR) if _material else ()
    _initial_selection = select_initial_material(None, _options)
    profile_ui = mo.ui.anywidget(
        MaterialSelect(
            options=list(_options),
            value=_initial_selection,
            label="Profile",
            disabled=_initial_selection is None,
        )
    )
    return (profile_ui,)


@app.cell
def _(face_ui, material_ui, profile_ui):
    base_context = load_context(material_ui, face_ui, profile_ui)
    return (base_context,)


@app.cell
def _(base_context):
    rows = emission_menu(base_context.checkpoint_results)
    _options = {row["label"]: row["value"] for row in rows if not row["disabled"]} or {
        "Incoherent": "incoherent"
    }
    emission_ui = mo.ui.radio(
        options=_options,
        value=next(iter(_options)),
        label="Emission",
        inline=True,
    )
    return (emission_ui,)


@app.cell
def _(base_context, emission_ui):
    context = select_emission(base_context, emission_ui.value)
    return (context,)


@app.cell(hide_code=True)
def _(context, emission_ui, face_ui, material_ui, profile_ui):
    parts = [
        mo.hstack(
            [material_ui, face_ui, profile_ui, emission_ui],
            wrap=True,
        )
    ]
    if context.selected_profile not in (None, context.selected_material):
        parts.append(
            mo.md(
                "*This profile selects a checkpoint directly; the face selector does not "
                "change the loaded checkpoint.*"
            )
        )
    if context.load_error is not None:
        parts.append(mo.callout(mo.md(context.load_error), kind="warn"))
    mo.vstack(parts)
    return


@app.cell
def _(context):
    energy_controls = make_energy_controls(mo, context.results)
    return (energy_controls,)


@app.cell
def _(context, energy_controls):
    energy_values = energy_controls.value
    pinned_results = (
        select_thickness(context.results, energy_values["thickness"])
        if energy_values["thickness"] is not None
        else context.results
    )
    return energy_values, pinned_results


@app.cell
def _(pinned_results):
    heatmap_energy_ui = make_heatmap_energy_control(mo, pinned_results)
    return (heatmap_energy_ui,)


@app.cell
def _(app_theme, context, heatmap_energy_ui, pinned_results):
    heatmap_widget = make_heatmap_widget(
        mo,
        alt,
        pinned_results,
        context.settings,
        energy=heatmap_energy_ui.value,
        theme=app_theme,
    )
    return (heatmap_widget,)


@app.cell
def _(context):
    polar_controls = make_dimension_controls(mo, context.results, varying_key="tilt_deg")
    azimuth_controls = make_dimension_controls(
        mo,
        context.results,
        varying_key="tilt_azim_deg",
    )
    return azimuth_controls, polar_controls


@app.cell
def _(context):
    scan_thickness_ui = make_scan_thickness_control(mo, context.results)
    return (scan_thickness_ui,)


@app.cell
def _(context, scan_thickness_ui):
    scan_results = (
        select_results(context.results, thickness_ang=scan_thickness_ui.value)
        if scan_thickness_ui.value is not None
        else context.results
    )
    return (scan_results,)


@app.cell
def _(scan_results):
    scan_heatmap_energy_ui = make_heatmap_energy_control(mo, scan_results)
    return (scan_heatmap_energy_ui,)


@app.cell
def _(app_theme, context, scan_heatmap_energy_ui, scan_results):
    scan_heatmap_widgets = make_scan_heatmap_widgets(
        mo,
        alt,
        scan_results,
        context.settings,
        context.cases,
        energy=scan_heatmap_energy_ui.value,
        theme=app_theme,
    )
    return (scan_heatmap_widgets,)


@app.cell
def _(context):
    detector_controls = make_detector_controls(mo, context.results)
    return (detector_controls,)


@app.cell
def _(context, detector_controls):
    detector_values = detector_controls.value
    constraints = {
        "thickness_ang": detector_values["thickness"],
        "tilt_azim_deg": detector_values["azimuth"],
    }
    constraints = {key: value for key, value in constraints.items() if value is not None}
    detector_results = (
        select_results(context.results, **constraints) if constraints else context.results
    )
    return detector_results, detector_values


@app.cell
def _(
    app_theme,
    context,
    energy_controls,
    energy_values,
    heatmap_energy_ui,
    heatmap_widget,
    pinned_results,
):
    _axes = resolve_axis_pair(energy_values["axes"])
    heatmap_selection = None if heatmap_widget is None else heatmap_widget.value

    def energy_tab():
        return render_energy_comparison(
            mo,
            context=context,
            pinned_results=pinned_results,
            controls=energy_controls,
            values=energy_values,
            axes=_axes,
            heatmap_energy_ui=heatmap_energy_ui,
            heatmap_widget=heatmap_widget,
            heatmap_selection=heatmap_selection,
            theme=app_theme,
        )

    return (energy_tab,)


@app.cell
def _(app_theme, context, polar_controls):
    _values = polar_controls.value
    _axes = resolve_axis_pair(_values["axes"])
    _specification = DimensionComparisonSpec(
        varying_key="tilt_deg",
        varying_label="polar tilt",
        varying_plural="polar angles",
        pinned_angle_key="tilt_azim_deg",
        pinned_angle_label="azimuth",
        description=(
            "Coherent CXR line spectra at pinned beam energy, azimuth, and thickness. "
            "Polar tilt varies across selected curves."
        ),
    )

    def polar_tab():
        return render_dimension_comparison(
            mo,
            context=context,
            controls=polar_controls,
            values=_values,
            axes=_axes,
            spec=_specification,
            theme=app_theme,
        )

    return (polar_tab,)


@app.cell
def _(app_theme, azimuth_controls, context):
    _values = azimuth_controls.value
    _axes = resolve_axis_pair(_values["axes"])
    _specification = DimensionComparisonSpec(
        varying_key="tilt_azim_deg",
        varying_label="azimuth",
        varying_plural="azimuths",
        pinned_angle_key="tilt_deg",
        pinned_angle_label="polar tilt",
        description=(
            "Coherent CXR line spectra at pinned beam energy, polar tilt, and thickness. "
            "Azimuth varies across selected curves."
        ),
    )

    def azimuth_tab():
        return render_dimension_comparison(
            mo,
            context=context,
            controls=azimuth_controls,
            values=_values,
            axes=_axes,
            spec=_specification,
            theme=app_theme,
        )

    return (azimuth_tab,)


@app.cell
def _(
    app_theme,
    context,
    scan_heatmap_energy_ui,
    scan_heatmap_widgets,
    scan_results,
    scan_thickness_ui,
):
    selections = {
        key: None if widget is None else widget.value
        for key, widget in scan_heatmap_widgets.items()
    }

    def scans_tab():
        return render_scans(
            mo,
            context=context,
            scan_results=scan_results,
            thickness_ui=scan_thickness_ui,
            heatmap_energy_ui=scan_heatmap_energy_ui,
            heatmap_widgets=scan_heatmap_widgets,
            heatmap_selections=selections,
            theme=app_theme,
        )

    def rankings_tab():
        return render_rankings(mo, context=context)

    return rankings_tab, scans_tab


@app.cell
def _(
    app_theme,
    context,
    detector_controls,
    detector_results,
    detector_values,
):
    _axes = resolve_axis_pair(detector_values["axes"], include_y_domain=True)

    def detector_tab():
        return render_detectors(
            mo,
            context=context,
            detector_results=detector_results,
            controls=detector_controls,
            values=detector_values,
            axes=_axes,
            theme=app_theme,
        )

    return (detector_tab,)


@app.cell
def _(
    azimuth_tab,
    context,
    detector_tab,
    energy_tab,
    polar_tab,
    rankings_tab,
    scans_tab,
):
    if not context.has_data:
        detail = (
            context.load_error or "Run `pyrite run standard -m <material>` to create a checkpoint."
        )
        view = mo.callout(
            mo.md(f"**No checkpoint data available.** {detail}"),
            kind="info",
        )
    else:
        view = mo.ui.tabs(
            {
                "Explore": lambda: mo.accordion(
                    {
                        "Compare beam energies": energy_tab,
                        "Compare polar angles": polar_tab,
                        "Compare azimuths": azimuth_tab,
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
                "Instruments": detector_tab,
            },
            lazy=True,
        )

    mo.vstack(
        [
            view,
            mo.md(
                "*3D trajectory and crystal-structure views live in `trace_app.py` "
                "(`marimo run src/pyrite/apps/trace_app.py`).*"
            ),
            mo.md(
                "*Pixel-detector observations: `pyrite app pixels launch`. Case baskets and "
                "cross-material comparison: `pyrite app compare launch`.*"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
