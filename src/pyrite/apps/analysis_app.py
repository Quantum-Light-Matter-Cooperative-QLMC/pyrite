# /// script
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.24.0"
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
        make_case_axes,
        make_detector_controls,
        make_dimension_controls,
        make_energy_controls,
        make_heatmap_energy_control,
        make_scan_thickness_control,
    )
    from pyrite.apps.analysis_ui.interactive import make_heatmap_widget, make_scan_heatmap_widgets
    from pyrite.apps.analysis_ui.views import (
        CASE_BASKET_CAP,
        render_case_comparison,
        render_cross_material,
        render_detectors,
        render_dimension_comparison,
        render_energy_comparison,
        render_rankings,
        render_scans,
    )
    from pyrite.apps.analyze import (
        analysis_checkpoint_manifest,
        comparison_stem,
        emission_menu,
        face_menu,
        get_default_material,
        initial_material,
        material_menu,
        profile_menu,
        select_initial_material,
    )
    from pyrite.materials import CATALOG
    from pyrite.results import (
        case_label,
        case_table_rows,
        select_results,
        select_thickness,
        slim_case_record,
        sweep_values,
    )
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
                    "Explore spectra, optimize geometry, inspect instrument response, and compare materials.",
                    eyebrow="PyRITE - a Python toolkit for Radiation from Interaction and Transport of Electrons",
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
def _():
    get_case_basket, set_case_basket = mo.state([])
    return get_case_basket, set_case_basket


@app.cell
def _(context):
    case_picker_ui = mo.ui.table(
        case_table_rows(context.results),
        selection="multi",
        label="cases in this checkpoint",
    )
    return (case_picker_ui,)


@app.cell
def _(case_picker_ui, context, set_case_basket):
    def add_selected(click_count):
        selected = case_picker_ui.value
        if selected:
            varying = set(sweep_values(context.results))
            additions = []
            for row in selected:
                record = context.results[row["name"]][row["E0_keV"]]
                label = case_label(
                    record["case"],
                    material_label=context.selected_material,
                    face=context.selected_face,
                    varying=varying,
                )
                additions.append(
                    slim_case_record(
                        record,
                        material=context.selected_material,
                        label=label,
                        face=context.selected_face,
                    )
                )

            def update(old):
                merged = list(old)
                for entry in additions:
                    key = (
                        entry["case"].get("material"),
                        entry["case"].get("face"),
                        entry["case"].get("label"),
                    )
                    merged = [
                        existing
                        for existing in merged
                        if (
                            existing["case"].get("material"),
                            existing["case"].get("face"),
                            existing["case"].get("label"),
                        )
                        != key
                    ]
                    merged.append(entry)
                return merged[-CASE_BASKET_CAP:]

            set_case_basket(update)
        return click_count + 1

    case_add_ui = mo.ui.button(value=0, on_click=add_selected, label="Add selected cases")
    return (case_add_ui,)


@app.cell
def _(get_case_basket):
    _options = {
        f"{index}: {entry['case']['label']}": index for index, entry in enumerate(get_case_basket())
    }
    case_remove_select_ui = mo.ui.multiselect(_options, label="remove from basket")
    return (case_remove_select_ui,)


@app.cell
def _(case_remove_select_ui, set_case_basket):
    def remove_selected(click_count):
        indexes = set(case_remove_select_ui.value)
        if indexes:
            set_case_basket(
                lambda old: [entry for index, entry in enumerate(old) if index not in indexes]
            )
        return click_count + 1

    def clear_basket(click_count):
        set_case_basket(lambda _old: [])
        return click_count + 1

    case_remove_ui = mo.ui.button(value=0, on_click=remove_selected, label="Remove selected")
    case_clear_ui = mo.ui.button(value=0, on_click=clear_basket, label="Clear basket")
    return case_clear_ui, case_remove_ui


@app.cell
def _():
    case_controls = make_case_axes(mo)
    return (case_controls,)


@app.cell
def _():
    energies = set()
    for material_key in CATALOG.material_keys:
        manifest = analysis_checkpoint_manifest(material_key)
        if manifest:
            energies.update(manifest["energies_keV"])
    cross_material_energy_options = {f"{energy:g} keV": energy for energy in sorted(energies)} or {
        "— no data —": None
    }
    compare_all_energies_ui = mo.ui.checkbox(value=True, label="Compare all beam energies")
    return compare_all_energies_ui, cross_material_energy_options


@app.cell
def _(compare_all_energies_ui, cross_material_energy_options):
    cross_material_energy_ui = mo.ui.dropdown(
        cross_material_energy_options,
        value=next(iter(cross_material_energy_options)),
        label="beam energy",
        disabled=compare_all_energies_ui.value,
    )
    return (cross_material_energy_ui,)


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
    case_add_ui,
    case_clear_ui,
    case_controls,
    case_picker_ui,
    case_remove_select_ui,
    case_remove_ui,
    context,
    get_case_basket,
):
    basket = get_case_basket()
    _values = case_controls.value
    _axes = resolve_axis_pair(_values["axes"])

    def case_tab():
        return render_case_comparison(
            mo,
            basket=basket,
            picker=case_picker_ui,
            add_button=case_add_ui,
            remove_selector=case_remove_select_ui,
            remove_button=case_remove_ui,
            clear_button=case_clear_ui,
            controls=case_controls,
            values=_values,
            axes=_axes,
            settings=context.settings,
            theme=app_theme,
        )

    return (case_tab,)


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
def _(app_theme, compare_all_energies_ui, context, cross_material_energy_ui):
    compare_all = compare_all_energies_ui.value
    energy = cross_material_energy_ui.value

    def cross_material_tab():
        return render_cross_material(
            mo,
            settings=context.settings,
            analysis_checkpoint_manifest=analysis_checkpoint_manifest,
            comparison_stem=lambda material: comparison_stem(material, DEFAULT_CHECKPOINT_DIR),
            compare_all_ui=compare_all_energies_ui,
            energy_ui=cross_material_energy_ui,
            compare_all=compare_all,
            energy=energy,
            theme=app_theme,
        )

    return (cross_material_tab,)


@app.cell
def _(
    azimuth_tab,
    case_tab,
    context,
    cross_material_tab,
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
                        "Compare any cases": case_tab,
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
                "Compare": cross_material_tab,
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
        ]
    )
    return


if __name__ == "__main__":
    app.run()
