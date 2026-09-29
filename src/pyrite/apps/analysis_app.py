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
        load_context,
        make_checkpoint_picker,
        resolve_axis_pair,
        select_emission,
        slice_results,
    )
    from pyrite.apps.analysis_ui.controls import (
        make_component_controls,
        make_detector_axes,
        make_detector_choice,
        make_map_quantity_control,
        make_slice_controls,
        make_spectra_axes,
        make_vary_control,
        make_varying_control,
    )
    from pyrite.apps.analysis_ui.interactive import is_heatmap_sweep, make_map_widget
    from pyrite.apps.analysis_ui.navigation import DETECTORS, MAP, SPECTRA, make_view_nav
    from pyrite.apps.analysis_ui.views import (
        SPECTRA_SPECS,
        render_detectors,
        render_map,
        render_map_trends,
        render_sidebar,
        render_spectra,
    )
    from pyrite.apps.analyze import (
        emission_menu,
        get_default_material,
        initial_material,
        material_menu,
        select_initial_material,
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
                    "Compare spectra, map geometry scans, and inspect detector response.",
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
    checkpoint_ui = make_checkpoint_picker(mo, material_ui.value["value"], DEFAULT_CHECKPOINT_DIR)
    return (checkpoint_ui,)


@app.cell
def _(checkpoint_ui, material_ui):
    base_context = load_context(material_ui, checkpoint_ui, DEFAULT_CHECKPOINT_DIR)
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


@app.cell
def _(base_context):
    # Built from the checkpoint, not the emission view, so toggling emission
    # keeps the slice. One name per dropdown: a view reruns only when a
    # dimension it reads changes.
    slice_controls = make_slice_controls(mo, base_context.checkpoint_results)
    energy_ui = slice_controls["energy"]
    tilt_ui = slice_controls["tilt"]
    azimuth_ui = slice_controls["azimuth"]
    thickness_ui = slice_controls["thickness"]
    return azimuth_ui, energy_ui, slice_controls, thickness_ui, tilt_ui


@app.cell(hide_code=True)
def _(base_context, checkpoint_ui, emission_ui, material_ui, slice_controls):
    render_sidebar(
        mo,
        material_ui=material_ui,
        checkpoint_ui=checkpoint_ui,
        emission_ui=emission_ui,
        slice_controls=slice_controls,
        results=base_context.checkpoint_results,
    )
    return


@app.cell
def _(context):
    no_data_notice = None
    if not context.has_data and context.notice is not None and context.load_error is None:
        # The material has checkpoints but no standard flat one: ask, never guess.
        no_data_notice = mo.callout(
            mo.md(f"**No checkpoint selected.** {context.notice}"), kind="info"
        )
    elif not context.has_data:
        detail = (
            context.load_error or "Run `pyrite run standard -m <material>` to create a checkpoint."
        )
        no_data_notice = mo.callout(
            mo.md(f"**No checkpoint data available.** {detail}"),
            kind="info",
        )
    no_data_notice
    return


@app.cell
def _():
    # No dependencies: the selected view survives material/emission changes.
    view_nav = make_view_nav(mo)
    view_nav
    return (view_nav,)


@app.cell
def _():
    # No dependencies: view-local choices survive view and checkpoint changes.
    vary_ui = make_vary_control(mo)
    spectrum_components_ui = make_component_controls(mo)
    map_quantity_ui = make_map_quantity_control(mo)
    detector_ui = make_detector_choice(mo)
    return detector_ui, map_quantity_ui, spectrum_components_ui, vary_ui


@app.cell
def _(view_nav):
    mo.stop(view_nav.value != SPECTRA)
    spectra_axes_controls = make_spectra_axes(mo)
    return (spectra_axes_controls,)


@app.cell
def _(base_context, vary_ui, view_nav):
    mo.stop(view_nav.value != SPECTRA)
    spectra_varying_ui = make_varying_control(
        mo,
        base_context.checkpoint_results,
        varying_key=vary_ui.value,
        unit=SPECTRA_SPECS[vary_ui.value].unit,
    )
    return (spectra_varying_ui,)


@app.cell
def _(
    app_theme,
    azimuth_ui,
    context,
    energy_ui,
    spectra_axes_controls,
    spectra_varying_ui,
    spectrum_components_ui,
    thickness_ui,
    tilt_ui,
    vary_ui,
    view_nav,
):
    mo.stop(not context.has_data or view_nav.value != SPECTRA)
    render_spectra(
        mo,
        context=context,
        slice_values={
            "energy": energy_ui.value,
            "tilt": tilt_ui.value,
            "azimuth": azimuth_ui.value,
            "thickness": thickness_ui.value,
        },
        spec=SPECTRA_SPECS[vary_ui.value],
        vary_ui=vary_ui,
        varying_ui=spectra_varying_ui,
        components=spectrum_components_ui,
        axes_controls=spectra_axes_controls,
        axes=resolve_axis_pair(spectra_axes_controls.value),
        theme=app_theme,
    )
    return


@app.cell
def _(app_theme, context, energy_ui, map_quantity_ui, thickness_ui, view_nav):
    mo.stop(not context.has_data or view_nav.value != MAP)
    map_results = slice_results(context.results, thickness=thickness_ui.value)
    heatmap_mode = is_heatmap_sweep(map_results)
    map_widget = (
        make_map_widget(
            mo,
            alt,
            map_results,
            context.settings,
            context.cases,
            quantity=map_quantity_ui.value,
            energy=energy_ui.value,
            theme=app_theme,
        )
        if heatmap_mode
        else None
    )
    return heatmap_mode, map_results, map_widget


@app.cell
def _(app_theme, context, map_results, view_nav):
    mo.stop(view_nav.value != MAP)
    map_trends = render_map_trends(mo, context=context, map_results=map_results, theme=app_theme)
    return (map_trends,)


@app.cell
def _(
    app_theme,
    context,
    energy_ui,
    heatmap_mode,
    map_quantity_ui,
    map_results,
    map_trends,
    map_widget,
    view_nav,
):
    mo.stop(view_nav.value != MAP)
    render_map(
        mo,
        context=context,
        map_results=map_results,
        heatmap_mode=heatmap_mode,
        quantity_ui=map_quantity_ui,
        map_widget=map_widget,
        selection=None if map_widget is None else map_widget.value,
        energy=energy_ui.value,
        trends=map_trends,
        theme=app_theme,
    )
    return


@app.cell
def _(view_nav):
    mo.stop(view_nav.value != DETECTORS)
    detector_axes_controls = make_detector_axes(mo)
    return (detector_axes_controls,)


@app.cell
def _(
    app_theme,
    azimuth_ui,
    context,
    detector_axes_controls,
    detector_ui,
    thickness_ui,
    tilt_ui,
    view_nav,
):
    mo.stop(not context.has_data or view_nav.value != DETECTORS)
    render_detectors(
        mo,
        context=context,
        detector_results=slice_results(
            context.results, azimuth=azimuth_ui.value, thickness=thickness_ui.value
        ),
        detector_ui=detector_ui,
        tilt=tilt_ui.value,
        axes_controls=detector_axes_controls,
        axes=resolve_axis_pair(detector_axes_controls.value),
        theme=app_theme,
    )
    return


@app.cell(hide_code=True)
def _():
    note = (
        "*3D trajectory and crystal-structure views live in `trace_app.py` "
        "(`marimo run src/pyrite/apps/trace_app.py`).*"
    )
    mo.vstack(
        [
            mo.md(note),
            mo.md(
                "*Pixel-detector observations: `pyrite app pixels launch`. Case baskets and "
                "cross-material comparison: `pyrite app compare launch`.*"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
