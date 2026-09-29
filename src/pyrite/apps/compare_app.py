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
    from pyrite.apps.analysis_ui import load_context, resolve_axis_pair, select_emission
    from pyrite.apps.analysis_ui.controls import make_case_axes
    from pyrite.apps.analysis_ui.views.cases import CASE_BASKET_CAP, render_case_comparison
    from pyrite.apps.analysis_ui.views.materials import render_cross_material
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
    from pyrite.campaign.config import default_settings
    from pyrite.materials import CATALOG
    from pyrite.results import case_label, case_table_rows, slim_case_record, sweep_values
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
                    "Compare",
                    "Compare hand-picked cases across checkpoints, and the best lines of every "
                    "material with data.",
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
def _(get_case_basket):
    case_controls = make_case_axes(mo, get_case_basket())
    return (case_controls,)


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
    emission_ui,
    face_ui,
    get_case_basket,
    material_ui,
    profile_ui,
):
    basket = get_case_basket()
    _values = case_controls.value
    _axes = resolve_axis_pair(_values["axes"])

    def case_tab():
        parts = [mo.hstack([material_ui, face_ui, profile_ui, emission_ui], wrap=True)]
        if context.selected_profile not in (None, context.selected_material):
            parts.append(
                mo.md(
                    "*This profile selects a checkpoint directly; the face selector does not "
                    "change the loaded checkpoint.*"
                )
            )
        if context.load_error is not None:
            parts.append(mo.callout(mo.md(context.load_error), kind="warn"))
        parts.append(
            render_case_comparison(
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
        )
        return mo.vstack(parts)

    return (case_tab,)


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
def _(app_theme, compare_all_energies_ui, cross_material_energy_ui):
    # Independent of the case picker: every catalog material's checkpoint is read.
    compare_all = compare_all_energies_ui.value
    energy = cross_material_energy_ui.value
    cross_material_settings = default_settings()

    def cross_material_tab():
        return render_cross_material(
            mo,
            settings=cross_material_settings,
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
def _(case_tab, cross_material_tab):
    mo.ui.tabs(
        {
            "Compare any cases": case_tab,
            "Compare materials": cross_material_tab,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
