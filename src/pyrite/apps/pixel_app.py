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
    from pyrite.apps.analysis_ui import make_checkpoint_picker, selected_checkpoint_stem
    from pyrite.apps.analysis_ui.pickers import checkpoint_notice
    from pyrite.apps.analysis_ui.pixels import (
        discover_observations,
        load_selected_observation,
        make_observation_selector,
        make_pixel_image_controls,
        make_pixel_selection_controls,
        resolve_pixel_image,
    )
    from pyrite.apps.analysis_ui.views.pixels import render_pixel_detector
    from pyrite.apps.analyze import (
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
                    "Pixel Detector",
                    "Inspect stored counting observations: detector images, one pixel's "
                    "geometry, true spectra, and measured histogram.",
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
    checkpoint_ui = make_checkpoint_picker(mo, material_ui.value["value"], DEFAULT_CHECKPOINT_DIR)
    return (checkpoint_ui,)


@app.cell
def _(checkpoint_ui):
    # Observations are discovered per checkpoint stem; the checkpoint itself is never loaded.
    observation_stem = selected_checkpoint_stem(checkpoint_ui)
    pixel_inventory = discover_observations(observation_stem)
    observation_ui = make_observation_selector(mo, pixel_inventory)
    return observation_stem, observation_ui, pixel_inventory


@app.cell(hide_code=True)
def _(checkpoint_ui, material_ui, observation_stem):
    _parts = [mo.hstack([material_ui, checkpoint_ui], wrap=True)]
    if observation_stem is not None:
        _parts.append(mo.md(f"*Observations for checkpoint stem `{observation_stem}`.*"))
    else:
        _notice = checkpoint_notice(material_ui.value["value"], None, DEFAULT_CHECKPOINT_DIR)
        if _notice is not None:
            _parts.append(mo.callout(mo.md(_notice), kind="info"))
    mo.vstack(_parts)
    return


@app.cell
def _(observation_ui, pixel_inventory):
    pixel_state = load_selected_observation(pixel_inventory, observation_ui)
    pixel_image_controls = (
        None
        if pixel_state.observation is None
        else make_pixel_image_controls(mo, pixel_state.observation)
    )
    pixel_selection_controls = (
        None
        if pixel_state.observation is None
        else make_pixel_selection_controls(mo, pixel_state.observation)
    )
    return pixel_image_controls, pixel_selection_controls, pixel_state


@app.cell
def _(pixel_image_controls, pixel_state):
    # Memoized per observation and image settings, so a pixel change reuses the image.
    pixel_resolved = (
        None
        if pixel_state.observation is None
        else resolve_pixel_image(pixel_state.observation, pixel_image_controls.value)
    )
    return (pixel_resolved,)


@app.cell
def _(
    app_theme,
    observation_ui,
    pixel_image_controls,
    pixel_resolved,
    pixel_selection_controls,
    pixel_state,
):
    render_pixel_detector(
        mo,
        state=pixel_state,
        selector=observation_ui,
        image_controls=pixel_image_controls,
        selection_controls=pixel_selection_controls,
        resolved=pixel_resolved,
        theme=app_theme,
    )
    return


if __name__ == "__main__":
    app.run()
