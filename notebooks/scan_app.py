# /// script
# [tool.marimo.display]
# theme = "dark"
# ///


import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    from _design import context_rail, directional_state, page_title, style_sheet

    from cxr_mc.config import (
        COLLAPSE_AZIMUTH,
        default_settings,
        format_penetration_watchdog_summary,
        gate_cases_by_penetration,
        material_sweep,
    )
    from cxr_mc.materials import CATALOG
    from cxr_mc.plots import stream_chunk
    from cxr_mc.run import checkpoint_path_for, load_checkpoint, run_sweep
    from cxr_mc.sweep import build_cases, geometry_table

    return (
        COLLAPSE_AZIMUTH,
        CATALOG,
        Path,
        build_cases,
        checkpoint_path_for,
        context_rail,
        default_settings,
        directional_state,
        format_penetration_watchdog_summary,
        gate_cases_by_penetration,
        geometry_table,
        material_sweep,
        mo,
        page_title,
        load_checkpoint,
        run_sweep,
        style_sheet,
        stream_chunk,
    )


@app.cell(hide_code=True)
def _(mo, page_title, style_sheet):
    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Bulk-crystal CXR scan",
                "Preview catalog-backed geometry, checkpoint state, and penetration exclusions before starting resumable Monte Carlo work.",
                eyebrow="Beamline control / scan",
            ),
        ]
    )
    return


@app.cell
def _(CATALOG, mo):
    material_options = {CATALOG.material(key).label: key for key in CATALOG.material_keys}
    material_ui = mo.ui.dropdown(
        material_options,
        value=CATALOG.material("hopg").label,
        label="Material to scan",
    )
    material_ui
    return (material_ui,)


@app.cell
def _(
    CATALOG,
    Path,
    build_cases,
    checkpoint_path_for,
    default_settings,
    format_penetration_watchdog_summary,
    gate_cases_by_penetration,
    load_checkpoint,
    material_sweep,
    material_ui,
):
    MATERIAL = material_ui.value

    settings = default_settings()
    sweep = material_sweep(MATERIAL)  # full parametric grid (data/materials.toml)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    cases, dropped = gate_cases_by_penetration(cases)
    penetration_summary = format_penetration_watchdog_summary(dropped)
    checkpoint_path = Path(checkpoint_path_for(MATERIAL))
    checkpoint_results = load_checkpoint(MATERIAL) if checkpoint_path.exists() else {}
    requested_pairs = {(case["name"], case["E0_keV"]) for case in cases}
    cached_pairs = {
        (name, energy)
        for name, energy_records in checkpoint_results.items()
        for energy in energy_records
    }
    cached_count = len(requested_pairs & cached_pairs)
    remaining_count = len(requested_pairs - cached_pairs)
    configuration_count = len({case["name"] for case in cases})
    material_label = CATALOG.material(MATERIAL).label
    return (
        MATERIAL,
        cached_count,
        cases,
        checkpoint_path,
        configuration_count,
        dropped,
        material_label,
        penetration_summary,
        remaining_count,
        settings,
    )


@app.cell(hide_code=True)
def _(
    MATERIAL,
    cached_count,
    cases,
    checkpoint_path,
    configuration_count,
    context_rail,
    directional_state,
    dropped,
    geometry_table,
    material_label,
    mo,
    penetration_summary,
    remaining_count,
):
    exclusion_detail = penetration_summary or "No cases excluded by penetration watchdog."
    preview = mo.vstack(
        [
            context_rail(
                mo,
                {
                    "Material": material_label,
                    "Checkpoint": "Cached" if checkpoint_path.exists() else "Ready",
                    "Cases": len(cases),
                    "Configs": configuration_count,
                    "Cached": cached_count,
                    "Remaining": remaining_count,
                },
            ),
            directional_state(
                mo,
                "Scan preview ready",
                f"Checkpoint: `{checkpoint_path}`. Changing material updates this preview only; no compute starts.",
                "Run scan" if cached_count == 0 else "Resume scan",
            ),
            mo.accordion(
                {
                    f"Penetration exclusions ({len(dropped)})": mo.md(exclusion_detail),
                    "Geometry preview": mo.lazy(lambda: geometry_table(cases)),
                },
            ),
        ]
    )
    preview
    return


@app.cell
def _(cached_count, mo):
    run_scan_ui = mo.ui.run_button(label="Resume scan" if cached_count else "Run scan")
    run_scan_ui
    return (run_scan_ui,)


@app.cell
def _(
    COLLAPSE_AZIMUTH,
    MATERIAL,
    cached_count,
    cases,
    checkpoint_path,
    directional_state,
    mo,
    remaining_count,
    run_scan_ui,
    run_sweep,
    settings,
    stream_chunk,
):
    mo.stop(
        not run_scan_ui.value,
        mo.callout(
            f"Ready. {cached_count} cached; {remaining_count} remaining. Scan starts only from the button above.",
            kind="info",
        ),
    )
    # Run (resumes from the checkpoint, skipping cached cases). The per-tilt
    # photon-counting tables stream live; all the figures are in the analysis app.
    results = {}
    try:
        with mo.status.spinner(
            title="Running checkpointed scan",
            subtitle=f"Writing `{checkpoint_path}` after each completed configuration group.",
        ):
            run_sweep(
                cases,
                results,
                on_chunk=lambda batch: stream_chunk(
                    results, batch, settings, collapse_azimuth=COLLAPSE_AZIMUTH
                ),
            )
    except EOFError as error:
        if remaining_count == 0:
            output = directional_state(
                mo,
                "Checkpoint already complete",
                f"All requested cases already exist in `{checkpoint_path}`.",
                f"cxr analyze {MATERIAL}",
            )
        else:
            output = mo.callout(f"Scan failed while reading checkpoint: `{error}`", kind="danger")
    else:
        output = directional_state(
            mo,
            "Scan complete",
            f"Results saved at `{checkpoint_path}`. Checkpoint remains resumable.",
            f"cxr analyze {MATERIAL}",
            kind="success",
        )
    output
    return


if __name__ == "__main__":
    app.run()
