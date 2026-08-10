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

    from cxr_mc.apps._design import (
        context_rail,
        directional_state,
        page_title,
        scan_grid,
        style_sheet,
    )
    from cxr_mc.campaign.config import (
        COLLAPSE_AZIMUTH,
        default_settings,
        format_penetration_watchdog_summary,
        gate_cases_by_penetration,
        material_sweep,
    )
    from cxr_mc.campaign.sweep import (
        build_cases,
        case_cost,
        geometry_table,
        scan_grid_rows,
        sweep_cost_weights,
    )
    from cxr_mc.materials import CATALOG
    from cxr_mc.plots import stream_chunk
    from cxr_mc.runs.run import checkpoint_path_for, load_checkpoint, run_sweep

    return (
        COLLAPSE_AZIMUTH,
        CATALOG,
        Path,
        build_cases,
        case_cost,
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
        scan_grid,
        scan_grid_rows,
        style_sheet,
        sweep_cost_weights,
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
    scan_grid_rows,
    sweep_cost_weights,
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
    } & requested_pairs
    cached_count = len(cached_pairs)
    remaining_count = len(requested_pairs - cached_pairs)
    configuration_count = len({case["name"] for case in cases})
    material_label = CATALOG.material(MATERIAL).label

    # Compute-weighted progress: cost weights over kept + excluded cases so the
    # preview grid sizes every cell (the excluded band included) by its relative
    # matmul cost, and the meter can denominate by compute instead of case count.
    grid_cases = cases + dropped
    cost_weights, _cost_total = sweep_cost_weights(grid_cases)
    run_cost_total = sum(cost_weights[key] for key in requested_pairs)
    cached_cost = sum(cost_weights[key] for key in cached_pairs)
    grid_energies, grid_rows = scan_grid_rows(
        grid_cases, cached=cached_pairs, excluded=dropped, weights=cost_weights
    )
    return (
        MATERIAL,
        cached_cost,
        cached_count,
        cached_pairs,
        cases,
        checkpoint_path,
        configuration_count,
        cost_weights,
        dropped,
        grid_energies,
        grid_rows,
        material_label,
        penetration_summary,
        remaining_count,
        run_cost_total,
        settings,
    )


@app.cell(hide_code=True)
def _(
    MATERIAL,
    cached_cost,
    cached_count,
    cases,
    checkpoint_path,
    configuration_count,
    context_rail,
    directional_state,
    dropped,
    geometry_table,
    grid_energies,
    grid_rows,
    material_label,
    mo,
    penetration_summary,
    remaining_count,
    run_cost_total,
    scan_grid,
):
    exclusion_detail = penetration_summary or "No cases excluded by penetration watchdog."
    compute_done = 0.0 if run_cost_total <= 0 else cached_cost / run_cost_total
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
                    "Est. compute": f"{compute_done:.0%} done",
                },
            ),
            scan_grid(mo, grid_energies, grid_rows),
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
    cached_cost,
    cached_count,
    cached_pairs,
    case_cost,
    cases,
    checkpoint_path,
    cost_weights,
    directional_state,
    dropped,
    mo,
    remaining_count,
    run_cost_total,
    run_scan_ui,
    run_sweep,
    scan_grid,
    scan_grid_rows,
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
    # Run (resumes from the checkpoint, skipping cached cases). The determinate
    # meter is COST-weighted -- it advances by each finished case's estimated
    # matmul work (sweep.case_cost), not a flat 1/N, so a heavy high-energy case
    # visibly moves the bar more than a cheap low-energy one. The per-tilt
    # photon-counting tables still stream live; all figures are in the analysis app.
    results = {}
    done_cost = {"total": float(cached_cost)}
    rendered = {"permille": 0}

    def _permille(cost):
        return 0 if run_cost_total <= 0 else round(1000.0 * cost / run_cost_total)

    def _advance(bar):
        target = _permille(done_cost["total"])
        bar.update(
            increment=target - rendered["permille"],
            subtitle=f"{done_cost['total'] / run_cost_total:.0%} of estimated compute done"
            if run_cost_total > 0
            else "compute complete",
        )
        rendered["permille"] = target

    try:
        with mo.status.progress_bar(
            total=1000,
            title="Running checkpointed scan",
            subtitle=f"Writing `{checkpoint_path}` after each completed configuration group.",
            completion_title="Scan complete",
        ) as bar:
            _advance(bar)  # seed the bar with the already-cached compute

            def _on_case(case):
                done_cost["total"] += case_cost(case)
                _advance(bar)

            run_sweep(
                cases,
                results,
                on_case=_on_case,
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
                f"pyrite app analysis launch {MATERIAL}",
            )
        else:
            output = mo.callout(f"Scan failed while reading checkpoint: `{error}`", kind="danger")
    else:
        # Final state grid: pairs run THIS session read as done-this-session
        # (bright cyan) over the pre-existing cached ones, excluded band in rose.
        done_pairs = {(case["name"], case["E0_keV"]) for case in cases} - cached_pairs
        final_energies, final_rows = scan_grid_rows(
            cases + dropped,
            cached=cached_pairs,
            done=done_pairs,
            excluded=dropped,
            weights=cost_weights,
        )
        output = mo.vstack(
            [
                scan_grid(mo, final_energies, final_rows),
                directional_state(
                    mo,
                    "Scan complete",
                    f"Results saved at `{checkpoint_path}`. Checkpoint remains resumable.",
                    f"pyrite app analysis launch {MATERIAL}",
                    kind="success",
                ),
            ]
        )
    output
    return


if __name__ == "__main__":
    app.run()
