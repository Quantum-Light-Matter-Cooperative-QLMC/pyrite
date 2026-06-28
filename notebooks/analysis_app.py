import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    # A dense spectrum (fine grid x several beam energies) can exceed Vega-Lite's
    # default 5000-row cap; vegafusion (shipped with marimo[recommended]) lifts it,
    # falling back to disabling the cap outright.
    try:
        alt.data_transformers.enable("vegafusion")
    except Exception:
        alt.data_transformers.disable_max_rows()

    from cxr_mc.config import default_settings, trajectory_sweep
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
    from cxr_mc.plots.altair_spectra import spectrum_chart
    from cxr_mc.plots.altair_sweeps import metric_vs_chart, scan_charts
    from cxr_mc.plots.altair_trajectories import (
        penetration_survival_chart,
        trajectory_chart,
    )
    from cxr_mc.results import filter_results, records, show_top, sweep_values
    from cxr_mc.run import cases_from_results, load_checkpoint
    from cxr_mc.sweep import MATERIAL_LABELS, build_cases

    return (
        MATERIAL_LABELS,
        build_cases,
        cases_from_results,
        default_settings,
        eaglexo_charge_chart,
        eaglexo_detected_chart,
        filter_results,
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
        show_top,
        spectrum_chart,
        sweep_values,
        timepix_detected_chart,
        trajectory_chart,
        trajectory_sweep,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bulk-crystal CXR — analysis & visualization (marimo)

    Loads the checkpoint written by **`scan_app.py`** and draws every figure — no
    sweep runs here (only the cheap, CPU-only electron transport behind the
    penetration figures). Interactive Vega-Lite charts (intrinsic spectra, detector
    views, parametric scans, survival) replace the matplotlib `browse()` sliders;
    pick the material + polar tilt below and everything re-renders reactively. A few
    figures stay on matplotlib (efficiency curves, the datashader trajectory grid,
    the cross-material comparison).
    """)
    return


@app.cell
def _(mo):
    material_ui = mo.ui.dropdown(
        [
            "hopg",
            "diamond",
            "silicon",
            "mose2",
            "wse2",
            "ptse2",
            "hfse2",
            "zrse2",
            "ws2",
            "mos2",
        ],
        value="hopg",
        label="Material (match the scan you ran)",
    )
    material_ui
    return (material_ui,)


@app.cell
def _(cases_from_results, default_settings, filter_results, load_checkpoint, material_ui):
    MATERIAL = material_ui.value
    settings = default_settings()
    _results = load_checkpoint(MATERIAL)  # {name: {E0: record}}
    cases = cases_from_results(_results)  # rebuild the case list from the records
    res = filter_results(_results, cases)  # all loaded cases for this material
    return MATERIAL, cases, res, settings


@app.cell
def _(MATERIAL, mo, records, res, settings, show_top):
    # Compact, ranked "best geometries" table (bright + well-defined line).
    mo.stop(
        not records(res),
        mo.md(f"**No checkpoint for `{MATERIAL}`** — run `scan_app.py` for it first."),
    )
    show_top(res, settings, top_n=15, select="quality_peak")
    return


@app.cell
def _(mo, records, res, sweep_values):
    # What's actually in this checkpoint -- swept knobs and their values (a pickle
    # accumulates every case ever run). Slice with results.select_results if needed.
    sweep_values(res) if records(res) else mo.md("*(load a checkpoint above)*")
    return


@app.cell
def _(mo, records, res):
    # Tilt selector -- drives every per-tilt chart below (replaces the browse slider).
    _tilts = sorted({r["case"]["tilt_deg"] for r in records(res)})
    _opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    tilt_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="polar tilt")
    tilt_ui
    return (tilt_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intrinsic spectra

    The coherent line spectrum at the selected polar tilt, one line per beam energy
    (brem as a faint dashed underlay). Pan/zoom directly in the chart.
    """)
    return


@app.cell
def _(mo, res, settings, spectrum_chart, tilt_ui):
    _chart = spectrum_chart(res, settings, tilt_deg=tilt_ui.value)
    _chart if _chart is not None else mo.md("*No spectra — run the scan first.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry selection & parameter scans

    `scan_charts` auto-picks a heatmap (both axes sweep many values) or line scans
    (an axis is fixed / sparse), one chart per quantity. Below it: the best dozen
    geometries (matplotlib) and explicit 1-D metric scans.
    """)
    return


@app.cell
def _(cases, mo, res, scan_charts, settings):
    _charts = scan_charts(res, settings, cases=cases, line_metric="prominence")
    mo.vstack(_charts) if _charts else mo.md("*No results to scan.*")
    return


@app.cell
def _(metric_vs_chart, mo, res, settings):
    _line = metric_vs_chart(res, settings, x="tilt_deg", metric="line_flux", hue="E0_keV")
    _peak = metric_vs_chart(res, settings, x="tilt_deg", metric="peak_flux", hue="E0_keV")
    mo.vstack([c for c in (_line, _peak) if c is not None]) or mo.md("*No results.*")
    return


@app.cell
def _(plot_best_spectra, res, settings):
    plot_best_spectra(res, settings, top_n=12, select="quality_peak")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eagle XO detector response

    The Raptor Eagle XO direct-detection CCD (`solid_angle x QE(E)`): soft PXR lines
    pass at ~90% QE while hard brem is crushed by the thin sensor. Photon-density
    (detected vs incident, with the QE envelope), then the recorded-charge density —
    a CCD integrates charge, weighting each photon by E/W_Si.
    """)
    return


@app.cell
def _(plot_eaglexo_efficiency):
    plot_eaglexo_efficiency(sensor="4240")  # QE + solid angle + resolution
    return


@app.cell
def _(eaglexo_detected_chart, mo, res, settings, tilt_ui):
    _chart = eaglexo_detected_chart(res, settings, tilt_deg=tilt_ui.value)
    _chart if _chart is not None else mo.md("*No detector spectra yet.*")
    return


@app.cell
def _(eaglexo_charge_chart, mo, res, settings, tilt_ui):
    _chart = eaglexo_charge_chart(res, settings, tilt_deg=tilt_ui.value)
    _chart if _chart is not None else mo.md("*No charge spectra yet.*")
    return


@app.cell
def _(cases, mo, plot_eaglexo_charge_map, res, records, settings):
    # Geometry map of the recorded charge rate (best per cell); auto-lines a thin axis.
    plot_eaglexo_charge_map(res, settings, cases=cases) if records(res) else mo.md("*No results.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Timepix3 detector response (optional comparison)

    The Si quad forward model: photoabsorption → charge sharing → per-pixel
    threshold counting. Detected vs incident at the selected tilt.
    """)
    return


@app.cell
def _(plot_timepix_efficiency):
    plot_timepix_efficiency(thickness_um=300.0, bias_v=100.0)
    return


@app.cell
def _(mo, res, settings, timepix_detected_chart, tilt_ui):
    _chart = timepix_detected_chart(res, settings, tilt_deg=tilt_ui.value)
    _chart if _chart is not None else mo.md("*No detector spectra yet.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Electron penetration

    Surviving-electron fraction vs depth (one curve per beam energy), an interactive
    low-Ne trajectory cross-section, and the dense datashader penetration grid
    (matplotlib). These run the cheap CPU-only transport directly — no checkpoint
    needed.
    """)
    return


@app.cell
def _(MATERIAL, build_cases, penetration_survival_chart, settings, trajectory_sweep):
    traj_sweep = trajectory_sweep(MATERIAL, n_tilts=9, energies=(30, 60))
    traj_cases = build_cases(traj_sweep, settings.n_electrons, settings.n_electrons_brem)
    penetration_survival_chart(traj_cases, Ne=500)
    return (traj_cases,)


@app.cell
def _(mo, traj_cases, trajectory_chart):
    # Interactive vector cross-section (low Ne) -- pan/zoom/hover individual tracks.
    _chart = trajectory_chart(traj_cases[0], Ne=40) if traj_cases else None
    _chart if _chart is not None else mo.md("*No trajectory cases.*")
    return


@app.cell
def _(plot_trajectory_grid, traj_cases):
    # Dense datashader penetration grid (one beam energy); stays on matplotlib.
    plot_trajectory_grid(traj_cases, energy=30, Ne=120)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-material comparison

    For every material whose checkpoint exists, the single best geometry's dominant
    line: energy vs flux, coloured by line quality. (Run the scan for several
    materials first.)
    """)
    return


@app.cell
def _(MATERIAL_LABELS, load_checkpoint, mo, plot_material_comparison, settings):
    _by_material = {}
    for _m in MATERIAL_LABELS:
        _r = load_checkpoint(_m)
        if _r:
            _by_material[MATERIAL_LABELS[_m]] = _r
    (
        plot_material_comparison(_by_material, settings, select="quality_peak")
        if len(_by_material) >= 2
        else mo.md("*Run `scan_app.py` for more materials to populate this comparison.*")
    )
    return


if __name__ == "__main__":
    app.run()
