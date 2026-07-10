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

    from cxr_mc.config import PENETRATION_TILT_DEG, default_settings, trajectory_sweep
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
    from cxr_mc.results import filter_results, records, sweep_values, top_geometries
    from cxr_mc.run import cases_from_results, load_checkpoint
    from cxr_mc.sweep import MATERIAL_LABELS, build_cases

    return (
        MATERIAL_LABELS,
        PENETRATION_TILT_DEG,
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
def _(mo):
    material_ui = mo.ui.dropdown(
        [
            "hopg",
            "hbn",
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
def _(mo, records, res, sweep_values):
    # What's actually in this checkpoint -- swept knobs and their values (a pickle
    # accumulates every case ever run). Slice with results.select_results if needed.
    # Tucked in a collapsible accordion: secondary detail, open it when you need it.
    mo.accordion(
        {
            "Checkpoint contents (swept knobs & values)": (
                sweep_values(res) if records(res) else mo.md("*(load a checkpoint above)*")
            )
        }
    )
    return


@app.cell
def _(mo, records, res):
    # Tilt selector -- drives every per-tilt chart below (replaces the browse slider).
    _tilts = sorted({r["case"]["tilt_deg"] for r in records(res)})
    _opts = {f"{t:g} deg": t for t in _tilts} or {"— no data —": None}
    tilt_ui = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="polar tilt")
    tilt_ui
    return (tilt_ui,)


@app.cell
def _(mo):
    # Spectral-plot controls: CXR-only (no brem background) toggle, adjustable
    # x-axis (photon energy) limits, and a lin/log y switch -- shared by the
    # intrinsic-spectra chart and the Timepix3 detected-vs-incident chart (the
    # two line-per-energy spectral views). Leave x-limits at 0 for autoscale.
    brem_ui = mo.ui.checkbox(value=True, label="show brem background")
    xmin_ui = mo.ui.number(value=0.0, label="x-min (eV, 0 = auto)")
    xmax_ui = mo.ui.number(value=0.0, label="x-max (eV, 0 = auto)")
    ylog_ui = mo.ui.switch(value=False, label="log y")
    mo.hstack([brem_ui, xmin_ui, xmax_ui, ylog_ui])
    return brem_ui, xmax_ui, xmin_ui, ylog_ui


@app.cell
def _(xmax_ui, xmin_ui):
    # (0, 0) -> None (autoscale); otherwise an explicit (lo, hi) domain.
    x_domain = (xmin_ui.value, xmax_ui.value) if (xmin_ui.value or xmax_ui.value) else None
    return (x_domain,)


@app.cell
def _(PENETRATION_TILT_DEG, mo):
    _angle_opts = {f"{t:g} deg": t for t in PENETRATION_TILT_DEG}
    penetration_angle_ui = mo.ui.dropdown(
        _angle_opts, value="-15 deg", label="polar tilt (penetration)"
    )
    return (penetration_angle_ui,)


@app.cell
def _(mo):
    # The dense grid draws ONE beam energy at a time (a panel per tilt); the
    # survival/track views above it already show both energies from
    # trajectory_sweep's default (30, 60), so this only needs to pick between
    # those two.
    penetration_energy_ui = mo.ui.dropdown(
        {"30 keV": 30.0, "60 keV": 60.0}, value="30 keV", label="beam energy (dense grid)"
    )
    return (penetration_energy_ui,)


@app.cell
def _(
    MATERIAL,
    MATERIAL_LABELS,
    brem_ui,
    build_cases,
    cases,
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    load_checkpoint,
    metric_vs_chart,
    mo,
    penetration_angle_ui,
    penetration_energy_ui,
    penetration_survival_chart,
    plot_best_spectra,
    plot_eaglexo_charge_map,
    plot_eaglexo_efficiency,
    plot_material_comparison,
    plot_timepix_efficiency,
    plot_trajectory_grid,
    records,
    res,
    scan_charts,
    settings,
    spectrum_chart,
    tilt_ui,
    timepix_detected_chart,
    top_geometries,
    trajectory_chart,
    trajectory_sweep,
    x_domain,
    ylog_ui,
):
    # All figures live in one lazy, tabbed layout. Each tab value is a zero-arg
    # builder closure (not a pre-built object), so `lazy=True` defers the *compute*
    # — only the open tab runs. The expensive matplotlib panels are nested in
    # `lazy` accordions; the builders return a Figure, which marimo renders on
    # expand. Closures capture `tilt_ui.value`, so switching tilt rebuilds the
    # tabs and the next view recomputes with the new tilt.
    _tilt = tilt_ui.value

    def _rankings_tab():
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

    def _spectra_tab():
        _md = mo.md(
            "The INTRINSIC coherent CXR line spectrum at the selected polar tilt, "
            "one line per beam energy; toggle the brem background off above for "
            "the CXR-only view. Pan/zoom in the chart, or set explicit x-limits "
            "and switch to log y with the controls above."
        )
        _chart = spectrum_chart(
            res,
            settings,
            tilt_deg=_tilt,
            include_brem=brem_ui.value,
            x_domain=x_domain,
            y_type="log" if ylog_ui.value else "linear",
        )
        return mo.vstack(
            [_md, _chart if _chart is not None else mo.md("*No spectra — run the scan first.*")]
        )

    def _scans_tab():
        # Each plot is a separate mo.lazy(fn) so the tab returns instantly with
        # placeholders; scans, metric lines, and the best-spectra accordion each
        # compute and render independently as they become visible.
        def _scan_charts_item():
            charts = scan_charts(res, settings, cases=cases, line_metric="prominence")
            return mo.vstack(charts) if charts else mo.md("*No scan results.*")

        def _metric_lines_item():
            line = metric_vs_chart(res, settings, x="tilt_deg", metric="line_flux", hue="E0_keV")
            peak = metric_vs_chart(res, settings, x="tilt_deg", metric="peak_flux", hue="E0_keV")
            parts = [c for c in (line, peak) if c is not None]
            return mo.vstack(parts) if parts else mo.md("*No metric results.*")

        return mo.vstack(
            [
                mo.md(
                    "`scan_charts` auto-picks a heatmap or line scans (one per swept quantity); "
                    "then 1-D metric scans vs polar tilt. Each section loads as it becomes "
                    "visible. The best-spectra panel loads on expand."
                ),
                mo.lazy(_scan_charts_item, show_loading_indicator=True),
                mo.lazy(_metric_lines_item, show_loading_indicator=True),
                mo.accordion(
                    {
                        "Best spectra (matplotlib)": lambda: plot_best_spectra(
                            res, settings, top_n=12, select="quality_peak"
                        )
                    },
                    lazy=True,
                ),
            ]
        )

    def _detectors_tab():
        # Eagle XO and Timepix3 render into a nested mo.accordion (lazy=True)
        # rather than a dropdown or nested mo.ui.tabs: a mo.ui.dropdown created
        # inside a lazy builder isn't in marimo's reactive DAG, so it can't drive
        # re-renders, and nested mo.ui.tabs-in-tabs is a known marimo rendering
        # bug (charts inside the inner tab silently render blank -- this is what
        # produced the "blank detector-tab plots" bug: marimo-team/marimo#6919).
        # An accordion lazily defers each section's compute exactly like the
        # inner tabs did, without the nesting problem.
        def _eaglexo_inner():
            _md = mo.md(
                "Raptor Eagle XO direct-detection CCD (`solid_angle x QE(E)`): soft PXR "
                "lines pass at ~90% QE, hard brem is crushed by the thin sensor. "
                "Photon density (detected vs incident), then recorded-charge density."
            )
            _detected = eaglexo_detected_chart(res, settings, tilt_deg=_tilt, x_domain=x_domain)
            _charge = eaglexo_charge_chart(res, settings, tilt_deg=_tilt, x_domain=x_domain)
            _parts = [_md, *(c for c in (_detected, _charge) if c is not None)]
            _parts.append(
                mo.accordion(
                    {
                        "Efficiency: QE + solid angle + resolution (matplotlib)": lambda: (
                            plot_eaglexo_efficiency(sensor="4240")
                        ),
                        "Charge geometry map (matplotlib)": lambda: (
                            plot_eaglexo_charge_map(res, settings, cases=cases)
                            if records(res)
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
            _detected = timepix_detected_chart(res, settings, tilt_deg=_tilt, x_domain=x_domain)
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

        return mo.accordion({"Eagle XO": _eaglexo_inner, "Timepix3": _timepix_inner}, lazy=True)

    def _penetration_tab():
        _angle = penetration_angle_ui.value
        _md = mo.md(
            "Surviving-electron fraction vs depth (one curve per beam energy) and "
            "an interactive low-Ne track cross-section, at the polar tilt selected "
            "in this tab (default -15 deg, a low nonzero angle -- avoids both the fully-"
            "normal and grazing-incidence edge cases). These run the cheap CPU-only "
            "transport directly — no checkpoint needed. For a stacked/multilayer "
            "material (e.g. mos2 on sapphire) the cascade is transported through "
            "the FULL stack, not just the top film. Dense grid loads on expand."
        )
        _sweep = trajectory_sweep(MATERIAL, energies=(30, 60))
        _traj = build_cases(_sweep, settings.n_electrons, settings.n_electrons_brem)
        if not _traj:
            return mo.vstack([_md, mo.md("*No trajectory cases.*")])
        _survival = penetration_survival_chart(_traj, Ne=500, tilt=_angle)
        # Pick the lowest energy at the selected tilt for the single-track view.
        _nc = min(_traj, key=lambda c: (abs(c["tilt_deg"] - _angle), c["E0_keV"]))
        _track = trajectory_chart(_nc, Ne=40)
        _grid_energy = penetration_energy_ui.value
        _parts = [_md, penetration_angle_ui, *(p for p in (_survival, _track) if p is not None)]
        _parts.append(penetration_energy_ui)
        _parts.append(
            mo.accordion(
                {
                    "Dense penetration grid (datashader, matplotlib)": lambda: plot_trajectory_grid(
                        _traj, energy=_grid_energy, Ne=120
                    )
                },
                lazy=True,
            )
        )
        return mo.vstack(_parts)

    def _cross_material_tab():
        _md = mo.md(
            "For every material whose checkpoint exists, the single best geometry's "
            "dominant line: energy vs flux, coloured by line quality."
        )
        _by_material = {}
        for _m in MATERIAL_LABELS:
            _r = load_checkpoint(_m)
            if _r:
                _by_material[MATERIAL_LABELS[_m]] = _r
        if len(_by_material) >= 2:
            return mo.vstack(
                [_md, plot_material_comparison(_by_material, settings, select="quality_peak")]
            )
        return mo.vstack(
            [_md, mo.md("*Run `scan_app.py` for more materials to populate this comparison.*")]
        )

    mo.ui.tabs(
        {
            "Intrinsic spectra": _spectra_tab,
            "Top geometries": _rankings_tab,
            "Geometry & scans": _scans_tab,
            "Detectors": _detectors_tab,
            "Penetration": _penetration_tab,
            "Cross-material": _cross_material_tab,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
