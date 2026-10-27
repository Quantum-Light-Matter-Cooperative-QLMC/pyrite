"""Opt-in optical radiation and instrument diagnostics; Plotly only."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="PyRITE · Optical Diagnostics",
    html_head_file="../_assets/branding/head.html",
)


@app.cell
def _():
    import json
    from dataclasses import asdict

    import marimo as mo

    from pyrite._backend import BACKEND
    from pyrite.apps._design import page_title, style_sheet
    from pyrite.apps.angular_maps import form_template, heatmap_figure
    from pyrite.apps.optical_diagnostics import (
        ScreenConfig,
        cherenkov_cone_map,
        collection_convergence,
        finite_source_demo,
        inference_demo,
        load_saved_optical,
        report_rows,
    )
    from pyrite.apps.radiation_figures import (
        optical_scan_figure,
        optical_source_figure,
        saved_optical_figure,
    )

    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Optical radiation diagnostics",
                "Finite-screen TR/DR, instrument inference and the bulk Cherenkov cone",
            ),
            mo.md("""
        Bounded reference models; synthetic calibration, no experimental anchor.
        Finite PEC TR/DR collection, Gaussian instrument inference and the Cherenkov
        cone are **separate exercises**. Fields retain spectral V·s/m normalization;
        pulse signals already include physical population. No transport runs, and
        saved artifacts can be inspected independently.
        """),
        ]
    )
    return (
        BACKEND,
        ScreenConfig,
        asdict,
        cherenkov_cone_map,
        collection_convergence,
        finite_source_demo,
        form_template,
        heatmap_figure,
        inference_demo,
        json,
        load_saved_optical,
        mo,
        optical_scan_figure,
        optical_source_figure,
        report_rows,
        saved_optical_figure,
    )


@app.cell
def _(mo):
    view = mo.ui.radio(
        [
            "Source and collection",
            "Convergence",
            "Instrument and inference",
            "Cherenkov cone",
            "Saved outputs",
        ],
        value="Source and collection",
        label="View",
        inline=True,
    )
    view
    return (view,)


@app.cell
def _(BACKEND, ScreenConfig, form_template, mo, view):
    mo.stop(view.value not in ("Source and collection", "Convergence"))
    _d = ScreenConfig()
    screen_form = (
        mo.Html(
            form_template(
                {
                    "Screen and beam": ("outer", "hole", "beta", "duration", "charge"),
                    "Collection": ("distance", "aperture", "low", "high", "points"),
                    "Numerics": ("grid", "pitch", "device"),
                }
            )
        )
        .batch(
            outer=mo.ui.number(0.1, 1e6, value=_d.outer_um, label="Screen radius (µm)"),
            hole=mo.ui.number(0, 1e6, value=_d.hole_um, label="Hole radius (µm; 0 = disc TR)"),
            beta=mo.ui.number(0.01, 0.9999999, value=_d.beta, step=1e-4, label="β"),
            duration=mo.ui.number(0, 1e6, value=_d.duration_fs, label="Gaussian RMS (fs)"),
            charge=mo.ui.number(1e-9, 1e4, value=_d.charge_pc, label="Charge (pC)"),
            distance=mo.ui.number(1, 1e7, value=_d.distance_um, label="Plane distance (µm)"),
            aperture=mo.ui.number(0.1, 1e6, value=_d.aperture_um, label="Aperture radius (µm)"),
            low=mo.ui.number(1e-4, 100, value=_d.low_eV, step=0.01, label="Band low (eV)"),
            high=mo.ui.number(1e-4, 100, value=_d.high_eV, step=0.01, label="Band high (eV)"),
            points=mo.ui.number(2, 1000, value=_d.energy_points, label="Band points"),
            grid=mo.ui.number(16, 4096, value=_d.grid, label="FFT samples per side"),
            pitch=mo.ui.number(1e-3, 100, value=_d.pitch_um, step=0.05, label="Pitch (µm)"),
            device=mo.ui.dropdown(
                ["cpu", *(["cuda"] if BACKEND.name == "cuda" else [])],
                value="cpu",
                label="Field device",
            ),
        )
        .form(submit_button_label="Compute selected diagnostic", bordered=True)
    )
    mo.vstack(
        [
            screen_form,
            mo.md(
                "Two identical offset-free numerical histories; temporal-only Gaussian law. "
                "Require screen/hole/edge scales ≫ wavelength and a window "
                "(samples × pitch) wider than the screen. FFT maps and collection need "
                "convergence; check the *Convergence* view at the same settings."
            ),
        ]
    )
    return (screen_form,)


@app.cell
def _(
    ScreenConfig,
    asdict,
    collection_convergence,
    finite_source_demo,
    mo,
    optical_source_figure,
    report_rows,
    screen_form,
    view,
):
    mo.stop(screen_form.value is None)
    _v = screen_form.value
    try:
        _config = ScreenConfig(
            hole_um=_v["hole"],
            outer_um=_v["outer"],
            beta=_v["beta"],
            duration_fs=_v["duration"],
            charge_pc=_v["charge"],
            distance_um=_v["distance"],
            aperture_um=_v["aperture"],
            low_eV=_v["low"],
            high_eV=_v["high"],
            energy_points=int(_v["points"]),
            grid=int(_v["grid"]),
            pitch_um=_v["pitch"],
            device=_v["device"],
        )
        _rail = mo.md(
            f"Window **{_config.window_um:g} µm** ({_config.grid}² at "
            f"{_config.pitch_um:g} µm) · device **{_config.device}**"
        )
        if view.value == "Convergence":
            with mo.status.spinner("Refining pitch/window and direct-Green comparison…"):
                _report = collection_convergence(_config)
            _content = mo.vstack(
                [
                    _rail,
                    mo.md(
                        "Independent pitch/window refinements, hemisphere FFT bias and "
                        "direct-Green collection comparison at the band's geometric-mean "
                        "energy. Numerical diagnostics, not physical validation."
                    ),
                    mo.ui.table(report_rows(_report), selection=None, page_size=25),
                ]
            )
        else:
            with mo.status.spinner("Solving finite-screen fields and plane collection…"):
                _band, _field, _reading = finite_source_demo(_config)
            _content = mo.vstack(
                [
                    _rail,
                    optical_source_figure(_band, _field),
                    mo.md(
                        "Field maps show one electron at the band centre; collected spectrum "
                        "is per physical bunch. Synthetic η=25%, gain=10¹² V/J; 1 kHz "
                        "repetition affects average power only."
                    ),
                    mo.ui.table([asdict(_reading)], selection=None),
                    mo.md(f"Source law/provenance: `{_band.source_record}`"),
                ]
            )
    except ValueError as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


@app.cell
def _(form_template, mo, view):
    mo.stop(view.value != "Instrument and inference")
    infer_form = (
        mo.Html(
            form_template(
                {
                    "Source": ("sigma", "low", "high"),
                    "Interferometer": ("mirror", "nodes", "noise"),
                }
            )
        )
        .batch(
            sigma=mo.ui.number(1, 1e5, value=200, label="Synthetic Gaussian RMS (fs)"),
            low=mo.ui.number(1e-5, 10, value=0.0005, step=1e-4, label="Detector band low (eV)"),
            high=mo.ui.number(1e-4, 10, value=0.025, step=1e-3, label="Detector band high (eV)"),
            mirror=mo.ui.number(0.01, 100, value=1.2, step=0.1, label="Mirror travel ± (mm)"),
            nodes=mo.ui.number(5, 10_001, value=61, label="Mirror nodes"),
            noise=mo.ui.number(1e-3, 1e6, value=15, label="Declared RMS error (µV)"),
        )
        .form(submit_button_label="Compute interferogram and conditional fit", bordered=True)
    )
    mo.vstack(
        [
            infer_form,
            mo.md(
                "Synthetic flat photon templates and 25% detector absorptance; fixed 45° "
                "linear polarization. No measured calibration and no finite-source handoff. "
                "A 200 fs turnover is ~0.0033 eV: keep it inside the band to identify the "
                "duration. OPD is twice the mirror travel. Noise-free expectation with "
                "declared errors; no systematic uncertainty propagation."
            ),
        ]
    )
    return (infer_form,)


@app.cell
def _(infer_form, inference_demo, json, mo, optical_scan_figure, report_rows):
    mo.stop(infer_form.value is None)
    _v = infer_form.value
    try:
        with mo.status.spinner("Forward model and profiled fit…"):
            _scan, _fit, _interferogram = inference_demo(
                _v["sigma"], _v["low"], _v["high"], _v["mirror"], int(_v["nodes"]), _v["noise"]
            )
        _summary = {
            "RMS_fs": _fit.rms_duration_fs,
            "conditional_95_interval_fs": _fit.confidence_interval_95_fs,
            "identified": _fit.is_identified_95,
            "multimodal": _fit.is_multimodal,
            "turnover_eV": _fit.turnover_energy_eV,
            "temporal_factor_at_band_edges": _fit.temporal_factor_at_band_edges,
        }
        _content = mo.vstack(
            [
                optical_scan_figure(_scan, _fit),
                mo.ui.table(report_rows(_summary), selection=None),
                mo.accordion(
                    {
                        "Assumptions and provenance": mo.md(
                            "```json\n"
                            + json.dumps(
                                {
                                    "assumptions": _fit.model_assumptions,
                                    "provenance": _interferogram.metadata,
                                },
                                indent=2,
                                default=str,
                            )
                            + "\n```"
                        )
                    }
                ),
                mo.md(
                    "Null confidence endpoints are open. This conditional interval is not a "
                    "full experimental uncertainty budget. Refine spectral integration and "
                    "delay step/range independently before interpreting a result."
                ),
            ]
        )
    except ValueError as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


@app.cell
def _(mo, view):
    mo.stop(view.value != "Cherenkov cone")
    index_ui = mo.ui.slider(1.0, 3.0, step=0.005, value=1.46, label="Refractive index n")
    photon_ui = mo.ui.number(0.1, 50, value=2.5, step=0.1, label="Photon energy (eV)")
    kinetic_ui = mo.ui.range_slider(
        1, 8, step=0.05, value=[2, 5], label="Electron kinetic energy (log10 keV)"
    )
    bins_ui = mo.ui.slider(20, 1800, step=10, value=360, label="Polar bins over 0–90°")
    log_ui = mo.ui.checkbox(value=True, label="Log color")
    mo.vstack(
        [
            mo.hstack([index_ui, photon_ui, log_ui], justify="start", gap=2, wrap=True),
            mo.hstack([kinetic_ui, bins_ui], justify="start", gap=2, wrap=True),
        ]
    )
    return bins_ui, index_ui, kinetic_ui, log_ui, photon_ui


@app.cell
def _(
    bins_ui,
    cherenkov_cone_map,
    heatmap_figure,
    index_ui,
    kinetic_ui,
    log_ui,
    mo,
    photon_ui,
):
    try:
        _panel, _threshold, _cone = cherenkov_cone_map(
            index_ui.value,
            photon_ui.value,
            (10.0 ** kinetic_ui.value[0], 10.0 ** kinetic_ui.value[1]),
            240,
            int(bins_ui.value),
        )
        _content = mo.vstack(
            [
                mo.md(
                    f"Threshold **{_threshold:,.1f} keV** · saturated cone "
                    f"**{_cone:.2f}°** (β → 1). Constant n: the yield is the same at every "
                    "photon energy inside the medium's transparent band."
                ),
                heatmap_figure(
                    [_panel],
                    value_label="photons/m/eV/sr (bin-averaged)",
                    angle_label="Polar angle from the electron velocity (deg)",
                    energy_label="Electron kinetic energy (keV)",
                    logarithmic=log_ui.value,
                    dynamic_range=1e4,
                    log_energy=True,
                ),
                mo.md(
                    "Bulk Frank–Tamm source of one electron in an unbounded, lossless, "
                    "nondispersive medium (`frank-tamm-homogeneous-dielectric`). Internal "
                    "yield only: no interfaces, escape, total internal reflection, "
                    "multiple scattering or slowing down. The delta-function cone is "
                    "deposited in its polar bin, so values scale with bin width."
                ),
            ]
        )
    except ValueError as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


@app.cell
def _(mo, view):
    mo.stop(view.value != "Saved outputs")
    saved_path = mo.ui.text(label="Existing NPZ artifact path", full_width=True)
    saved_kind = mo.ui.dropdown(
        ["Fit", "Interferogram", "Planar source"], value="Fit", label="Artifact type"
    )
    load_run = mo.ui.run_button(label="Read saved artifact")
    mo.vstack([saved_path, saved_kind, load_run])
    return load_run, saved_kind, saved_path


@app.cell
def _(json, load_run, load_saved_optical, mo, saved_kind, saved_optical_figure, saved_path):
    mo.stop(not load_run.value)
    try:
        _saved = load_saved_optical(saved_path.value, saved_kind.value)
        _content = mo.vstack(
            [
                saved_optical_figure(_saved, saved_kind.value),
                mo.md(f"```json\n{json.dumps(_saved.metadata, indent=2)}\n```"),
            ]
        )
    except (OSError, ValueError, KeyError) as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


if __name__ == "__main__":
    app.run()
