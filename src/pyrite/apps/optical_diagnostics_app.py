"""Opt-in optical radiation and instrument diagnostics; Plotly only."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from dataclasses import asdict

    import marimo as mo

    from pyrite.apps.optical_diagnostics import (
        collection_convergence,
        finite_source_demo,
        inference_demo,
        load_saved_optical,
    )
    from pyrite.apps.radiation_figures import (
        optical_scan_figure,
        optical_source_figure,
        saved_optical_figure,
    )

    mo.md("""
    # Optical radiation diagnostics

    Bounded reference models; synthetic calibration, no experimental anchor.
    Finite PEC TR/DR collection and Gaussian instrument inference are **separate exercises**.
    Fields retain spectral V·s/m normalization; pulse signals already include physical population.
    No transport or GPU job runs automatically. Saved artifacts can be inspected independently.
    """)
    return (
        asdict,
        collection_convergence,
        finite_source_demo,
        inference_demo,
        json,
        load_saved_optical,
        mo,
        optical_scan_figure,
        optical_source_figure,
        saved_optical_figure,
    )


@app.cell
def _(mo):
    view = mo.ui.radio(
        ["Source and collection", "Convergence", "Instrument and inference", "Saved outputs"],
        value="Source and collection",
        label="View",
        inline=True,
    )
    view
    return (view,)


@app.cell
def _(mo, view):
    mo.stop(view.value not in ("Source and collection", "Convergence"))
    hole = mo.ui.dropdown(
        {"Disc TR": 0.0, "5 µm aperture DR": 5.0}, value="Disc TR", label="20 µm PEC screen"
    )
    distance = mo.ui.slider(50, 200, step=25, value=100, label="Collection distance (µm)")
    duration = mo.ui.number(0, 10, value=1, label="Gaussian RMS duration (fs)")
    charge = mo.ui.number(0.00001, 0.01, value=0.001, label="Physical charge (pC)")
    source_run = mo.ui.run_button(label="Compute selected diagnostic")
    mo.vstack(
        [
            mo.hstack([hole, distance, duration, charge]),
            mo.md(
                "β=0.95; 0.8–1.2 eV; 10 µm collection radius; 128² grid at 0.5 µm pitch. Two identical offset-free numerical histories; temporal-only Gaussian law. Require screen/hole/edge scales ≫ wavelength. FFT maps and collection need convergence."
            ),
            source_run,
        ]
    )
    return charge, distance, duration, hole, source_run


@app.cell
def _(
    asdict,
    charge,
    collection_convergence,
    distance,
    duration,
    finite_source_demo,
    hole,
    json,
    mo,
    optical_source_figure,
    source_run,
    view,
):
    mo.stop(not source_run.value)
    try:
        if view.value == "Convergence":
            _report = collection_convergence(hole.value, distance.value)
            _content = mo.vstack(
                [
                    mo.md(
                        "Independent pitch/window refinements, hemisphere FFT bias and direct-Green collection comparison. These are numerical diagnostics, not physical validation. Energy, angular acceptance and spectral/delay integration convergence remain separate requirements."
                    ),
                    mo.md(f"```json\n{json.dumps(_report, indent=2)}\n```"),
                ]
            )
        else:
            _band, _field, _reading = finite_source_demo(
                hole.value, duration.value, charge.value, distance.value
            )
            _content = mo.vstack(
                [
                    optical_source_figure(_band, _field),
                    mo.md(
                        "Field maps show one electron; collected spectrum is per physical bunch. Synthetic η=25%, gain=10¹² V/J; 1 kHz repetition affects average power only."
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
def _(mo, view):
    mo.stop(view.value != "Instrument and inference")
    sigma = mo.ui.slider(50, 500, value=200, step=10, label="Synthetic Gaussian RMS duration (fs)")
    band = mo.ui.dropdown(
        {"Turnover covered: 0.0005–0.025 eV": 0.0005, "High band only: 0.02–0.025 eV": 0.02},
        value="Turnover covered: 0.0005–0.025 eV",
        label="Synthetic detector band",
    )
    infer_run = mo.ui.run_button(label="Compute synthetic interferogram and conditional fit")
    mo.vstack(
        [
            sigma,
            band,
            infer_run,
            mo.md(
                "Synthetic flat photon templates and 25% detector absorptance; fixed 45° linear polarization. No measured calibration, no finite-source handoff in this exercise. A 200 fs turnover is ~0.0033 eV. No unique profile reconstruction. Mirror travel ±1.2 mm, 61 nodes; OPD is twice mirror travel. Noise-free expectation with declared 15 µV RMS errors; no systematic uncertainty propagation."
            ),
        ]
    )
    return band, infer_run, sigma


@app.cell
def _(band, infer_run, inference_demo, json, mo, optical_scan_figure, sigma):
    mo.stop(not infer_run.value)
    try:
        _scan, _fit, _interferogram = inference_demo(sigma.value, band.value)
        _summary = {
            "RMS_fs": _fit.rms_duration_fs,
            "conditional_95_interval_fs": _fit.confidence_interval_95_fs,
            "identified": _fit.is_identified_95,
            "multimodal": _fit.is_multimodal,
            "turnover_eV": _fit.turnover_energy_eV,
            "temporal_factor_at_band_edges": _fit.temporal_factor_at_band_edges,
            "assumptions": _fit.model_assumptions,
            "provenance": _interferogram.metadata,
        }
        _content = mo.vstack(
            [
                optical_scan_figure(_scan, _fit),
                mo.md(f"```json\n{json.dumps(_summary, indent=2)}\n```"),
                mo.md(
                    "Null confidence endpoints are open. This conditional interval is not a full experimental uncertainty budget. Refine spectral integration and delay step/range independently before interpreting a result."
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
    saved_path = mo.ui.text(label="Existing NPZ artifact path")
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
