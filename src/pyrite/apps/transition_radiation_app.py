# /// script
# [tool.marimo.display]
# theme = "light"
# ///
"""Interactive transition-radiation and coherent-bunch reference demo."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pyrite._backend import BACKEND
    from pyrite.apps._design import page_title, style_sheet
    from pyrite.apps.angular_maps import (
        form_template,
        heatmap_figure,
        material_options,
        transport_cores,
    )
    from pyrite.apps.transition_radiation import (
        DEFAULT_LINE_HISTORIES,
        DemoConfig,
        comparison_figure,
        map_panels,
        straight_track_demo,
        transport_figure,
        transported_demo,
    )

    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "TR and CTR",
                "Surface interference and coherent bunch emission",
                eyebrow="PyRITE physics demo",
            ),
            mo.md("""
        **Reference:** 30 keV electrons, free-standing 20 nm silicon at 5° tilt, 150° backward
        observation, 200–1000 eV photons. TR sums the entry/exit fields of each
        electron; CTR also keeps phases between electrons. The idealized
        1 as RMS, 1 nm FWHM, 0.0001 pC bunch makes that difference visible.

        Every control below is open to physical values: forward (< 90°) and backward
        observation, relativistic beams, any catalog medium and any photon band the
        dielectric tables cover. The analytic plots use straight tracks and the exact
        planar-film solver; **Run transported comparison** adds a production Monte Carlo
        run. These are source densities at a direction, before detector response.
        """),
        ]
    )
    return (
        BACKEND,
        DEFAULT_LINE_HISTORIES,
        DemoConfig,
        comparison_figure,
        form_template,
        heatmap_figure,
        map_panels,
        material_options,
        mo,
        straight_track_demo,
        transport_cores,
        transport_figure,
        transported_demo,
    )


@app.cell
def _(DemoConfig, form_template, material_options, mo):
    _d = DemoConfig()
    _materials = material_options()
    scene_form = (
        mo.Html(
            form_template(
                {
                    "Beam and bunch": ("energy", "duration", "spot", "charge"),
                    "Film": ("material", "thickness", "tilt", "angle"),
                    "Photon band": ("start", "stop", "energy_points", "log_energy"),
                    "Map angles": ("angle_min", "angle_max", "angle_points"),
                }
            )
        )
        .batch(
            energy=mo.ui.number(0.1, 1e8, value=_d.energy_keV, label="Beam (keV)", full_width=True),
            duration=mo.ui.number(
                1e-3, 1e15, value=_d.duration_as, label="RMS duration (as)", full_width=True
            ),
            spot=mo.ui.number(
                1e-3, 1e10, value=_d.spot_nm, label="Spot FWHM (nm)", full_width=True
            ),
            charge=mo.ui.number(
                1e-9, 1e5, value=_d.charge_pc, label="Charge (pC)", full_width=True
            ),
            material=mo.ui.dropdown(
                _materials,
                value=next(k for k, v in _materials.items() if v == _d.material),
                label="Medium",
                searchable=True,
                full_width=True,
            ),
            thickness=mo.ui.number(
                0.01, 1e7, value=_d.thickness_nm, label="Film (nm)", full_width=True
            ),
            tilt=mo.ui.number(
                -89.5, 89.5, value=_d.tilt_deg, step=0.5, label="Tilt (deg)", full_width=True
            ),
            angle=mo.ui.number(
                -180,
                180,
                value=_d.angle_deg,
                step=0.1,
                label="Spectrum observation (deg from beam)",
                full_width=True,
            ),
            start=mo.ui.number(1, 1e9, value=_d.start_eV, label="Start (eV)", full_width=True),
            stop=mo.ui.number(2, 1e9, value=_d.stop_eV, label="Stop (eV)", full_width=True),
            energy_points=mo.ui.number(
                2, 100_000, value=_d.energy_points, label="Points", full_width=True
            ),
            log_energy=mo.ui.checkbox(value=_d.log_energy, label="Logarithmic spacing"),
            angle_min=mo.ui.number(
                -180, 180, value=_d.angle_min_deg, step=0.1, label="Min (deg)", full_width=True
            ),
            angle_max=mo.ui.number(
                -180, 180, value=_d.angle_max_deg, step=0.1, label="Max (deg)", full_width=True
            ),
            angle_points=mo.ui.number(
                2, 20_000, value=_d.angle_points, label="Samples", full_width=True
            ),
        )
        .form(submit_button_label="Update comparison", bordered=True)
    )
    scene_form
    return (scene_form,)


@app.cell
def _(DemoConfig, mo, scene_form, straight_track_demo):
    _values = scene_form.value
    try:
        config = (
            DemoConfig()
            if _values is None
            else DemoConfig(
                energy_keV=_values["energy"],
                thickness_nm=_values["thickness"],
                angle_deg=_values["angle"],
                duration_as=_values["duration"],
                spot_nm=_values["spot"],
                charge_pc=_values["charge"],
                material=_values["material"],
                tilt_deg=_values["tilt"],
                start_eV=_values["start"],
                stop_eV=_values["stop"],
                energy_points=int(_values["energy_points"]),
                log_energy=_values["log_energy"],
                angle_min_deg=_values["angle_min"],
                angle_max_deg=_values["angle_max"],
                angle_points=int(_values["angle_points"]),
            )
        )
        analytic = straight_track_demo(config)
    except ValueError as _error:
        mo.stop(True, mo.callout(mo.md(str(_error)), kind="danger"))
    return analytic, config


@app.cell
def _(analytic, mo):
    _energy = analytic["energy"]
    per_bunch_ui = mo.ui.checkbox(label="Show photons per physical bunch", value=False)
    log_ui = mo.ui.checkbox(label="Log color", value=True)
    cut_ui = mo.ui.slider(
        steps=[float(e) for e in _energy],
        value=float(_energy[len(_energy) // 2]),
        label="Angular-cut photon energy (eV)",
        show_value=True,
    )
    mo.hstack([per_bunch_ui, log_ui, cut_ui], justify="start", gap=2, wrap=True)
    return cut_ui, log_ui, per_bunch_ui


@app.cell
def _(
    analytic,
    comparison_figure,
    config,
    cut_ui,
    heatmap_figure,
    log_ui,
    map_panels,
    mo,
    per_bunch_ui,
):
    _unit = "bunch" if per_bunch_ui.value else "electron"
    mo.vstack(
        [
            heatmap_figure(
                map_panels(analytic, config, per_bunch=per_bunch_ui.value),
                value_label=f"photons/{_unit}/eV/sr",
                angle_label="Signed angle from beam (deg), lab x–z plane",
                logarithmic=log_ui.value,
                dynamic_range=1e10,
            ),
            comparison_figure(
                analytic, config, per_bunch=per_bunch_ui.value, energy_eV=cut_ui.value
            ),
        ]
    )
    return


@app.cell
def _(config, mo):
    mo.md(f"""
    Physical population **N = {config.population:,.1f} electrons/bunch**.
    Straight identical tracks use the Gaussian intensity form factor and
    **1 + (N − 1)|F|²** per electron. A point bunch gives N² per bunch;
    a long bunch tends to the incoherent N scaling. No sampled offsets are
    used in these analytic plots. Gray map cells are directions parallel to
    the film faces. A single interface is a different target from a film:
    their difference includes entry/exit interference and absorption.
    """)
    return


@app.cell
def _(BACKEND, DEFAULT_LINE_HISTORIES, mo, transport_cores):
    transport_button = mo.ui.run_button(label="Run transported comparison")
    trials_ui = mo.ui.number(1, 1_000_000, value=DEFAULT_LINE_HISTORIES, label="Line histories")
    core_ui = mo.ui.dropdown(
        [c for c in transport_cores(BACKEND.name) if c != "auto"],
        value="per-electron",
        label="Transport core",
    )
    mo.vstack(
        [
            mo.md(f"""
    **Production comparison:** two fixed-count runs (history TR and bunch CTR) with
    sampled time and transverse offsets, twice as many continuum as line histories.
    It retains signed CTR estimates; negative bins indicate unresolved sampling.
    Array backend: **{BACKEND.name}** (fixed at launch; use `--backend cuda` for the
    CUDA core and fused TR kernel). Cached results are reused when changing display
    units or angular-cut energy.
    """),
            mo.hstack([trials_ui, core_ui, transport_button], justify="start", gap=2),
        ]
    )
    return core_ui, transport_button, trials_ui


@app.cell
def _(config, core_ui, mo, transport_button, transported_demo, trials_ui):
    mo.stop(not transport_button.value, mo.md("Click **Run transported comparison** to compute."))
    try:
        production, messages, elapsed = transported_demo(
            config, trials=int(trials_ui.value), transport_core=core_ui.value
        )
    except ValueError as _error:
        mo.stop(True, mo.callout(mo.md(str(_error)), kind="danger"))
    return elapsed, messages, production


@app.cell
def _(config, core_ui, elapsed, messages, mo, per_bunch_ui, production, transport_figure):
    mo.vstack(
        [
            mo.md(
                f"{core_ui.value} core: **{elapsed:.2f} s**. Signed values use a symmetric "
                "log scale. Combined CXR excludes the separately gridded bremsstrahlung continuum."
            ),
            transport_figure(production, config, per_bunch=per_bunch_ui.value),
            mo.callout(mo.md("\n\n".join(messages)), kind="warn")
            if messages
            else mo.md("No runtime warnings. Fixed counts do not provide a precision certificate."),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Reproduce and interpret

    ```bash
    pyrite run tr_demo
    pyrite run ctr_demo
    pyrite app transition-radiation export
    ```

    Profiles save a separate `transition.h5` component. `emission = incoherent`
    refers to PXR/CBS; `transition_radiation = bunch` independently selects CTR.
    The production comparison includes incoherent lines, so it does not claim
    additive intensities between TR and coherent PXR/CBS.

    **Limits:** flat infinite interfaces, independent-atom Chantler dielectric
    response (crude near the plasma energy and edges), no diffracted TR or
    optical/IR model (see the optical diagnostics app). The production photon
    floor is max(plasma energy, 12 eV); below a few hundred eV the analytic map is
    only as good as that dielectric response. Scalar acceptance metadata does not
    integrate narrow fringes: use the dedicated planar-detector integrator for
    channel counts. Adaptive precision, electron-block transport, channeling,
    grooves and temporal profiles are refused for TR. Large N with few sampled
    histories can overwhelm the CTR signal with noise; do not clip negative bins
    or count the form factor twice. Static HTML contains the initial analytic
    comparison; use the live notebook for controls and transport.
    """)
    return


@app.cell
def _(mo):
    dtr_run = mo.ui.run_button(label="Compute experimental DTR reference")
    dtr_two_beam = mo.ui.checkbox(label="Include repeated-diffraction two-beam comparison")
    mo.vstack(
        [
            mo.md(
                "Experimental DTR: Born and two-beam solvers remain unverified. Born PXR-only is its DTR-off reference; two-beam rows contain coupled PXR/DTR with no separate DTR-off interpretation. Retain signed cross terms."
            ),
            dtr_two_beam,
            dtr_run,
        ]
    )
    return dtr_run, dtr_two_beam


@app.cell
def _(dtr_run, dtr_two_beam, mo):
    mo.stop(not dtr_run.value)
    from pyrite.apps.radiation_figures import dtr_reference_figure
    from pyrite.apps.transition_radiation import cached_dtr_reference

    _columns, _metadata = cached_dtr_reference(dtr_two_beam.value)
    mo.vstack([dtr_reference_figure(_columns), mo.md(f"Reference provenance: `{_metadata}`")])
    return


if __name__ == "__main__":
    app.run()
