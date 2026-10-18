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

    from pyrite.apps._design import page_title, style_sheet
    from pyrite.apps.transition_radiation import (
        DemoConfig,
        comparison_figure,
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

        The initial plots use straight tracks and the exact planar-film solver.
        **Run transported comparison** adds a small production Monte Carlo run.
        These are source densities at a direction, before detector response.
        """),
        ]
    )
    return (
        DemoConfig,
        comparison_figure,
        mo,
        straight_track_demo,
        transport_figure,
        transported_demo,
    )


@app.cell
def _(mo):
    scene_form = (
        mo.md("""
    **Film and bunch** — submit to update the comparison.

    {energy} {thickness} {angle}

    {duration} {spot} {charge}
    """)
        .batch(
            energy=mo.ui.slider(20, 100, value=30, step=5, label="Beam (keV)"),
            thickness=mo.ui.slider(1, 100, value=20, step=1, label="Film (nm)"),
            angle=mo.ui.slider(100, 175, value=150, step=1, label="Observation (deg from beam)"),
            duration=mo.ui.number(0.1, 100, value=1, step=0.1, label="RMS duration (as)"),
            spot=mo.ui.number(0.1, 100, value=1, step=0.1, label="Spot FWHM (nm)"),
            charge=mo.ui.number(1e-6, 0.001, value=0.0001, step=1e-6, label="Charge (pC)"),
        )
        .form(submit_button_label="Update comparison")
    )
    scene_form
    return (scene_form,)


@app.cell
def _(DemoConfig, scene_form, straight_track_demo):
    _values = scene_form.value
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
        )
    )
    analytic = straight_track_demo(config)
    return analytic, config


@app.cell
def _(mo):
    per_bunch_ui = mo.ui.checkbox(label="Show photons per physical bunch", value=False)
    cut_ui = mo.ui.slider(200, 1000, step=10, value=500, label="Angular-cut photon energy (eV)")
    mo.hstack([per_bunch_ui, cut_ui])
    return cut_ui, per_bunch_ui


@app.cell
def _(analytic, comparison_figure, config, cut_ui, per_bunch_ui):
    comparison_figure(analytic, config, per_bunch=per_bunch_ui.value, energy_eV=cut_ui.value)
    return


@app.cell
def _(config, mo):
    mo.md(f"""
    Physical population **N = {config.population:,.1f} electrons/bunch**.
    Straight identical tracks use the Gaussian intensity form factor and
    **1 + (N − 1)|F|²** per electron. A point bunch gives N² per bunch;
    a long bunch tends to the incoherent N scaling. No sampled offsets are
    used in these analytic plots. A single interface is a different target
    from a film: their difference includes entry/exit interference and absorption.
    """)
    return


@app.cell
def _(mo):
    transport_button = mo.ui.run_button(label="Run transported comparison")
    mo.vstack(
        [
            mo.md("""
    **Production comparison:** 32 line histories and 64 continuum histories,
    two fixed-count CPU runs, with sampled time and transverse offsets.
    It retains signed CTR estimates; negative bins indicate unresolved sampling.
    Changing the submitted scene requires another click. Cached results are
    reused when changing display units or angular-cut energy.
    """),
            transport_button,
        ]
    )
    return (transport_button,)


@app.cell
def _(config, mo, transport_button, transported_demo):
    mo.stop(not transport_button.value, mo.md("Click **Run transported comparison** to compute."))
    production, messages, elapsed = transported_demo(config)
    return elapsed, messages, production


@app.cell
def _(config, elapsed, messages, mo, per_bunch_ui, production, transport_figure):
    mo.vstack(
        [
            mo.md(
                f"CPU comparison: **{elapsed:.2f} s**. Signed values use a symmetric log scale. "
                "Combined CXR excludes the separately gridded bremsstrahlung continuum."
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
    optical/IR model. The production photon floor is max(plasma energy, 12 eV);
    this demo stays at 200 eV and above. Scalar acceptance metadata does not
    integrate narrow fringes: use the dedicated planar-detector integrator for
    channel counts. Adaptive precision, electron-block transport, channeling,
    grooves and temporal profiles are refused for TR. Large N with few sampled
    histories can overwhelm the CTR signal with noise; do not clip negative bins
    or count the form factor twice. Static HTML contains the initial analytic
    comparison; use the live notebook for controls and transport.
    """)
    return


if __name__ == "__main__":
    app.run()
