"""Local CPU exploration of the maintained experimental axial example."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from experimental_axial_channeling import compute, plot

    return compute, mo, plot


@app.cell
def _(mo):
    mo.md(r"""
    # Experimental axial channeling

    **Not rigorously validated.** Evolve a pure 2D Gaussian in Si[001] at
    5 MeV and inspect density, initial-state overlap and periodic-grid norm.
    This coherent CPU diagnostic has no scattering or absorption and does
    not enable axial production transport or predict measured dechanneling.
    The small finite FFT grid demonstrates the workflow, without a convergence
    claim. Refine supercell extent, samples per cell and propagation step
    independently before physical interpretation.
    """)
    return


@app.cell
def _(mo):
    controls = (
        mo.md("""
    Depth (Å): {depth}

    Gaussian waist (Å): {waist}
    """)
        .batch(
            depth=mo.ui.slider(200, 1_000, step=100, value=500),
            waist=mo.ui.slider(0.3, 0.9, step=0.1, value=0.5),
        )
        .form(submit_button_label="Recompute diagnostic")
    )
    controls  # noqa: B018 - marimo displays this cell's final expression.
    return (controls,)


@app.cell
def _(compute, controls):
    parameters = controls.value or {"depth": 500, "waist": 0.5}
    data = compute(depth_ang=float(parameters["depth"]), waist_ang=float(parameters["waist"]))
    return (data,)


@app.cell
def _(data, plot):
    plot(data, view="density")
    return


@app.cell
def _(data, plot):
    plot(data, view="evolution")
    return


@app.cell
def _(data, mo):
    mo.md(f"""
    Final retained grid norm: **{data["retained_norm"][-1]:.4f}**;
    squared overlap with the initial normalized state:
    **{data["initial_state_overlap"][-1]:.4f}**. Density is per incident
    electron per Å², without conditioning on survival; its integral equals
    the retained grid norm.
    Pre-normalization sampled entry norm: **{data["sampled_norms"][0]:.4f}**.

    `cells=(2, 2)` counts crystal periods; `shape=(64, 64)` samples the whole
    periodically repeated supercell. The Gaussian is centered at (1, 1) Å,
    with zero carrier tilt. One supercell edge connects to its opposite edge.
    The solver periodizes and normalizes the entry on this finite grid; norm
    preservation does not establish an isolated-beam approximation. Increasing
    extent without increasing shape reduces resolution.

    There is no retained eigenstate cutoff here. Periodic crossings, omitted
    spectral weight and numerical sinks are not physical escape. Coherent norm
    is not scattering survival; changing overlap is not population loss.
    Overlap is a pure-state quantity; mixed ensembles have no single
    wavefunction phase. Config fields use bare numeric Å, eV and radians.
    """)
    return


if __name__ == "__main__":
    app.run()
