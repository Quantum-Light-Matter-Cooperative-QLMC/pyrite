"""Local CPU exploration of the maintained experimental planar example."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from experimental_planar_channeling import compute, plot

    return compute, mo, plot


@app.cell
def _(mo):
    mo.md(r"""
    # Experimental planar channeling

    **Not rigorously validated.** Explore the transverse populations and
    forward radiation of a fixed-impact 1D Gaussian in Si(110) at 17 MeV.
    This direct density-matrix diagnostic runs locally on CPU; it does not
    run production transport, write checkpoints, or predict detector counts.

    The small basis and fixed grids demonstrate the interface, without a
    convergence claim. Refine basis size, Bloch cells, depth step and photon
    grid independently before interpreting rates or line shapes.
    """)
    return


@app.cell
def _(mo):
    controls = (
        mo.md("""
    Depth (Å): {depth}

    Effective incidence to the planes (mrad): {incidence}
    """)
        .batch(
            depth=mo.ui.slider(500, 5_000, step=500, value=2_000),
            incidence=mo.ui.slider(-0.2, 0.2, step=0.02, value=0.0),
        )
        .form(submit_button_label="Recompute diagnostic")
    )
    controls  # noqa: B018 - marimo displays this cell's final expression.
    return (controls,)


@app.cell
def _(compute, controls):
    parameters = controls.value or {"depth": 2_000, "incidence": 0.0}
    data = compute(
        depth_ang=float(parameters["depth"]),
        incidence_rad=float(parameters["incidence"]) * 1e-3,
    )
    return (data,)


@app.cell
def _(data, plot):
    plot(data)
    return


@app.cell
def _(data, mo):
    mo.md(f"""
    Final retained population: **{data["retained_population"][-1]:.4f}**;
    below-barrier population: **{data["bound_population"][-1]:.4f}** per
    incident electron. Entry retained/free/outside weights:
    **{data["entry"]["retained"]:.4f} / {data["entry"]["free"]:.4f} /
    {data["entry"]["outside"]:.4f}**.

    Entry weights describe the initial projection, not physical escape.
    Evolution populations are unconditioned on survival. The radiation is
    photons/(incident electron · eV · sr) at θ = 0, including capture, with
    absorption disabled, zero in-plane divergence and no instrument response.
    No finite-aperture integration or charge scaling has been applied.

    Parameters use Å, eV and radians internally. Incidence is the effective
    refracted angle to the planes, not laboratory target tilt. This 1D
    reduction is not a model of electron or photon OAM. Simplified attenuation,
    side-face escape and accumulated in-plane diffusion limit the broader model.
    """)
    return


if __name__ == "__main__":
    app.run()
