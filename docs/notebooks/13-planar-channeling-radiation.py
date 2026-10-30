# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.24.0",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=False)
def _():
    import os

    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from pyrite.experimental.quantum import propagate_planar_source, score_planar_source
    from pyrite.quantum import (
        QuantumDynamics,
        QuantumNumerics,
        QuantumRegion,
        QuantumSetup,
        TransverseStateSpec,
    )
    from pyrite.units import ureg

    return (
        QuantumDynamics,
        QuantumNumerics,
        QuantumRegion,
        QuantumSetup,
        TransverseStateSpec,
        mo,
        np,
        plt,
        propagate_planar_source,
        score_planar_source,
        ureg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Planar channeling and radiation

    Project a **1 Å** Gaussian into Si(110), evolve its populations over
    **0.2 µm**, and score forward radiation at **17 MeV**.
    This experimental CPU diagnostic uses a restricted finite basis;
    its physical predictions are not rigorously validated.
    """)
    return


@app.cell
def _(QuantumDynamics, QuantumNumerics, QuantumRegion, QuantumSetup, TransverseStateSpec, ureg):
    source = TransverseStateSpec.recipe("gaussian", waist=1 * ureg.angstrom, dimension=1)
    setup = QuantumSetup(
        region=QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
        dynamics=QuantumDynamics(kind="channeling"),
        numerics=QuantumNumerics(basis_size=8, bloch_cells=4, step_length=2 * ureg.nm),
    )
    entry = setup.project_state(
        source,
        crystal="silicon",
        kinetic_energy_eV=17 * ureg.MeV,
        incidence_rad=0 * ureg.mrad,
        impact="fixed",
    )
    depth = 0.2 * ureg.um
    return depth, entry


@app.cell
def _(depth, entry, propagate_planar_source):
    history = propagate_planar_source(entry, depth_ang=depth)
    return (history,)


@app.cell
def _(history, plt):
    fig_populations, _ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    _depth_um = history.z_ang / 10_000
    _ax.plot(_depth_um, history.trace, label="Retained subspace")
    _ax.plot(_depth_um, history.bound_population, label="Below barrier")
    for _band in range(3):
        _ax.plot(_depth_um, history.band_populations[:, _band], "--", lw=1, label=f"Band {_band}")
    _ax.set(xlabel="Depth (µm)", ylabel="Population per incident electron")
    _ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9)
    fig_populations  # noqa: B018 - displayed by marimo
    return


@app.cell
def _(depth, entry, np, score_planar_source, ureg):
    radiation = score_planar_source(
        entry,
        depth_ang=depth,
        photon_energy_eV=np.linspace(1_000.0, 50_000.0, 1_025) * ureg.eV,
        theta_rad=0 * ureg.mrad,
        sample_ang=10 * ureg.nm,
        absorption=False,
    )
    return (radiation,)


@app.cell
def _(plt, radiation):
    fig_radiation, _ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    _ax.plot(radiation.photon_energy_eV / 1_000, radiation.spectrum, label="Total")
    _ax.plot(
        radiation.photon_energy_eV / 1_000,
        radiation.capture_spectrum,
        "--",
        label="Capture component",
    )
    _ax.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="Forward source density · absorption disabled",
        ylim=(0, None),
    )
    _ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _ax.legend()
    fig_radiation  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Populations are normalized to incident electrons, without conditioning on
    survival. The spectrum includes capture, but no aperture, instrument
    response or beam-charge scaling. Incidence is the effective refracted
    angle to the planes, not laboratory tilt.

    Try changing depth or incidence, then refine the basis and depth step.
    This fixed-impact diagnostic is distinct from the restricted
    lattice-averaged Gaussian admitted by production transport.
    """)
    return


if __name__ == "__main__":
    app.run()
