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

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting PXR line energies

    Predict ideal HOPG resonance energies before simulating transport.
    `photon_energies()` takes beam energy in **eV**, angles in **degrees**,
    and returns a leading material axis even for one material. It is
    analytic, so the controls below recompute instantly.
    """)
    return


@app.cell
def _():
    from pyrite.materials import load_material_catalog
    from pyrite.radiation import photon_energies

    catalog = load_material_catalog()
    return catalog, photon_energies


@app.cell
def _(mo):
    # Live in Molab or a local marimo session; the static page shows the defaults.
    energy_keV = mo.ui.slider(10, 200, step=5, value=30, label="Electron energy (keV)")
    tilt_deg = mo.ui.slider(0, 75, step=5, value=30, label="Sample tilt (°)")
    observation_deg = mo.ui.slider(30, 160, step=5, value=135, label="Observation angle (°)")
    reflection = mo.ui.dropdown(
        {"(002)": (0, 0, 2), "(004)": (0, 0, 4), "(006)": (0, 0, 6)},
        value="(002)",
        label="Reflection",
    )
    mo.vstack([energy_keV, tilt_deg, observation_deg, reflection])
    return energy_keV, observation_deg, reflection, tilt_deg


@app.cell
def _(catalog, energy_keV, np, observation_deg, photon_energies, reflection, tilt_deg):
    from pyrite.units import ureg

    angles_deg = np.linspace(30, 160, 131)
    reference_keV = np.array([60.0, 100.0, 200.0])
    # The analytic API takes numeric eV/degree magnitudes, not Pint objects.
    beam_energies_eV = (np.r_[reference_keV, energy_keV.value] * ureg.keV).to("eV").magnitude
    lines_eV = photon_energies(
        "hopg",
        beam_energies_eV[:, None],
        tilt_deg=tilt_deg.value,
        theta_obs_deg=angles_deg[None, :],
        hkl=reflection.value,
        catalog=catalog,
    )[0]
    selected_eV = float(
        photon_energies(
            "hopg",
            (energy_keV.value * ureg.keV).to("eV").magnitude,
            tilt_deg=tilt_deg.value,
            theta_obs_deg=observation_deg.value,
            hkl=reflection.value,
            catalog=catalog,
        )[0]
    )
    return angles_deg, lines_eV, reference_keV, selected_eV


@app.cell
def _(
    angles_deg,
    energy_keV,
    lines_eV,
    observation_deg,
    plt,
    reference_keV,
    reflection,
    selected_eV,
    tilt_deg,
):
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for _energy, _line in zip(reference_keV, lines_eV[:-1], strict=True):
        axis.plot(angles_deg, _line / 1_000, lw=1, alpha=0.5, label=f"{_energy:g} keV")
    axis.plot(
        angles_deg, lines_eV[-1] / 1_000, "k", lw=2, label=f"{energy_keV.value:g} keV (selected)"
    )
    axis.plot(observation_deg.value, selected_eV / 1_000, "ro", zorder=3)
    axis.annotate(
        f"{selected_eV:.0f} eV",
        (observation_deg.value, selected_eV / 1_000),
        textcoords="offset points",
        xytext=(-8, 8),
        ha="right",
    )
    axis.set(
        xlabel="Lab observation angle (°)",
        ylabel="Vacuum line energy (keV)",
        title=f"HOPG {reflection.selected_key}, sample tilt {tilt_deg.value:g}°",
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Electron energy", fontsize=8)
    figure  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    These are **vacuum line centers**, not intensities or counts. Faster
    electrons raise every line, most strongly near forward observation, because
    the resonance energy scales as $\mathbf v\cdot\mathbf g/(1-\hat{\mathbf n}\cdot\mathbf v)$.
    Tilting the sample rotates $\mathbf g$ relative to the beam and slides the
    whole curve; higher orders of the same planes, such as (004), appear at
    integer multiples of the (002) energy.

    The model assumes straight electrons at their entrance energy and an ideal
    lattice; it excludes transport, mosaic spread, in-medium dispersion and
    response. A kinematically allowed line need not have measurable intensity.
    Compare the selected point with notebook 01's spectrum, then see the
    [physics and validation](https://pyrite.readthedocs.io/en/latest/physics/index.html)
    documentation.
    """)
    return


if __name__ == "__main__":
    app.run()
