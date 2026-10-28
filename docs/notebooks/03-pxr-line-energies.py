# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.1", "pyrite-mc==0.22.4", "matplotlib>=3.11", "numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    return mo, np, os, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # Predicting PXR line energies

    [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/03-pxr-line-energies.py)

    Predict where a HOPG reflection appears before running Monte Carlo transport.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup

    `photon_energies()` uses electron kinetic energy in **eV** and angles in degrees.
    It returns a leading material axis, even for one material. Here the basal (002)
    reflection, sample tilt, and lab observation angle are explicit.
    """)
    return


@app.cell
def _():
    from pyrite.materials import load_material_catalog
    from pyrite.radiation import photon_energies

    catalog = load_material_catalog()
    beam_energy_eV = 30_000.0
    tilt_deg = 30.0
    hkl = (0, 0, 2)
    return catalog, beam_energy_eV, hkl, photon_energies, tilt_deg


@app.cell
def _(beam_energy_eV, catalog, hkl, np, photon_energies, tilt_deg):
    observation_deg = np.linspace(30, 160, 131)
    line_eV = photon_energies(
        "hopg",
        beam_energy_eV,
        tilt_deg=tilt_deg,
        theta_obs_deg=observation_deg,
        hkl=hkl,
        catalog=catalog,
    )[0]
    energy_scan_eV = np.array([30_000.0, 60_000.0, 100_000.0])
    energy_lines_eV = photon_energies(
        "hopg",
        energy_scan_eV[:, None],
        tilt_deg=tilt_deg,
        theta_obs_deg=observation_deg[None, :],
        hkl=hkl,
        catalog=catalog,
    )[0]
    return energy_lines_eV, energy_scan_eV, line_eV, observation_deg


@app.cell
def _(energy_lines_eV, energy_scan_eV, observation_deg, plt):
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for _energy, _line in zip(energy_scan_eV, energy_lines_eV, strict=True):
        axis.plot(observation_deg, _line / 1_000, label=f"{_energy / 1_000:g} keV electrons")
    axis.set(
        xlabel="Lab observation angle (°)",
        ylabel="Vacuum line energy (keV)",
        title="HOPG (002), sample tilt 30°",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure  # noqa: B018 - marimo renders the final expression
    return (figure,)


@app.cell
def _(beam_energy_eV, catalog, hkl, photon_energies):
    tilt_scan_deg = [0.0, 30.0, 60.0]
    tilting_lines_eV = photon_energies(
        "hopg",
        beam_energy_eV,
        tilt_deg=tilt_scan_deg,
        theta_obs_deg=135.0,
        hkl=hkl,
        catalog=catalog,
    )[0]
    second_order_eV = photon_energies(
        "hopg",
        beam_energy_eV,
        tilt_deg=30.0,
        theta_obs_deg=135.0,
        hkl=(0, 0, 4),
        catalog=catalog,
    )[0]
    return second_order_eV, tilt_scan_deg, tilting_lines_eV


@app.cell
def _(mo, second_order_eV, tilt_scan_deg, tilting_lines_eV):
    mo.md(
        f"At 135° observation, tilts {tilt_scan_deg}° give "
        f"{[round(float(x), 1) for x in tilting_lines_eV]} eV. "
        f"The (004) line at 30° tilt is {second_order_eV:.1f} eV."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and next steps

    These are **vacuum line centres**, not intensities or detector counts. A straight
    entrance-energy electron and one ideal crystal orientation are assumed. Transport,
    energy loss, mosaic averaging, in-medium dispersion and detector response are
    excluded. An allowed energy does not guarantee nonzero intensity; unreachable
    resonances return NaN. Compare with the captured spectrum in notebook 01 only
    with those approximations in mind.

    Try changing `tilt_deg` or `hkl`. Consult the
    [radiation API](https://pyrite.readthedocs.io/en/latest/api.html#advanced-transport-and-radiation)
    and [physics and validation](https://pyrite.readthedocs.io/en/latest/physics/index.html)
    for conventions and model limitations.
    """)
    return


@app.cell
def _(catalog, hkl, mo, pyrite):
    import hashlib

    tutorial_provenance = {
        "tutorial": "03-pxr-line-energies",
        "pyrite_version": pyrite.__version__,
        "backend": "cpu",
        "seed": None,
        "table_identity": "not used: deterministic catalog geometry only",
        "material": "hopg",
        "hkl": list(hkl),
        "catalog_cif_sha256": hashlib.sha256(catalog.crystals["hopg"].cif.read_bytes()).hexdigest(),
    }
    mo.json(tutorial_provenance)
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
