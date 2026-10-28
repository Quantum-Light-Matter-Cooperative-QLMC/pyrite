# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.22.5",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.
    return mo, np, plt, pyrite


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting PXR line energies

    Predict ideal HOPG (002) resonance energies before simulating transport.
    `photon_energies()` takes beam energy in **eV**, angles in **degrees**,
    and returns a leading material axis even for one material.
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
    return beam_energy_eV, catalog, hkl, photon_energies, tilt_deg


@app.cell
def _(catalog, hkl, np, photon_energies, tilt_deg):
    observation_deg = np.linspace(30, 160, 131)
    energy_scan_eV = np.array([30_000.0, 60_000.0, 100_000.0])
    energy_lines_eV = photon_energies(
        "hopg",
        energy_scan_eV[:, None],
        tilt_deg=tilt_deg,
        theta_obs_deg=observation_deg[None, :],
        hkl=hkl,
        catalog=catalog,
    )[0]
    return energy_lines_eV, energy_scan_eV, observation_deg


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
    return


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


@app.cell(hide_code=True)
def _(mo, second_order_eV, tilt_scan_deg, tilting_lines_eV):
    mo.md(
        f"At 135° observation, tilts {tilt_scan_deg}° give "
        f"{[round(float(x), 1) for x in tilting_lines_eV]} eV. "
        f"The (004) line at 30° tilt is {second_order_eV:.1f} eV."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    These are **vacuum line centers**, not intensities or counts. The model
    assumes straight electrons at their entrance energy and an ideal lattice;
    it excludes transport, mosaic spread, in-medium dispersion and response.
    A kinematically allowed line need not have measurable intensity.

    Try changing `tilt_deg`, electron energy or `hkl`. The (004) reflection
    appears at twice the (002) energy for the same geometry. Compare with
    notebook 01's spectrum and the
    [physics and validation](https://pyrite.readthedocs.io/en/latest/physics/index.html)
    documentation.
    """)
    return


@app.cell(hide_code=True)
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
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
