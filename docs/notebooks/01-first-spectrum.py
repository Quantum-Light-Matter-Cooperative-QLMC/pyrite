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
    # Your first X-ray spectrum

    [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/01-first-spectrum.py)

    Simulate a small crystalline target and compare PXR + CBS with bremsstrahlung.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup

    Complete notebook 00 once to install PyRITE and fetch/verify its reference tables.
    This example uses 30 keV electrons and a 100 nm HOPG slab. Its fixed history
    counts keep it CPU-friendly; the curves are **illustrative, not converged**.
    The sampled electron histories do not represent a physical bunch population.
    """)
    return


@app.cell
def _():
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    return (
        Beam,
        CxrReflections,
        Detector,
        FixedSampling,
        Footprint,
        Numerics,
        PointBunch,
        PointSpot,
        Slab,
        simulate,
    )


@app.cell
def _(Beam, Detector, Footprint, PointBunch, PointSpot, Slab):
    beam = Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch())
    target = Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
    detector = Detector(observation_angle_deg=135.0)
    return beam, detector, target


@app.cell
def _(CxrReflections, FixedSampling, Numerics):
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=32_768, brem_electrons=128),
        cxr_reflections=CxrReflections(n_families=2),
    )
    return (numerics,)


@app.cell
def _(beam, detector, numerics, simulate, target):
    result = simulate(beam, target, detector, numerics=numerics)
    return (result,)


@app.cell
def _(result):
    cxr = result.cxr.incoherent
    brem = result.brem
    return brem, cxr


@app.cell
def _(brem, cxr, plt):
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    axis.plot(cxr.photon_energy_eV / 1_000, cxr.density, label="PXR + CBS")
    axis.plot(brem.photon_energy_eV / 1_000, brem.density, label="Bremsstrahlung")
    axis.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="30 keV electrons → 100 nm HOPG",
        xlim=(0.3, 3.7),
        yscale="log",
        ylim=(1e-14, 1e-7),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure  # noqa: B018 - marimo renders the final expression
    return (figure,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    PXR and CBS are combined here because their amplitudes interfere. Their line
    structure differs from the broad bremsstrahlung continuum. Plot each component
    on its **native photon-energy grid**; adding array entries by index would mix
    different energies. The vertical axis is source spectral density per incident
    electron, per eV, per steradian: it is neither physical beam flux nor acquired
    counts. Low history counts can leave noisy features and empty bins.

    Try doubling the history counts, then change only the target thickness. Keep a
    record of the resolved seed when comparing configurations; different backends need not produce
    identical random trajectories. Continue to notebook 02 to separate source
    yields from beam-current-scaled photon rates. Notebook 03 predicts vacuum line
    energies; the [results how-to](https://pyrite.readthedocs.io/en/latest/guides/working-with-results.html)
    covers saved checkpoints. See
    [model limitations](https://pyrite.readthedocs.io/en/latest/physics/capabilities.html)
    and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html)
    before interpreting these small-sample results quantitatively.
    """)
    return


@app.cell
def _(mo, pyrite, result):
    tutorial_provenance = {
        "tutorial": "01-first-spectrum",
        "pyrite_version": pyrite.__version__,
        "backend": result.provenance["backend"],
        "seed": result.case["seed"],
        "table_identity": dict(result.provenance["xsgen_tables"]),
        "source_identity": result.provenance["identity_digest"],
    }
    mo.json(tutorial_provenance)
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
