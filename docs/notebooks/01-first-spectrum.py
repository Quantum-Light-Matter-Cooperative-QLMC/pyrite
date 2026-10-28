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

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.

    tables_ready = True
    return mo, plt, pyrite, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Your first X-ray spectrum

    Simulate a small crystalline target and compare PXR + CBS with bremsstrahlung.

    ## Setup

    This example uses 30 keV electrons incident upon a 100 nm HOPG slab. Its small
    trial counts keep it CPU-friendly; the curves are **only illustrative**.
    The sampled electron histories do not represent a physical bunch population.
    """)
    return


@app.cell
def _(tables_ready):
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    beam = Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch())
    target = Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
    detector = Detector(observation_angle_deg=135.0)

    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=32_768, brem_electrons=128),
        cxr_reflections=CxrReflections(n_families=2),
    )

    assert tables_ready  # Wait for reference-table installation before transport.
    result = simulate(beam, target, detector, numerics=numerics)

    cxr = result.cxr.incoherent
    brem = result.brem
    return brem, cxr, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(brem, cxr, plt):
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    axis.plot(cxr.photon_energy_eV / 1_000, cxr.density, label="PXR + CBS")
    axis.plot(brem.photon_energy_eV / 1_000, brem.density, label="Bremsstrahlung")
    axis.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (electron · eV · sr)",
        title="30 keV electrons → 100 nm HOPG",
        xlim=(0.3, 3.7),
        yscale="log",
        ylim=(1e-14, 1e-7),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    PXR and CBS are combined here because their amplitudes interfere. Their line
    structure differs from the broad bremsstrahlung continuum. Each component is plotted
    on its **native photon-energy grid**; adding array entries by index would mix
    different energies. The vertical axis is source spectral density per incident
    electron, per eV, per steradian, i.e., integrating over the spectrum, beam current,
    and detection solid-angle cone would translate to Phs

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


@app.cell(hide_code=True)
def _(mo, pyrite, result):
    tutorial_provenance = {
        "tutorial": "01-first-spectrum",
        "pyrite_version": pyrite.__version__,
        "backend": result.provenance["backend"],
        "seed": result.case["seed"],
        "table_identity": dict(result.provenance["xsgen_tables"]),
        "source_identity": result.provenance["identity_digest"],
    }
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
