# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.27.0",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Set the backend before importing PyRITE; restart the kernel to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    with mo.status.spinner(title="Preparing PyRITE reference tables, warming JIT caches"):
        _ = pyrite.fetch_tables()  # Fetch data tables
        _ = pyrite.execution.warm_cache()

    tables_ready = True
    return mo, np, plt, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding simulation results

    Convert an electron-normalized PXR/CBS spectrum into a photon rate for
    a specified beam current and angular acceptance. A **0.5 pC** bunch at
    **100 kHz** gives **50 nA** average current; the **8,192 sampled histories**
    are numerical estimators, not bunch electrons.
    """)
    return


@app.cell
def _(tables_ready):
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab
    from pyrite.units import ureg

    beam = Beam(
        kinetic_energy=30 * ureg.keV,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
        bunch_charge=0.5 * ureg.pC,
        repetition_rate=100 * ureg.kHz,
    )
    target = Slab(
        "hopg",
        thickness=100 * ureg.nm,
        tilt=30 * ureg.degree,
        footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
    )
    detector = Detector(observation_angle=135 * ureg.degree)
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=8_192, brem_electrons=128),
        cxr_reflections=CxrReflections(n_families=1),
    )
    assert tables_ready  # Reference tables must be ready before transport.
    result = simulate(beam, target, detector, numerics=numerics)
    return beam, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the native spectra

    `to_dataframe()` returns a Polars table: one row per energy sample of
    one component. `component` identifies CXR, bremsstrahlung or characteristic
    radiation; `mode` distinguishes incoherent/coherent CXR (null for other
    components). `photon_energy_eV` is in eV, and `density` is in
    **photons / (incident electron · eV · sr)**. `normalization="source"`
    identifies the source density used here.

    Components keep their own energy grids, so rows with the same index need
    not describe the same energy. This table has no summed `total`, beam-current
    scaling or detector efficiency. Integrating `density` over an energy band
    gives photons / (incident electron · sr).
    """)
    return


@app.cell
def _(result):
    # Inspect component spectra on their native energy grids.
    result.to_dataframe()
    return


@app.cell
def _(plt, result):
    cxr = result.cxr.incoherent
    brem = result.brem
    fig_source, ax_source = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax_source.plot(cxr.photon_energy_eV / 1_000, cxr.density, label="PXR + CBS")
    ax_source.plot(brem.photon_energy_eV / 1_000, brem.density, label="Bremsstrahlung")
    if result.characteristic is not None:
        line = result.characteristic
        ax_source.plot(line.photon_energy_eV / 1_000, line.density, label="Characteristic")
    ax_source.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="Raw source components on native grids",
        xlim=(0.3, 3.7),
        yscale="log",
    )
    ax_source.grid(alpha=0.25)
    ax_source.legend(fontsize=8)
    fig_source  # noqa: B018 - marimo displays the figure
    return (cxr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explicit current and matching acceptance

    `.quantify().rate(current=...)` changes per-primary yield into photons
    per second using $I/|e|$. It keeps the per-sr directional measure.
    `accepted_spectra(assume_uniform=True)` uses this scalar case's acceptance
    with explicit acknowledgment that density is uniform over it. Each
    component keeps its native sampling and energy grid. `to_energy("keV")`
    converts the energy coordinate and reciprocal density together.
    """)
    return


@app.cell
def _(beam, result):
    from pyrite.units import Q_

    mean_current = Q_(beam.bunch_charge_pc, "pC") * Q_(beam.rep_rate_hz, "Hz")
    per_sr = result.cxr.incoherent.quantify().rate(current=mean_current)
    accepted = result.accepted_spectra(assume_uniform=True)
    into_acceptance = {
        name: component.rate(current=mean_current).to_energy("keV")
        for name, component in accepted.items()
    }
    per_current = accepted["cxr"].rate_per_current()
    # One native table per component; write_csv/write_parquet can export it.
    into_acceptance["cxr"].to_dataframe().head(5)
    return accepted, into_acceptance, mean_current, per_current, per_sr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this beam, **50 nA ≈ 3.12 × 10¹¹ electrons / s**. An accepted
    per-current coefficient is computed directly with $1/|e|$, even at zero
    beam current. A second current transformation is rejected. The native
    tables retain semantic columns; `dataframe_units()` supplies physical
    units for a JSON sidecar. These are expected photon rates, not acquired
    counts after detector response, exposure and a possible Poisson draw.

    Integrate each original component with its declared sampling, then add
    its integral. A mixed node/bin-mean `combined_cxr()` may be plotted but
    refuses integration; `integrated_yield()` integrates its parts honestly.
    No union-grid interpolation or zero-filled tails are needed for totals.
    """)
    return


@app.cell
def _(accepted, mean_current, np, per_current, per_sr, result):
    from scipy.constants import elementary_charge

    band_eV = (600.0, 2_000.0)
    represented = result.integrated_yield(band_eV, mode="incoherent")
    electron_rate = mean_current.to("A").magnitude / elementary_charge
    accepted_total_rate = (
        represented["total"].inside * electron_rate * result.accepted_solid_angle_sr()
    )
    # Metadata reports unsupported portions of the requested window separately.
    print("Represented accepted band rate [photons/s]:", accepted_total_rate)
    print("CXR uncovered windows [eV]:", represented["cxr"].uncovered_eV)
    assert np.allclose(
        (per_current.density * mean_current).to("1/eV/s").magnitude,
        accepted["cxr"].rate(current=mean_current).density.magnitude,
    )
    assert per_sr.measure == "per_energy_per_steradian"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot accepted components on their native grids
    """)
    return


@app.cell
def _(into_acceptance, plt):
    fig_rate, ax_rate = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for _name, _view in into_acceptance.items():
        ax_rate.plot(_view.photon_energy.magnitude, _view.density.magnitude, label=_name)
    ax_rate.axvspan(0.6, 2.0, alpha=0.12, color="grey", label="Integration band")
    ax_rate.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (s · keV)",
        title="Native components scaled by explicit current and matching acceptance",
        xlim=(0.3, 3.7),
        yscale="log",
    )
    ax_rate.grid(alpha=0.25)
    ax_rate.legend(fontsize=8)
    fig_rate  # noqa: B018 - marimo displays the figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    `to_dataframe()` preserves each component's native energy grid.
    Quantity tables record paired energy/density units, measure and
    normalization. `combined_cxr()` combines same-grid line components;
    incoherent and coherent CXR are alternative choices. The **600–2,000 eV**
    represented total sums native component integrals and reports uncovered
    support separately; the photon-rate plot retains separate components. These are _expected source rates_,
    not measured counts: no exposure, efficiency or Poisson noise is applied.
    The scalar acceptance assumes locally uniform angular density.

    Try doubling `bunch_charge_pc` without changing the history count.
    Notebook 06 adds detector response and counts. See the
    [results how-to](https://pyrite.readthedocs.io/en/latest/guides/working-with-results.html)
    for saving and reloading results.
    """)
    return


if __name__ == "__main__":
    app.run()
