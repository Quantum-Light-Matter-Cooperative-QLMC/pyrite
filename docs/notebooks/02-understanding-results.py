# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.25.0",
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

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.

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
    ## Put components on one grid and choose units

    `spectral_flux()` returns a wide Polars table: one row per energy, one
    column per component, plus `total`. It combines all native energy nodes,
    linearly interpolates each component between its nodes, and uses zero
    outside that component's range. Columns can now be added at the same energy.

    `units` chooses the normalization of **every component and `total`**:

    | `units=`               | Spectral values                         | Scaling from `"electron"`          |
    | ---------------------- | --------------------------------------- | ---------------------------------- |
    | `"electron"`           | photons / (incident electron · eV · sr) | None                               |
    | `"sr"`                 | photons / (s · eV · sr)                 | Multiply by physical electrons / s |
    | `"detector"` (default) | photons / (s · eV) into acceptance      | Also multiply by accepted sr       |
    | `"nA"`                 | photons / (s · nA · eV) into acceptance | Accepted rate per 1 nA of current  |

    The energy columns remain in eV and keV; selecting `photon_energy_keV`
    does **not** change a density from per eV to per keV (that needs ×1,000).
    These are expected rates before detector efficiency and exposure, not
    acquired counts. This scalar-detector conversion assumes intensity is
    uniform across its accepted solid angle.
    """)
    return


@app.cell
def _(result):
    # Each table has a common, interpolated energy grid and explicit units.
    per_sr = result.spectral_flux(units="sr")
    into_acceptance = result.spectral_flux(units="detector")
    into_acceptance  # noqa: B018 - display the common-grid table in marimo
    return into_acceptance, per_sr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this beam, **50 nA ≈ 3.12 × 10¹¹ electrons / s**. Thus the `"sr"`
    table is the electron-normalized table multiplied by that rate; the
    `"detector"` table also includes the detector's accepted solid angle.
    Integrating a `"detector"` column over energy in **eV** gives photons / s.
    The following check applies those conversions to the CXR band alone.
    """)
    return


@app.cell
def _(beam, cxr, into_acceptance, np, per_sr, result):
    from scipy.constants import elementary_charge

    band_eV = (600.0, 2_000.0)
    lo, hi = band_eV
    energy = cxr.photon_energy_eV
    nodes = np.r_[lo, energy[(energy > lo) & (energy < hi)], hi]

    # Integrating photon number flux density per electron per sr
    cxr_yield_per_electron_sr = float(np.trapezoid(np.interp(nodes, energy, cxr.density), nodes))

    current_nA = beam.bunch_charge_pc * beam.rep_rate_hz / 1_000  # Current
    electrons_per_second = current_nA * 1e-9 / elementary_charge  # electron number flux density

    flux_sr_grid = per_sr.get_column("photon_energy_eV").to_numpy()
    flux_sr_cxr = per_sr.get_column("cxr").to_numpy()
    cxr_rate_sr = float(np.trapezoid(np.interp(nodes, flux_sr_grid, flux_sr_cxr), nodes))

    # For a scalar detector, the accepted rate assumes locally uniform intensity.
    result.accepted_solid_angle_sr()
    assert np.isclose(cxr_rate_sr, cxr_yield_per_electron_sr * electrons_per_second, rtol=1e-10)
    assert np.array_equal(into_acceptance.get_column("photon_energy_eV").to_numpy(), flux_sr_grid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot the complete spectrum summed over all components
    """)
    return


@app.cell
def _(into_acceptance, plt):
    fig_rate, ax_rate = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax_rate.plot(
        into_acceptance.get_column("photon_energy_eV").to_numpy() / 1_000,
        into_acceptance.get_column("total").to_numpy(),
        label="Total accepted photon rate",
    )
    ax_rate.axvspan(0.6, 2.0, alpha=0.12, color="grey", label="CXR integration band")
    ax_rate.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (s · eV)",
        title="Source spectrum scaled by current and solid-angle acceptance",
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

    `to_dataframe()` preserves each component's native energy grid;
    `spectral_flux()` interpolates onto a common one. `"electron"` gives
    the electron-normalized density, `"sr"` scales by the physical electron
    rate, and `"detector"` also multiplies by accepted solid angle.

    The **600–2,000 eV** band integral uses CXR only; the photon-rate plot
    includes all computed components. These are _expected source rates_,
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
