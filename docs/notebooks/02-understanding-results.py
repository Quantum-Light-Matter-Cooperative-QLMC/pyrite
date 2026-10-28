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

    # Set the backend before importing PyRITE; restart the kernel to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.

    tables_ready = True
    return mo, np, plt, pyrite, tables_ready


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

    beam = Beam(
        energy_keV=30.0,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
        bunch_charge_pc=0.5,
        rep_rate_hz=100_000.0,
    )
    target = Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(5, 5))
    detector = Detector(observation_angle_deg=135.0)
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
    ### Examine the returned data structures
    by looking at component spectra on their own native energy grids
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
        title="Response-free source components on native grids",
        xlim=(0.3, 3.7),
        yscale="log",
    )
    ax_source.grid(alpha=0.25)
    ax_source.legend(fontsize=8)
    fig_source  # noqa: B018 - marimo displays the figure
    return (cxr,)


@app.cell
def _(result):
    # Each table has a common, interpolated energy grid and explicit units.
    # per_electron = result.spectral_flux(units="electron")
    # per_sr = result.spectral_flux(units="sr")
    into_acceptance = result.spectral_flux(units="detector")
    return (into_acceptance,)


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
    acceptance_sr = result.accepted_solid_angle_sr()
    assert np.isclose(cxr_rate_sr, cxr_yield_per_electron_sr * electrons_per_second, rtol=1e-10)
    assert np.array_equal(into_acceptance.get_column("photon_energy_eV").to_numpy(), flux_sr_grid)
    return (
        acceptance_sr,
        band_eV,
        current_nA,
        cxr_rate_sr,
        cxr_yield_per_electron_sr,
    )


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
    includes all computed components. These are *expected source rates*,
    not measured counts: no exposure, efficiency or Poisson noise is applied.
    The scalar acceptance assumes locally uniform angular density.

    Try doubling `bunch_charge_pc` without changing the history count.
    Notebook 06 adds detector response and counts. See the
    [results how-to](https://pyrite.readthedocs.io/en/latest/guides/working-with-results.html)
    for saving and reloading results.
    """)
    return


@app.cell(hide_code=True)
def _(
    acceptance_sr,
    band_eV,
    current_nA,
    cxr_rate_sr,
    cxr_yield_per_electron_sr,
    mo,
    pyrite,
    result,
):
    band_summary = {
        "band_eV": list(band_eV),
        "mean_current_nA": current_nA,
        "sampled_cxr_histories": 8_192,
        "accepted_solid_angle_sr": acceptance_sr,
        "cxr_yield_photons_per_electron_sr": cxr_yield_per_electron_sr,
        "cxr_rate_photons_per_s_sr": cxr_rate_sr,
        "cxr_rate_into_acceptance_photons_per_s": cxr_rate_sr * acceptance_sr,
    }
    tutorial_provenance = {
        "tutorial": "02-understanding-results",
        "pyrite_version": pyrite.__version__,
        "backend": result.provenance["backend"],
        "seed": result.case["seed"],
        "table_identity": dict(result.provenance["xsgen_tables"]),
        "source_identity": result.provenance["identity_digest"],
        "band_eV": band_summary["band_eV"],
        "mean_current_nA": band_summary["mean_current_nA"],
        "accepted_solid_angle_sr": band_summary["accepted_solid_angle_sr"],
        "sampled_cxr_histories": band_summary["sampled_cxr_histories"],
    }
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
