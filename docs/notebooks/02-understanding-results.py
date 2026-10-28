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

    # Set the backend before importing PyRITE; restart the kernel to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.

    return mo, np, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # Understanding simulation results

    [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/02-understanding-results.py)

    Turn an electron-normalized source spectrum into an energy-integrated yield
    and a physical photon rate, without confusing either with detector counts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A source and its numerical sampling

    This independently runnable example follows notebook 01. A **0.5 pC** bunch
    at **100 kHz** corresponds to **50 nA** average beam current, regardless of
    how many Monte Carlo histories we sample. The low history counts below are
    chosen for teaching, **not** precision or convergence.
    """)
    return


@app.cell
def _():
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
        cxr_reflections=CxrReflections(n_families=2),
    )
    return beam, detector, numerics, simulate, target


@app.cell
def _(beam, detector, numerics, simulate, target):
    result = simulate(beam, target, detector, numerics=numerics)
    return (result,)


@app.cell
def _(mo, result):
    # Long-form rows retain the original energy nodes of each radiation family.
    native_rows = result.to_dataframe()
    native_overview = native_rows.group_by("component").len().sort("component").to_dicts()
    mo.json({"native_grid_sizes": native_overview, "sample_rows": native_rows.head(3).to_dicts()})
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
    per_electron = result.spectral_flux(units="electron")
    per_sr = result.spectral_flux(units="sr")
    into_acceptance = result.spectral_flux(units="detector")
    return into_acceptance, per_electron, per_sr


@app.cell
def _(beam, cxr, into_acceptance, mo, np, per_sr, result):
    from scipy.constants import elementary_charge

    band_eV = (600.0, 2_000.0)
    lo, hi = band_eV
    energy = cxr.photon_energy_eV
    if not energy[0] <= lo < hi <= energy[-1]:
        raise ValueError("selected band must fit entirely within the CXR grid")
    nodes = np.r_[lo, energy[(energy > lo) & (energy < hi)], hi]
    # Insert exact band endpoints before integrating a sampled density.
    yield_per_electron_sr = float(np.trapezoid(np.interp(nodes, energy, cxr.density), nodes))
    current_nA = beam.bunch_charge_pc * beam.rep_rate_hz / 1_000
    electrons_per_second = current_nA * 1e-9 / elementary_charge
    flux_sr_grid = per_sr.get_column("photon_energy_eV").to_numpy()
    flux_sr_cxr = per_sr.get_column("cxr").to_numpy()
    cxr_rate_sr = float(np.trapezoid(np.interp(nodes, flux_sr_grid, flux_sr_cxr), nodes))
    # For a scalar detector, the accepted rate assumes locally uniform intensity.
    acceptance_sr = result.accepted_solid_angle_sr()
    assert np.isclose(cxr_rate_sr, yield_per_electron_sr * electrons_per_second, rtol=1e-10)
    assert np.array_equal(into_acceptance.get_column("photon_energy_eV").to_numpy(), flux_sr_grid)
    band_summary = {
        "band_eV": list(band_eV),
        "mean_current_nA": current_nA,
        "sampled_cxr_histories": 8_192,
        "accepted_solid_angle_sr": acceptance_sr,
        "cxr_yield_photons_per_electron_sr": yield_per_electron_sr,
        "cxr_rate_photons_per_s_sr": cxr_rate_sr,
        "cxr_rate_into_acceptance_photons_per_s": cxr_rate_sr * acceptance_sr,
    }
    mo.json(band_summary)
    return (band_summary,)


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


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    `result.to_dataframe()` keeps each component's **native energy grid**.
    `result.spectral_flux()` forms a common grid by interpolation: `"electron"`
    returns source density, `"sr"` applies the physical electron rate, and
    `"detector"` also applies the accepted solid angle. Here the band integral
    intentionally uses **CXR only**, whereas the second plot includes all
    computed components. Integrating over energy removes the `/eV` dimension.

    These are *expected emitted photon rates*, not acquired counts: no exposure
    time, quantum efficiency, energy redistribution or Poisson counting has been
    applied. The scalar-detector acceptance uses a uniform-angular-density
    approximation. See the [results how-to](https://pyrite.readthedocs.io/en/latest/guides/working-with-results.html)
    for saved checkpoints, and the [model limitations](https://pyrite.readthedocs.io/en/latest/physics/capabilities.html)
    and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html)
    before using a spectrum quantitatively.

    Try doubling `bunch_charge_pc` **without** changing histories: the
    electron-normalized yield stays the same, but the photon rate doubles.
    Then change the number of histories at fixed beam current to examine Monte
    Carlo variation. Continue to notebook 03 for ideal PXR line energies.
    """)
    return


@app.cell
def _(band_summary, mo, pyrite, result):
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
    mo.json(tutorial_provenance)
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
