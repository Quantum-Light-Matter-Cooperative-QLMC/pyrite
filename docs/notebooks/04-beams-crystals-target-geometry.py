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

    # Select "cuda" before the first PyRITE import, then restart the kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Installed pins are reused without downloading again.

    tables_ready = True
    return mo, np, plt, pyrite, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Beams, crystals and target geometry

    Configure a 30 keV electron beam, finite HOPG crystal and 135° observation.
    `TransverseSpot` sets the incident **FWHM** in mm; `Slab` uses thickness
    in Å and footprint widths in mm. `CrystalCut` fixes the lattice
    orientation, while `tilt_deg` rotates the entrance face in the lab.
    """)
    return


@app.cell
def _():
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, TransverseSpot
    from pyrite.detectors import Detector
    from pyrite.geometry import CrystalCut, Footprint, Slab
    from pyrite.materials import load_material_catalog

    catalog = load_material_catalog()
    material = "hopg"
    assert material in catalog.crystals
    beam = Beam(
        energy_keV=30.0,
        transverse=TransverseSpot(fwhm_x_mm=0.45),
        longitudinal=PointBunch(),
    )
    target = Slab(
        material,
        thickness_ang=1_000.0,  # 100 nm
        tilt_deg=30.0,
        footprint=Footprint(width_mm=2.0, height_mm=2.0),
        cut=CrystalCut(beam_uvw=(0, 0, 2)),
    )
    detector = Detector(observation_angle_deg=135.0)
    return beam, detector, material, simulate, target


@app.cell
def _(beam, np, plt, target):
    # Analytic INPUT profile, not a plot of Monte Carlo electron histories.
    sigma_mm = beam.transverse.fwhm_x_mm / (2 * np.sqrt(2 * np.log(2)))
    x_mm = np.linspace(-1.25, 1.25, 401)
    profile = np.exp(-0.5 * (x_mm / sigma_mm) ** 2)
    fig_profile, _ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    _ax.plot(x_mm, profile, label="Incident Gaussian spot (relative density)")
    half_width = target.footprint.width_mm / 2
    _ax.axvspan(-half_width, half_width, alpha=0.12, color="grey", label="Target footprint")
    _ax.set(xlabel="Transverse position x (mm)", ylabel="Relative density", ylim=(0, 1.08))
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.2)
    fig_profile  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate PXR/CBS

    Use a small history budget to illustrate the crystal source spectrum.
    Bremsstrahlung is disabled so the result isolates PXR/CBS.
    """)
    return


@app.cell
def _(beam, detector, simulate, tables_ready, target):
    assert tables_ready  # Wait for reference-table installation before transport.
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics

    result = simulate(
        beam,
        target,
        detector,
        physics=Physics(brem_source="none"),
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=2_048, brem_electrons=1),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )
    return (result,)


@app.cell
def _(plt, result):
    spectrum = result.cxr.incoherent
    fig_spectrum, _ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    _ax.plot(spectrum.photon_energy_eV / 1_000, spectrum.density, label="PXR + CBS")
    _ax.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (incident electron · eV · sr)",
        title="30 keV beam on a 100 nm crystal",
        xlim=(0.3, 3.7),
        yscale="log",
    )
    _ax.grid(alpha=0.2)
    _ax.legend()
    fig_spectrum  # noqa: B018 - marimo renders the final expression
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    The first plot shows the *configured entrance spot*, not sampled
    trajectories. The second is the *response-free* PXR/CBS source density,
    not detector counts. The (002) `CrystalCut` sets the lattice axis;
    the selected diffraction families are controlled separately by numerics.

    Try changing `beam.energy_keV`, `target.thickness_ang`, or `material`
    one at a time. Other crystals may need different cuts and energy windows.
    These small-sample spectra are illustrative; consult
    [validation](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html)
    before making quantitative predictions.
    """)
    return


@app.cell(hide_code=True)
def _(beam, material, mo, pyrite, result, target):
    tutorial_provenance = {
        "tutorial": "04-beams-crystals-target-geometry",
        "pyrite_version": pyrite.__version__,
        "backend": result.provenance["backend"],
        "seed": result.case["seed"],
        "table_identity": dict(result.provenance["xsgen_tables"]),
        "source_identity": result.provenance["identity_digest"],
        "beam_energy_keV": beam.energy_keV,
        "spot_fwhm_x_mm": beam.transverse.fwhm_x_mm,
        "material": material,
        "thickness_ang": target.thickness_ang,
        "cut_beam_uvw": list(target.cut.beam_uvw),
        "sampled_cxr_histories": 2_048,
    }
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
