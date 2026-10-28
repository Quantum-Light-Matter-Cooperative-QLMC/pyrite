# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.1", "pyrite-mc==0.22.5", "matplotlib>=3.11", "numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Select "cuda" before the first PyRITE import, then restart the kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Installed pins are reused without downloading again.

    return mo, np, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # Beams, crystals and target geometry

    Describe an electron beam and a finite crystal, then run one small PXR/CBS simulation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose a beam and crystal

    An electron **energy** (keV) and a transverse **FWHM** (mm) describe the
    incident beam. The crystal has its own thickness (Å), transverse footprint
    (full widths in mm), and tilt (degrees). A crystal's **lattice cut** is
    distinct from its lab-frame tilt and from the reflections used in scoring.

    Material keys come from the catalog. Change `"hopg"` to another crystal
    key such as `"silicon"` to explore a different target.
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
def _(beam, mo, np, plt, target):
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Run one source calculation

    Keep the physics and numerical settings separate from the beam and geometry.
    The sample histories below are a numerical estimator—not the number of
    electrons in a physical bunch. Bremsstrahlung is disabled to focus on
    the crystal's PXR/CBS source spectrum.
    """)
    return


@app.cell
def _(beam, detector, simulate, target):
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpret and experiment

    The first plot is the *configured entrance distribution*; the second is the
    *computed response-free source density*. It is not a measured spectrum or
    detector count rate. The `CrystalCut(beam_uvw=(0, 0, 2))` places a direct
    lattice _axis along the entrance-face normal; `tilt_deg` then rotates that
    face in the laboratory. Neither chooses which diffraction planes the
    radiation calculation sums. That is a separate numerical/reflection policy.

    Try changing **one parameter at a time**: first `beam.energy_keV`, then
    `target.thickness_ang`, then `material`. A different crystal may need a
    different cut or energy range. Only the finite sample is illustrated; do not
    use the curve quantitatively without convergence and validation checks.
    Compare the [beam/target how-to](https://pyrite.readthedocs.io/en/latest/guides/index.html),
    [physics](https://pyrite.readthedocs.io/en/latest/physics/index.html), and
    [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html).
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
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
