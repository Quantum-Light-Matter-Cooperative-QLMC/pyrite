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
    # Parameter sweeps

    Map the PXR/CBS yield as beam energy and HOPG thickness vary,
    at a fixed 30° target tilt and 135° observation angle.
    """)
    return


@app.cell
def _(np):
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    energies_keV = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    thicknesses_nm = np.array([25.0, 50.0, 75.0, 100.0, 125.0])
    band_eV = (600.0, 2_000.0)
    detector = Detector(observation_angle_deg=135.0)
    physics = Physics(brem_source="none")
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=2_048, brem_electrons=2_048),
        cxr_reflections=CxrReflections(n_families=2),
    )
    return (
        Beam,
        Footprint,
        PointBunch,
        PointSpot,
        Slab,
        band_eV,
        detector,
        energies_keV,
        numerics,
        physics,
        simulate,
        thicknesses_nm,
    )


@app.cell
def _(
    Beam,
    Footprint,
    PointBunch,
    PointSpot,
    Slab,
    band_eV,
    detector,
    energies_keV,
    np,
    numerics,
    physics,
    simulate,
    tables_ready,
    thicknesses_nm,
):
    assert tables_ready
    # Rows are thicknesses; columns are beam energies.
    band_yield = np.zeros((len(thicknesses_nm), len(energies_keV)))
    runs = []
    for _row, _thickness_nm in enumerate(thicknesses_nm):
        for _col, _energy_keV in enumerate(energies_keV):
            _result = simulate(
                Beam(
                    energy_keV=float(_energy_keV), transverse=PointSpot(), longitudinal=PointBunch()
                ),
                Slab(
                    "hopg",
                    thickness_ang=float(_thickness_nm * 10),
                    tilt_deg=30.0,
                    footprint=Footprint(5, 5),
                ),
                detector,
                physics=physics,
                numerics=numerics,
            )
            _spectrum = _result.cxr.incoherent
            _grid = _spectrum.photon_energy_eV
            _lo, _hi = band_eV
            assert _grid[0] <= _lo < _hi <= _grid[-1]
            # Include interpolated band edges on each result's native grid.
            _nodes = np.r_[_lo, _grid[(_grid > _lo) & (_grid < _hi)], _hi]
            band_yield[_row, _col] = np.trapezoid(
                np.interp(_nodes, _grid, _spectrum.density), _nodes
            )
            runs.append(
                {
                    "energy_keV": float(_energy_keV),
                    "thickness_nm": float(_thickness_nm),
                    "seed": int(_result.case["seed"]),
                    "source_identity": _result.provenance["identity_digest"],
                }
            )
    assert np.isfinite(band_yield).all() and np.all(band_yield >= 0)
    sweep_result = _result
    sweep_yield = band_yield.copy()
    return runs, sweep_result, sweep_yield


@app.cell
def _(energies_keV, plt, sweep_yield, thicknesses_nm):
    fig_sweep, _ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    _image = _ax.pcolormesh(energies_keV, thicknesses_nm, sweep_yield, shading="nearest")
    _ax.set(
        xlabel="Beam energy (keV)",
        ylabel="HOPG thickness (nm)",
        title="PXR/CBS yield in the 600–2,000 eV band",
        xticks=energies_keV,
        yticks=thicknesses_nm,
    )
    fig_sweep.colorbar(_image, ax=_ax, label="Photons / (incident electron · sr)")
    fig_sweep  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each tile integrates a source spectrum over the same photon-energy band;
    it is a yield per incident electron and steradian, before detector response.
    Here, yield grows with thickness and falls with beam energy.
    All 25 runs use 2,048 sampled histories per population and seed 1.

    Try shifting the energy band or replacing the thickness axis with target tilt.
    Use [notebook 07](https://pyrite.readthedocs.io/en/latest/notebooks/designing/07-sampling-convergence-performance.html)
    to choose a larger sampling budget for a selected region.
    """)
    return


@app.cell(hide_code=True)
def _(band_eV, energies_keV, mo, pyrite, runs, sweep_result, sweep_yield, thicknesses_nm):
    tutorial_provenance = {
        "tutorial": "08-parameter-sweeps",
        "pyrite_version": pyrite.__version__,
        "backend": sweep_result.provenance["backend"],
        "seed": sweep_result.case["seed"],
        "table_identity": dict(sweep_result.provenance["xsgen_tables"]),
        "energies_keV": energies_keV.tolist(),
        "thicknesses_nm": thicknesses_nm.tolist(),
        "band_eV": list(band_eV),
        "yield_per_electron_sr": sweep_yield.tolist(),
        "sampled_cxr_histories": 2_048,
        "sampled_brem_histories": 2_048,
        "runs": runs,
    }
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
