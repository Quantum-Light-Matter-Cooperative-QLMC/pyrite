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

    # Select the backend before importing PyRITE; restart to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Complete installed tables need no download.
    return mo, np, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # From X-rays to detector counts

    [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/06-detector-counts.py)

    Predict an ideal pixel detector's expected counts and compare a seeded Poisson realization.

    ## Define the physical scene

    A 30 keV beam illuminates 100 nm HOPG. The 16 × 16 mm panel sits 100 mm
    away, at 135° from the incident beam direction. Its 16 × 16 pixels have
    1 mm pitch. A 0.01 pC bunch at 1 MHz gives **10 nA** mean current.
    These physical beam populations are independent of sampled histories.
    """)
    return


@app.cell
def _():
    from pyrite import Scene, simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.detectors import PixelGrid, PlanarDetector, PlanarPose
    from pyrite.geometry import Footprint, Slab

    beam = Beam(
        energy_keV=30.0,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
        bunch_charge_pc=0.01,
        rep_rate_hz=1_000_000.0,
    )
    panel = PlanarDetector(
        PlanarPose.from_observation(distance_mm=100.0, polar_deg=135.0),
        pixels=PixelGrid(shape=(16, 16), pitch_mm=(1.0, 1.0)),
    )
    scene = Scene(
        beam,
        Slab("hopg", 1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        panel,
    )
    return beam, panel, scene, simulate


@app.cell
def _(mo):
    mo.md(r"""
    ## Specify what is measured

    Geometry belongs to `Scene`; response and exposure belong to `Observation`.
    `IdealPhotonCounter()` has unit efficiency and no energy broadening. Count
    photons in the half-open reporting interval **[0.5, 4) keV** during **1 s**.
    `mode="poisson"` retains the fractional expectation and adds integer draws.

    Score 4 × 4 angular tiles across the panel: neighboring pixels share source
    spectra within each tile, while their pixel-centre ray geometry is retained.
    Pixel pitch alone does not set the angular scoring resolution.
    """)
    return


@app.cell
def _():
    from pyrite.config import AngularSampling
    from pyrite.detectors import IdealPhotonCounter
    from pyrite.observations import Acquisition, Observation

    acquisition = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(500.0, 4_000.0),
        mode="poisson",
        seed=23,
    )
    observation = Observation(
        detector="detector",
        response=IdealPhotonCounter(),
        sampling=AngularSampling((4, 4)),
        acquisition=acquisition,
    )
    return acquisition, observation


@app.cell
def _(observation, scene, simulate):
    from pyrite.config import CxrReflections, FixedSampling, Numerics

    result = simulate(
        scene,
        observation=observation,
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=512, brem_electrons=128),
            cxr_reflections=CxrReflections(n_families=2),
        ),
    )
    return (result,)


@app.cell
def _(panel, result):
    batch = result.acquire(
        region=(slice(None), slice(None)),
        components=("cxr", "brem", "characteristic"),
    )
    expected_image = batch.expected.sum(axis=-1).reshape(panel.pixels.shape)
    count_image = batch.total_counts.reshape(panel.pixels.shape)
    return batch, count_image, expected_image


@app.cell
def _(count_image, expected_image, plt):
    fig_counts, _axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    _maximum = max(float(expected_image.max()), float(count_image.max()))
    for _ax, _image, _title in zip(
        _axes,
        (expected_image, count_image),
        ("Expected counts", "Poisson realization (seed 23)"),
        strict=True,
    ):
        _artist = _ax.imshow(_image, origin="lower", vmin=0, vmax=_maximum)
        _ax.set(xlabel="Pixel column", ylabel="Pixel row", title=_title)
        _ax.set_xticks((0, 5, 10, 15))
        _ax.set_yticks((0, 5, 10, 15))
    fig_counts.colorbar(_artist, ax=_axes, label="Counts / pixel in 1 s ([0.5, 4) keV)")
    fig_counts  # noqa: B018 - marimo captures the final expression
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpret and explore

    Both images already integrate energy and pixel solid angle, apply the
    ideal response, and scale by physical electrons during the exposure.
    **Do not multiply by current, solid angle or exposure again.** They include
    PXR/CBS, bremsstrahlung and characteristic emission in the reporting window;
    underflow, overflow and below-threshold expectations remain separate.
    No downstream filter or background is included.

    The nearly uniform expectation and noisier integer image share one color
    scale. Counting fluctuations are conditional on the estimated source:
    they exclude its Monte Carlo uncertainty, calibration errors and backgrounds.
    The modest histories, two reflection families and coarse angular tiles are
    illustrative; they do not establish convergence or detector calibration.

    First change only `acquisition.seed`: the expectation stays fixed while
    counts change. Then double `exposure_s`: expected counts double, while the
    realized total need not double exactly. More source histories refine the
    expectation; they do not increase the physical exposure.

    Consult the [detector how-to](https://pyrite.readthedocs.io/en/latest/guides/detector-observations.html),
    [observation reference](https://pyrite.readthedocs.io/en/latest/reference/python-api/observations.html),
    [response physics](https://pyrite.readthedocs.io/en/latest/physics/detectors/detector-response.html),
    and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html).
    """)
    return


@app.cell(hide_code=True)
def _(acquisition, batch, beam, count_image, expected_image, mo, np, pyrite, result):
    tutorial_provenance = {
        "tutorial": "06-detector-counts",
        "pyrite_version": pyrite.__version__,
        "backend": result.provenance["backend"],
        "seed": result.case["seed"],
        "table_identity": dict(result.provenance["xsgen_tables"]),
        "source_identity": result.provenance["identity_digest"],
        "observation_identity": batch.observation_digest,
        "acquisition_seed": acquisition.seed,
        "exposure_s": acquisition.exposure_s,
        "current_nA": beam.rep_rate_hz * beam.bunch_charge_pc / 1_000,
        "measured_edges_eV": batch.measured_edges_eV.tolist(),
        "components": list(batch.components),
        "pixel_shape": list(expected_image.shape),
        "angular_tiles": [4, 4],
        "sampled_cxr_histories": 512,
        "sampled_brem_histories": 128,
        "expected_total": float(np.sum(expected_image)),
        "realized_total": int(np.sum(count_image)),
        "outside_window_expected": {
            "underflow": float(batch.underflow_expected.sum()),
            "overflow": float(batch.overflow_expected.sum()),
            "below_cut": float(batch.below_cut_expected.sum()),
        },
    }
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
