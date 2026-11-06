# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.28.0",
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

    # Select "cuda" before the first PyRITE import, then restart the kernel.
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
    # Parameter sweeps

    Map PXR/CBS yield from an HOPG slab two ways. First, `Sweep` and `run_sweep`
    run one independent transport per grid point, here over target tilt and
    thickness. Second, an observation-angle scan shares one transport per tilt,
    because the electrons do not depend on where the detector sits. Both use a
    30 keV beam and report a source yield per incident electron and steradian,
    before detector response.
    """)
    return


@app.cell
def _(np):
    from pyrite import Scene, Sweep
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector
    from pyrite.geometry import Slab
    from pyrite.units import ureg

    tilts = np.array([15.0, 30.0, 45.0, 60.0]) * ureg.degree
    thicknesses = np.array([25.0, 50.0, 75.0, 100.0, 125.0]) * ureg.nm
    band_eV = (600.0, 2_000.0)
    beam = Beam(
        kinetic_energy=30 * ureg.keV,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
    )
    size = {"width": 5 * ureg.mm, "height": 5 * ureg.mm}
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=1_024, brem_electrons=1_024),
        cxr_reflections=CxrReflections(n_families=1),
    )
    return (
        Detector,
        Physics,
        Scene,
        Slab,
        Sweep,
        band_eV,
        beam,
        size,
        numerics,
        thicknesses,
        tilts,
        ureg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent transports: `Sweep` and `run_sweep`

    `Sweep` expands a base scene over dotted paths into a Cartesian product of
    cases; each case gets its own seed. `run_sweep` executes them. It needs a
    checkpoint location, and here a temporary one keeps the example from mixing
    with other studies. See
    [Persisted parameter studies](18-persisted-parameter-studies.ipynb)
    for saving and resuming.
    """)
    return


@app.cell
def _(
    Detector,
    Physics,
    Scene,
    Slab,
    Sweep,
    beam,
    size,
    numerics,
    thicknesses,
    tilts,
    ureg,
):
    sweep = Sweep(
        base=Scene(
            beam,
            Slab("hopg", thickness=thicknesses[0], tilt=tilts[0], **size),
            Detector(observation_angle=135 * ureg.degree),
        ),
        axes={"target.tilt": tuple(tilts), "target.thickness": tuple(thicknesses)},
    )
    cases = sweep.cases(numerics=numerics, physics=Physics(brem_source="none"))
    assert len(cases) == len(tilts) * len(thicknesses)
    return (cases,)


@app.cell
def _(band_eV, cases, mo, np, tables_ready, thicknesses, tilts):
    from tempfile import TemporaryDirectory

    from pyrite.execution import run_sweep

    assert tables_ready
    store = {}
    with mo.status.spinner(title="Running the tilt–thickness sweep"):
        with TemporaryDirectory(prefix="pyrite-sweep-") as _checkpoints:
            assert run_sweep(cases, store, checkpoint_dir=_checkpoints, progress=False)

    # Cases follow axis order, so the first axis (tilt) gives the rows.
    _yields = []
    for _case in cases:
        _record = store[_case["name"]][_case["E0_keV"]]
        _energy, _density = _record["E_grid"], _record["spec"]
        _inside = (_energy >= band_eV[0]) & (_energy <= band_eV[1])
        _yields.append(np.trapezoid(_density[_inside], _energy[_inside]))
    sweep_yield = np.reshape(_yields, (len(tilts), len(thicknesses)))
    assert np.isfinite(sweep_yield).all() and np.all(sweep_yield >= 0)
    return (sweep_yield,)


@app.cell
def _(plt, sweep_yield, thicknesses, tilts):
    fig_sweep, _ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    _image = _ax.pcolormesh(
        thicknesses.to("nm").magnitude,
        tilts.to("degree").magnitude,
        sweep_yield,
        shading="nearest",
    )
    _ax.set(
        xlabel="HOPG thickness (nm)",
        ylabel="Target tilt (°)",
        title="PXR/CBS yield in the 600–2,000 eV band at 135° observation",
        xticks=thicknesses.to("nm").magnitude,
        yticks=tilts.to("degree").magnitude,
    )
    fig_sweep.colorbar(_image, ax=_ax, label="Photons / (incident electron · sr)")
    fig_sweep  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Yield grows with thickness at every tilt except 45°, where it is essentially
    zero. That is geometry: with this target's tilt convention, a 45° tilt puts
    the 135° observation direction parallel to the slab face, so photons leave
    along a long, strongly absorbing path. All 20 runs use 1,024 sampled histories
    per population and distinct seeds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One transport per tilt: `directional_sources`

    Changing the observation angle changes only the scoring. The **experimental**
    `directional_sources` runs one transport and scores every requested direction
    on it. You supply one energy grid shared by all angles; it must resolve every
    line, because no automatic per-angle window is applied. Angles within one
    tilt share electron histories, so their noise is correlated along the angle
    axis. A different seed per tilt keeps the rows independent.
    """)
    return


@app.cell
def _(Slab, band_eV, beam, size, mo, np, numerics, tables_ready, tilts, ureg):
    from pyrite.radiation import directional_sources

    assert tables_ready
    observation_angles = np.linspace(45.0, 175.0, 27) * ureg.degree
    energy_grid = np.arange(band_eV[0], band_eV[1] + 0.5, 1.0) * ureg.eV

    angle_yield = np.zeros((len(tilts), len(observation_angles)))
    with mo.status.progress_bar(total=len(tilts), title="Scoring angles per tilt") as _progress:
        _seed = 1
        for _row, _tilt in enumerate(tilts):
            _source = directional_sources(
                beam,
                Slab("hopg", thickness=100 * ureg.nm, tilt=_tilt, **size),
                # This experimental helper still takes numeric eV and degree grids.
                energy_eV=energy_grid.to("eV").magnitude,
                angle_deg=observation_angles.to("degree").magnitude,
                numerics=numerics,
                seed=_seed,
            )["pxr-cbs"]
            assert _source.supported.all()
            angle_yield[_row] = np.trapezoid(
                _source.density, energy_grid.to("eV").magnitude, axis=1
            )
            _seed += 1
            _progress.update()
    return angle_yield, observation_angles


@app.cell
def _(angle_yield, observation_angles, plt, tilts):
    fig_angles, _ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for _tilt, _yield in zip(tilts, angle_yield, strict=True):
        _ax.plot(
            observation_angles.to("degree").magnitude,
            _yield,
            label=f"{_tilt.to('degree').magnitude:g}° tilt",
        )
    _ax.set(
        xlabel="Observation angle (°)",
        ylabel="Photons / (incident electron · sr)",
        title="PXR/CBS yield in the 600–2,000 eV band, 100 nm HOPG",
    )
    _ax.legend()
    fig_angles  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each curve drops sharply where the observation direction runs parallel to the
    slab face, at 90° plus the tilt: the same null as the 45° row above, now seen
    at every tilt. One run per point would need 108 transports for this figure;
    here four suffice. Scoring still costs time, so the saving is largest when
    transport dominates. Spectra stay on one explicit 1 eV grid, so refine it
    and the sampling before quantitative use, and compare against a few
    independent `simulate` runs.

    Try another band, or add `"bremsstrahlung"` to `components`. Use
    [notebook 07](07-sampling-convergence-performance.ipynb)
    to choose a larger sampling budget for a selected region.
    """)
    return


if __name__ == "__main__":
    app.run()
