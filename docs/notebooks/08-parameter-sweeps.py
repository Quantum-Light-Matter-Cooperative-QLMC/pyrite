# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.32.0",
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

    tilts = np.array([5.0, 25.0, 45.0, 65.0, 85.0]) * ureg.degree
    thicknesses = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]) * ureg.nm
    band_eV = (50.0, 2_000.0)
    tilt_azimuth = 180.0 * ureg.degree
    beam_energy = 30 * ureg.keV
    beam = Beam(
        kinetic_energy=beam_energy,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
    )
    size = {"width": 5 * ureg.mm, "height": 5 * ureg.mm}
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=256, brem_histories=256),
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
        beam_energy,
        numerics,
        size,
        thicknesses,
        tilt_azimuth,
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
    numerics,
    size,
    thicknesses,
    tilt_azimuth,
    tilts,
    ureg,
):
    sweep = Sweep(
        base=Scene(
            beam,
            Slab(
                "hopg",
                thickness=thicknesses[0],
                tilt=tilts[0],
                tilt_azimuth=tilt_azimuth,
                **size,
            ),
            Detector(observation_angle=135 * ureg.degree),
        ),
        axes={"target.tilt": tuple(tilts), "target.thickness": tuple(thicknesses)},
    )
    cases = sweep.cases(numerics=numerics, physics=Physics(brem_source="mc"))
    assert len(cases) == len(tilts) * len(thicknesses)
    return (cases,)


@app.cell
def _(band_eV, cases, mo, np, tables_ready, thicknesses, tilts):
    from pathlib import Path

    from pyrite.execution import run_sweep

    # Completed cases persist here, so a rerun resumes instead of recomputing.
    _checkpoints = Path.home() / ".cache" / "pyrite-sweep"
    _checkpoints.mkdir(parents=True, exist_ok=True)

    assert tables_ready
    store = {}
    with mo.status.spinner(title="Running the tilt–thickness sweep"):
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
def _(band_eV, plt, sweep_yield, thicknesses, tilts):
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
        title=f"PXR/CBS yield in the {band_eV[0]:g}–{band_eV[1]:,.0f} eV band at 135° observation",
        xticks=thicknesses.to("nm").magnitude,
        yticks=tilts.to("degree").magnitude,
    )
    fig_sweep.colorbar(_image, ax=_ax, label="Photons / (incident electron · sr)")
    fig_sweep  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Yield generally grows with thickness, and the 85° tilt is brightest. The 45°
    row stays near zero: for that tilt, 135° falls in a broad minimum of the
    angular distribution, which the next section maps. All 30 runs use 256
    sampled histories per population and distinct seeds, so neighbouring cells
    differ partly by noise.
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

    The line energy moves with angle, so a fixed band mixes the line with a
    varying amount of continuum. Instead, `photon_energies` gives the line
    energy at every tilt and angle, and each component is integrated over that
    line energy ±5%. Requesting `"characteristic"` and `"bremsstrahlung"` on the
    same transport gives the background under each window. `mo.persistent_cache`
    stores the scored windows on disk, so rerunning with unchanged inputs
    skips the scoring; change any input and it recomputes.
    """)
    return


@app.cell
def _(
    Slab,
    band_eV,
    beam,
    beam_energy,
    mo,
    np,
    numerics,
    size,
    tables_ready,
    tilt_azimuth,
    tilts,
    ureg,
):
    from pyrite.radiation import directional_sources, photon_energies

    assert tables_ready
    observation_angles = np.linspace(4, 180.0, 181) * ureg.degree
    energy_grid = np.arange(band_eV[0], band_eV[1] + 0.5, 1.0) * ureg.eV
    line_window = 0.05  # integrate over line energy * (1 ± 5%)
    components = ("pxr-cbs", "characteristic", "bremsstrahlung")

    # The analytic and experimental helpers take numeric eV and degree values;
    # plain arrays also hash cleanly into the cache key below.
    _tilts_deg = tilts.to("degree").magnitude
    _angles_deg = observation_angles.to("degree").magnitude
    _energy_eV = energy_grid.to("eV").magnitude
    line_energy_eV = photon_energies(
        "hopg",
        beam_energy.to("eV").magnitude,
        tilt_deg=_tilts_deg[:, None],
        tilt_azim_deg=tilt_azimuth.to("degree").magnitude,
        theta_obs_deg=_angles_deg[None, :],
    )[0]

    def _window_yield(source, centers_eV, fraction):
        # Integrate each angle's density over its own line window. Windows
        # leaving the energy grid, or angles with no line (NaN), give NaN.
        _low, _high = centers_eV * (1 - fraction), centers_eV * (1 + fraction)
        _inside = (_energy_eV >= _low[:, None]) & (_energy_eV <= _high[:, None])
        if source.sampling == "bin-mean":  # characteristic: per-bin means
            _counts = np.where(_inside, source.density * np.diff(source.bin_edges_eV), 0.0)
            _yield = _counts.sum(axis=1)
        else:  # point samples
            _yield = np.array(
                [
                    np.trapezoid(_d[_m], _energy_eV[_m])
                    for _d, _m in zip(source.density, _inside, strict=True)
                ]
            )
        _covered = (_low >= _energy_eV[0]) & (_high <= _energy_eV[-1])
        return np.where(_covered, _yield, np.nan)

    @mo.persistent_cache
    def _score_line_windows():
        _yields = {_name: np.full(line_energy_eV.shape, np.nan) for _name in components}
        with mo.status.progress_bar(total=len(_tilts_deg), title="Scoring angles per tilt") as _bar:
            for _row, _tilt_deg in enumerate(_tilts_deg):
                _sources = directional_sources(
                    beam,
                    Slab(
                        "hopg",
                        thickness=100 * ureg.nm,
                        tilt=_tilt_deg * ureg.degree,
                        tilt_azimuth=tilt_azimuth,
                        **size,
                    ),
                    energy_eV=_energy_eV,
                    angle_deg=_angles_deg,
                    numerics=numerics,
                    seed=_row + 1,  # independent rows
                    components=components,
                )
                assert all(_source.supported.all() for _source in _sources.values())
                for _name, _source in _sources.items():
                    _yields[_name][_row] = _window_yield(_source, line_energy_eV[_row], line_window)
                _bar.update()
        return _yields

    line_yield = _score_line_windows()
    return line_window, line_yield, observation_angles


@app.cell
def _(line_window, line_yield, observation_angles, plt, tilts):
    fig_cxr_angles, _ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for _tilt, _yield in zip(tilts, line_yield["pxr-cbs"], strict=True):
        _ax.plot(
            observation_angles.to("degree").magnitude,
            _yield,
            label=f"{_tilt.to('degree').magnitude:g}° tilt",
        )
    _ax.set(
        xlabel="Observation angle (°)",
        ylabel="Photons / (incident electron · sr)",
        title=f"PXR/CBS yield within ±{line_window:.0%} of the line, 100 nm HOPG",
        xlim=(0, 180),
    )
    _ax.grid(alpha=0.5)
    _ax.legend()
    fig_cxr_angles  # noqa: B018 - captured by marimo
    return


@app.cell
def _(line_window, line_yield, np, observation_angles, plt, tilts):
    # Background under each line: continuum plus characteristic (carbon K).
    _background = line_yield["bremsstrahlung"] + line_yield["characteristic"]
    # Exits along the slab face absorb line and background alike, by hundreds
    # of decades; their ratio is meaningless, so drop them.
    _kept = _background > 1e-6 * np.nanmax(_background, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        line_to_background = np.where(_kept, line_yield["pxr-cbs"] / _background, np.nan)

    fig_line_to_background, _ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for _tilt, _ratio in zip(tilts, line_to_background, strict=True):
        _ax.plot(
            observation_angles.to("degree").magnitude,
            _ratio,
            label=f"{_tilt.to('degree').magnitude:g}° tilt",
        )
    _ax.set(
        xlabel="Observation angle (°)",
        ylabel="PXR/CBS ÷ (bremsstrahlung + characteristic)",
        title=f"Line-to-background within ±{line_window:.0%} of the line, 100 nm HOPG",
        xlim=(0, 180),
        yscale="log",
    )
    _ax.grid(which="both", alpha=0.5)
    _ax.legend()
    fig_line_to_background  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the first figure, each curve has a narrow null where the exit runs along
    the slab face, at 90° minus the tilt (the slab normal leans toward −x here),
    and a broader minimum at larger angles. The 45° minimum near 130° is why the
    45° row of the sweep stays dark at 135°.

    The second figure divides the PXR/CBS window yield by the bremsstrahlung plus
    characteristic yield in the same window. Away from the minima the line
    carries roughly two to four times the background; near them the ratio falls
    by orders of magnitude, so a detector placed there would see mostly
    background. Directions where the background is absorbed by more than six
    decades are left blank. No detector response is applied: a detector whose
    resolution is broader than ±5% of the line collects more continuum.

    One run per point would need 905 transports for these figures; here five
    suffice. Scoring still costs time, so the saving is largest when transport
    dominates. Spectra stay on one explicit 1 eV grid, so refine it and the
    sampling before quantitative use, and compare against a few independent
    `simulate` runs.

    Use [notebook 07](07-sampling-convergence-performance.ipynb)
    to choose a larger sampling budget for a selected region.
    """)
    return


if __name__ == "__main__":
    app.run()
