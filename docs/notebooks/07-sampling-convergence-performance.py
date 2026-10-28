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


@app.cell(hide_code=False)
def _():
    import os

    # Select the backend before the first PyRITE import; restart to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Reuse the installed tables without network access.
    tables_ready = True
    return mo, np, plt, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sampling, convergence and performance

    Compare PXR/CBS yields for six fixed-history budgets and one
    adaptive stopping policy. All runs use a 30 keV beam, 100 nm HOPG,
    135° observation, and no bremsstrahlung.

    The **600–2,000 eV** band integral is less noisy than individual grid
    samples. Sampled histories are not physical bunch charge.
    """)
    return


@app.cell
def _():
    from time import perf_counter

    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import AdaptiveSampling, CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    beam = Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch())
    target = Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0, footprint=Footprint(5, 5))
    detector = Detector(observation_angle_deg=135.0)
    physics = Physics(brem_source="none")
    reflections = CxrReflections(n_families=2)
    band_eV = (600.0, 2_000.0)
    budgets = (256, 256*2, 256*3, 256*4, 256*5, 256*6, 256*7, 256*8, 256*9, 256*10)
    electron_block_size = 256  # Adaptive stopping checks after each transport block.
    spectrum_chunk_size = 512  # Segments per spectrum reduction, not electron histories.
    return (
        AdaptiveSampling,
        FixedSampling,
        Numerics,
        band_eV,
        beam,
        budgets,
        detector,
        electron_block_size,
        perf_counter,
        physics,
        reflections,
        simulate,
        spectrum_chunk_size,
        target,
    )


@app.cell
def _(
    FixedSampling,
    Numerics,
    band_eV,
    beam,
    budgets,
    detector,
    np,
    perf_counter,
    physics,
    reflections,
    simulate,
    spectrum_chunk_size,
    tables_ready,
    target,
):
    assert tables_ready  # Wait for reference tables before executing transport.
    # Both policies use the same per-electron CPU transport core and CXR chunk.
    # One untimed warmup reduces, but does not eliminate, first-call compilation costs.
    _ = simulate(
        beam,
        target,
        detector,
        physics=physics,
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=16, brem_electrons=16),
            cxr_reflections=reflections,
            cxr_chunk=spectrum_chunk_size,
            transport_core="per-electron",
        ),
    )
    fixed_rows = []
    for _n in budgets:
        _start = perf_counter()
        _result = simulate(
            beam,
            target,
            detector,
            physics=physics,
            numerics=Numerics(
                sampling=FixedSampling(cxr_electrons=_n, brem_electrons=_n),
                cxr_reflections=reflections,
                cxr_chunk=spectrum_chunk_size,
                transport_core="per-electron",
            ),
        )
        _elapsed = perf_counter() - _start
        _spectrum = _result.cxr.incoherent
        _grid = _spectrum.photon_energy_eV
        _lo, _hi = band_eV
        assert _grid[0] <= _lo < _hi <= _grid[-1]
        _nodes = np.r_[_lo, _grid[(_grid > _lo) & (_grid < _hi)], _hi]
        _yield = float(np.trapezoid(np.interp(_nodes, _grid, _spectrum.density), _nodes))
        assert np.isfinite(_yield) and _yield >= 0.0
        fixed_rows.append(
            {
                "histories": int(_result.case["Ne"]),
                "seed": int(_result.case["seed"]),
                "yield_per_electron_sr": _yield,
                "elapsed_s": _elapsed,
                "source_identity": _result.provenance["identity_digest"],
            }
        )
    assert [row["histories"] for row in fixed_rows] == list(budgets)
    return (fixed_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adaptive stopping

    `AdaptiveSampling` monitors the **per-electron** relative standard error
    (RSE) of the selected CXR band contribution, with additional maximum-share,
    effective-sample-size and stability guards. This is **not** an RSE estimate
    for each photon-energy bin. A run can reach its maximum without convergence.

    `block_electrons` determines how often stopping is checked; `cxr_chunk`
    separately controls the spectrum-reduction kernel. The loose RSE target
    is illustrative, and the default safeguards remain enabled.
    """)
    return


@app.cell
def _(
    AdaptiveSampling,
    Numerics,
    band_eV,
    beam,
    detector,
    electron_block_size,
    np,
    perf_counter,
    physics,
    reflections,
    simulate,
    spectrum_chunk_size,
    tables_ready,
    target,
):
    assert tables_ready  # Wait for reference tables before executing transport.

    policy = AdaptiveSampling(
        target_rse=0.025,
        max_electrons=8_192,
        block_electrons=electron_block_size,
        observables=("cxr",),
        band_eV=band_eV,
    )
    _start = perf_counter()
    adaptive_result = simulate(
        beam,
        target,
        detector,
        physics=physics,
        numerics=Numerics(
            sampling=policy,
            cxr_reflections=reflections,
            cxr_chunk=spectrum_chunk_size,
            transport_core="per-electron",
        ),
    )
    _elapsed = perf_counter() - _start
    _spectrum = adaptive_result.cxr.incoherent
    _grid = _spectrum.photon_energy_eV
    _lo, _hi = band_eV
    assert _grid[0] <= _lo < _hi <= _grid[-1]
    _nodes = np.r_[_lo, _grid[(_grid > _lo) & (_grid < _hi)], _hi]
    _yield = float(np.trapezoid(np.interp(_nodes, _grid, _spectrum.density), _nodes))
    assert np.isfinite(_yield) and _yield >= 0.0
    _stop = adaptive_result.provenance["adaptive_sampling"]
    _cxr_status = _stop["statistics"]["cxr"]
    adaptive_row = {
        "histories": int(_stop["realized_electrons"]),
        "stop_reason": _stop["stop_reason"],
        "statistics_limited": _stop["statistics_limited"],
        "rse": _cxr_status["relative_se"],
        "effective_sample_size": _cxr_status["effective_sample_size"],
        "max_electron_share": _cxr_status["max_electron_share"],
        "stability": _cxr_status["stability"],
        "guard_passes": _cxr_status["passes"],
        "seed": int(adaptive_result.case["seed"]),
        "yield_per_electron_sr": _yield,
        "elapsed_s": _elapsed,
        "target_rse": policy.target_rse,
        "max_electrons": policy.max_electrons,
        "source_identity": adaptive_result.provenance["identity_digest"],
    }
    stopping_diagnostics = _stop
    return adaptive_row, stopping_diagnostics


@app.cell(hide_code=True)
def _(adaptive_row, fixed_rows, mo, stopping_diagnostics):
    mo.json(
        {
            "fixed_budget": fixed_rows,
            "adaptive": adaptive_row,
            "adaptive_stopping_diagnostics": stopping_diagnostics,
            "note": "Elapsed times are environment-specific, not portable GPU/CPU benchmarks.",
        }
    )
    return


@app.cell
def _(adaptive_row, budgets, fixed_rows, np, plt):
    _counts = np.array(budgets, dtype=float)
    _yields = np.array([row["yield_per_electron_sr"] for row in fixed_rows])
    _times = np.array([row["elapsed_s"] for row in fixed_rows])
    fig_convergence, (_yield_ax, _time_ax) = plt.subplots(
        1, 2, figsize=(9.2, 3.9), layout="constrained"
    )
    _yield_ax.plot(_counts, _yields, "bo-", label="Fixed budget")
    _yield_ax.scatter(
        [adaptive_row["histories"]],
        [adaptive_row["yield_per_electron_sr"]],
        marker="D",
        color='red',
        label="Adaptive run",
        zorder=3,
    )
    _yield_ax.axhline(_yields[-1], color="grey", ls=":", label="Largest fixed budget")
    _yield_ax.set(
        xlabel="CXR Monte Carlo Trials",
        ylabel="PXR/CBS yield (photons / electron / sr)",
        title="[600, 2,000] eV band yield",
    )
    _yield_ax.legend(fontsize=8)
    _yield_ax.grid(alpha=0.25)
    _yield_ax.tick_params("x", rotation=45, rotation_mode="xtick")

    _time_ax.plot(_counts, _times, "bo-", label="Fixed budget")
    _time_ax.scatter(
        [adaptive_row["histories"]],
        [adaptive_row["elapsed_s"]],
        marker="D",
        color='red',
        label="Adaptive run",
        zorder=3,
    )
    _time_ax.set(
        xlabel="Sampled CXR electron histories",
        ylabel="Wall time (s)",
        title="Single-machine execution time",
    )
    _time_ax.grid(alpha=0.25)
    fig_convergence  # noqa: B018 - captured by marimo
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    The largest fixed-budget yield is only a reference, **not ground truth**.
    Both modes use the default seed 1, the same per-electron CPU core, the same CXR
    spectrum chunk size, and equal CXR/bremsstrahlung transport populations.
    Those choices matter even with bremsstrahlung scoring disabled: the second
    population affects transport energy cutoffs. Fixed-budget points share
    random draws, so their spread is **not** an independent-sample uncertainty.

    Check `stop_reason`, `statistics_limited`, the measured RSE, and every
    entry in `guard_passes` before calling an adaptive run converged. The
    requested `case["Ne"]` remains its *maximum*, for provenance and cache
    identity; the plotted count is `realized_electrons`. The stopping monitor
    measures per-electron CXR mass, whereas the plot integrates the final
    gridded spectral density. Changing reflection families or photon grids
    tests systematic discretization, not sampling convergence.

    Wall times depend on compilation, caching and hardware. The untimed
    warmup helps, but these are **not portable benchmarks**. These runs do
    not estimate uncertainty across independent seeds. Use `transport_core="cuda"` consistently for
    GPU comparisons. Try increasing
    the budgets and compare the [sampling reference](https://pyrite.readthedocs.io/en/latest/reference/python-api/config.html)
    and [performance guidance](https://pyrite.readthedocs.io/en/latest/computation/index.html).
    """)
    return


if __name__ == "__main__":
    app.run()
