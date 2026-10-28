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

    # Select the backend before the first PyRITE import; restart to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Reuse the installed tables without network access.
    tables_ready = True
    return mo, np, plt, pyrite, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sampling, convergence and performance

    Compare PXR/CBS yields for three sampled-history budgets and one
    adaptive stopping policy. All runs use a 30 keV beam, 100 nm HOPG,
    135° observation, and no bremsstrahlung.

    The **600–2,000 eV** band integral is less noisy than individual grid
    samples. Small CPU budgets illustrate the method, not validated precision;
    sampled histories are not physical bunch charge.
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
    budgets = (256, 1_024, 4_096, 8_192, 16_384, 32_768)
    return (
        AdaptiveSampling,
        FixedSampling,
        Numerics,
        band_eV,
        beam,
        budgets,
        detector,
        perf_counter,
        physics,
        reflections,
        simulate,
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
    tables_ready,
    target,
):
    assert tables_ready  # Wait for reference tables before executing transport.
    electron_batch_size = 256

    # One untimed warmup reduces, but does not eliminate, first-call compilation costs.
    _ = simulate(
        beam,
        target,
        detector,
        physics=physics,
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=16, brem_electrons=1),
            cxr_reflections=reflections,
            cxr_chunk=electron_batch_size,
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
                sampling=FixedSampling(cxr_electrons=_n, brem_electrons=1),
                cxr_reflections=reflections,
                cxr_chunk=512,
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
    return electron_batch_size, fixed_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adaptive stopping

    `AdaptiveSampling` estimates a batch-means relative standard error
    (RSE), subject to minimum-sample and contribution guards. It can reach
    its history limit **without meeting the target**. The deliberately
    loose RSE target here is for teaching.
    """)
    return


@app.cell
def _(
    AdaptiveSampling,
    Numerics,
    band_eV,
    beam,
    detector,
    electron_batch_size,
    np,
    perf_counter,
    physics,
    reflections,
    simulate,
    tables_ready,
    target,
):
    assert tables_ready  # Wait for reference tables before executing transport.

    policy = AdaptiveSampling(
        target_rse=0.4,
        max_electrons=32_768,
        block_electrons=electron_batch_size,
        observables=("cxr",),
        band_eV=band_eV,
    )
    _start = perf_counter()
    adaptive_result = simulate(
        beam,
        target,
        detector,
        physics=physics,
        numerics=Numerics(sampling=policy, cxr_reflections=reflections),
    )
    _elapsed = perf_counter() - _start
    _spectrum = adaptive_result.cxr.incoherent
    _grid = _spectrum.photon_energy_eV
    _lo, _hi = band_eV
    assert _grid[0] <= _lo < _hi <= _grid[-1]
    _nodes = np.r_[_lo, _grid[(_grid > _lo) & (_grid < _hi)], _hi]
    _yield = float(np.trapezoid(np.interp(_nodes, _grid, _spectrum.density), _nodes))
    assert np.isfinite(_yield) and _yield >= 0.0
    adaptive_row = {
        "histories": int(adaptive_result.case["Ne"]),
        "seed": int(adaptive_result.case["seed"]),
        "yield_per_electron_sr": _yield,
        "elapsed_s": _elapsed,
        "target_rse": policy.target_rse,
        "max_electrons": policy.max_electrons,
        "source_identity": adaptive_result.provenance["identity_digest"],
    }
    return adaptive_result, adaptive_row


@app.cell(hide_code=True)
def _(adaptive_row, fixed_rows, mo):
    mo.json(
        {
            "fixed_budget": fixed_rows,
            "adaptive": adaptive_row,
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
    _yield_ax.plot(_counts, _yields, "o-", label="Fixed budget")
    _yield_ax.scatter(
        [adaptive_row["histories"]],
        [adaptive_row["yield_per_electron_sr"]],
        marker="D",
        label="Adaptive run",
        zorder=3,
    )
    _yield_ax.axhline(_yields[-1], color="grey", ls=":", label="Largest fixed budget")
    _yield_ax.set(
        xlabel="Sampled CXR electron histories",
        ylabel="PXR/CBS yield (photons / electron / sr)",
        title="[600, 2,000] eV band yield",
        xscale="log",
    )
    _yield_ax.legend(fontsize=8)
    _yield_ax.grid(alpha=0.25)

    _time_ax.plot(_counts, _times, "o-", label="Fixed budget")
    _time_ax.scatter(
        [adaptive_row["histories"]],
        [adaptive_row["elapsed_s"]],
        marker="D",
        label="Adaptive run",
        zorder=3,
    )
    _time_ax.set(
        xlabel="Sampled CXR electron histories",
        ylabel="Wall time (s)",
        title="Single-machine execution time",
        xscale="log",
    )
    _time_ax.grid(alpha=0.25)
    fig_convergence  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    The largest fixed-budget yield is only a reference, **not ground truth**.
    These runs share a resolved default seed, so their differences are not
    an independent-sample error estimate. For uncertainty, use independent
    seeds and inspect the convergence diagnostics.

    An adaptive run may stop at its maximum without achieving its RSE
    target; check its diagnostics. Changing reflection families or energy
    grids tests systematic discretization, not sampling convergence.

    Wall times depend on compilation, caching and hardware. The untimed
    warmup helps, but these are **not portable benchmarks**. Try increasing
    the budgets and compare the [sampling reference](https://pyrite.readthedocs.io/en/latest/reference/python-api/config.html)
    and [performance guidance](https://pyrite.readthedocs.io/en/latest/computation/index.html).
    """)
    return


@app.cell(hide_code=True)
def _(adaptive_result, adaptive_row, band_eV, fixed_rows, mo, pyrite):
    tutorial_provenance = {
        "tutorial": "07-sampling-convergence-performance",
        "pyrite_version": pyrite.__version__,
        "backend": adaptive_result.provenance["backend"],
        "seed": adaptive_result.case["seed"],
        "table_identity": dict(adaptive_result.provenance["xsgen_tables"]),
        "source_identity": adaptive_result.provenance["identity_digest"],
        "band_eV": list(band_eV),
        "fixed_runs": fixed_rows,
        "adaptive_run": adaptive_row,
        "physics": "pxr-cbs only; brem disabled",
        "timing": "perf_counter, local wall seconds, after one untimed warmup",
    }
    mo.json(tutorial_provenance).style({"display": "none"})
    return


if __name__ == "__main__":
    app.run()
