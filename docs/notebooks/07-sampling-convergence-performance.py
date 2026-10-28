# /// script
# requires-python = ">=3.13"
# dependencies = ["endf-parserpy==0.17.0", "marimo>=0.25.1", "pyrite-mc==0.22.5", "matplotlib>=3.11", "numpy>=2"]
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

    Examine how the PXR/CBS yield changes with sampled electron histories,
    then try PyRITE's bounded adaptive stopping rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep the physics fixed

    We compare the **same** 30 keV beam, 100 nm HOPG slab and 135° observation
    direction for three numerical budgets. Bremsstrahlung is disabled to isolate
    the CXR estimator; a **[600, 2,000] eV** band integral avoids comparing
    individual noisy grid samples. Each yield is in photons per incident
    electron per steradian. Sampled histories are *not* physical bunch charge.

    This tutorial uses the CPU backend and intentionally small budgets.
    It illustrates a convergence workflow, **not validated precision**.
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
    budgets = (256, 1_024, 4_096)
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

    # One untimed warmup reduces, but does not eliminate, first-call compilation costs.
    _ = simulate(
        beam,
        target,
        detector,
        physics=physics,
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=16, brem_electrons=1),
            cxr_reflections=reflections,
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
    ## Adaptive sampling is a separate stopping policy

    `AdaptiveSampling` checks a batch-means relative standard error (RSE),
    subject to minimum histories, effective-sample and largest-contribution
    guards. It can reach its maximum **without meeting the target**. Its RSE
    estimate is not the difference between successive fixed-budget runs.
    The bounded example below uses a deliberately loose teaching target.
    """)
    return


@app.cell
def _(
    AdaptiveSampling,
    Numerics,
    band_eV,
    beam,
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

    policy = AdaptiveSampling(
        target_rse=0.25,
        min_electrons=256,
        max_electrons=2_048,
        block_electrons=256,
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


@app.cell
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
    ## Interpret and explore

    A stable band integral across increasing budgets is useful **evidence to
    investigate**, but not a confidence interval. The three fixed runs reuse
    PyRITE's resolved default seed; their overlapping histories can be
    correlated, so their spread is **not an independent-sample SEM**. The
    largest fixed budget is only a *reference estimate*, not ground truth.

    Adaptive sampling stops according to its internal checks or at its maximum.
    Reaching a configured target must be confirmed from the run's convergence
    diagnostics; **do not infer success from the final count**. A noisy or
    rare-event-dominated observable may need many more histories. Changing
    reflection families or energy bins is a *systematic discretization study*,
    distinct from sampling convergence.

    Wall times include Python, compilation/caching and scheduling overheads;
    the small untimed warmup does not make these hardware-comparable timings.
    Try increasing all budgets by 4×, repeating on a clean process, and testing
    the adaptive target with stronger guards. For quantitative uncertainty,
    use independently seeded runs where the execution interface supports them,
    and inspect validated batch-means diagnostics rather than assigning error
    bars to the three plotted points. See the
    [sampling policy reference](https://pyrite.readthedocs.io/en/latest/reference/python-api/config.html),
    [performance guidance](https://pyrite.readthedocs.io/en/latest/computation/index.html),
    and [validation](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html).
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
