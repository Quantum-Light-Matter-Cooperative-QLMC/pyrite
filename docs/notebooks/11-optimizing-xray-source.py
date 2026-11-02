# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc[optimization]==0.25.0",
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
    import warnings
    from dataclasses import replace
    from pathlib import Path
    from tempfile import TemporaryDirectory

    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna

    import pyrite

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mo.status.spinner(title="Preparing PyRITE reference tables, warming JIT caches"):
        _ = pyrite.fetch_tables()  # Fetch data tables
        _ = pyrite.execution.warm_cache()

    tables_ready = True

    checkpoint_directory = TemporaryDirectory(prefix="pyrite-optimization-tutorial-")
    return (
        Path,
        checkpoint_directory,
        mo,
        np,
        optuna,
        plt,
        replace,
        tables_ready,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimizing an X-ray source

    **Design question:** which graphite tilt and thickness maximize incoherent
    PXR/CBS emission between **820 and 900 eV**, observed at 90° from a 30 keV
    beam? We hold the material, beam and observation direction fixed.

    Search **30–60° tilt (1° increments)** and **10–50 nm thickness (5 nm
    increments)**: 279 feasible designs, respecting the **50 nm fabrication
    limit**. Unlike a search that evaluates slabs *above* that limit,
    every trial here is eligible.
    We compare against a clearly identified **45° / 50 nm reference design**.
    The 48-trial budget samples only part of the grid; untested locations are
    never interpolated or presented as measured values.

    Maximizing yield alone can push thickness to the allowed upper boundary:
    this is not an optimization of fabrication cost, signal-to-background, or
    detector-accepted photon rate. The optimization API is experimental.

    For local execution, install the optional optimizer:
    ```bash
    uv tool install "pyrite-mc[optimization]"
    ```
    """)
    return


@app.cell
def _(np):
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.optimization import (
        FixedWindowYield,
        Integer,
        Objective,
        OptimizationStudy,
        OptunaOptimizer,
        ParameterSpace,
        PyriteEvaluator,
    )
    from pyrite.units import ureg

    search_numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=32, brem_electrons=32),
        transport_core="per-electron",
        energy_bins=(
            EnergyBins()
            .with_cxr_energies(np.linspace(750, 1_000, 251) * ureg.eV)
            .with_brem_energies(np.array([750.0, 1_000.0]) * ureg.eV)
        ),
        cxr_reflections=CxrReflections(n_families=1),
    )
    base = Scene(
        Beam(kinetic_energy=30 * ureg.keV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "hopg",
            thickness=50 * ureg.nm,
            tilt=45 * ureg.degree,
            tilt_azimuth=180 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
            mosaic=False,
        ),
        Detector(observation_angle=90 * ureg.degree),
        physics=Physics(
            brem_source=None,
            inelastic_model="continuous",
            radiative_model="uncoupled",
            bremsstrahlung_model="eedl",
        ),
        numerics=search_numerics,
    )
    score = FixedWindowYield(820 * ureg.eV, 900 * ureg.eV)
    definition = OptimizationStudy(
        base,
        ParameterSpace(
            {
                "target.tilt": Integer(30 * ureg.degree, 60 * ureg.degree, step=1),
                "target.thickness": Integer(100 * ureg.angstrom, 500 * ureg.angstrom, step=50),
            }
        ),
        (Objective(score.name, "maximize", score.units),),
    )
    return (
        EnergyBins,
        FixedSampling,
        Numerics,
        OptunaOptimizer,
        PyriteEvaluator,
        base,
        definition,
        score,
        search_numerics,
        ureg,
    )


@app.cell
def _(score):
    def candidate_metrics(store):
        return score(store)

    return (candidate_metrics,)


@app.cell
def _(
    OptunaOptimizer,
    Path,
    PyriteEvaluator,
    candidate_metrics,
    checkpoint_directory,
    definition,
    mo,
    optuna,
    tables_ready,
    warnings,
):
    assert tables_ready
    run_options = {
        "checkpoint_path": str(Path(checkpoint_directory.name) / "search.h5"),
        "max_workers": 0,
        "cache_read": False,
        "cache_write": False,
    }
    evaluator = PyriteEvaluator(candidate_metrics, base_seed=21, run_options=run_options)
    with warnings.catch_warnings(action="ignore", category=optuna.exceptions.ExperimentalWarning):
        optimizer = OptunaOptimizer(
            definition.parameters,
            definition.objectives,
            constraints=definition.constraints,
            seed=21,
        )
    with mo.status.spinner(
        title="Searching X-ray source designs",
        subtitle="48 sampled geometries (32 histories each)",
    ):
        with mo.capture_stdout():
            search_result = definition.run(evaluator, optimizer, max_trials=48)
    assert len(search_result.trials) == 48
    assert all(_trial.state.value != "failed" for _trial in search_result.trials)
    finalist = search_result.best_trial
    assert finalist is not None
    return finalist, run_options, search_result


@app.cell
def _(finalist, np, plt, score, search_result):
    # Every point is an *evaluated* geometry; no contour interpolation of sparse data.
    _trials = sorted(search_result.trials, key=lambda trial: trial.number)
    _numbers = np.array([trial.number + 1 for trial in _trials])
    _yields = np.array([trial.objectives[score.name] for trial in _trials])
    _tilts = np.array([trial.parameters["target.tilt"] for trial in _trials])
    _thicknesses = np.array([trial.parameters["target.thickness"] / 10 for trial in _trials])
    _best = np.maximum.accumulate(_yields)

    fig_search, (_map, _history) = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    _points = _map.scatter(
        _tilts,
        _thicknesses,
        c=_yields,
        cmap="viridis",
        s=65,
        edgecolors="black",
        linewidths=0.35,
    )
    fig_search.colorbar(_points, ax=_map, label="820–900 eV yield (photons / electron / sr)")
    _map.plot(
        45,
        50,
        marker="D",
        ms=9,
        mfc="none",
        mec="C3",
        mew=1.8,
        linestyle="none",
        label="Reference: 45°, 50 nm",
    )
    _map.plot(
        finalist.parameters["target.tilt"],
        finalist.parameters["target.thickness"] / 10,
        marker="*",
        ms=17,
        mfc="none",
        mec="black",
        mew=1.5,
        linestyle="none",
        label="Selected design",
    )
    _map.set(
        xlabel="Graphite tilt (degrees)",
        ylabel="Thickness (nm)",
        title="Where the optimizer actually sampled",
        xlim=(29, 61),
        ylim=(5, 55),
    )
    _map.legend(fontsize=8, loc="upper left")

    _history.scatter(_numbers, _yields, s=18, alpha=0.55, label="Individual trial")
    _history.step(_numbers, _best, where="post", color="C3", label="Best observed so far")
    _history.set(
        xlabel="Evaluation number",
        title="How the search progressed",
        ylabel="820–900 eV yield (photons / electron / sr)",
        xlim=(1, len(_numbers)),
    )
    _history.legend(fontsize=8)
    fig_search  # noqa: B018
    return


@app.cell
def _(
    EnergyBins,
    FixedSampling,
    Numerics,
    PyriteEvaluator,
    base,
    candidate_metrics,
    definition,
    finalist,
    mo,
    np,
    replace,
    run_options,
    search_numerics,
    score,
    tables_ready,
    ureg,
):
    assert tables_ready
    validation_numerics = Numerics(
        transport_core="per-electron",
        cxr_reflections=search_numerics.cxr_reflections,
        sampling=FixedSampling(cxr_electrons=256, brem_electrons=256),
        energy_bins=(
            EnergyBins()
            .with_cxr_energies(np.linspace(750, 1_000, 251) * ureg.eV)
            .with_brem_energies(np.array([750.0, 1_000.0]) * ureg.eV)
        ),
    )
    validation_scene = replace(
        definition.scene(finalist.parameters),
        numerics=validation_numerics,
        physics=base.physics,
    )
    reference_scene = replace(base, numerics=validation_numerics)
    validation_evaluator = PyriteEvaluator(
        candidate_metrics,
        base_seed=1_001,
        run_options={
            **run_options,
            "checkpoint_path": run_options["checkpoint_path"].replace("search.h5", "validation.h5"),
        },
    )
    _reference = []
    _independent = []
    with mo.status.progress_bar(total=8, title="Comparing reference and winner") as _progress:
        with mo.capture_stdout():
            for _replicate in range(4):
                # Same held-out seed for both geometries; never reuse search seeds.
                _reference.append(validation_evaluator(reference_scene, _replicate)[score.name])
                _progress.update(subtitle=f"Reference, seed {_replicate + 1}/4")
                _independent.append(validation_evaluator(validation_scene, _replicate)[score.name])
                _progress.update(subtitle=f"Finalist, seed {_replicate + 1}/4")
    reference_yields = np.array(_reference)
    independent_yields = np.array(_independent)
    assert np.isfinite(reference_yields).all() and np.all(reference_yields >= 0)
    assert np.isfinite(independent_yields).all() and np.all(independent_yields >= 0)
    return independent_yields, reference_yields


@app.cell
def _(finalist, independent_yields, np, plt, reference_yields):
    fig_validation, _ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    # Grey connections denote evaluations made with the same held-out seed.
    for _ref, _winner in zip(reference_yields, independent_yields, strict=True):
        _ax.plot((0, 1), (_ref, _winner), color="0.7", alpha=0.6, zorder=1)
    for _index, _values, _label in (
        (0, reference_yields, "Reference"),
        (1, independent_yields, "Selected design"),
    ):
        _ax.scatter(np.full(len(_values), _index), _values, color="0.5", s=20, zorder=2)
        _ax.errorbar(
            _index,
            _values.mean(),
            yerr=_values.std(ddof=1) / np.sqrt(len(_values)),
            fmt="D",
            color=f"C{_index}",
            markersize=8,
            capsize=6,
            zorder=3,
            label=f"{_label}: mean ± SEM",
        )
    _ax.set(
        xticks=(0, 1),
        xticklabels=("Reference: 45°, 50 nm", "Selected geometry"),
        ylabel="820–900 eV yield (photons / electron / sr)",
        xlim=(-0.35, 1.35),
        title=f"Independent seeds · finalist {finalist.parameters['target.tilt']:g}° / "
        f"{finalist.parameters['target.thickness'] / 10:g} nm",
    )
    _ax.legend(fontsize=8)
    fig_validation  # noqa: B018
    return


@app.cell(hide_code=True)
def _(finalist, independent_yields, mo, reference_yields):
    _gain = (
        independent_yields.mean() / reference_yields.mean()
        if reference_yields.mean() > 0
        else float("nan")
    )
    mo.md(f"""
    ## Interpretation

    The selected design is **{finalist.parameters["target.tilt"]:g}° tilt**, **{finalist.parameters["target.thickness"] / 10:g} nm**.
    Independent mean yield is **{independent_yields.mean():.3g} photons/electron/sr**,
    versus **{reference_yields.mean():.3g}** for the 45°/50 nm reference
    (**{_gain:.2f}×** their mean ratio, when defined). Grey lines in the plot
    connect the four *paired* held-out seeds (1,001–1,004); diamonds show means
    and **SEM**. This comparison uses 256 electron histories per sample and the
    same **1 eV** photon grid for search, reference and finalist.

    Only actual evaluated geometries appear in the design map. Repeated trials
    at the same position overlap; color is a noisy search estimate, not a
    smooth physical response surface. The staircase in the right panel is
    selection progress, **not** a plot of yield versus tilt or thickness.

    The score is the integrated *incoherent PXR/CBS directional source yield*
    at 90°, **not** detector counts or accepted flux. Picking the maximum of
    48 noisy 32-history trials biases its search estimate upward (the winner's
    curse); rely on the independent comparison above. The error bars reflect
    Monte Carlo variation only, not grid, atomic-model or experimental errors.
    If the winner sits at the **50 nm fabrication limit**, that may mean more
    material simply emits more photons under this objective: for a meaningful
    engineering optimum, add cost or signal-to-background objectives rather
    than raising the fabrication limit without justification. A sampled maximum is not a global optimum.
    """)
    return


if __name__ == "__main__":
    app.run()
