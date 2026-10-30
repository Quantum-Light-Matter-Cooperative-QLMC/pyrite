# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc[nvidia,optimization]==0.24.0",
#     "matplotlib>=3.11",
#     "numpy>=2",
# ]
#
# [tool.pyrite]
# backend = "cuda"
# freshness = false
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    os.environ.setdefault("PYRITE_MC_BACKEND", "cuda")
    import subprocess

    import cupy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna

    import pyrite

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mo.status.spinner(title="Preparing PyRITE reference tables"):
        _ = pyrite.fetch_tables()
    tables_ready = True
    return cp, mo, np, plt, pyrite, subprocess, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tuning CUDA execution

    Tune **transport kernels first**, then **spectral reduction kernels**. The
    first study minimizes synchronized electron-transport time by changing
    electron batch and continuation schedule. The second holds the chosen
    transport configuration fixed and minimizes synchronized spectrum time by
    changing only the CXR segment chunk. Both studies run complete simulations
    and reject candidates whose spectral densities differ from their baseline.

    This is a short demonstration, not an exhaustive GPU benchmark. Increase
    trial counts and history counts for a representative production workload.

    This tutorial **needs an NVIDIA GPU**; on Molab, choose a GPU runtime. The
    tuner, `TransportKernelTuning` and `KernelLattice` are experimental.
    Timings depend on the device, driver, precision and the other work on the
    machine.
    """)
    return


@app.cell
def _(cp, np, subprocess):
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.units import ureg

    scene = Scene(
        Beam(kinetic_energy=30 * ureg.keV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "hopg",
            thickness=5 * ureg.um,
            tilt=45 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
        ),
        Detector(observation_angle=90 * ureg.degree),
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=65_536, brem_electrons=1_024),
            seed=7,
            energy_bins=EnergyBins().with_cxr_energies(np.linspace(750, 1_000, 251) * ureg.eV),
            cxr_reflections=CxrReflections(n_families=1),
        ),
    )

    def check_gpu_util():
        _pool = cp.get_default_memory_pool()
        _free, _total = cp.cuda.runtime.memGetInfo()

        _MiB = 1024**2

        print(f"GPU total:        {_total / _MiB:.1f} MiB")
        print(f"GPU free:         {_free / _MiB:.1f} MiB")
        print(f"GPU driver used:  {(_total - _free) / _MiB:.1f} MiB")
        print(f"CuPy pool used:   {_pool.used_bytes() / _MiB:.1f} MiB")
        print(f"CuPy reserved:    {_pool.total_bytes() / _MiB:.1f} MiB")
        print(f"CuPy pool limit:  {_pool.get_limit() / _MiB:.1f} MiB")

        _result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )

        print(_result.stdout or _result.stderr)

    return check_gpu_util, scene


@app.cell
def _(check_gpu_util, mo, scene, tables_ready):
    import inspect

    from pyrite.optimization import KernelLattice, tune_kernels

    assert tables_ready
    phase_objectives_active = "objective" in inspect.signature(tune_kernels).parameters

    def measure_stage(stage_scene, lattice, *, phase, trials, seed):
        options = dict(trials=trials, warmups=1, repeats=1, rtol=1e-4, seed=seed)
        title = f"{phase.capitalize()} kernel tuning"
        if not phase_objectives_active:
            with mo.status.spinner(
                title=title,
                subtitle="Running whole-simulation benchmarks",
            ):
                return tune_kernels(stage_scene, lattice, **options)

        with mo.status.progress_bar(
            total=trials + 1,
            title=title,
            show_eta=True,
        ) as _progress:

            def advance(done, total, state):
                _progress.update(
                    increment=1,
                    subtitle=f"{done}/{total} completed ({state})",
                )

            return tune_kernels(
                stage_scene, lattice, objective=phase, on_progress=advance, **options
            )
            check_gpu_util()

    transport_lattice = KernelLattice(
        electron_batch=(None, 2_048, 4_096, 8_192, 16_384),
        schedule=("full-history", "auto"),
    )
    transport_tuning = measure_stage(scene, transport_lattice, phase="transport", trials=5, seed=11)
    return (
        KernelLattice,
        measure_stage,
        phase_objectives_active,
        transport_tuning,
    )


@app.cell
def _(KernelLattice, measure_stage, scene, transport_tuning):
    from dataclasses import replace

    # Carry the selected transport settings into the spectral study.
    _scene = replace(scene, numerics=scene.numerics.with_kernel_tuning(transport_tuning.best))
    spectral_lattice = KernelLattice(cxr_chunk=(None, 8_192, 16_284, 32_768, 65_536))
    spectral_tuning = measure_stage(_scene, spectral_lattice, phase="spectrum", trials=5, seed=12)

    tuning = spectral_tuning  # Includes the winning transport settings.
    return spectral_tuning, tuning


@app.cell(hide_code=True)
def _(mo, phase_objectives_active, spectral_tuning, transport_tuning):
    def describe(trial):
        t = trial.tuning.transport
        values = {
            "batch": t.electron_batch,
            "schedule": t.schedule,
            "rows": t.continuation_rows,
            "cxr chunk": trial.tuning.cxr_chunk,
        }
        return ", ".join(f"{k} {'default' if v is None else v}" for k, v in values.items())

    metric_label = "synchronized phase" if phase_objectives_active else "whole-simulation"

    def table(label, study):
        _rows = [
            f"### {label}: {metric_label} seconds",
            "| Trial | Settings | Median (s) | Outcome |",
            "|---:|---|---:|---|",
            f"| baseline | {describe(study.baseline)} | {study.baseline.median_s:.3f} | reference |",
        ]
        for _trial in study.trials:
            if _trial.tuning is None or _trial.repeat_of is not None:
                continue
            _time = "—" if _trial.median_s is None else f"{_trial.median_s:.3f}"
            _outcome = "accepted" if _trial.state == "complete" else f"rejected: {_trial.reason}"
            _rows.append(f"| {_trial.number} | {describe(_trial)} | {_time} | {_outcome} |")
        _rows.append(
            f"**Chosen:** {describe(study.chosen)}; "
            f"**{study.speedup:.2f}×** this phase's baseline median."
        )
        return "\n".join(_rows)

    _env = ", ".join(f"{k} `{v}`" for k, v in spectral_tuning.environment.items())
    mo.md(
        f"Measured on {_env}.\n\n"
        + table("Transport", transport_tuning)
        + "\n\n"
        + table("Spectrum", spectral_tuning)
    )
    return


@app.cell
def _(np, phase_objectives_active, plt, spectral_tuning):
    _done = sorted(
        (t for t in spectral_tuning.trials if t.state == "complete"), key=lambda t: t.median_s
    )
    _shown = [spectral_tuning.baseline, *_done[:8]]
    fig_timing, _ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for _i, _trial in enumerate(_shown):
        _samples = np.asarray(_trial.samples_s)
        _ax.scatter(np.full(_samples.size, _i), _samples, color="0.6", s=12)
        _ax.scatter(_i, np.median(_samples), color="C3" if _i == 0 else "C0", marker="_", s=400)
    _ax.set(
        xticks=range(len(_shown)),
        xticklabels=["baseline", *(str(t.number) for t in _shown[1:])],
        xlabel="Trial (accepted, fastest first)",
        ylabel="Spectrum time (s)" if phase_objectives_active else "Full run time (s)",
        title=f"{spectral_tuning.environment.get('device', 'GPU')}: spectral configurations",
    )
    fig_timing  # noqa: B018
    return


@app.cell
def _(mo, np, plt, pyrite, scene, tuning):
    with mo.status.progress_bar(total=2, title="Checking spectra after tuning") as _progress:
        baseline_result = pyrite.simulate(scene)
        _progress.update(subtitle="Original configuration")
        tuned_result = pyrite.simulate(
            scene, numerics=scene.numerics.with_kernel_tuning(tuning.best)
        )
        _progress.update(subtitle="Transport + spectrum winners")
    _energy = baseline_result.cxr.incoherent.photon_energy_eV
    _base = baseline_result.cxr.incoherent.density
    _tuned = tuned_result.cxr.incoherent.density
    assert np.array_equal(_energy, tuned_result.cxr.incoherent.photon_energy_eV)
    assert np.allclose(_tuned, _base, rtol=1e-4, atol=1e-12)
    _floor = 1e-6 * np.abs(_base).max()
    _relative = (_tuned - _base) / np.maximum(np.abs(_base), _floor)

    fig_check, (_top, _bottom) = plt.subplots(
        2, 1, figsize=(7, 4.6), sharex=True, layout="constrained", height_ratios=(3, 1)
    )
    _top.plot(_energy, _base, label="Baseline settings")
    _top.plot(_energy, _tuned, "--", label="Chosen settings")
    _top.set(ylabel="photons / electron / eV / sr")
    _top.legend()
    _bottom.plot(_energy, _relative)
    _bottom.set(xlabel="Photon energy (eV)", ylabel="relative diff.")
    fig_check  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo, phase_objectives_active):
    _phase_note = (
        "Transport and spectrum studies report their **synchronized phase time**. "
        "This is not total wall time; increase repeats for production tuning."
    )
    _fallback_note = (
        "PyRITE v0.24.0 fallback: transport and spectrum settings are tuned "
        "**separately**, but both use **whole-simulation wall time**. "
        "Upgrade PyRITE for per-phase timings and per-trial progress."
    )
    mo.md(_phase_note if phase_objectives_active else _fallback_note)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each trial runs a full CUDA simulation in its **own process**. The first
    (untimed) call absorbs compilation; one synchronized repeat measures the
    selected phase. The transport phase includes its trajectory setup and
    pre-spectral line-grid work. The spectrum phase includes CXR and background
    scoring. The phase timer does **not** isolate one CUDA kernel launch.

    **Transport controls:** electron batch and schedule only. **Spectrum
    controls:** CXR segment chunk only (bremsstrahlung chunk is also available
    in `KernelLattice`). For the second stage, the first stage's winner is held
    fixed. Every candidate's complete spectrum is checked for agreement with
    the stage baseline at `rtol=1e-4`; numerical equivalence is a guard, not a
    claim of scientific validation.

    The default experiment uses four transport candidates and three spectrum
    candidates, instead of one 16-trial mixed search. A slow configuration can
    still take time; the progress display updates **between** complete worker
    processes, not inside a single CUDA launch. Increase `repeats` to 3 or more
    for stable medians, and enlarge the electron count for production tuning.
    You can then apply `scene.numerics.with_kernel_tuning(tuning.best)` in a
    new process. Winners are GPU-, driver- and workload-specific.

    On v0.24.0, the two parameter groups are tuned separately using complete
    simulation wall times; after upgrading to the phase-timing API, each
    study targets the named synchronized phase.
    """)
    return


if __name__ == "__main__":
    app.run()
