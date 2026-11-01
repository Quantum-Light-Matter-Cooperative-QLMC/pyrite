# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc[nvidia,optimization]==0.25.0",
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
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna

    import pyrite

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mo.status.spinner(title="Preparing PyRITE reference tables"):
        _ = pyrite.fetch_tables()
    tables_ready = True
    return mo, np, plt, pyrite, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tuning CUDA execution

    A simulation has two phases: **electron transport**, then **spectrum
    scoring** of the transported segments. PyRITE tunes their execution
    controls as two independent Optuna studies:

    - **Transport** candidates change how histories are launched: histories
      per launch (`electron_batch` cap and `scratch_mib` device budget), the
      `schedule` (whole histories or bounded continuation rounds), the
      continuation row quantum and the CUDA threads per block. Each must
      reproduce the baseline's transported arrays **bit for bit**.
    - **Spectrum** candidates change how kept segments are scored: the CXR and
      bremsstrahlung segment chunks and the CXR line-reduction launch (threads
      and energy nodes per block). They rescore *one* kept transport and must
      reproduce its spectra to rounding.

    None of these controls changes the physics or the result's identity, only
    how the work is laid onto the GPU. Their best values depend on the scene
    and the device, so this tutorial reads the GPU first and scales the search
    ranges to it. A large GPU such as Molab's **RTX Pro 6000** (96 GB) can
    hold far more concurrent histories than the defaults, which were chosen
    for 16 GB cards.

    This tutorial **needs an NVIDIA GPU**; on Molab, choose a GPU runtime. The
    tuner, `pyrite.phases` and the kernel-tuning fields are experimental.
    Timings depend on the device, driver, precision and other work on the
    machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import cupy

    _device = cupy.cuda.Device()
    _props = cupy.cuda.runtime.getDeviceProperties(_device.id)
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    gpu = {
        "name": _props["name"].decode(),
        "sms": int(_props["multiProcessorCount"]),
        "threads_per_sm": int(_props["maxThreadsPerMultiProcessor"]),
        "compute_capability": f"{_props['major']}.{_props['minor']}",
        "total_GiB": total_bytes / 2**30,
        "free_GiB": free_bytes / 2**30,
    }
    resident_threads = gpu["sms"] * gpu["threads_per_sm"]
    mo.md(
        f"**{gpu['name']}** (compute {gpu['compute_capability']}): {gpu['sms']} SMs, "
        f"{resident_threads:,} resident threads at full occupancy, "
        f"{gpu['free_GiB']:.1f} of {gpu['total_GiB']:.1f} GiB free."
    )
    return free_bytes, gpu, resident_threads


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose a workload and a budget

    Two workloads stress different phases. The **spectrum-heavy** case is a
    thin HOPG crystal at 30 keV, where scoring the CXR lines dominates. The
    **transport-heavy** case is 100 µm of HOPG at 1 MeV, where each electron
    makes many more steps. More Monte Carlo electrons fill a large GPU better
    and make timings more stable, but every candidate runs the phase
    `1 + repeats` times. The trial budget is per phase; a budget smaller than
    the grid searches it with TPE instead of trying every combination.
    """)
    return


@app.cell
def _(mo):
    workload = mo.ui.dropdown(
        options={
            "Spectrum-heavy: HOPG 5 µm, 30 keV": "spectrum",
            "Transport-heavy: HOPG 100 µm, 1 MeV": "transport",
        },
        value="Spectrum-heavy: HOPG 5 µm, 30 keV",
        label="Workload",
    )
    electrons = mo.ui.dropdown(
        options={f"{n:,}": n for n in (4_096, 16_384, 65_536, 262_144)},
        value="16,384",
        label="Monte Carlo electrons (CXR)",
    )
    trial_budget = mo.ui.slider(8, 200, value=40, step=4, label="Trials per phase")
    repeats = mo.ui.slider(2, 7, value=3, label="Timed repeats per trial")
    timeout_min = mo.ui.slider(5, 80, value=30, step=5, label="Time limit per phase (min)")
    mo.vstack([workload, electrons, trial_budget, repeats, timeout_min])
    return electrons, repeats, timeout_min, trial_budget, workload


@app.cell
def _(electrons, np, workload):
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.units import ureg

    _kinetic, _thickness = {
        "spectrum": (30 * ureg.keV, 5 * ureg.um),
        "transport": (1 * ureg.MeV, 100 * ureg.um),
    }[workload.value]
    scene = Scene(
        Beam(kinetic_energy=_kinetic, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "hopg",
            thickness=_thickness,
            tilt=45 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
        ),
        Detector(observation_angle=90 * ureg.degree),
        numerics=Numerics(
            sampling=FixedSampling(
                cxr_electrons=electrons.value, brem_electrons=max(1_024, electrons.value // 16)
            ),
            seed=7,
            # A fixed 1 eV window around the HOPG (002) line keeps the spectrum
            # grid the same for every candidate and every electron count.
            energy_bins=EnergyBins().with_cxr_energies(np.linspace(750, 1_000, 251) * ureg.eV),
            cxr_reflections=CxrReflections(n_families=1),
        ),
    )
    return (scene,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the time goes

    `simulate_transport` stops after transport; `score_spectra` scores the
    kept segments. Timing them separately shows which phase is worth tuning.
    The first, untimed run compiles the kernels.
    """)
    return


@app.cell(hide_code=True)
def _(mo, scene, tables_ready):
    from time import perf_counter

    from pyrite.phases import score_spectra, simulate_transport

    assert tables_ready
    with mo.status.progress_bar(total=100, title="One untimed run, then timed phases") as _bar:

        def show(event):
            _bar.update(
                increment=0,
                subtitle=f"{event.phase}: {event.stage} {event.done:,}/{event.total:,}",
            )

        score_spectra(simulate_transport(scene))  # compile kernels first
        _started = perf_counter()
        transported = simulate_transport(scene, progress=show)
        _bar.update(increment=50)
        transport_s = perf_counter() - _started
        _started = perf_counter()
        _ = score_spectra(transported, progress=show)
        _bar.update(increment=50)
        spectrum_s = perf_counter() - _started
    del transported  # releases the kept segments on the device
    mo.md(
        f"Transport **{transport_s:.2f} s**, spectrum **{spectrum_s:.2f} s** "
        f"({spectrum_s / (transport_s + spectrum_s):.0%} of the run)."
    )
    return spectrum_s, transport_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A search space scaled to this GPU

    Each control lists its candidate values; `None` keeps the built-in
    default. The ranges below are derived from the device read above:

    - `scratch_mib` runs from the 512 MiB default up to half of the free
      memory, because the scratch budget, not `electron_batch`, usually sets
      how many histories share one launch.
    - `electron_batch` caps histories per launch from a few thousand up to
      several times the GPU's resident-thread capacity.
    - `kernel_threads` and `cxr_threads` are threads per block. Small blocks
      spread a short launch over more SMs; large blocks amortize scheduling.
    - `continuation_rows` is the rows each history may emit per continuation
      round; `"full-history"` ignores it, so those pairs are skipped.

    Widen or narrow any tuple; an empty tuple leaves that control untuned.
    """)
    return


@app.cell(hide_code=True)
def _(electrons, free_bytes, mo, resident_threads, scene, trial_budget):
    from pyrite.optimization import KernelSearchSpace

    def doubling(start, stop):
        values, value = [], start
        while value <= stop:
            values.append(value)
            value *= 2
        return tuple(values)

    _scratch_cap = max(1_024, int(0.5 * free_bytes / 2**20))
    _batch_cap = max(4 * resident_threads, electrons.value)
    space = KernelSearchSpace(
        # Transport phase
        electron_batch=(None, *doubling(4_096, _batch_cap)),
        schedule=("full-history", "continuation", "auto"),
        continuation_rows=(None, 64, 128, 512, 1_024),
        scratch_mib=(None, *doubling(1_024, _scratch_cap)),
        kernel_threads=(None, 32, 64, 256),
        # Spectrum phase
        cxr_chunk=(None, *doubling(16_384, 1_048_576)),
        brem_chunk=(None, 1_024, 4_096, 16_384),
        cxr_threads=(None, 128, 256, 1_024),
        cxr_energies_per_block=(None, 1, 2, 4),
    )
    _base = scene.numerics.kernel_tuning
    grid = {_p: len(space.choices(_base, _p)) for _p in ("transport", "spectrum")}
    # tune_kernels tries every combination when the budget covers the grid.
    trials = {_p: min(n, trial_budget.value) for _p, n in grid.items()}
    _rows = [
        f"| {_p} | {grid[_p]:,} | {trials[_p]}"
        f"{' (every combination)' if trials[_p] == grid[_p] else ' (TPE)'} |"
        for _p in grid
    ]
    _dims = "\n".join(
        f"- `{_name}`: {', '.join('default' if v is None else f'{v:,}' if isinstance(v, int) else v for v in getattr(space, _name))}"
        for _p in ("transport", "spectrum")
        for _name in space.dimensions(_p)
    )
    mo.md(
        "| Phase | Valid combinations | Trials run |\n|---|---:|---|\n"
        + "\n".join(_rows)
        + "\n\n"
        + _dims
    )
    return space, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the studies

    Trials go to an SQLite file next to the notebook. If the session drops,
    for instance at Molab's idle shutdown, rerunning this cell resumes the
    same study: finished trials are kept, and only the remaining budget runs.
    A changed workload, space or device starts a new study under a new name.
    """)
    return


@app.cell
def _(electrons, mo, repeats, scene, space, timeout_min, trial_budget, trials, workload):
    from pyrite.optimization import tune_kernels

    _total = sum(trials.values())
    _name = f"tutorial-15-{workload.value}-{electrons.value}"
    with mo.status.progress_bar(
        total=_total + 2, title="Tuning transport, then spectrum", show_eta=True
    ) as _bar:

        def advance(update):
            if update.simulation is not None:
                _sim = update.simulation
                _detail = (
                    f"run {update.run}/{update.runs}: {_sim.stage} {_sim.done:,}/{_sim.total:,}"
                )
            else:
                _detail = update.state
            _finished = update.state in ("baseline", "complete", "rejected")
            _bar.update(
                increment=1 if _finished else 0,
                subtitle=f"{update.phase} {update.trial}/{update.trials}: {_detail}",
            )

        tuning = tune_kernels(
            scene,
            space,
            trials=trial_budget.value,
            warmups=1,
            repeats=repeats.value,
            timeout=60.0 * timeout_min.value,
            storage="sqlite:///pyrite-kernel-tuning.sqlite3",
            study_name=_name,
            on_progress=advance,
        )
    return (tuning,)


@app.cell(hide_code=True)
def _(mo, spectrum_s, transport_s, tuning):
    _DEFAULT = {
        "electron_batch": 65_536,
        "schedule": "auto",
        "continuation_rows": 256,
        "scratch_mib": 512,
        "kernel_threads": 32,
        "cxr_chunk": "case",
        "brem_chunk": "case",
        "cxr_threads": 512,
        "cxr_energies_per_block": 3,
    }

    def settings(tuning_config, phase):
        t = tuning_config.transport
        if phase == "transport":
            values = {
                "batch": t.electron_batch,
                "schedule": t.schedule,
                "rows": t.continuation_rows,
                "scratch MiB": t.scratch_mib,
                "threads": t.kernel_threads,
            }
        else:
            values = {
                "cxr chunk": tuning_config.cxr_chunk,
                "brem chunk": tuning_config.brem_chunk,
                "cxr threads": tuning_config.cxr_threads,
                "energies/block": tuning_config.cxr_energies_per_block,
            }
        return (
            ", ".join(
                f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                for k, v in values.items()
                if v is not None
            )
            or "defaults"
        )

    def table(study, top=12):
        _done = sorted(
            (t for t in study.trials if t.state == "complete" and t.repeat_of is None),
            key=lambda t: t.median_s,
        )
        _rejected = sum(t.state == "rejected" for t in study.trials)
        _rows = [
            f"### {study.phase.capitalize()}: fastest {min(top, len(_done))} of {len(_done)} "
            f"completed trials ({_rejected} rejected)",
            "| Trial | Settings | Median (s) | vs baseline |",
            "|---:|---|---:|---:|",
            f"| baseline | {settings(study.baseline.tuning, study.phase)} "
            f"| {study.baseline.median_s:.3f} | 1.00× |",
        ]
        for _trial in _done[:top]:
            _rows.append(
                f"| {_trial.number} | {settings(_trial.tuning, study.phase)} "
                f"| {_trial.median_s:.3f} | {study.baseline.median_s / _trial.median_s:.2f}× |"
            )
        _verdict = "confirmed" if study.improved else "kept the baseline"
        _rows.append(
            f"\n**Chosen:** {settings(study.chosen.tuning, study.phase)} ({_verdict}); "
            f"**{study.speedup:.2f}×** the baseline phase median in the interleaved re-timing."
        )
        return "\n".join(_rows)

    _whole = (transport_s + spectrum_s) / (
        transport_s / tuning.transport.speedup + spectrum_s / tuning.spectrum.speedup
    )
    _env = ", ".join(f"{k} `{v}`" for k, v in tuning.environment.items())
    mo.md(
        f"Measured on {_env}.\n\n"
        + "\n\n".join(table(_study) for _study in tuning.phases)
        + f"\n\n**Estimated whole-run speedup: {_whole:.2f}×**. A phase's gain counts "
        "only in proportion to its share of the run. Built-in defaults: "
        + ", ".join(f"`{k}` {v}" for k, v in _DEFAULT.items())
        + "."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which controls matter

    Each panel plots every completed trial's phase time against one control's
    value, with the baseline as a dashed line. A control whose points sit at
    the same height for every value barely matters on this workload and GPU;
    one with a clear trend is worth tuning per device.
    """)
    return


@app.cell
def _(np, plt, space, tuning):
    _panels = [
        (_study, _name) for _study in tuning.phases for _name in space.dimensions(_study.phase)
    ]
    _cols = 3
    fig_controls, _axes = plt.subplots(
        -(-len(_panels) // _cols),
        _cols,
        figsize=(10, 2.6 * -(-len(_panels) // _cols)),
        layout="constrained",
        squeeze=False,
    )
    for _ax, (_study, _name) in zip(_axes.flat, _panels, strict=False):
        _labels = ["default" if v is None else str(v) for v in getattr(space, _name)]
        for _trial in _study.trials:
            if _trial.state != "complete" or _trial.repeat_of is not None:
                continue
            _holder = _trial.tuning.transport if _study.phase == "transport" else _trial.tuning
            _value = getattr(_holder, _name)
            _x = _labels.index("default" if _value is None else str(_value))
            _ax.scatter(
                _x + np.random.default_rng(_trial.number).uniform(-0.15, 0.15),
                _trial.median_s,
                s=14,
                color="C0",
                alpha=0.7,
            )
        _ax.axhline(_study.baseline.median_s, color="C3", ls="--", lw=1)
        _ax.set_xticks(range(len(_labels)), _labels, rotation=30, fontsize=7)
        _ax.set_title(f"{_study.phase}: {_name}", fontsize=9)
        _ax.set_ylabel("phase time (s)", fontsize=8)
        _ax.set_ylim(bottom=0)
    for _ax in list(_axes.flat)[len(_panels) :]:
        _ax.set_visible(False)
    fig_controls.suptitle(tuning.environment.get("device", "GPU"))
    fig_controls  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply and check the joined winner

    `tuning.best` joins the transport winner with the spectrum winner.
    Applying it through ordinary `Numerics` must leave the spectrum unchanged
    up to the summation-order differences that chunking and the reduction
    launch can introduce.
    """)
    return


@app.cell
def _(mo, np, plt, pyrite, scene, tuning):
    with mo.status.progress_bar(total=2, title="Checking spectra after tuning") as _progress:
        baseline_result = pyrite.simulate(scene)
        _progress.update(subtitle="Original configuration")
        tuned_result = pyrite.simulate(
            scene, numerics=scene.numerics.with_kernel_tuning(tuning.best)
        )
        _progress.update(subtitle="Joined transport + spectrum winners")
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


@app.cell
def _(gpu, mo, tuning):
    import json
    from dataclasses import asdict

    _record = {
        "gpu": gpu,
        "environment": tuning.environment,
        "best": asdict(tuning.best),
        "phases": [asdict(_study) for _study in tuning.phases],
    }
    mo.download(
        data=json.dumps(_record, indent=1, default=str).encode(),
        filename="pyrite-kernel-tuning.json",
        label="Download the tuning record (JSON)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the numbers mean

    Every candidate runs in one CUDA **worker process** that compiles kernels
    and loads tables once and releases the device memory pool before each
    candidate; `isolate="candidate"` restarts it per candidate instead, at a
    large cost. A candidate gets one untimed warm-up and the chosen number of
    synchronized timed runs of its phase. The fastest accepted candidate is
    then re-timed in interleaved pairs with the baseline, and replaces it only
    for at least a 2% confirmed gain (`min_speedup`). "Kept the baseline"
    means the defaults were already as fast here, not that tuning failed.

    A candidate is **rejected**, not timed, when its transport differs by a
    single bit, its spectrum leaves the tolerance, or it needs the
    out-of-memory chunk-halving retry. A large `scratch_mib` that does not fit
    beside the tables is therefore reported, not silently slowed.

    The transport timer includes trajectory setup and line-grid resolution;
    the spectrum timer includes CXR, characteristic and bremsstrahlung
    scoring. Neither isolates one kernel launch. The equality checks are a
    tuning guard, not scientific validation. Winners are GPU-, driver- and
    workload-specific: keep the downloaded record with its device, re-tune for
    a production workload, and apply
    `scene.numerics.with_kernel_tuning(tuning.best)` in a new process.
    """)
    return


if __name__ == "__main__":
    app.run()
