# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "endf-parserpy==0.17.0",
#     "marimo>=0.23.3",
#     "pyrite-mc==0.24.0",
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

    # Select the backend before importing PyRITE, then restart to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    with mo.status.spinner(title="Preparing PyRITE reference tables"):
        _ = pyrite.fetch_tables()

    tables_ready = True
    return mo, np, plt, pyrite, tables_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Persisted parameter studies

    A single `simulate()` returns one in-memory `Result`. A **`Sweep`** expands
    a Cartesian product of scenes into cases that a checkpoint-aware runner
    can save, reload and resume. Here we vary beam energy and HOPG thickness
    (four inexpensive CPU cases), inspect the persisted results, and run the
    same sweep a second time without repeating completed transport.

    We use a temporary workspace so rerunning this example cannot mix with
    production checkpoints. Replace it with a persistent directory to resume
    the study after restarting Python.
    """)
    return


@app.cell
def _(np):
    from tempfile import TemporaryDirectory

    from pyrite import Scene, Sweep
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector
    from pyrite.execution import run_sweep
    from pyrite.geometry import Footprint, Slab
    from pyrite.results import Settings, records, select_results, summary_table, sweep_values
    from pyrite.storage import checkpoint_manifest, load_checkpoint
    from pyrite.units import ureg

    energies_keV = np.array([30.0, 40.0])
    thicknesses_nm = np.array([50.0, 100.0])
    scene = Scene(
        Beam(
            kinetic_energy=30 * ureg.keV,
            transverse=PointSpot(),
            longitudinal=PointBunch(),
        ),
        Slab(
            "hopg",
            thickness=50 * ureg.nm,
            tilt=30 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
        ),
        Detector(observation_angle=135 * ureg.degree),
    )
    sweep = Sweep(
        base=scene,
        axes={
            "beam.kinetic_energy": tuple(float(e) * ureg.keV for e in energies_keV),
            "target.thickness": tuple(float(t) * ureg.nm for t in thicknesses_nm),
        },
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=64, brem_electrons=64),
        cxr_reflections=CxrReflections(n_families=1),
        seed=101,
    )
    physics = Physics(
        brem_source="none",
        inelastic_model="continuous",
        radiative_model="uncoupled",
    )
    cases = sweep.cases(numerics=numerics, physics=physics)
    assert len(cases) == 4
    assert len({(case["name"], case["E0_keV"]) for case in cases}) == 4
    return (
        Settings,
        TemporaryDirectory,
        cases,
        checkpoint_manifest,
        energies_keV,
        load_checkpoint,
        records,
        run_sweep,
        select_results,
        summary_table,
        sweep_values,
        thicknesses_nm,
    )


@app.cell
def _(TemporaryDirectory, cases, mo, run_sweep, tables_ready):
    assert tables_ready
    workspace = TemporaryDirectory(prefix="pyrite-persisted-sweep-")
    first_store = {}
    with mo.status.spinner(title="Running and checkpointing four cases"):
        completed = run_sweep(
            cases,
            first_store,
            checkpoint_dir=workspace.name,
            resume=True,
            max_workers=0,
            progress=False,
        )
    assert completed
    assert sum(map(len, first_store.values())) == len(cases)
    return first_store, workspace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reload and resume

    Each case is identified by its saved `name` and electron energy; the
    lowering step also assigns distinct deterministic seeds (101–104).
    Finished cases are written as components beneath `hopg/` in the temporary
    checkpoint root. `load_checkpoint` reconstructs the nested
    `{name: {electron_energy_keV: record}}` store.

    The second run starts with an **empty** in-memory dictionary. Its progress
    callback must report all four cases cached and zero new cases. This tests
    disk-backed resume rather than merely reusing the first Python dictionary.
    """)
    return


@app.cell
def _(cases, first_store, load_checkpoint, np, records, run_sweep, workspace):
    saved_store = load_checkpoint("hopg", checkpoint_dir=workspace.name)
    assert len(records(saved_store)) == len(cases)

    # The loaded arrays are exactly the ones produced in the first execution.
    for _case in cases:
        _key = _case["name"]
        _energy = _case["E0_keV"]
        for _field in ("E_grid", "spec", "brem"):
            np.testing.assert_array_equal(
                first_store[_key][_energy][_field],
                saved_store[_key][_energy][_field],
            )

    resumed_store = {}
    resume_events = []

    def _record_progress(new, total, cached):
        resume_events.append((new, total, cached))

    resumed = run_sweep(
        cases,
        resumed_store,
        checkpoint_dir=workspace.name,
        resume=True,
        max_workers=0,
        progress=False,
        on_progress=_record_progress,
    )
    assert resumed and resume_events
    assert resume_events[-1] == (0, len(cases), len(cases))
    for _case in cases:
        _key = _case["name"]
        _energy = _case["E0_keV"]
        np.testing.assert_array_equal(
            saved_store[_key][_energy]["spec"],
            resumed_store[_key][_energy]["spec"],
        )
    return resume_events, saved_store


@app.cell(hide_code=True)
def _(cases, checkpoint_manifest, mo, workspace):
    manifest = checkpoint_manifest("hopg", checkpoint_dir=workspace.name)
    assert manifest is not None
    assert manifest["n_records"] == len(cases)
    assert manifest["completed_case_set"]["count"] == len(cases)
    seeds = sorted({int(case["seed"]) for case in cases})
    assert seeds == [101, 102, 103, 104]
    mo.md(
        f"**Recorded case seeds:** {seeds}. "
        "**Completed case-set SHA-256:** "
        f"`{manifest['completed_case_set']['sha256']}`. "
        "This hash verifies the set of stored case labels and energies; "
        "it is *not* a physics-content hash. Keep the inputs and table versions "
        "with your saved checkpoint for a reproducible study."
    )
    return


@app.cell
def _(Settings, records, saved_store, select_results, summary_table, sweep_values):
    available = sweep_values(saved_store, fields=("E0_keV", "thickness_ang"))
    selected = select_results(saved_store, E0_keV=30.0)
    assert len(records(selected)) == 2
    # Saved checkpoint records are dictionaries, not simulate() Result objects.
    # summary_table reports current-scaled photon rates and selected geometry.
    selected_summary = summary_table(records(selected), Settings())
    assert len(selected_summary) == 2
    selected_summary  # noqa: B018
    return available, selected_summary


@app.cell
def _(energies_keV, np, plt, records, saved_store, thicknesses_nm):
    # Integrate the *saved* incoherent PXR/CBS source spectrum on its native grid.
    band_yield = np.full((len(thicknesses_nm), len(energies_keV)), np.nan)
    for _record in records(saved_store):
        _case = _record["case"]
        _row = int(np.flatnonzero(np.isclose(thicknesses_nm, _case["thickness_ang"] / 10))[0])
        _col = int(np.flatnonzero(np.isclose(energies_keV, _case["E0_keV"]))[0])
        band_yield[_row, _col] = np.trapezoid(_record["spec"], _record["E_grid"])
    assert np.isfinite(band_yield).all()

    fig, _ax = plt.subplots(figsize=(6.8, 4), layout="constrained")
    _heatmap = _ax.pcolormesh(energies_keV, thicknesses_nm, band_yield, shading="nearest")
    _ax.set(
        title="Saved incoherent PXR/CBS source yield",
        xlabel="Electron energy (keV)",
        ylabel="HOPG thickness (nm)",
        xticks=energies_keV,
        yticks=thicknesses_nm,
    )
    fig.colorbar(_heatmap, ax=_ax, label="Photons / (incident electron · sr)")
    fig  # noqa: B018 - captured by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reuse this for a larger study

    `select_results` filters on **stored case fields** such as `E0_keV` and
    `thickness_ang`, not on input quantity objects; the legacy record stores
    angles in degrees, thickness in ångström and energy in keV.
    `summary_table` uses `Settings()` for read-time response choices and
    returns photon **rates**, while the figure integrates the raw, unscaled
    source density (not pixel counts). The 64 histories and one reflection
    family here are for workflow demonstration, not converged predictions.

    A fixed seed makes case identity and results reproducible for identical
    software, datasets and execution settings; changing physics, numerics or
    axes should use a separate checkpoint root. If interrupted, rerun the same
    `run_sweep(cases, {}, checkpoint_dir=..., resume=True)` command with that
    root to pick up unfinished cases. A temporary directory disappears when
    this notebook closes, so use a real path for cross-session persistence.

    For the equivalent **named profile / CLI** workflow, see
    [Configure and inspect a campaign](https://pyrite.readthedocs.io/en/latest/guides/configuration-cookbook.html)
    and [Checkpoint layout](https://pyrite.readthedocs.io/en/latest/reference/results-and-apps.html).
    """)
    return


if __name__ == "__main__":
    app.run()
