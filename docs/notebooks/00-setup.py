# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.1", "pyrite-mc==0.22.5", "matplotlib>=3.11", "numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.

    return mo, np, os, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # Environment and GPU setup

    Verify your Python environment, choose a backend, and complete one tiny CPU simulation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Install and fetch tables once

    Use Python **3.13 or newer**. Locally, create an environment and install:

    ```bash
    uv venv --python 3.13
    uv pip install 'pyrite-mc[apps]==0.22.5'
    uv run pyrite tables fetch
    uv run pyrite tables verify
    uv run marimo edit docs/notebooks/00-setup.py
    ```

    The reference data download is about 100 MB. Fetch verifies archive SHA-256
    before installation; `tables verify` checks installed pins and payloads.
    The setup cell calls `_ = pyrite.fetch_tables()` to install missing pinned tables
    automatically. Complete installations are reused without a network request.
    The CLI commands above remain useful for manual verification or troubleshooting.
    See the [tables reference](https://pyrite.readthedocs.io/en/latest/repo-design/cli/cli-reference-tables.html).

    **Molab:** open the badge above; its inline dependency manifest installs the
    CPU environment. The first cell fetches missing tables automatically; you can
    run `pyrite tables verify` in Molab's terminal. Start with CPU. If your Molab
    session offers an NVIDIA RTX Pro 6000, select that GPU in Molab, then install
    the NVIDIA extra into the
    same environment:

    ```bash
    uv pip install 'pyrite-mc[nvidia]==0.22.5'
    nvidia-smi
    ```

    Use the same extra for a local NVIDIA GPU. A compatible NVIDIA driver is still
    required; the current extra supplies CUDA 13 CuPy/toolkit packages. GPU type,
    availability and account quotas depend on the host. This notebook does not
    require a GPU or allocate GPU transport by default.
    """)
    return


@app.cell
def _(mo, os, pyrite):
    import importlib.util
    import platform

    environment = {
        "python": platform.python_version(),
        "pyrite": pyrite.__version__,
        "cupy_installed": importlib.util.find_spec("cupy") is not None,
        "selected_backend": os.environ["PYRITE_MC_BACKEND"],
    }
    if environment["cupy_installed"]:
        try:
            import cupy as cp

            environment["cupy_version"] = cp.__version__
            environment["cuda_device_count"] = cp.cuda.runtime.getDeviceCount()
            environment["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
        except Exception as _error:
            environment["cuda_status"] = f"Unavailable: {type(_error).__name__}"
    else:
        environment["cuda_status"] = "CuPy not installed; CPU execution is available"
    mo.json(environment)
    return (environment,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Choose the backend before import

    The first code cell sets `PYRITE_MC_BACKEND` to `cpu` unless the environment
    already selected another backend. For GPU execution, set it to `cuda` **before
    importing PyRITE**. Restart the kernel after changing the backend or installing
    CuPy; rerunning a later cell does not undo imports cached in a live kernel.
    The smoke result below records the actual backend, which should match your
    selection. `nvidia-smi` confirms driver/GPU visibility; importing CuPy alone
    does not prove a compatible device exists.
    """)
    return


@app.cell
def _():
    from pyrite import simulate
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab

    return (
        Beam,
        CxrReflections,
        Detector,
        FixedSampling,
        Footprint,
        Numerics,
        Physics,
        PointBunch,
        PointSpot,
        Slab,
        simulate,
    )


@app.cell
def _(
    Beam,
    CxrReflections,
    Detector,
    FixedSampling,
    Footprint,
    Numerics,
    Physics,
    PointBunch,
    PointSpot,
    Slab,
    simulate,
):
    smoke = simulate(
        Beam(energy_keV=30.0, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab("hopg", thickness_ang=100.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        Detector(observation_angle_deg=135.0),
        physics=Physics(brem_source="none"),
        numerics=Numerics(
            sampling=FixedSampling(cxr_electrons=16, brem_electrons=1),
            cxr_reflections=CxrReflections(n_families=1),
        ),
    )
    return (smoke,)


@app.cell
def _(mo, np, smoke):
    smoke_summary = {
        "backend": smoke.provenance["backend"],
        "energy_nodes": int(smoke.cxr.incoherent.photon_energy_eV.size),
        "finite_spectrum": bool(np.isfinite(smoke.cxr.incoherent.density).all()),
        "nonzero_spectrum": bool(np.any(smoke.cxr.incoherent.density > 0)),
    }
    assert smoke_summary["finite_spectrum"] and smoke_summary["nonzero_spectrum"]
    mo.json(smoke_summary)
    return (smoke_summary,)


@app.cell(hide_code=True)
def _(mo, pyrite, smoke):
    tutorial_provenance = {
        "tutorial": "00-setup",
        "pyrite_version": pyrite.__version__,
        "backend": smoke.provenance["backend"],
        "seed": smoke.case["seed"],
        "table_identity": dict(smoke.provenance["xsgen_tables"]),
        "source_identity": smoke.provenance["identity_digest"],
    }
    # Preserve machine-readable export metadata without displaying it to readers.
    mo.json(tutorial_provenance).style({"display": "none"})
    return (tutorial_provenance,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Continue

    This tiny, fixed-seed run verifies that the selected backend can produce finite,
    nonzero spectral data with the installed tables. It is **not a convergence or
    physics-validation test**. The histories are numerical samples, not a physical
    beam charge. Check the [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html)
    for scientific evidence and limitations.

    Continue with **01 — Your first X-ray spectrum**. Keep the core sequence on CPU
    until workload size justifies a GPU. GPU outputs can differ across architectures;
    compare physically meaningful quantities with justified tolerances, not exact
    random-history identity.
    """)
    return


if __name__ == "__main__":
    app.run()
