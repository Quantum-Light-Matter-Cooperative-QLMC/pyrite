# /// script
# requires-python = ">=3.13"
# dependencies = ["endf-parserpy==0.17.0", "marimo>=0.25.1", "pyrite-mc==0.22.5", "matplotlib>=3.11", "numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=False)
def _():
    import os

    # Change to "cuda" before the first PyRITE import; restart an existing kernel.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import numpy as np

    import pyrite

    _ = pyrite.fetch_tables()  # Fetch missing pinned tables once; reuse installed data.
    tables_ready = True
    return tables_ready, mo, np, os, pyrite


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Environment and GPU setup

    Verify your Python environment, choose a backend, and complete one tiny CPU simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install and fetch tables once

    **Add Packages**
    To add or change a dependency, open MoLab's built-in package manager on the
    left and enter the package name and version, for example `endf-parserpy==0.17.0`.

    The required interaction data download is about 100 MB. The setup cell calls `_ = pyrite.fetch_tables()`
    to install missing tables automatically -- see the [tables reference](https://pyrite.readthedocs.io/en/latest/repo-design/cli/cli-reference-tables.html).

    If an existing session reports **MF=26/MT=527 photon yield table is missing**,
    open MoLab's package manager, enter `endf-parserpy==0.17.0`, and restart
    the kernel.

    **NVIDIA GPU:** start with CPU. If your MoLab session offers an NVIDIA
    RTX Pro 6000, first add `pyrite-mc[nvidia]` as a dependency, then select
    that GPU in the Molab menu before restarting the kernel.
    """)
    return


@app.cell(hide_code=True)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose the backend before import

    The first code cell sets `PYRITE_MC_BACKEND` to `cpu` unless the environment
    already selected another backend. For GPU execution, set it to `cuda` **before
    importing PyRITE**. Restart the kernel after changing the backend or installing
    CuPy.
    """)
    return


@app.cell(hide_code=False)
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
    tables_ready,
):
    assert tables_ready  # Wait for reference-table installation before transport.
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


@app.cell(hide_code=True)
def _(mo, np, smoke):
    smoke_summary = {
        "backend": smoke.provenance["backend"],
        "energy_nodes": int(smoke.cxr.incoherent.photon_energy_eV.size),
        "finite_spectrum": bool(np.isfinite(smoke.cxr.incoherent.density).all()),
        "nonzero_spectrum": bool(np.any(smoke.cxr.incoherent.density > 0)),
    }
    assert smoke_summary["finite_spectrum"] and smoke_summary["nonzero_spectrum"]
    mo.json(smoke_summary)
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continue

    This tiny, fixed-seed run verifies that the selected backend can produce finite,
    nonzero spectral data with the installed tables.

    Continue with **01 — Your first X-ray spectrum**. Keep the core sequence on CPU
    until workload size justifies a GPU. GPU outputs can differ across architectures;
    compare physically meaningful quantities with justified tolerances, not exact
    random-history identity.
    """)
    return


if __name__ == "__main__":
    app.run()
