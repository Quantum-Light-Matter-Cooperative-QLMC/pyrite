# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.1", "pyrite-mc==0.22.4", "matplotlib>=3.11", "numpy>=2"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Select the backend before the first PyRITE import; restart to change it.
    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import pyrite

    pyrite.fetch_tables()  # Complete installed tables require no network access.
    return mo, np, plt, pyrite


@app.cell
def _(mo):
    mo.md(r"""
    # Angular radiation distributions

    [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/docs/notebooks/05-angular-radiation-distributions.py)

    Compute a detector-free PXR/CBS energy–angle map and compare spectra at three observation directions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Define an angular slice

    The incident electron travels along lab **+z**. Angles are signed in the
    **x–z plane**: 0° is forward, +90° is +x, and ±180° is backward. The result
    is a pointwise source density **per steradian**, not a count in an angular bin.

    `directional_source()` is **experimental**. Use a finite slab so every sampled
    direction has an escape path. The explicit energy grid is deliberately limited
    to 100–2,000 eV; it is not guaranteed to contain every possible PXR line.
    """)
    return


@app.cell
def _():
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import CxrReflections, FixedSampling, Numerics
    from pyrite.geometry import Footprint, Slab
    from pyrite.radiation import directional_source

    beam = Beam(
        energy_keV=30.0,
        transverse=PointSpot(),
        longitudinal=PointBunch(),
        bunch_charge_pc=1.0,
        rep_rate_hz=50_000.0,
    )
    target = Slab(
        "silicon",
        thickness_ang=1_000.0,
        tilt_deg=30.0,
        footprint=Footprint(5.0, 5.0),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=512, brem_electrons=512),
        cxr_reflections=CxrReflections(n_families=1),
    )
    return beam, directional_source, numerics, target


@app.cell
def _(beam, directional_source, np, numerics, target):
    angles_deg = np.linspace(-180.0, 180.0, 48, endpoint=False)
    energies_eV = np.linspace(100.0, 2_000.0, 257)
    source = directional_source(
        beam,
        target,
        energy_eV=energies_eV,
        angle_deg=angles_deg,
        numerics=numerics,
        direction_batch=12,
        seed=1,
    )
    assert source.density.shape == (angles_deg.size, energies_eV.size)
    return (source,)


@app.cell
def _(mo, np, source):
    summary = {
        "source_component": source.component,
        "density_units": source.units,
        "angles": int(source.angle_deg.size),
        "energy_nodes": int(source.energy_eV.size),
        "supported_directions": int(np.count_nonzero(source.supported)),
        "mean_current_nA": source.current_na,
        "kinematic_line_ceiling_eV": source.provenance["line_window_ceiling_eV"],
    }
    mo.json(summary)
    return


@app.cell
def _(np, plt, source):
    from matplotlib.colors import LogNorm

    # Unsupported directions are NaN, never zero. A finite slab typically has none.
    _density = np.where(source.supported[:, None], source.density, np.nan)
    _positive = _density[np.isfinite(_density) & (_density > 0)]
    if _positive.size == 0:
        raise ValueError("no positive source values; inspect sampling and photon-energy window")
    _high = float(np.max(_positive))
    _low = _high * 1e-5
    _display = np.ma.masked_invalid(np.where(_density > 0, _density, np.nan))

    fig_map, (_map_ax, _cut_ax) = plt.subplots(2, 1, figsize=(7.4, 6.5), layout="constrained")
    _mesh = _map_ax.pcolormesh(
        source.angle_deg,
        source.energy_eV / 1_000,
        _display.T,
        norm=LogNorm(vmin=_low, vmax=_high),
        shading="auto",
        rasterized=True,
    )
    fig_map.colorbar(_mesh, ax=_map_ax, label="Photons / (electron · eV · sr)")
    _map_ax.set(
        title="30 keV electrons → 100 nm Si: escaped PXR/CBS",
        xlabel="Signed lab angle (°)",
        ylabel="Photon energy (keV)",
    )

    _rate = source.rate()  # photons / (s · eV · sr), no angular acceptance
    for _direction in (0.0, 90.0, 135.0):
        _i = int(np.argmin(np.abs(source.angle_deg - _direction)))
        if source.supported[_i]:
            _cut_ax.plot(source.energy_eV / 1_000, _rate[_i], label=f"{source.angle_deg[_i]:g}°")
    _cut_ax.set(
        xlabel="Photon energy (keV)",
        ylabel="Photons / (s · eV · sr)",
        yscale="log",
        xlim=(source.energy_eV[0] / 1_000, source.energy_eV[-1] / 1_000),
    )
    _cut_ax.legend(title="Observation direction")
    _cut_ax.grid(alpha=0.25)
    fig_map  # noqa: B018 - captured by marimo
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpret the map

    The upper panel is **per electron, per eV, per sr**; the lower panel
    multiplies by the physical electron rate using the configured **50 nA**
    mean current. Neither plot integrates solid angle, applies a detector
    response, or predicts measured photon counts. **PXR + CBS** includes their
    mutual interference, but not characteristic lines or bremsstrahlung.

    The angular grid omits the duplicate +180° endpoint. Masked/unsupported
    directions (if any) are not physical zeros. Modest Monte Carlo sampling,
    one reflection family, and a restricted photon window mean the figure is
    illustrative, **not convergence or validation evidence**.

    Try changing the target tilt or increasing histories, then refine the
    energy and angular grids independently. For interface restrictions, see the
    [directional-source reference](https://pyrite.readthedocs.io/en/latest/reference/directional-sources.html)
    and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html).
    """)
    return


@app.cell
def _(mo, np, pyrite, source):
    tutorial_provenance = {
        "tutorial": "05-angular-radiation-distributions",
        "pyrite_version": pyrite.__version__,
        "backend": source.provenance["backend"],
        "seed": source.case["seed"],
        "table_identity": dict(source.provenance["xsgen_tables"]),
        "source_identity": source.provenance["identity_digest"],
        "component": source.component,
        "current_nA": source.current_na,
        "energy_bounds_eV": [float(source.energy_eV[0]), float(source.energy_eV[-1])],
        "angles": int(source.angle_deg.size),
        "supported": int(np.count_nonzero(source.supported)),
        "sampled_cxr_histories": int(source.case["Ne"]),
    }
    mo.json(tutorial_provenance)
    return (tutorial_provenance,)


if __name__ == "__main__":
    app.run()
