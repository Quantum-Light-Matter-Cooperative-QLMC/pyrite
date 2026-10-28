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

    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from pyrite.quantum import gaussian, hermite_gauss, laguerre_gauss, mixture, superpose

    return gaussian, hermite_gauss, laguerre_gauss, mixture, mo, np, plt, superpose


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Constructing transverse quantum states

    Compare Gaussian, Hermite–Gaussian and Laguerre–Gaussian states,
    then distinguish coherent interference from an incoherent mixture.
    Coordinates and the **1.5 Å** waist are transverse to the beam.
    """)
    return


@app.cell
def _(gaussian, hermite_gauss, laguerre_gauss, np):
    coordinates_ang = np.linspace(-5.0, 5.0, 101)
    ground = gaussian(waist_ang=1.5, dimension=2)
    hermite = hermite_gauss(1, 0, waist_ang=1.5)
    positive = laguerre_gauss(ell=1, waist_ang=1.5)
    negative = laguerre_gauss(ell=-1, waist_ang=1.5)
    return coordinates_ang, ground, hermite, negative, positive


@app.cell
def _(coordinates_ang, ground, hermite, np, plt, positive):
    fig_modes, _axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained", sharey=True)
    for _ax, _state, _label in zip(
        _axes, (ground, hermite, positive), ("Gaussian", "HG₁₀", "LG₀¹"), strict=True
    ):
        _density = np.abs(_state.wavefunction(coordinates_ang, coordinates_ang)) ** 2
        _image = _ax.imshow(_density.T, origin="lower", extent=(-5, 5, -5, 5), cmap="magma", vmin=0)
        _ax.set(xlabel="x (Å)", title=_label)
        fig_modes.colorbar(_image, ax=_ax, shrink=0.7, label="Density (Å⁻²)")
    _axes[0].set_ylabel("y (Å)")
    fig_modes  # noqa: B018 - displayed by marimo
    return


@app.cell
def _(mixture, negative, positive, superpose):
    coherent = superpose([(1.0, positive), (1.0j, negative)])
    incoherent = mixture([(1.0, positive), (1.0, negative)])
    return coherent, incoherent


@app.cell
def _(coherent, coordinates_ang, incoherent, np, plt):
    coherent_density = np.abs(coherent.wavefunction(coordinates_ang, coordinates_ang)) ** 2
    mixed_density = sum(
        _weight * np.abs(_state.wavefunction(coordinates_ang, coordinates_ang)) ** 2
        for _weight, _state in incoherent.components
    )
    fig_coherence, _axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained", sharey=True)
    for _ax, _density, _label in zip(
        _axes,
        (coherent_density, mixed_density),
        ("Coherent sum", "Incoherent mixture"),
        strict=True,
    ):
        _image = _ax.imshow(
            _density.T,
            origin="lower",
            extent=(-5, 5, -5, 5),
            cmap="magma",
            vmin=0,
            vmax=float(coherent_density.max()),
        )
        _ax.set(xlabel="x (Å)", title=_label)
    _axes[0].set_ylabel("y (Å)")
    fig_coherence.colorbar(_image, ax=_axes, shrink=0.8, label="Density (Å⁻²)")
    fig_coherence  # noqa: B018 - displayed by marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Coherent sums add amplitudes, including their relative phase; mixtures
    add weighted densities and have no single wavefunction phase.
    Each distribution integrates to one over the transverse plane.

    Try changing the coherent relative phase from `1j` to `1.0`, or changing
    the mixture weights. These are standalone states; their construction
    does not imply admission to experimental channeling transport.
    """)
    return


if __name__ == "__main__":
    app.run()
