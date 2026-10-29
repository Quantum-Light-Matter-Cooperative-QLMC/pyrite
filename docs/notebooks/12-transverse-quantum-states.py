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


@app.cell
def _():
    import os

    os.environ.setdefault("PYRITE_MC_BACKEND", "cpu")
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from pyrite.quantum import gaussian, hermite_gauss, laguerre_gauss, mixture, superpose

    return (
        gaussian,
        hermite_gauss,
        laguerre_gauss,
        mixture,
        mo,
        np,
        plt,
        superpose,
    )


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
    hermite_32 = hermite_gauss(3, 2, waist_ang=1.5)
    positive = laguerre_gauss(ell=1, waist_ang=1.5)
    negative = laguerre_gauss(ell=-1, waist_ang=1.5)
    positive_lg_32 = laguerre_gauss(ell=3, p=2, waist_ang=1.5)
    return (
        coordinates_ang,
        ground,
        hermite_32,
        negative,
        positive,
        positive_lg_32,
    )


@app.cell
def _(coordinates_ang, ground, hermite_32, np, plt, positive, positive_lg_32):

    fig_modes, _axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained", sharey=True)
    flattened_axes = [axis for _axis_pair in _axes for axis in _axis_pair]
    plot_tuple = zip(
        flattened_axes,
        (ground, hermite_32, positive, positive_lg_32),
        ("Gaussian", r"$\mathrm{HG}_{32}$", r"$\mathrm{LG}_0^1$", r"$\mathrm{LG}_2^3$"),
        strict=True,
    )
    for _ax, _state, _label in plot_tuple:
        _density = np.abs(_state.wavefunction(coordinates_ang, coordinates_ang)) ** 2
        _image = _ax.imshow(_density.T, origin="lower", extent=(-5, 5, -5, 5), cmap="magma", vmin=0)
        _ax.set(xlabel="x (Å)", title=_label)
        fig_modes.colorbar(_image, ax=_ax, shrink=0.7, label="Density (Å⁻²)")
    _axes[0][0].set_ylabel("y (Å)")
    _axes[1][0].set_ylabel("y (Å)")
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
        _ax.set(xlabel=r"$x\,\mathrm{(\AA)}$", title=_label)
    _axes[0].set_ylabel(r"$y\,\mathrm{(\AA)}$")
    fig_coherence.suptitle(r"Summing $\mathrm{LG_0^1}$ and $\mathrm{LG_0^{-1}}$")
    fig_coherence.colorbar(_image, ax=_axes, shrink=0.8, label=r"Density ($\mathrm{\AA^{-2}}$)")
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
