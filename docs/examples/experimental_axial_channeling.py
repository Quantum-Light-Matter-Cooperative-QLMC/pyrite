"""Shared, bounded experimental axial diagnostic for the guide and notebook."""

import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from pyrite.experimental.quantum import AxialDiagnosticConfig, axial_diagnostic
from pyrite.quantum import gaussian


def calculate(depth_ang=500.0, waist_ang=0.5):
    """Evolve a pure Gaussian in a periodic Si[001] supercell on CPU."""
    config = AxialDiagnosticConfig(
        crystal="silicon",
        axis=(0, 0, 1),
        kinetic_energy_eV=5e6,
        cells=(2, 2),
        shape=(64, 64),
        depths_ang=tuple(np.linspace(0.0, depth_ang, 51)),
        step_ang=2.0,
        mode="coherent",
        backend="cpu",
    )
    entry = gaussian(waist_ang=waist_ang, dimension=2, center_ang=(1.0, 1.0))
    result = axial_diagnostic(config, entry=entry)
    waves = result.wavefunctions_per_ang
    overlap = np.abs(np.sum(waves[0].conj() * waves, axis=(1, 2))) ** 2
    overlap *= result.grid.cell_area_ang2**2
    return result, overlap


def compute(depth_ang=500.0, waist_ang=0.5):
    """Explicit regeneration only; no scattering, transport tables or GPU."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, overlap = calculate(depth_ang, waist_ang)
    snapshots = [0, len(result.depths_ang) // 2, len(result.depths_ang) - 1]
    return {
        "config": asdict(result.config),
        "source": {"recipe": "gaussian", "waist_ang": waist_ang, "center_ang": [1.0, 1.0]},
        "depth_ang": result.depths_ang,
        "snapshot_indices": snapshots,
        "density_per_ang2": result.density_per_ang2[snapshots],
        "x_ang": result.grid.x,
        "y_ang": result.grid.y,
        "extent_ang": result.grid.extent_ang,
        "cell_area_ang2": result.grid.cell_area_ang2,
        "sampled_norms": result.sampled_norms,
        "retained_norm": result.surviving,
        "initial_state_overlap": overlap,
        "units": {"density": "per incident electron / angstrom^2", "depth": "angstrom"},
        "experimental_model": result.experimental_model,
        "warnings": list(dict.fromkeys(str(item.message) for item in caught)),
    }


def plot(data, view="density"):
    """Render actual density snapshots or pure-state overlap and grid norm."""
    depth_nm = np.asarray(data["depth_ang"]) / 10
    if view == "evolution":
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.plot(depth_nm, data["retained_norm"], label="Retained grid norm")
        ax.plot(depth_nm, data["initial_state_overlap"], label="Initial-state overlap")
        ax.set(xlabel="Depth (nm)", ylabel="Probability", ylim=(0, 1.06), xlim=(0, depth_nm[-1]))
        ax.legend(fontsize=9)
        return fig
    if view != "density":
        raise ValueError("view must be density or evolution")
    density = np.asarray(data["density_per_ang2"])
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), layout="constrained", sharey=True)
    x, y = np.asarray(data["x_ang"]), np.asarray(data["y_ang"])
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = (-dx / 2, x[-1] + dx / 2, -dy / 2, y[-1] + dy / 2)
    for ax, values, index in zip(axes, density, data["snapshot_indices"], strict=True):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=float(density.max()),
            cmap="magma",
            interpolation="nearest",
        )
        ax.set(xlabel="x (Å)", title=f"z = {depth_nm[index]:g} nm")
    axes[0].set_ylabel("y (Å)")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Density (incident electron⁻¹ Å⁻²)")
    return fig


if __name__ == "__main__":
    output = compute()
    plot(output)
    plot(output, view="evolution")
    plt.show()
