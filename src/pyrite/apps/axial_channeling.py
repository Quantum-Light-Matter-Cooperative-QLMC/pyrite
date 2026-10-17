"""Bounded, cached solves and plotting for the experimental axial demo."""

from functools import lru_cache

import numpy as np

from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    AxialDiagnosticResult,
    axial_diagnostic,
)
from pyrite.montecarlo.channeling.states import gaussian, laguerre_gauss, mixture


@lru_cache(maxsize=4)
def solve_axial_demo(
    config: AxialDiagnosticConfig,
    preset: str = "Gaussian",
    waist_ang: float = 0.5,
    impact_ang: tuple[float, float] = (0.0, 0.0),
) -> AxialDiagnosticResult:
    """Reuse completed small solves; larger statistics belong on remote compute."""
    if (
        max(config.shape) > 48
        or max(config.cells) > 3
        or config.depths_ang[-1] > 200
        or config.step_ang < 1
        or config.n_electrons > 32
        or len(config.depths_ang) > 21
    ):
        raise ValueError(
            "app limit: 48² grid, 3² cells, 200 Å, step >= 1 Å, 32 electrons, 21 depths"
        )
    if not np.isfinite(waist_ang) or waist_ang <= 0 or not np.all(np.isfinite(impact_ang)):
        raise ValueError("waist must be positive and impact finite")
    if preset == "Gaussian":
        entry = gaussian(waist_ang=waist_ang, center_ang=impact_ang)
    elif preset in ("Vortex +1", "Vortex -1"):
        entry = laguerre_gauss(
            1 if preset == "Vortex +1" else -1, waist_ang=waist_ang, center_ang=impact_ang
        )
    elif preset == "Equal ±1 mixture":
        entry = mixture(
            [
                (0.5, laguerre_gauss(ell, waist_ang=waist_ang, center_ang=impact_ang))
                for ell in (-1, 1)
            ]
        )
    else:
        raise ValueError("unknown axial entry preset")
    return axial_diagnostic(config, entry=entry)


def axial_demo_figure(result: AxialDiagnosticResult, sample: int, conditional: bool = False):
    """Draw stored arrays only; phase is available solely for a coherent pure state."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    extent = (0, result.grid.extent_ang[0], 0, result.grid.extent_ang[1])
    for ax, values, title, unit in (
        (axes[0], result.potential_eV, "Axial potential", "eV"),
        (
            axes[1],
            result.density(conditional=conditional)[sample],
            "Survivor-conditioned density" if conditional else "Density per incident electron",
            "Å⁻²",
        ),
    ):
        mesh = ax.imshow(values.T, origin="lower", extent=extent, aspect="equal")
        ax.set(title=title, xlabel="x (Å)", ylabel="y (Å)")
        fig.colorbar(mesh, ax=ax, label=unit)
    if result.wavefunctions_per_ang is not None:
        mesh = axes[2].imshow(
            np.angle(result.wavefunctions_per_ang[sample]).T,
            origin="lower",
            extent=extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[2].set(title="Coherent envelope phase", xlabel="x (Å)", ylabel="y (Å)")
        fig.colorbar(mesh, ax=axes[2], label="rad; carrier tilt factored out")
    else:
        axes[2].text(0.5, 0.5, "Ensemble density\nNo unique phase", ha="center", va="center")
        axes[2].set_axis_off()
    axes[3].plot(result.depths_ang, result.surviving)
    axes[3].axvline(result.depths_ang[sample], ls="--", color="0.5")
    axes[3].set(
        title="MCWF grid survival" if result.config.mode == "mcwf" else "Coherent norm",
        xlabel="Depth (Å)",
        ylabel="Incident-electron fraction",
        ylim=(0, 1.05),
    )
    return fig
