"""Bounded, cached solves and plotting for the experimental axial demo."""

from functools import lru_cache

import numpy as np

from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    AxialDiagnosticResult,
    axial_diagnostic,
)
from pyrite.montecarlo.channeling.states import MixedState, gaussian, laguerre_gauss, mixture

#: Interactive budget in grid samples x split steps x trajectories (<= ~1 min on one core).
APP_WORK_LIMIT = 1.1e9
#: Recorded depths kept per app solve.
APP_MAX_DEPTHS = 41


def axial_demo_depths(depth_ang: float, step_ang: float) -> tuple[float, ...]:
    """Return up to ``APP_MAX_DEPTHS`` evenly spaced step multiples from 0 to ``depth_ang``."""
    steps = round(depth_ang / step_ang)
    if steps < 1:
        raise ValueError("depth must span at least one step")
    index = np.unique(np.rint(np.linspace(0, steps, APP_MAX_DEPTHS)).astype(int))
    return tuple(float(i * step_ang) for i in index)


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
        or config.step_ang < 1
        or config.n_electrons > 32
        or len(config.depths_ang) > APP_MAX_DEPTHS
    ):
        raise ValueError(
            f"app limit: 48² grid, 3² cells, step >= 1 Å, 32 electrons, {APP_MAX_DEPTHS} depths"
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
    if config.mode == "mcwf":
        trajectories = config.n_electrons
    else:
        trajectories = len(entry.components) if isinstance(entry, MixedState) else 1
    steps = config.depths_ang[-1] / config.step_ang
    if config.shape[0] * config.shape[1] * steps * trajectories > APP_WORK_LIMIT:
        raise ValueError(
            f"app limit: grid samples x steps x trajectories <= {APP_WORK_LIMIT:.1e}; "
            "use a coarser step, fewer electrons, or remote compute"
        )
    return axial_diagnostic(config, entry=entry)


def _centered(values: np.ndarray, grid):
    """Roll an ``(x, y)`` periodic map so the projected origin sits mid-panel.

    Returns the ``imshow`` array and pixel-edge extent [Å]; sample ``(i, j)`` is
    at ``(i dx, j dy)``, so the origin string and default impact land centered.
    """
    shift = (grid.shape[0] // 2, grid.shape[1] // 2)
    step = [e / n for e, n in zip(grid.extent_ang, grid.shape, strict=True)]
    extent = []
    for n, k, d in zip(grid.shape, shift, step, strict=True):
        extent += [-(k + 0.5) * d, (n - k - 0.5) * d]
    return np.roll(values, shift, axis=(0, 1)).T, tuple(extent)


def axial_demo_figure(result: AxialDiagnosticResult, sample: int, conditional: bool = False):
    """Draw stored arrays only; phase is available solely for a coherent pure state."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, values, title, unit in (
        (axes[0], result.potential_eV, "Axial potential", "eV"),
        (
            axes[1],
            result.density(conditional=conditional)[sample],
            "Survivor-conditioned density" if conditional else "Density per incident electron",
            "Å⁻²",
        ),
    ):
        shown, extent = _centered(values, result.grid)
        mesh = ax.imshow(shown, origin="lower", extent=extent, aspect="equal")
        ax.set(title=title, xlabel="x (Å)", ylabel="y (Å)")
        fig.colorbar(mesh, ax=ax, label=unit)
    if result.wavefunctions_per_ang is not None:
        shown, extent = _centered(np.angle(result.wavefunctions_per_ang[sample]), result.grid)
        mesh = axes[2].imshow(
            shown,
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
    axes[3].plot(result.depths_ang / 1e4, result.surviving)
    axes[3].axvline(result.depths_ang[sample] / 1e4, ls="--", color="0.5")
    axes[3].set(
        title="MCWF grid survival" if result.config.mode == "mcwf" else "Coherent norm",
        xlabel="Depth (µm)",
        ylabel="Incident-electron fraction",
        ylim=(0, 1.05),
    )
    return fig
