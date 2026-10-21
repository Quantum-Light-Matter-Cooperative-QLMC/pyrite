"""Bounded, cached solves and plotting for the experimental axial demo."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    AxialDiagnosticResult,
    axial_diagnostic,
)
from pyrite.montecarlo.channeling.axial_emission import (
    AxialLineCatalog,
    LevelDensity,
    axial_line_catalog,
    photon_angular_momentum,
)
from pyrite.montecarlo.channeling.dipole import dipole_rate_per_ang
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid
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
    levels = _level_observable(config)
    return axial_diagnostic(config, entry=entry, observable=levels, record_metrics=True)


@lru_cache(maxsize=4)
def _catalog(crystal, axis, kinetic_energy_eV, temperature_K) -> AxialLineCatalog | None:
    profile = axial_profile(crystal, axis, temperature_K)
    if len(profile.strings) != 1 or np.hypot(*profile.strings[0].position_ang) > 1e-9:
        return None  # on-column orbitals need one string per cell, at the origin
    catalog = axial_line_catalog(profile, kinetic_energy_eV)
    return catalog if catalog.lines else None


class _RolledLevelDensity:
    """``LevelDensity`` about string 0 after rolling the origin to the grid middle.

    The roll keeps the string's Wigner–Seitz cell contiguous on the periodic
    grid; the carrier phase it shifts is global, so ``rho`` is unchanged.
    """

    def __init__(self, catalog: AxialLineCatalog, grid: RealSpaceGrid) -> None:
        self.shift = (grid.shape[0] // 2, grid.shape[1] // 2)
        center = tuple(
            k * e / n for k, e, n in zip(self.shift, grid.extent_ang, grid.shape, strict=True)
        )
        self.levels = LevelDensity(catalog, grid, center)

    def __call__(self, phi, tilt_inv_ang=(0.0, 0.0)):
        return self.levels(np.roll(phi, self.shift, axis=(0, 1)), tilt_inv_ang)


def _level_observable(config: AxialDiagnosticConfig) -> _RolledLevelDensity | None:
    catalog = _catalog(config.crystal, config.axis, config.kinetic_energy_eV, config.temperature_K)
    if catalog is None:
        return None
    period = catalog.profile.period_ang
    grid = RealSpaceGrid((period[0] * config.cells[0], period[1] * config.cells[1]), config.shape)
    return _RolledLevelDensity(catalog, grid)


@dataclass(frozen=True)
class AxialRadiation:
    """Dipole lines of string 0 accumulated over the stored depths.

    ``polarization[d, j]`` is line ``j``'s ``T_ab`` (``a, b = +, -``) at depth
    ``d`` [Å²]; ``cumulative[d, j]`` integrates it over depth (trapezoid on the
    recorded depths) [Å³], so ``photons(d)`` is the line yield per incident
    electron emitted before depth ``d``.
    """

    catalog: AxialLineCatalog
    depths_ang: np.ndarray
    polarization: np.ndarray
    cumulative: np.ndarray

    def photons(self, sample: int) -> np.ndarray:
        """Photons per incident electron per line emitted before ``depths_ang[sample]``."""
        strength = np.trace(self.cumulative[sample], axis1=1, axis2=2).real
        deltas = np.array([line.delta_eV for line in self.catalog.lines])
        return dipole_rate_per_ang(deltas, np.sqrt(np.maximum(strength, 0)), self.catalog.gamma)

    def coherency(self, sample: int, line: int) -> np.ndarray:
        """Cartesian ``C_ab = sum d_a d_b^*`` of a line's accumulated emission [Å³]."""
        u = np.array([[1.0, 1.0], [1j, -1j]]) / np.sqrt(2.0)
        return u @ self.cumulative[sample, line] @ u.conj().T


def axial_radiation(result: AxialDiagnosticResult) -> AxialRadiation | None:
    """Return line emission from a demo solve, or ``None`` without a line catalog.

    Validation: cr-axial-helicity-emission
    """
    observable = _level_observable(result.config)
    if observable is None or result.observables is None:
        return None
    catalog = observable.levels.catalog
    rho = [observable.levels.unpack(values) for values in result.observables]
    polarization = np.array(
        [[line.polarization(levels[line.initial]) for line in catalog.lines] for levels in rho]
    )
    steps = np.diff(result.depths_ang)[:, None, None, None]
    increments = 0.5 * (polarization[1:] + polarization[:-1]) * steps
    cumulative = np.concatenate([np.zeros_like(polarization[:1]), np.cumsum(increments, axis=0)])
    return AxialRadiation(catalog, np.asarray(result.depths_ang), polarization, cumulative)


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
    elif result.coherence_numerator_per_ang2 is not None:
        from pyrite.montecarlo.channeling.axial_state_diagnostics import coherence_maps

        phase, _, _ = coherence_maps(
            result.coherence_numerator_per_ang2[sample],
            result.coherence_second_moment[sample],
            result.density_per_ang2[sample],
            result.coherence_reference_index,
            result.config.n_electrons if result.config.mode == "mcwf" else None,
        )
        shown, extent = _centered(phase, result.grid)
        mesh = axes[2].imshow(
            shown, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        axes[2].set(
            title="Ensemble relative phase (physical carrier)", xlabel="x (Å)", ylabel="y (Å)"
        )
        fig.colorbar(mesh, ax=axes[2], label="arg g¹(r,r₀) (rad)")
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


def axial_coherence_figure(result: AxialDiagnosticResult, sample: int):
    """Plot cached normalized spatial coherence; never reconstruct from density."""
    import matplotlib.pyplot as plt
    from pyrite.montecarlo.channeling.axial_state_diagnostics import coherence_maps

    if result.coherence_numerator_per_ang2 is None:
        raise ValueError("coherence was not recorded; rerun with record_coherence=True")
    phase, visibility, sem = coherence_maps(
        result.coherence_numerator_per_ang2[sample],
        result.coherence_second_moment[sample],
        result.density_per_ang2[sample],
        result.coherence_reference_index,
        result.config.n_electrons if result.config.mode == "mcwf" else None,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, title, cmap, bounds in (
        (axes[0], phase, "Relative phase arg g¹", "twilight", (-np.pi, np.pi)),
        (axes[1], visibility, "Coherence visibility |g¹|", "viridis", (0, 1)),
    ):
        shown, extent = _centered(values, result.grid)
        image = ax.imshow(
            shown,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=bounds[0],
            vmax=bounds[1],
        )
        ax.set(xlabel="x (Å)", ylabel="y (Å)", title=title)
        fig.colorbar(image, ax=ax)
    ref = result.coherence_reference_index
    purity = float(result.metrics.purity[sample]) if result.metrics is not None else float("nan")
    fig.suptitle(
        f"r₀ = pixel {ref}; survival={result.surviving[sample]:.3g}; "
        f"empirical purity={purity:.3g}; "
        + ("2σ MC sampling mask" if sem is not None else "deterministic ensemble")
    )
    return fig


def axial_spectral_population_data(result: AxialDiagnosticResult):
    """Return orthonormal spectral populations, independently of local line weights."""
    if result.spectral_populations is None:
        return None
    mode = result.config.spectral_representation
    basis = (
        "trajectory-sector transverse-energy bins"
        if mode == "energy_bins"
        else "fixed-carrier orthonormal transverse eigenspace projectors"
    )
    return (
        result.spectral_populations,
        result.spectral_labels,
        (
            f"{basis}; periodic FFT Hamiltonian, reference q={result.config.tilt_rad}; "
            f"barrier Umax={result.spectral_barrier_eV:.4g} eV; omitted survival may "
            "include uncomputed bound AND above-barrier states; neither is physical escape"
        ),
    )


def catalog_rate(catalog: AxialLineCatalog, line, polarization: np.ndarray) -> np.ndarray:
    """Photons per Å per incident electron from ``T_ab`` stacked over depth."""
    strength = np.trace(polarization, axis1=-2, axis2=-1).real
    return dipole_rate_per_ang(line.delta_eV, np.sqrt(np.maximum(strength, 0)), catalog.gamma)


def axial_radiation_figures(radiation: AxialRadiation, sample: int, line: int):
    """Helicity-split line spectrum and the selected line's far-field maps."""
    import matplotlib.pyplot as plt

    from pyrite.apps.channeling import far_field_figure

    catalog = radiation.catalog
    photons = radiation.photons(sample)
    energies_keV = np.array([catalog.photon_energy_eV(item) for item in catalog.lines]) / 1e3
    trace = np.trace(radiation.cumulative[sample], axis1=1, axis2=2).real
    plus = np.divide(
        radiation.cumulative[sample, :, 0, 0].real, trace, out=np.zeros_like(trace), where=trace > 0
    )
    width = 0.01 * energies_keV.max()
    spectrum, (ax, rate_ax) = plt.subplots(1, 2, figsize=(13, 3.6), constrained_layout=True)
    ax.bar(energies_keV, photons * plus, width, label="J_z = +1 (on-axis helicity +)")
    ax.bar(energies_keV, photons * (1 - plus), width, bottom=photons * plus, label="J_z = −1")
    ax.bar(energies_keV[line], photons[line], width, fill=False, edgecolor="k", lw=1.2)
    ax.set(
        title=f"Line yield before {radiation.depths_ang[sample] / 1e4:.4g} µm (string 0)",
        xlabel="Forward photon energy (keV)",
        ylabel="Photons per incident electron",
        xlim=(0, 1.1 * energies_keV.max()),
    )
    ax.legend(frameon=False, fontsize=8)
    depths_um = radiation.depths_ang / 1e4
    for j, item in enumerate(catalog.lines):
        rate = 1e4 * catalog_rate(catalog, item, radiation.polarization[:, j])
        rate_ax.plot(
            depths_um, rate, lw=2 if j == line else 1, label=f"{item.initial}→{item.final}"
        )
    rate_ax.axvline(depths_um[sample], ls="--", color="0.5")
    rate_ax.set(
        title="Line emission rate vs depth",
        xlabel="Depth (µm)",
        ylabel="Photons per µm per incident electron",
        ylim=(0, None),
    )
    rate_ax.legend(frameon=False, fontsize=7, ncol=2)
    chosen = catalog.lines[line]
    title = (
        f"Line {chosen.initial}→{chosen.final}, {catalog.photon_energy_eV(chosen) / 1e3:.3g} keV "
        "forward"
    )
    if photons[line] > 1e-12:
        far = far_field_figure(catalog.gamma, radiation.coherency(sample, line), title)
        moments = photon_angular_momentum(
            radiation.cumulative[sample, line], catalog.gamma, 1.0 / catalog.gamma
        )
    else:
        far, moments = None, None
    return spectrum, far, moments


def axial_line_options(radiation: AxialRadiation) -> tuple[dict[str, int], str]:
    """Dropdown labels for every line and the label of the brightest at the last depth."""
    catalog = radiation.catalog
    labels = [
        f"{item.initial}→{item.final} · {catalog.photon_energy_eV(item) / 1e3:.3g} keV"
        for item in catalog.lines
    ]
    strongest = int(np.argmax(radiation.photons(len(radiation.depths_ang) - 1)))
    return dict(zip(labels, range(len(labels)), strict=True)), labels[strongest]


def axial_population_data(result: AxialDiagnosticResult):
    """Stored string-0 level weights, retaining degeneracy labels and residuals."""
    observable = _level_observable(result.config)
    if observable is None or result.observables is None:
        # A complete orthonormal finite-position basis remains available when
        # the string-local level catalog is unsupported.
        return (
            (
                result.density_per_ang2.reshape(len(result.depths_ang), -1)
                * result.grid.cell_area_ang2
            ),
            None,
            "finite position-grid basis",
        )
    levels = observable.levels.catalog.levels
    populations = np.array(
        [
            [np.trace(block).real for block in observable.levels.unpack(row)]
            for row in result.observables
        ]
    )
    labels = [
        f"L{i}: {level.energy_eV:.3g} eV (g={len(level.states)})" for i, level in enumerate(levels)
    ]
    orbitals = np.concatenate(observable.levels._orbitals).reshape(
        -1, int(np.prod(result.grid.shape))
    )
    gram = orbitals.conj() @ orbitals.T * result.grid.cell_area_ang2
    defect = np.linalg.norm(gram - np.eye(len(gram)), ord=2)
    return (
        populations,
        labels,
        (
            f"string-0 local projection weights; incomplete/nonorthogonal sampling, "
            f"Gram defect ||G-I||₂={defect:.3g}; signed residual is not loss"
        ),
    )
