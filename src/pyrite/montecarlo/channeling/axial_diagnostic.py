"""Experimental diagnostic assembly of the existing axial solvers; CPU only."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from pyrite.materials.channeling.axial import AxialProfile, axial_profile

from .axial_dissipator import axial_dissipator
from .axial_mcwf import AxialMCWF
from .realspace import RealSpaceGrid, SplitStepPropagator
from .states import EntryState, MixedState, gaussian, momentum_inv_ang
from .vortex import CylindricalProjection


@dataclass(frozen=True)
class AxialDiagnosticConfig:
    """Solver configuration; lengths in Å, kinetic energy in eV, tilt in rad.

    ``cells`` counts periodic projected crystal cells; ``shape`` is the total
    FFT grid. Depths must be increasing nonnegative multiples of ``step_ang``.
    MCWF uses the existing per-electron Philox streams keyed by seed/electron.
    ``oam_string`` optionally selects a profile string; the projection uses a
    disk of radius ``oam_radius_ang`` and reports modes -8 through 8.
    """

    crystal: str = "silicon"
    axis: tuple[int, int, int] = (0, 0, 1)
    kinetic_energy_eV: float = 5e6
    temperature_K: float = 293.15
    cells: tuple[int, int] = (1, 1)
    shape: tuple[int, int] = (32, 32)
    depths_ang: tuple[float, ...] = (0.0, 50.0, 100.0)
    step_ang: float = 5.0
    tilt_rad: tuple[float, float] = (0.0, 0.0)
    mode: Literal["coherent", "mcwf"] = "coherent"
    n_electrons: int = 16
    seed: int = 0
    mechanisms: tuple[str, ...] = ("tds", "electronic")
    oam_string: int | None = None
    oam_radius_ang: float = 0.5


def _validate(config: AxialDiagnosticConfig) -> None:
    for name in ("kinetic_energy_eV", "temperature_K", "step_ang", "oam_radius_ang"):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, minimum in (("cells", 1), ("shape", 2)):
        values = getattr(config, name)
        if len(values) != 2 or any(not isinstance(v, Integral) or v < minimum for v in values):
            raise ValueError(f"{name} needs two integers >= {minimum}")
    if (
        len(config.axis) != 3
        or any(not isinstance(v, Integral) for v in config.axis)
        or not any(config.axis)
    ):
        raise ValueError("axis needs three integer indices, not all zero")
    if len(config.tilt_rad) != 2 or not np.all(np.isfinite(config.tilt_rad)):
        raise ValueError("tilt_rad needs two finite angles")
    depths = np.asarray(config.depths_ang)
    if (
        depths.ndim != 1
        or not depths.size
        or not np.all(np.isfinite(depths))
        or np.any(depths < 0)
        or np.any(np.diff(depths) <= 0)
    ):
        raise ValueError("depths_ang must be nonempty, finite, increasing and nonnegative")
    if not np.allclose(
        depths / config.step_ang, np.rint(depths / config.step_ang), rtol=0, atol=1e-9
    ):
        raise ValueError("depths_ang must be multiples of step_ang")
    if config.mode not in ("coherent", "mcwf"):
        raise ValueError("mode must be coherent or mcwf")
    if not isinstance(config.n_electrons, Integral) or config.n_electrons < 1:
        raise ValueError("n_electrons must be a positive integer")
    if not isinstance(config.seed, Integral) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(config.mechanisms) - {"tds", "electronic"}:
        raise ValueError("mechanisms must contain only tds and electronic")


@dataclass(frozen=True)
class AxialDiagnosticResult:
    """Stored diagnostic, with incident-electron density in Å⁻².

    Integrating ``density_per_ang2`` over the periodic grid equals ``surviving``.
    Inputs are normalized on the periodic grid separately for each mixture
    component, matching AxialMCWF; ``sampled_norms`` records the pre-normalization
    norms and exposes finite-grid sampling effects. ``wavefunctions_per_ang``
    exists only for coherent pure states (carrier tilt factored out). Coherent
    survival is unitary norm, not scattering survival. ``oam_populations`` is
    per incident electron inside the selected disk, truncated to ``oam_ells``;
    it is neither a full-cell nor a survivor-conditioned OAM distribution.
    """

    config: AxialDiagnosticConfig
    entry: EntryState
    profile: AxialProfile
    grid: RealSpaceGrid
    depths_ang: np.ndarray
    potential_eV: np.ndarray
    density_per_ang2: np.ndarray
    surviving: np.ndarray
    sampled_norms: tuple[float, ...]
    wavefunctions_per_ang: np.ndarray | None
    jumps: dict[str, int]
    oam_ells: np.ndarray | None
    oam_populations: np.ndarray | None
    oam_origin_ang: tuple[float, float] | None
    experimental_model: str = "experimental axial diagnostic; existing #508 solvers"

    def density(self, *, conditional: bool = False) -> np.ndarray:
        """Return density [Å⁻²], optionally conditioned on survival (zero if empty)."""
        if not conditional:
            return self.density_per_ang2
        return np.divide(
            self.density_per_ang2,
            self.surviving[:, None, None],
            out=np.zeros_like(self.density_per_ang2),
            where=self.surviving[:, None, None] > 0,
        )


def axial_diagnostic(
    config: AxialDiagnosticConfig = AxialDiagnosticConfig(),
    *,
    entry: EntryState | None = None,
) -> AxialDiagnosticResult:
    """Assemble and run the existing experimental axial diagnostic on the CPU.

    ``entry`` uses the existing 2D pure/mixture interface; omitted means a
    Gaussian of waist 0.5 Å at the coordinate origin. No amplitudes are averaged
    across mixture components or MCWF electrons. This adds no physics and does
    not enable ``Physics(channeling_model="axial")``. See the axial diagnostic
    guide for periodic boundaries, convergence and validation limits.

    Validation: cr-axial-realspace-propagation, cr-axial-realspace-dissipator
    (existing solvers; their ledger statuses are unchanged).
    """
    _validate(config)
    entry = gaussian(waist_ang=0.5) if entry is None else entry
    if entry.dimension != 2:
        raise ValueError("entry must be a 2D state")
    profile = axial_profile(config.crystal, config.axis, config.temperature_K)
    grid = RealSpaceGrid(
        (profile.period_ang[0] * config.cells[0], profile.period_ang[1] * config.cells[1]),
        config.shape,
    )
    potential = profile.on_grid(config.cells, config.shape)
    components = grid.ensemble(entry)
    norms = tuple(grid.norm(phi) for _, phi in components)
    if any(not np.isfinite(norm) or norm <= 0 for norm in norms):
        raise ValueError("entry has no finite positive sampled norm on this grid")
    projection = None
    origin = None
    if config.oam_string is not None:
        if not isinstance(config.oam_string, Integral) or not 0 <= config.oam_string < len(
            profile.strings
        ):
            raise ValueError("oam_string must index a profile string")
        origin = profile.strings[config.oam_string].position_ang
        projection = CylindricalProjection(grid, origin, config.oam_radius_ang, 8)
    waves = None
    oam = None
    if config.mode == "coherent":
        solver = SplitStepPropagator(
            grid, config.kinetic_energy_eV, potential_eV=potential, tilt_rad=config.tilt_rad
        )
        density = np.zeros((len(config.depths_ang), *grid.shape))
        if projection is not None:
            oam = np.zeros((len(config.depths_ang), len(projection.ells)))
        tilt = tuple(momentum_inv_ang(config.kinetic_energy_eV) * t for t in config.tilt_rad)
        for (weight, phi), norm in zip(components, norms, strict=True):
            history = np.stack(
                [
                    state
                    for _, state in solver.trajectory(
                        phi / np.sqrt(norm), config.depths_ang, step_ang=config.step_ang
                    )
                ]
            )
            density += weight * np.abs(history) ** 2
            if projection is not None:
                oam += weight * np.stack([projection(state, tilt) for state in history])
            if not isinstance(entry, MixedState):
                waves = history
        surviving = density.sum(axis=(1, 2)) * grid.cell_area_ang2
        jumps = {}
    else:
        dissipator = axial_dissipator(
            profile,
            config.kinetic_energy_eV,
            config.cells,
            config.shape,
            mechanisms=config.mechanisms,
        )

        def observe(phi, tilt):
            values = np.abs(phi).ravel() ** 2
            return values if projection is None else np.concatenate((values, projection(phi, tilt)))

        mcwf = AxialMCWF(dissipator, config.kinetic_energy_eV, potential).run(
            entry,
            config.depths_ang,
            config.n_electrons,
            seed=config.seed,
            step_ang=config.step_ang,
            tilt_rad=config.tilt_rad,
            observable=observe,
        )
        count = grid.shape[0] * grid.shape[1]
        density = mcwf.observables[:, :count].reshape(-1, *grid.shape)
        if projection is not None:
            oam = mcwf.observables[:, count:]
        surviving, jumps = mcwf.surviving, mcwf.jumps
    return AxialDiagnosticResult(
        config,
        entry,
        profile,
        grid,
        np.asarray(config.depths_ang),
        potential,
        density,
        surviving,
        norms,
        waves,
        jumps,
        None if projection is None else projection.ells,
        oam,
        origin,
    )
