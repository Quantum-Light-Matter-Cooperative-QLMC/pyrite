"""Experimental diagnostic assembly of the existing axial solvers and bounded ensemble recording."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from pyrite.materials.channeling.axial import AxialProfile, axial_profile

from .axial_dissipator import axial_dissipator
from .axial_mcwf import AxialMCWF
from .metrics import RECORDING_BUDGET_BYTES, AxialMetricsObservable, EnsembleMetrics
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
    backend: Literal["cpu", "cuda"] = "cpu"
    # Opt-in orthonormal FFT-supercell spectral measurements. Electronic jumps
    # change the carrier sector, so choose energy bins instead of state indices.
    spectral_states: int = 0
    spectral_representation: Literal["states", "energy_bins"] = "energy_bins"
    spectral_energy_bins: int = 8
    spectral_cutoff_eV: float | None = None
    record_coherence: bool = False
    coherence_reference_index: tuple[int, int] | None = None
    record_current: bool = False
    record_current_divergence: bool = False
    # (x_start, y_start, width_pixels, height_pixels), with periodic wrap.
    current_contour: tuple[int, int, int, int] | None = None


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
    from .axial_flux import FluxContour, samples_per_cell

    samples_per_cell(config.cells, config.shape)
    if (
        config.record_current_divergence or config.current_contour is not None
    ) and not config.record_current:
        raise ValueError("enable record_current to record divergence or contour flux")
    if config.current_contour is not None:
        bounds = config.current_contour
        if not isinstance(bounds, tuple) or len(bounds) != 4:
            raise ValueError("current_contour needs (start_x, start_y, width, height)")
        FluxContour(bounds[:2], bounds[2:]).validate(config.shape)
    if not isinstance(config.spectral_states, Integral) or not 0 <= config.spectral_states <= 256:
        raise ValueError("spectral_states must be an integer between 0 and 256")
    if config.spectral_representation not in ("states", "energy_bins"):
        raise ValueError("unknown spectral representation")
    if (
        not isinstance(config.spectral_energy_bins, Integral)
        or not 1 <= config.spectral_energy_bins <= 64
    ):
        raise ValueError("spectral_energy_bins must be between 1 and 64")
    if config.spectral_cutoff_eV is not None and not np.isfinite(config.spectral_cutoff_eV):
        raise ValueError("spectral cutoff must be finite")
    if (
        config.spectral_states
        and config.spectral_representation == "states"
        and config.mode == "mcwf"
        and "electronic" in config.mechanisms
    ):
        raise ValueError("electronic jumps change the carrier sector: use energy_bins")


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
    ``observables`` holds the optional caller observable, shape ``(depth, k)``,
    weighted like ``density_per_ang2`` (per incident electron).
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
    observables: np.ndarray | None = None
    metrics: EnsembleMetrics | None = None
    numerical_sink: np.ndarray | None = None
    jump_history: dict[str, np.ndarray] | None = None
    experimental_model: str = "experimental axial diagnostic; existing #508 solvers"
    spectral_populations: np.ndarray | None = None
    spectral_energies_eV: np.ndarray | None = None
    spectral_labels: tuple[str, ...] | None = None
    spectral_represented: np.ndarray | None = None
    spectral_omitted: np.ndarray | None = None
    spectral_barrier_eV: float | None = None
    coherence_reference_index: tuple[int, int] | None = None
    coherence_numerator_per_ang2: np.ndarray | None = None
    coherence_second_moment: np.ndarray | None = None
    # The physical Bloch carrier is included. Units are per depth.
    transverse_current_per_ang2: np.ndarray | None = None  # (depth, component, x, y)
    transverse_divergence_per_ang3: np.ndarray | None = None
    contour_flux_per_ang: np.ndarray | None = None  # outward (-x,+x,-y,+y)

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


def _coherent(config, grid, potential, components, norms, observe):
    """Weighted sum of ``observe`` over coherent component histories, and the last history."""
    solver = SplitStepPropagator(
        grid, config.kinetic_energy_eV, potential_eV=potential, tilt_rad=config.tilt_rad
    )
    k = momentum_inv_ang(config.kinetic_energy_eV)
    tilt = (k * config.tilt_rad[0], k * config.tilt_rad[1])
    stacked = 0.0
    history = None
    for (weight, phi), norm in zip(components, norms, strict=True):
        history = np.stack(
            [
                state
                for _, state in solver.trajectory(
                    phi / np.sqrt(norm), config.depths_ang, step_ang=config.step_ang
                )
            ]
        )
        stacked = stacked + weight * np.stack([observe(state, tilt) for state in history])
    return stacked, history


class _DiagnosticRecording:
    """Compose bounded recorded observables and declare device resource needs."""

    def __init__(
        self,
        grid,
        potential,
        config,
        projection,
        observable,
        record_metrics,
        spectral=None,
        coherence=None,
        current=None,
    ):
        self.metrics = (
            AxialMetricsObservable(
                grid, potential, config.kinetic_energy_eV, len(config.depths_ang)
            )
            if record_metrics
            else None
        )
        self.spectral = spectral
        self.coherence = coherence
        self.current = current
        self.current_width = 0 if current is None else current.recording_width
        self.spectral_width = 0 if spectral is None else spectral.recording_width
        self.coherence_width = 0 if coherence is None else coherence.recording_width
        self.observers = [
            f
            for f in (projection, observable, spectral, coherence, current, self.metrics)
            if f is not None
        ]
        self.cuda_compatible = all(getattr(f, "cuda_compatible", False) for f in self.observers)
        self.width = 0 if self.metrics is None else self.metrics.width
        extras = (
            (0 if projection is None else len(projection.ells))
            + (0 if observable is None else getattr(observable, "recording_width", 4096))
            + self.spectral_width
            + self.coherence_width
            + self.current_width
        )
        self.recording_bytes = (
            len(config.depths_ang) * 16 * (int(np.prod(grid.shape)) + self.width + extras)
        )
        if (
            record_metrics or spectral is not None or coherence is not None or current is not None
        ) and (self.recording_bytes > RECORDING_BUDGET_BYTES):
            raise ValueError("combined ensemble recording exceeds 128 MiB budget")
        self.scratch_bytes = 16 * int(np.prod(grid.shape)) * 100 + sum(
            int(getattr(f, "scratch_bytes", 0))
            for f in (spectral, coherence, current)
            if f is not None
        )
        self.electrons = config.n_electrons if config.mode == "mcwf" else None

    def __call__(self, phi, tilt):
        xp = np
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as xp
        parts = [xp.abs(phi).ravel() ** 2]
        parts.extend(xp.asarray(observer(phi, tilt)) for observer in self.observers)
        return xp.concatenate(parts)

    def finish(self, stacked, surviving, density):
        if self.metrics is None:
            return None
        return self.metrics.finish(
            stacked[:, -self.width :], surviving, density, electrons=self.electrons
        )


def _validate_backend(config):
    if config.backend not in ("cpu", "cuda") or (
        config.backend == "cuda" and config.mode != "mcwf"
    ):
        raise ValueError("CUDA diagnostic backend requires MCWF mode")


def axial_diagnostic(
    config: AxialDiagnosticConfig = AxialDiagnosticConfig(),
    *,
    entry: EntryState | None = None,
    observable: Callable[[np.ndarray, tuple[float, float]], np.ndarray] | None = None,
    record_metrics: bool = False,
) -> AxialDiagnosticResult:
    """Assemble the experimental axial diagnostic with optional stored ensemble metrics.

    ``entry`` uses the existing 2D pure/mixture interface; omitted means a
    Gaussian of waist 0.5 Å at the coordinate origin. No amplitudes are averaged
    across mixture components or MCWF electrons. This adds no physics and does
    not enable ``Physics(channeling_model="axial")``. ``observable(phi, tilt)``
    (``AxialMCWF`` signature, tilt in Å⁻¹) must be linear in ``|phi><phi|``;
    it is averaged with the same weights as the density. See the axial diagnostic
    guide for periodic boundaries, convergence and validation limits.

    ``record_metrics`` records full-grid empirical ensemble purity (probe SEM),
    position-basis squared coherence and conditional transverse H moments;
    undefined conditional values are NaN. ``backend="cuda"`` is MCWF-only;
    optional caller/OAM observables must support device arrays explicitly.

    Validation: cr-quantum-ensemble-metrics,
    cr-axial-realspace-propagation, cr-axial-realspace-dissipator
    (existing solvers; their ledger statuses are unchanged).
    """
    _validate(config)
    _validate_backend(config)
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
    from .axial_state_diagnostics import (
        AxialSpectralObservable,
        ReferenceCoherenceObservable,
        choose_reference,
    )

    initial_q = tuple(momentum_inv_ang(config.kinetic_energy_eV) * t for t in config.tilt_rad)
    spectral = None
    if config.spectral_states:
        edges = None
        if config.spectral_representation == "energy_bins":
            # Bin definitions are fixed for the entire solve; the actual
            # eigenvalues used per electron follow its instantaneous H(q).
            from .axial_state_diagnostics import transverse_spectrum

            sample = transverse_spectrum(
                grid, potential, config.kinetic_energy_eV, initial_q, config.spectral_states
            )
            lo, hi = float(sample.energies_eV[0]), float(sample.energies_eV[-1])
            pad = max((hi - lo) * 0.05, 1e-5)
            edges = np.linspace(lo - pad, hi + pad, config.spectral_energy_bins + 1)
            barrier = float(np.max(potential))
            if edges[0] < barrier < edges[-1]:
                edges = np.unique(np.r_[edges, barrier])
        spectral = AxialSpectralObservable(
            grid,
            potential,
            config.kinetic_energy_eV,
            initial_q,
            count=config.spectral_states,
            cutoff_eV=config.spectral_cutoff_eV,
            representation=config.spectral_representation,
            energy_edges_eV=edges,
        )
    coherence = None
    if config.record_coherence:
        ref = config.coherence_reference_index
        if ref is None:
            ref = choose_reference(grid, components)
        coherence = ReferenceCoherenceObservable(grid, ref)
    projection = None
    origin = None
    if config.oam_string is not None:
        if not isinstance(config.oam_string, Integral) or not 0 <= config.oam_string < len(
            profile.strings
        ):
            raise ValueError("oam_string must index a profile string")
        origin = profile.strings[config.oam_string].position_ang
        projection = CylindricalProjection(grid, origin, config.oam_radius_ang, 8)
    from .axial_flux import AxialFluxObservable, FluxContour

    current = None
    if config.record_current:
        bounds = config.current_contour
        current = AxialFluxObservable(
            grid,
            config.kinetic_energy_eV,
            divergence=config.record_current_divergence,
            contour=None if bounds is None else FluxContour(bounds[:2], bounds[2:]),
        )
    observe = _DiagnosticRecording(
        grid,
        potential,
        config,
        projection,
        observable,
        record_metrics,
        spectral,
        coherence,
        current,
    )
    numerical_sink = np.zeros(len(config.depths_ang))
    jump_history = None

    if config.mode == "coherent":
        stacked, waves = _coherent(config, grid, potential, components, norms, observe)
        waves = None if isinstance(entry, MixedState) else waves
        jumps = {}
    else:
        dissipator = axial_dissipator(
            profile,
            config.kinetic_energy_eV,
            config.cells,
            config.shape,
            mechanisms=config.mechanisms,
        )
        backend = {} if config.backend == "cpu" else {"backend": config.backend}
        mcwf = AxialMCWF(dissipator, config.kinetic_energy_eV, potential, **backend).run(
            entry,
            config.depths_ang,
            config.n_electrons,
            seed=config.seed,
            step_ang=config.step_ang,
            tilt_rad=config.tilt_rad,
            observable=observe,
        )
        stacked, waves, jumps = mcwf.observables, None, mcwf.jumps
        numerical_sink = (
            np.searchsorted(np.sort(mcwf.sink_depths_ang), config.depths_ang, side="right")
            / config.n_electrons
        )
        jump_history = getattr(mcwf, "jump_history", None)
    count = grid.shape[0] * grid.shape[1]
    stop = count if projection is None else count + len(projection.ells)
    # A complex caller observable promotes the stack; density and OAM are real.
    density = stacked[:, :count].real.reshape(-1, *grid.shape)
    oam = None if projection is None else stacked[:, count:stop].real
    extra_stop = stacked.shape[1] - (
        observe.width + observe.spectral_width + observe.coherence_width + observe.current_width
    )
    extra = None if observable is None else stacked[:, stop:extra_stop]
    spec_values = None
    spec_represented = None
    coh_numerator = None
    coh_second = None
    if spectral is not None:
        spec = stacked[:, extra_stop : extra_stop + observe.spectral_width].real
        spec_values = spec[:, :-1]
        spec_represented = spec[:, -1]
        extra_stop += observe.spectral_width
    if coherence is not None:
        n = count
        coh = stacked[:, extra_stop : extra_stop + observe.coherence_width]
        coh_numerator = coh[:, :n].reshape(-1, *grid.shape)
        coh_second = coh[:, n:].real.reshape(-1, *grid.shape)
        extra_stop += observe.coherence_width
    current_maps = None
    current_divergence = None
    current_sides = None
    if current is not None:
        current_maps, current_divergence, current_sides = current.unpack(
            stacked[:, extra_stop : extra_stop + observe.current_width]
        )
    if config.mode == "coherent":
        surviving = density.sum(axis=(1, 2)) * grid.cell_area_ang2
    else:
        surviving = mcwf.surviving
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
        extra,
        observe.finish(stacked, surviving, density),
        numerical_sink,
        jump_history,
        spectral_populations=spec_values,
        spectral_energies_eV=None if spectral is None else spectral.energies_eV,
        spectral_labels=None if spectral is None else tuple(spectral.labels),
        spectral_represented=spec_represented,
        spectral_omitted=None if spectral is None else np.maximum(surviving - spec_represented, 0),
        spectral_barrier_eV=None if spectral is None else spectral.barrier_eV,
        coherence_reference_index=(
            None
            if coherence is None
            else (int(coherence.reference_index[0]), int(coherence.reference_index[1]))
        ),
        coherence_numerator_per_ang2=coh_numerator,
        coherence_second_moment=coh_second,
        transverse_current_per_ang2=current_maps,
        transverse_divergence_per_ang3=current_divergence,
        contour_flux_per_ang=current_sides,
    )
