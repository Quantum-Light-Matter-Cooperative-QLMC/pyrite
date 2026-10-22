"""Incoherent incident-energy quadrature for planar and axial channeling.

Each energy node is an *independent* quantum calculation.  Physical densities,
probabilities and photon intensities may be averaged; wavefunction amplitudes,
eigenbasis density matrices and energy-dependent transition catalogs may not.

The entry convention matches ``BeamSpec.energy_spread_frac``: the RMS kinetic
energy spread is ``sigma_E = E0 * energy_spread_frac``.  Gaussian quadrature
is truncated to a bounded, positive kinetic-energy interval and renormalized;
its omitted probability mass is recorded rather than silently discarded.

This module does not sample a different energy per MCWF electron (issue #627)
or model correlated energy--arrival-time phase space (issue #628).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import erf, sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnergyQuadrature:
    """Incident-normalized nodes and provenance of the energy distribution.

    ``weights`` sum to one over the *retained* interval.  For a Gaussian,
    ``excluded_probability`` is the parent normal distribution's mass outside
    that interval, including nonphysical energies at or below zero.  ``mean_eV``
    and ``rms_eV`` describe the discretized, renormalized distribution.
    """

    energies_eV: np.ndarray
    weights: np.ndarray
    method: str
    reference_eV: float
    input_sigma_eV: float
    integration_bounds_eV: tuple[float, float]
    excluded_probability: float
    quadrature_order: int

    @property
    def mean_eV(self) -> float:
        return float(np.dot(self.weights, self.energies_eV))

    @property
    def rms_eV(self) -> float:
        return float(np.sqrt(np.dot(self.weights, (self.energies_eV - self.mean_eV) ** 2)))

    @property
    def is_monoenergetic(self) -> bool:
        return self.energies_eV.size == 1


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


@dataclass(frozen=True)
class IncidentEnergyEnsemble:
    """Bounded energy ensemble with independent energy-specific solver runs.

    Use exactly one spread description: ``energy_spread_frac`` (RMS fraction
    of ``kinetic_energy_eV``), ``energy_spread_eV`` (absolute RMS), or explicit
    ``energy_nodes_eV`` and ``node_weights``. The zero-spread default executes
    exactly one calculation at the specified energy, without quadrature.

    ``order`` controls the Gauss--Legendre quadrature over the truncated
    Gaussian, while ``cutoff_sigma`` controls the truncation.  Increase both
    independently to assess convergence. ``max_nodes`` is a hard work bound.
    Explicit nodes are *not* implicitly sorted or resampled.
    """

    kinetic_energy_eV: float
    energy_spread_frac: float = 0.0
    energy_spread_eV: float | None = None
    energy_nodes_eV: Sequence[float] | None = None
    node_weights: Sequence[float] | None = None
    order: int = 17
    cutoff_sigma: float = 4.0
    max_nodes: int = 65

    def quadrature(self) -> EnergyQuadrature:
        e0 = float(self.kinetic_energy_eV)
        frac = float(self.energy_spread_frac)
        if not np.isfinite(e0) or e0 <= 0:
            raise ValueError("kinetic_energy_eV must be finite and positive")
        if not np.isfinite(frac) or frac < 0:
            raise ValueError("energy_spread_frac must be finite and nonnegative")
        if not isinstance(self.max_nodes, int) or not 1 <= self.max_nodes <= 4096:
            raise ValueError("max_nodes must be an integer from 1 to 4096")
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order must be a positive integer")
        if not np.isfinite(self.cutoff_sigma) or self.cutoff_sigma <= 0:
            raise ValueError("cutoff_sigma must be finite and positive")
        explicit = self.energy_nodes_eV is not None or self.node_weights is not None
        if explicit:
            if self.energy_nodes_eV is None or self.node_weights is None:
                raise ValueError("explicit energies require matching node_weights")
            if frac != 0 or self.energy_spread_eV is not None:
                raise ValueError("explicit energy nodes cannot be combined with an RMS spread")
            energy = np.asarray(self.energy_nodes_eV, dtype=float)
            weight = np.asarray(self.node_weights, dtype=float)
            if (
                energy.ndim != 1
                or weight.ndim != 1
                or energy.size == 0
                or energy.shape != weight.shape
                or energy.size > self.max_nodes
                or not np.all(np.isfinite(energy))
                or np.any(energy <= 0)
                or not np.all(np.isfinite(weight))
                or np.any(weight < 0)
                or not np.any(weight > 0)
            ):
                raise ValueError(
                    "energy nodes and nonnegative weights must be finite, aligned and bounded"
                )
            mask = weight > 0
            energy, weight = energy[mask], weight[mask]
            weight = weight / weight.sum()
            method = "explicit"
            sigma = 0.0
            excluded = 0.0
            bounds = (float(energy.min()), float(energy.max()))
        else:
            if self.energy_spread_eV is not None:
                if frac != 0:
                    raise ValueError("provide either RMS fraction or RMS energy, not both")
                sigma = float(self.energy_spread_eV)
            else:
                sigma = e0 * frac
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError("RMS energy spread must be finite and nonnegative")
            if sigma == 0:
                energy, weight = np.array([e0]), np.array([1.0])
                method, excluded, bounds = "mono", 0.0, (e0, e0)
            else:
                if self.order < 3 or self.order > self.max_nodes:
                    raise ValueError("Gaussian quadrature order must be 3..max_nodes")
                low = max(float(np.nextafter(0.0, 1.0)), e0 - self.cutoff_sigma * sigma)
                high = e0 + self.cutoff_sigma * sigma
                if not np.isfinite(high) or high <= low:
                    raise ValueError("Gaussian quadrature has invalid positive-energy bounds")
                x, w = np.polynomial.legendre.leggauss(self.order)
                energy = low + 0.5 * (x + 1.0) * (high - low)
                scaled = (energy - e0) / sigma
                # Quadrature includes the Gaussian PDF; normalization removes
                # the finite-interval Gaussian mass and constants in the PDF.
                weight = w * np.exp(-0.5 * scaled**2)
                weight = weight / weight.sum()
                interval_mass = _normal_cdf((high - e0) / sigma) - _normal_cdf((low - e0) / sigma)
                excluded = max(0.0, 1.0 - interval_mass)
                bounds, method = (low, high), "truncated_gaussian"
        # Keep the node representation immutable (the dataclass alone would
        # not prevent a caller from mutating shared arrays between solves).
        energy = np.array(energy, dtype=float, copy=True)
        weight = np.array(weight, dtype=float, copy=True)
        energy.setflags(write=False)
        weight.setflags(write=False)
        return EnergyQuadrature(energy, weight, method, e0, sigma, bounds, excluded, energy.size)


class _Accumulator:
    """Incremental shaped sum; does not retain all energy-node histories."""

    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}

    def add(self, weight: float, **values: np.ndarray | float | None) -> None:
        for name, value in values.items():
            if value is None:
                if name in self.values:
                    raise ValueError(f"{name} is unavailable at an energy node")
                continue
            arr = np.asarray(value)
            if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains nonfinite or nonnumeric values")
            if name not in self.values:
                self.values[name] = weight * np.asarray(arr, dtype=np.result_type(arr, float))
            else:
                if arr.shape != self.values[name].shape:
                    raise ValueError(f"{name} shape differs between incident energies")
                self.values[name] += weight * arr

    def add_energy_ordered_bands(self, weight: float, value: np.ndarray) -> None:
        """Accumulate only band indices present at every incident energy node."""
        arr = np.asarray(value)
        if arr.ndim < 1 or not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            raise ValueError("band_populations contains nonfinite or nonnumeric values")
        previous = self.values.get("band_populations")
        if previous is None:
            self.values["band_populations"] = weight * np.asarray(
                arr, dtype=np.result_type(arr, float)
            )
            return
        if arr.shape[:-1] != previous.shape[:-1]:
            raise ValueError("band_populations depth shape differs between incident energies")
        n_bands = min(arr.shape[-1], previous.shape[-1])
        self.values["band_populations"] = previous[..., :n_bands] + weight * arr[..., :n_bands]

    def get(self, name: str) -> np.ndarray | None:
        return self.values.get(name)

    def require(self, name: str) -> np.ndarray:
        value = self.get(name)
        if value is None:
            raise ValueError(f"{name} was not accumulated")
        return value


def _matching(name: str, first: np.ndarray, following: np.ndarray) -> None:
    if not np.array_equal(first, following):
        raise ValueError(f"{name} differs across energy nodes; use a shared physical grid")


def _planar_solver(factory: Callable[[float], Any], energy_eV: float) -> Any:
    solver = factory(float(energy_eV))
    if not np.isclose(float(solver.energy_eV), energy_eV, rtol=1e-12, atol=1e-8):
        raise ValueError("solver_factory must recompute a solver at each energy node")
    return solver


def planar_solver_factory(
    profile: Any,
    *,
    n_cells: int,
    max_order: int,
    mechanisms: Sequence[str] = ("tds", "electronic"),
    margin_eV: float | None = None,
) -> Callable[[float], Any]:
    """Create an energy-specific planar solver builder (not a cached solver).

    All Hamiltonians, electronic and TDS scattering kernels are rebuilt at
    each incident energy.  If a custom profile/dissipator is needed, provide a
    custom ``solver_factory(E_eV)`` to the ensemble functions instead.
    """
    from .density import DensityMatrixSolver
    from .operators import electronic_kernel, tds_kernel

    kinds = tuple(mechanisms)
    if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"tds", "electronic"}:
        raise ValueError("mechanisms must be unique and drawn from tds/electronic")

    def build(energy_eV: float) -> DensityMatrixSolver:
        terms = []
        if "tds" in kinds:
            terms.append(tds_kernel(profile, energy_eV, n_cells=n_cells, max_order=max_order))
        if "electronic" in kinds:
            terms.append(
                electronic_kernel(profile, energy_eV, n_cells=n_cells, max_order=max_order)
            )
        kernel = terms[0]
        for term in terms[1:]:
            kernel = kernel + term
        if margin_eV is None:
            return DensityMatrixSolver(profile, energy_eV, kernel)
        return DensityMatrixSolver(profile, energy_eV, kernel, margin_eV=margin_eV)

    return build


@dataclass(frozen=True)
class PlanarEnergyDiagnostic:
    """Ensemble-averaged, incident-normalized planar physical observables.

    ``band_populations`` refers to *energy-ordered band indices*, not one
    shared eigenstate; individual eigenbasis matrices are never averaged.
    ``per_node`` can retain their energy-specific results if requested.
    """

    quadrature: EnergyQuadrature
    z_ang: np.ndarray
    kinetic_energy_eV: np.ndarray
    trace: np.ndarray
    bound_population: np.ndarray
    band_populations: np.ndarray
    free_flux: np.ndarray
    overflow_flux: np.ndarray
    reprojection_flux: np.ndarray
    entry_flux: float
    spatial_x_ang: np.ndarray | None
    spatial_density_per_ang: np.ndarray | None
    per_node: tuple[Any, ...] | None
    plane_wave_density: np.ndarray

    @property
    def total_flux(self) -> np.ndarray:
        return self.entry_flux + self.free_flux + self.overflow_flux + self.reprojection_flux

    @property
    def balance_defect(self) -> np.ndarray:
        return self.trace + self.total_flux - 1.0


def planar_energy_diagnostic(
    ensemble: IncidentEnergyEnsemble,
    solver_factory: Callable[[float], Any],
    thickness_ang: float,
    dz_ang: float,
    *,
    retain_node_results: bool = False,
    **propagate_kwargs: Any,
) -> PlanarEnergyDiagnostic:
    """Run ``DensityMatrixSolver.propagate`` independently for each energy.

    A shared spatial grid (``spatial_x_ang``) allows density averaging in
    physical space; returned ``final_density`` matrices are deliberately *not*
    combined. ``initial`` must be recomputed per energy by callers where it
    refers to eigenbasis coordinates (or omitted for normal beam entry).
    """
    if propagate_kwargs.get("initial") is not None:
        raise ValueError(
            "energy-dependent eigenbasis initial matrices are incompatible; use normal entry"
        )
    nodes = ensemble.quadrature()
    acc = _Accumulator()
    first = None
    reference_basis = None
    kept = [] if retain_node_results else None
    for energy, weight in zip(nodes.energies_eV, nodes.weights, strict=True):
        solver = _planar_solver(solver_factory, float(energy))
        result = solver.propagate(thickness_ang, dz_ang, **propagate_kwargs)
        # Convert *each* final eigenbasis rho to the common Bloch plane-wave
        # basis before ensemble averaging coherences. Never sum rho_eigen.
        basis = (
            solver.n_cells,
            np.asarray(solver.orders),
            solver.profile.period_ang,
            np.asarray(solver.profile.potential_eV),
        )
        if reference_basis is None:
            reference_basis = basis
        else:
            if (
                basis[0] != reference_basis[0]
                or not np.array_equal(basis[1], reference_basis[1])
                or basis[2] != reference_basis[2]
                or not np.array_equal(basis[3], reference_basis[3])
            ):
                raise ValueError("planar coherence requires an identical physical plane-wave basis")
        vectors = solver.basis.vectors
        plane_rho = vectors @ result.final_density @ vectors.conj().swapaxes(-1, -2)
        if first is None:
            first = result
        else:
            _matching("depth", first.z_ang, result.z_ang)
            if (first.spatial_x_ang is None) != (result.spatial_x_ang is None):
                raise ValueError("spatial grid must be present at every energy")
            if first.spatial_x_ang is not None:
                _matching("spatial grid", first.spatial_x_ang, result.spatial_x_ang)
        acc.add(
            float(weight),
            kinetic_energy_eV=result.kinetic_energy_eV,
            trace=result.trace,
            bound_population=result.bound_population,
            free_flux=result.free_flux,
            overflow_flux=result.overflow_flux,
            reprojection_flux=result.reprojection_flux,
            entry_flux=result.entry_flux,
            spatial_density_per_ang=result.spatial_density_per_ang,
            plane_wave_density=plane_rho,
        )
        acc.add_energy_ordered_bands(float(weight), result.band_populations)
        if kept is not None:
            kept.append(result)
    assert first is not None
    return PlanarEnergyDiagnostic(
        nodes,
        np.asarray(first.z_ang),
        acc.require("kinetic_energy_eV"),
        acc.require("trace"),
        acc.require("bound_population"),
        acc.require("band_populations"),
        acc.require("free_flux"),
        acc.require("overflow_flux"),
        acc.require("reprojection_flux"),
        float(acc.require("entry_flux")),
        first.spatial_x_ang,
        acc.get("spatial_density_per_ang"),
        None if kept is None else tuple(kept),
        acc.require("plane_wave_density"),
    )


@dataclass(frozen=True)
class PlanarEnergySpectrum:
    """Averaged planar photon intensities on a common detector-energy grid.

    ``stokes`` is derived from the *intensity-weighted* Q and U numerator;
    per-node line catalogs and rates are retained only in ``per_node``.
    """

    quadrature: EnergyQuadrature
    photon_energy_eV: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    spectrum: np.ndarray
    capture_spectrum: np.ndarray
    stokes: np.ndarray
    bound_population: np.ndarray
    total_line_yield: float
    total_capture_yield: float
    per_node: tuple[Any, ...] | None


def planar_energy_spectrum(
    ensemble: IncidentEnergyEnsemble,
    solver_factory: Callable[[float], Any],
    thickness_ang: float,
    photon_energy_eV: np.ndarray,
    theta_rad: Any,
    phi_rad: Any = 0.0,
    *,
    retain_node_results: bool = False,
    **spectrum_kwargs: Any,
) -> PlanarEnergySpectrum:
    """Integrate energy-specific ``channeling_spectrum`` intensities, not amplitudes.

    Reject analytic ``energy_spread_eV`` broadening: it would count the same
    incident spread twice.  All other spectrum controls are passed through.
    Catalogs, depth-rate arrays and eigenbasis populations are not combined.
    """
    from .emission import channeling_spectrum

    analytic_spread = spectrum_kwargs.pop("energy_spread_eV", 0.0)
    if not np.isfinite(analytic_spread) or analytic_spread != 0:
        raise ValueError(
            "explicit energy quadrature is incompatible with analytic energy_spread_eV"
        )
    nodes = ensemble.quadrature()
    acc = _Accumulator()
    kept = [] if retain_node_results else None
    first = None
    for energy, weight in zip(nodes.energies_eV, nodes.weights, strict=True):
        result = channeling_spectrum(
            _planar_solver(solver_factory, float(energy)),
            thickness_ang,
            photon_energy_eV,
            theta_rad,
            phi_rad,
            energy_spread_eV=0.0,
            **spectrum_kwargs,
        )
        if first is None:
            first = result
        else:
            _matching("photon energy grid", first.photon_energy_eV, result.photon_energy_eV)
            _matching("detector theta grid", first.theta_rad, result.theta_rad)
            _matching("detector phi grid", first.phi_rad, result.phi_rad)
            _matching("depth", first.z_ang, result.z_ang)
        # Undefined Stokes at zero intensity must contribute zero numerator.
        stokes_numer = np.where(
            np.asarray(result.spectrum)[..., None] > 0,
            np.nan_to_num(result.stokes) * np.asarray(result.spectrum)[..., None],
            0.0,
        )
        acc.add(
            float(weight),
            spectrum=result.spectrum,
            capture_spectrum=result.capture_spectrum,
            stokes_numer=stokes_numer,
            bound_population=result.bound_population,
            line_yield=result.total_line_yield,
            capture_yield=result.total_capture_yield,
        )
        if kept is not None:
            kept.append(result)
    assert first is not None
    spectrum = acc.require("spectrum")
    stokes_numer = acc.require("stokes_numer")
    with np.errstate(divide="ignore", invalid="ignore"):
        stokes = np.divide(
            stokes_numer,
            spectrum[..., None],
            out=np.full_like(stokes_numer, np.nan),
            where=spectrum[..., None] > 0,
        )
    return PlanarEnergySpectrum(
        nodes,
        first.photon_energy_eV,
        first.theta_rad,
        first.phi_rad,
        spectrum,
        acc.require("capture_spectrum"),
        stokes,
        acc.require("bound_population"),
        float(acc.require("line_yield").item()),
        float(acc.require("capture_yield").item()),
        None if kept is None else tuple(kept),
    )


@dataclass(frozen=True)
class AxialEnergyDiagnostic:
    """Axial physical-space ensemble; no unique ensemble wavefunction or phase.

    The OAM populations refer to fixed geometric modes and a common disk;
    ``observables`` must be a physically fixed, linear-in-density observable.
    The per-energy jump counts and wavefunctions remain in ``per_node``.
    """

    quadrature: EnergyQuadrature
    depths_ang: np.ndarray
    grid: Any
    potential_eV: np.ndarray
    density_per_ang2: np.ndarray
    surviving: np.ndarray
    oam_ells: np.ndarray | None
    oam_populations: np.ndarray | None
    observables: np.ndarray | None
    wavefunctions_per_ang: np.ndarray | None
    coherence_pairs: tuple[tuple[int, int], ...]
    coherence_per_ang2: np.ndarray | None
    per_node: tuple[Any, ...] | None

    def density(self, *, conditional: bool = False) -> np.ndarray:
        if not conditional:
            return self.density_per_ang2
        return np.divide(
            self.density_per_ang2,
            self.surviving[:, None, None],
            out=np.zeros_like(self.density_per_ang2),
            where=self.surviving[:, None, None] > 0,
        )


def axial_energy_diagnostic(
    ensemble: IncidentEnergyEnsemble,
    config: Any,
    *,
    entry: Any = None,
    observable_factory: Callable[[float], Callable | None] | None = None,
    coherence_pairs: Sequence[tuple[int, int]] = (),
    retain_node_results: bool = False,
) -> AxialEnergyDiagnostic:
    """Run the experimental axial solver separately at each incident energy.

    For energy-dependent level-projection observables, provide an
    ``observable_factory(E_eV)`` which constructs the level projector at that
    energy.  Such arrays must describe a *common physical labeling* across
    nodes; projectors onto unrelated eigenstates cannot be meaningfully added.
    """
    from .axial_diagnostic import axial_diagnostic

    nodes = ensemble.quadrature()
    pairs = tuple(tuple(pair) for pair in coherence_pairs)
    n_pixels = config.shape[0] * config.shape[1]
    if len(pairs) > 128 or any(
        len(pair) != 2 or any(not isinstance(i, int) or not 0 <= i < n_pixels for i in pair)
        for pair in pairs
    ):
        raise ValueError("coherence_pairs must contain at most 128 valid flattened pixel pairs")
    # Same periodic crystal geometry for all energies. The carrier correction
    # turns envelope coherences into a physically common lab-frame observable.
    if pairs:
        from pyrite.materials.channeling.axial import axial_profile

        profile = axial_profile(config.crystal, config.axis, config.temperature_K)
        extent = (profile.period_ang[0] * config.cells[0], profile.period_ang[1] * config.cells[1])
        coords = [
            (
                i // config.shape[1] * extent[0] / config.shape[0],
                i % config.shape[1] * extent[1] / config.shape[1],
            )
            for i in range(n_pixels)
        ]
        dx = np.array([coords[i][0] - coords[j][0] for i, j in pairs])
        dy = np.array([coords[i][1] - coords[j][1] for i, j in pairs])
    acc = _Accumulator()
    first = None
    kept = [] if retain_node_results else None
    unique_wave = None
    for energy, weight in zip(nodes.energies_eV, nodes.weights, strict=True):
        user_observer = None if observable_factory is None else observable_factory(float(energy))
        if pairs:

            def observer(phi, tilt, user_observer=user_observer):
                flat = phi.ravel()
                g1 = np.asarray([flat[i] * np.conj(flat[j]) for i, j in pairs])
                g1 *= np.exp(1j * (tilt[0] * dx + tilt[1] * dy))
                if user_observer is None:
                    return g1
                return np.concatenate((g1, np.asarray(user_observer(phi, tilt)).ravel()))
        else:
            observer = user_observer
        result = axial_diagnostic(
            replace(config, kinetic_energy_eV=float(energy)),
            entry=entry,
            observable=observer,
        )
        if first is None:
            first = result
            if nodes.is_monoenergetic:
                unique_wave = result.wavefunctions_per_ang
        else:
            _matching("depth", first.depths_ang, result.depths_ang)
            _matching("OAM labels", np.asarray(first.oam_ells), np.asarray(result.oam_ells))
            _matching("axial potential", first.potential_eV, result.potential_eV)
            if (
                first.grid.shape != result.grid.shape
                or first.grid.extent_ang != result.grid.extent_ang
            ):
                raise ValueError("axial physical-space grids must agree across energy nodes")
        if (first.oam_populations is None) != (result.oam_populations is None):
            raise ValueError("OAM projection must be defined at every energy")
        if (first.observables is None) != (result.observables is None):
            raise ValueError("observable must be defined at every energy")
        if result.observables is None:
            if pairs or user_observer is not None:
                raise ValueError("observable output is missing at an energy node")
            coherence = extra = None
        else:
            coherence = result.observables[:, : len(pairs)] if pairs else None
            extra = (
                result.observables[:, len(pairs) :]
                if pairs and user_observer is not None
                else result.observables
                if not pairs
                else None
            )
        acc.add(
            float(weight),
            density_per_ang2=result.density_per_ang2,
            surviving=result.surviving,
            coherence_per_ang2=coherence,
            oam_populations=result.oam_populations,
            observables=extra,
        )
        if kept is not None:
            kept.append(result)
    assert first is not None
    return AxialEnergyDiagnostic(
        nodes,
        first.depths_ang,
        first.grid,
        first.potential_eV,
        acc.require("density_per_ang2"),
        acc.require("surviving"),
        first.oam_ells,
        acc.get("oam_populations"),
        acc.get("observables"),
        unique_wave,
        pairs,
        acc.get("coherence_per_ang2"),
        None if kept is None else tuple(kept),
    )


@dataclass(frozen=True)
class AxialEnergySpectrum:
    """Incoherently averaged axial emission per common photon-energy bin.

    The output is *photons per incident electron per eV*, not a coherent
    superposition of lines. With ``detector_sigma_eV=0`` a delta line is
    deposited wholly in its bin; nonzero sigma describes detector resolution,
    **not** an additional incident-energy spread.
    """

    quadrature: EnergyQuadrature
    photon_energy_edges_eV: np.ndarray
    photon_density_per_eV: np.ndarray
    helicity_plus_per_eV: np.ndarray
    helicity_minus_per_eV: np.ndarray
    per_node: tuple[Any, ...] | None

    @property
    def photon_yield_per_bin(self) -> np.ndarray:
        return self.photon_density_per_eV * np.diff(self.photon_energy_edges_eV)


def axial_energy_spectrum(
    ensemble: IncidentEnergyEnsemble,
    config: Any,
    photon_energy_edges_eV: Sequence[float],
    *,
    entry: Any = None,
    sample: int = -1,
    detector_sigma_eV: float = 0.0,
    retain_node_results: bool = False,
) -> AxialEnergySpectrum:
    """Compute energy-specific axial MCWF/coherent solutions and dipole lines.

    The line catalogs and photon energies are recomputed at each incident
    energy. The incoming energies are *not* selected per MCWF trajectory.
    Spectral output uses bin-integrated line yields (or detector-smeared
    yields); no planar analytic energy-spread convolution is applied.
    """
    from importlib import import_module

    from scipy.special import ndtr

    from .axial_diagnostic import axial_diagnostic

    app = import_module("pyrite.apps.axial_channeling")
    edges = np.asarray(photon_energy_edges_eV, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("photon_energy_edges_eV must be finite, increasing bin edges")
    if not np.isfinite(detector_sigma_eV) or detector_sigma_eV < 0:
        raise ValueError("detector_sigma_eV must be finite and nonnegative")
    if not isinstance(sample, int):
        raise ValueError("sample must be an integer depth index")
    n_depths = len(config.depths_ang)
    if not -n_depths <= sample < n_depths:
        raise ValueError("sample is outside the stored depth range")
    nodes = ensemble.quadrature()
    summed = np.zeros((3, len(edges) - 1), dtype=float)
    kept = [] if retain_node_results else None
    widths = np.diff(edges)
    for energy, weight in zip(nodes.energies_eV, nodes.weights, strict=True):
        specific = replace(config, kinetic_energy_eV=float(energy))
        diag = axial_diagnostic(specific, entry=entry, observable=app._level_observable(specific))
        radiation = app.axial_radiation(diag)
        if radiation is None:
            if kept is not None:
                kept.append((diag, None))
            continue
        photons = radiation.photons(sample)
        energies = np.asarray(
            [radiation.catalog.photon_energy_eV(line) for line in radiation.catalog.lines]
        )
        polarization = radiation.cumulative[sample]
        trace = np.trace(polarization, axis1=-2, axis2=-1).real
        plus = np.divide(
            polarization[:, 0, 0].real, trace, out=np.zeros_like(trace), where=trace > 0
        )
        minus = np.divide(
            polarization[:, 1, 1].real, trace, out=np.zeros_like(trace), where=trace > 0
        )
        yields = np.stack([photons, photons * plus, photons * minus])
        if detector_sigma_eV == 0:
            indices = np.searchsorted(edges, energies, side="right") - 1
            good = (indices >= 0) & (indices < widths.size)
            for i in range(3):
                np.add.at(summed[i], indices[good], float(weight) * yields[i, good])
        else:
            fractions = ndtr((edges[1:, None] - energies) / detector_sigma_eV) - ndtr(
                (edges[:-1, None] - energies) / detector_sigma_eV
            )
            summed += float(weight) * yields @ fractions.T
        if kept is not None:
            kept.append((diag, radiation))
    result = summed / widths[None, :]
    return AxialEnergySpectrum(
        nodes, edges, result[0], result[1], result[2], None if kept is None else tuple(kept)
    )


def check_quadrature_convergence(
    ensemble: IncidentEnergyEnsemble,
    evaluate: Callable[[IncidentEnergyEnsemble], np.ndarray],
    *,
    orders: Sequence[int] = (9, 17, 33),
    rtol: float = 1e-2,
    atol: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Re-run an observable at increasing quadrature orders until it converges.

    ``evaluate`` must return a common-grid physical observable (e.g. a photon
    spectrum or spatial density). The cutoff is held fixed: assess the omitted
    Gaussian tails separately by changing ``ensemble.cutoff_sigma``. Raises if
    the final order fails; no partially converged value is presented as final.
    """
    if not orders or any(not isinstance(n, int) or n < 3 or n > ensemble.max_nodes for n in orders):
        raise ValueError("orders must contain valid quadrature orders")
    if any(a >= b for a, b in zip(orders, orders[1:], strict=False)):
        raise ValueError("quadrature orders must be strictly increasing")
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("convergence tolerances must be finite and nonnegative")
    if ensemble.quadrature().method != "truncated_gaussian":
        raise ValueError("quadrature convergence is only meaningful for Gaussian ensembles")
    previous = None
    for n in orders:
        value = np.asarray(evaluate(replace(ensemble, order=n)))
        if not np.all(np.isfinite(value)):
            raise ValueError("evaluation returned nonfinite physical observables")
        if previous is not None:
            if value.shape != previous.shape:
                raise ValueError("evaluations must use the same physical grid")
            if np.allclose(value, previous, rtol=rtol, atol=atol):
                return value, n
        previous = value
    raise RuntimeError(f"energy quadrature not converged by order {orders[-1]}")
