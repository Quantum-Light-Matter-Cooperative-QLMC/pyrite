"""Energy-resolved *individual* axial MCWF trajectories (issue #627).

An electron gets a fixed entrance kinetic energy; the energy-specific
Hamiltonian, dissipator, carrier tilt and (optional) line catalog are rebuilt
at that energy. No energy loss at jumps, amplitude averaging or energy-time
phase model is implied. This is a bounded, sequential CPU reference backend;
it deliberately does not claim CUDA parity before a CUDA MCWF backend exists.

Counter-addressed Philox streams ensure the same electron has the same
energy, entrance random stream and MCWF history regardless of chunk admission.
Observables are reduced in increasing global electron ID order, including
removed particles as zero. A single electron at a time bounds temporary history
storage and avoids summing energy-dependent Hilbert-space representations.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any, Literal

import numpy as np
from scipy.special import ndtr, ndtri

from .energy_ensemble import IncidentEnergyEnsemble
from .mcwf import mcwf_stream_key

# Stable, disjoint namespaces for entrance-energy and transverse/trajectory RNG.
_ENERGY_TAG = 0x6270
_ENTRY_TAG = 0x6271
_TRAJECTORY_TAG = 0x6272


def _rng(seed: int, electron_id: int, tag: int) -> np.random.Generator:
    """Independent counter-addressed Philox stream, insensitive to chunking."""
    key = int(mcwf_stream_key(int(seed) ^ int(tag), int(electron_id)))
    return np.random.Generator(np.random.Philox(key=key))


def sample_electron_energies(
    distribution: IncidentEnergyEnsemble,
    n_electrons: int,
    *,
    seed: int = 0,
    sampling: Literal["binned", "continuous"] = "binned",
) -> np.ndarray:
    """Draw fixed entrance kinetic energies with stream keyed by global ID.

    ``binned`` draws exactly from the weighted #626 nodes. ``continuous``
    samples the *truncated* Gaussian by inverse CDF, or the explicit discrete
    distribution as appropriate. Monoenergetic energy is returned exactly.
    Both modes are deterministic for a given seed and global electron count.
    """
    if not isinstance(n_electrons, int) or not 1 <= n_electrons <= 100_000:
        raise ValueError("n_electrons must be between 1 and 100000")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    if sampling not in ("binned", "continuous"):
        raise ValueError("sampling must be 'binned' or 'continuous'")
    q = distribution.quadrature()
    if q.is_monoenergetic:
        return np.full(n_electrons, float(q.energies_eV[0]))
    energies = np.empty(n_electrons)
    cumulative = np.cumsum(q.weights)
    for i in range(n_electrons):
        u = float(_rng(seed, i, _ENERGY_TAG).random())
        if sampling == "binned" or q.method == "explicit":
            energies[i] = q.energies_eV[min(np.searchsorted(cumulative, u), len(cumulative) - 1)]
        else:
            lo, hi = q.integration_bounds_eV
            sigma = q.input_sigma_eV
            low_cdf = ndtr((lo - q.reference_eV) / sigma)
            high_cdf = ndtr((hi - q.reference_eV) / sigma)
            energies[i] = q.reference_eV + sigma * ndtri(low_cdf + u * (high_cdf - low_cdf))
    energies.setflags(write=False)
    return energies


@dataclass(frozen=True)
class AxialEnergyMCWFOptions:
    """Bounded execution contract; ``chunk_size`` changes admission, not streams.

    ``work_limit`` bounds ``n_electrons * pixels * split_steps``; the
    ``storage_limit_bytes`` bounds returned arrays and per-electron energies,
    not temporary solver allocations (which depend on the dissipator tables).
    ``max_cached_solvers`` limits resident energy-dependent dissipators.
    """

    n_electrons: int = 64
    seed: int = 0
    sampling: Literal["binned", "continuous"] = "binned"
    chunk_size: int = 32
    max_bins: int = 32
    max_cached_solvers: int = 2
    max_distinct_energies: int = 2048
    checkpoint_every: int = 0
    work_limit: int = 100_000_000
    storage_limit_bytes: int = 128 * 1024 * 1024
    backend: Literal["cpu", "cuda"] = "cpu"


def _check_options(options: AxialEnergyMCWFOptions, config: Any) -> None:
    if options.backend != "cpu":
        raise NotImplementedError(
            "energy-resolved CUDA MCWF requires a production CUDA axial solver; "
            "CPU-only reference implementation does not claim CPU/CUDA parity"
        )
    for name, lo, hi in (
        ("n_electrons", 1, 100_000),
        ("seed", 0, 2**63 - 1),
        ("chunk_size", 1, 10_000),
        ("max_bins", 1, 256),
        ("max_cached_solvers", 1, 8),
        ("max_distinct_energies", 1, 100_000),
        ("work_limit", 1, 10**14),
        ("storage_limit_bytes", 1024, 2**40),
        ("checkpoint_every", 0, 100_000),
    ):
        value = getattr(options, name)
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if options.sampling not in ("binned", "continuous"):
        raise ValueError("invalid sampling method")
    if config.mode != "mcwf":
        raise ValueError("axial energy-resolved trajectories require mode='mcwf'")
    if (
        len(config.depths_ang) < 1
        or config.step_ang <= 0
        or not np.all(np.isfinite(config.depths_ang))
        or min(config.depths_ang) < 0
        or np.any(np.diff(config.depths_ang) <= 0)
    ):
        raise ValueError("depths must be finite, nonnegative and increasing")
    ratio = np.asarray(config.depths_ang) / config.step_ang
    if not np.allclose(ratio, np.rint(ratio), rtol=0, atol=1e-9):
        raise ValueError("depths must be multiples of step_ang")
    if any(d < 2 for d in config.shape) or any(d < 1 for d in config.cells):
        raise ValueError("invalid axial grid")
    work = options.n_electrons * prod(config.shape) * int(np.max(np.rint(ratio)))
    if work > options.work_limit:
        raise ValueError(f"work estimate {work} exceeds work_limit {options.work_limit}")


def _energy_bins(
    energies: np.ndarray,
    distribution: IncidentEnergyEnsemble,
    options: AxialEnergyMCWFOptions,
    edges: Sequence[float] | None,
    discrete: bool,
    nominal_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin IDs, bin edges, display centers). No silently excluded electrons."""
    if edges is not None:
        bounds = np.asarray(edges, dtype=float)
        if (
            bounds.ndim != 1
            or bounds.size < 2
            or bounds.size > options.max_bins + 1
            or not np.all(np.isfinite(bounds))
            or np.any(np.diff(bounds) <= 0)
            or bounds[0] < 0
        ):
            raise ValueError("energy_bin_edges_eV must be positive, increasing and bounded")
        ids = np.searchsorted(bounds, energies, side="right") - 1
        ids = np.where(energies == bounds[-1], len(bounds) - 2, ids)
        if np.any((ids < 0) | (ids >= len(bounds) - 1)):
            raise ValueError("some sampled energies are outside energy_bin_edges_eV")
        centers = (bounds[:-1] + bounds[1:]) / 2
        return ids.astype(int), bounds, centers
    if discrete:
        if nominal_centers is None:
            centers, ids = np.unique(energies, return_inverse=True)
        else:
            centers = np.unique(nominal_centers)
            ids = np.searchsorted(centers, energies)
            if np.any(ids >= len(centers)) or not np.array_equal(centers[ids], energies):
                raise ValueError("binned energies must match the #626 quadrature nodes")
        if centers.size > options.max_bins:
            raise ValueError("number of distinct incident energies exceeds max_bins")
        if centers.size == 1:
            width = max(centers[0] * 1e-8, 1e-8)
            bounds = np.array([max(0.0, centers[0] - width), centers[0] + width])
        else:
            mid = 0.5 * (centers[1:] + centers[:-1])
            bounds = np.r_[
                max(0.0, centers[0] - (mid[0] - centers[0])),
                mid,
                centers[-1] + (centers[-1] - mid[-1]),
            ]
        return ids.astype(int), bounds, centers
    # Continuous Gaussian, equal-width bins over *the same truncated distribution*.
    low, high = distribution.quadrature().integration_bounds_eV
    if low == high:
        return _energy_bins(energies, distribution, options, None, True)
    bounds = np.linspace(low, high, options.max_bins + 1)
    ids = np.clip(np.searchsorted(bounds, energies, side="right") - 1, 0, options.max_bins - 1)
    return ids.astype(int), bounds, (bounds[:-1] + bounds[1:]) / 2


@dataclass(frozen=True)
class AxialEnergyMCWFCheckpoint:
    """Bounded, resumable *raw sums* after the first ``processed_electrons`` IDs.

    Snapshot arrays are independent read-only copies. To resume, supply the
    same configuration, entrance/observable factories and distribution.
    ``signature`` checks energy assignments, grid/depth settings and seed,
    but cannot fingerprint arbitrary Python callback implementations.
    """

    processed_electrons: int
    signature: str
    bin_counts_processed: np.ndarray
    density_sums: np.ndarray
    survival_sums: np.ndarray
    coherence_sums: np.ndarray | None
    oam_sums: np.ndarray | None
    observable_sums: np.ndarray | None
    photon_sums: np.ndarray | None
    jumps_by_bin: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class AxialEnergyMCWFResult:
    """Energy-conditioned means are per *assigned* incident electron (not survivor).

    The unconditional result is the count-weighted sum of conditional bins.
    Density is in Å^-2, coherence is the laboratory-frame reduced matrix
    element between fixed pixels, survival is relative to incident electrons.
    ``jumps_by_bin`` contains *counts*, not probabilities. The radiation
    histogram, if requested, is photons per incident electron per eV and is
    integrated through the final stored depth, including removed electrons.
    """

    energies_eV: np.ndarray
    energy_bin_edges_eV: np.ndarray
    energy_bin_centers_eV: np.ndarray
    bin_counts: np.ndarray
    depths_ang: np.ndarray
    grid: Any
    density_per_ang2: np.ndarray
    density_by_bin_per_ang2: np.ndarray
    surviving: np.ndarray
    surviving_by_bin: np.ndarray
    oam_ells: np.ndarray | None
    oam_populations: np.ndarray | None
    oam_populations_by_bin: np.ndarray | None
    coherence_pairs: tuple[tuple[int, int], ...]
    coherence_per_ang2: np.ndarray | None
    coherence_by_bin_per_ang2: np.ndarray | None
    observables: np.ndarray | None
    observables_by_bin: np.ndarray | None
    jumps: dict[str, int]
    jumps_by_bin: tuple[dict[str, int], ...]
    photon_energy_edges_eV: np.ndarray | None
    photon_density_per_eV: np.ndarray | None
    photon_density_by_bin_per_eV: np.ndarray | None
    helicity_plus_per_eV: np.ndarray | None
    helicity_minus_per_eV: np.ndarray | None
    excluded_probability: float
    provenance: dict[str, Any]

    def density(self, *, conditional_on_survival: bool = False) -> np.ndarray:
        if not conditional_on_survival:
            return self.density_per_ang2
        return np.divide(
            self.density_per_ang2,
            self.surviving[:, None, None],
            out=np.zeros_like(self.density_per_ang2),
            where=self.surviving[:, None, None] > 0,
        )


def _radiation_bins(
    values: np.ndarray,
    levels: Any,
    catalog: Any,
    depths: np.ndarray,
    edges: np.ndarray,
    sigma_eV: float,
) -> np.ndarray:
    """Integrate each energy-specific polarization and deposit line *intensity*."""
    from .dipole import dipole_rate_per_ang

    rho = [levels.unpack(row) for row in values]
    pol = np.asarray(
        [[line.polarization(level[line.initial]) for line in catalog.lines] for level in rho]
    )
    if pol.shape[1] == 0:
        return np.zeros((3, len(edges) - 1))
    increments = 0.5 * (pol[1:] + pol[:-1]) * np.diff(depths)[:, None, None, None]
    total = increments.sum(axis=0)
    strength = np.maximum(0.0, np.trace(total, axis1=-2, axis2=-1).real)
    delta = np.array([line.delta_eV for line in catalog.lines])
    photons = dipole_rate_per_ang(delta, np.sqrt(strength), catalog.gamma)
    plus = np.divide(total[:, 0, 0].real, strength, out=np.zeros_like(strength), where=strength > 0)
    minus = np.divide(
        total[:, 1, 1].real, strength, out=np.zeros_like(strength), where=strength > 0
    )
    intensities = np.stack((photons, photons * plus, photons * minus))
    photon_energy = np.array([catalog.photon_energy_eV(line) for line in catalog.lines])
    counts = np.zeros((3, edges.size - 1))
    if sigma_eV == 0:
        ids = np.searchsorted(edges, photon_energy, side="right") - 1
        good = (ids >= 0) & (ids < counts.shape[1])
        for channel in range(3):
            np.add.at(counts[channel], ids[good], intensities[channel, good])
    else:
        fractions = ndtr((edges[1:, None] - photon_energy) / sigma_eV) - ndtr(
            (edges[:-1, None] - photon_energy) / sigma_eV
        )
        counts += intensities @ fractions.T
    return counts / np.diff(edges)[None, :]


def axial_energy_mcwf(
    distribution: IncidentEnergyEnsemble,
    config: Any,
    *,
    options: AxialEnergyMCWFOptions = AxialEnergyMCWFOptions(),
    entry: Any = None,
    entry_factory: Callable[[float, int, np.random.Generator], tuple[Any, tuple[float, float]]]
    | None = None,
    electron_energies_eV: Sequence[float] | None = None,
    energy_bin_edges_eV: Sequence[float] | None = None,
    coherence_pairs: Sequence[tuple[int, int]] = (),
    observable_factory: Callable[[float], Callable | None] | None = None,
    photon_energy_edges_eV: Sequence[float] | None = None,
    detector_sigma_eV: float = 0.0,
    solver_factory: Callable[[float, Any, np.ndarray], Any] | None = None,
    radiation_factory: Callable[[float, Any], Any] | None = None,
    checkpoint_callback: Callable[[AxialEnergyMCWFCheckpoint], None] | None = None,
    resume_from: AxialEnergyMCWFCheckpoint | None = None,
) -> AxialEnergyMCWFResult:
    """Run energy-resolved MCWFs in a strictly bounded, deterministic stream.

    ``entry_factory(E, global_id, rng) -> (entry, tilt_rad)`` supplies correlated
    incident energy and transverse phase space. It must be deterministic from
    its arguments; the RNG is the electron's separate entrance Philox stream.
    The *observable_factory(E)* must return a fixed-shape, physically comparable
    density-linear observable. Custom eigenstate populations only have a clear
    interpretation within their *own* energy bin. ``solver_factory`` is an
    optional dependency-injection point for reference benchmarks; it must
    return an energy-matched MCWF solver with the shared grid. No hooks are
    installed in the underlying solver, preserving its native scattering RNG.

    ``chunk_size`` only controls admission iteration, not summation order;
    each MCWF run contains exactly one globally addressed electron. This
    intentional conservative CPU reference design is slower than vectorized
    trajectories but provides exact chunk-size independence. A dedicated CUDA
    implementation must preserve these addressing/reduction contracts.
    """
    from pyrite.materials.channeling.axial import axial_profile

    from .axial_dissipator import axial_dissipator
    from .axial_mcwf import AxialMCWF
    from .realspace import RealSpaceGrid
    from .states import gaussian

    _check_options(options, config)
    if options.checkpoint_every and checkpoint_callback is None:
        raise ValueError("checkpoint_every requires checkpoint_callback")
    nodes = distribution.quadrature()
    if electron_energies_eV is None:
        energies = sample_electron_energies(
            distribution, options.n_electrons, seed=options.seed, sampling=options.sampling
        )
        discrete = options.sampling == "binned" or nodes.method in ("mono", "explicit")
    else:
        energies = np.asarray(electron_energies_eV, dtype=float)
        if (
            energies.shape != (options.n_electrons,)
            or not np.all(np.isfinite(energies))
            or np.any(energies <= 0)
        ):
            raise ValueError("electron_energies_eV must be positive, finite, one per electron")
        energies = np.array(energies, copy=True)
        energies.setflags(write=False)
        discrete = True
    distinct_energies = np.unique(energies).size
    if distinct_energies > options.max_distinct_energies:
        raise ValueError("distinct electron energies exceed max_distinct_energies")
    ids, bounds, centers = _energy_bins(
        energies,
        distribution,
        options,
        energy_bin_edges_eV,
        discrete,
        nominal_centers=(
            nodes.energies_eV
            if electron_energies_eV is None and options.sampling == "binned"
            else None
        ),
    )
    counts = np.bincount(ids, minlength=len(centers))
    nb, nd = len(centers), len(config.depths_ang)
    pairs = tuple(tuple(p) for p in coherence_pairs)
    npix = prod(config.shape)
    if len(pairs) > 128 or any(
        len(p) != 2 or any(not isinstance(k, int) or not 0 <= k < npix for k in p) for p in pairs
    ):
        raise ValueError("coherence_pairs must contain <=128 valid flattened pixel pairs")
    photon_edges = None
    if photon_energy_edges_eV is not None:
        photon_edges = np.asarray(photon_energy_edges_eV, float)
        if (
            photon_edges.ndim != 1
            or not 2 <= len(photon_edges) <= 4097
            or not np.all(np.isfinite(photon_edges))
            or np.any(np.diff(photon_edges) <= 0)
        ):
            raise ValueError("photon_energy_edges_eV requires 2..4097 increasing finite edges")
    if not np.isfinite(detector_sigma_eV) or detector_sigma_eV < 0:
        raise ValueError("detector_sigma_eV must be finite and nonnegative")
    # Conservative bound *before* constructing expensive material/displacement kernels.
    nbytes = (nb + 1) * nd * npix * 8 + (nb + 1) * nd * (1 + len(pairs) + 17) * 16
    nbytes += energies.nbytes + ids.nbytes + bounds.nbytes
    if photon_edges is not None:
        nbytes += (nb + 1) * 3 * (len(photon_edges) - 1) * 8
    capacity_factor = 2 if options.checkpoint_every or resume_from is not None else 1
    if nbytes * capacity_factor > options.storage_limit_bytes:
        raise ValueError(f"required core reductions {nbytes} bytes exceed storage_limit_bytes")
    profile = axial_profile(config.crystal, config.axis, config.temperature_K)
    grid = RealSpaceGrid(
        (profile.period_ang[0] * config.cells[0], profile.period_ang[1] * config.cells[1]),
        config.shape,
    )
    potential = profile.on_grid(config.cells, config.shape)
    projection = None
    if config.oam_string is not None:
        from .vortex import CylindricalProjection

        if not isinstance(config.oam_string, int) or not 0 <= config.oam_string < len(
            profile.strings
        ):
            raise ValueError("oam_string must index a profile string")
        projection = CylindricalProjection(
            grid, profile.strings[config.oam_string].position_ang, config.oam_radius_ang, 8
        )
    x = np.arange(config.shape[0]) * grid.extent_ang[0] / config.shape[0]
    y = np.arange(config.shape[1]) * grid.extent_ang[1] / config.shape[1]
    dx = np.asarray([x[p[0] // config.shape[1]] - x[p[1] // config.shape[1]] for p in pairs])
    dy = np.asarray([y[p[0] % config.shape[1]] - y[p[1] % config.shape[1]] for p in pairs])
    density_bins = np.zeros((nb, nd, *config.shape))
    survival_bins = np.zeros((nb, nd))
    coh_bins = np.zeros((nb, nd, len(pairs)), complex) if pairs else None
    oam_bins = np.zeros((nb, nd, len(projection.ells))) if projection is not None else None
    additional_bins = None
    photons_bins = np.zeros((nb, 3, len(photon_edges) - 1)) if photon_edges is not None else None
    jumps_bins = [dict.fromkeys(("tds", "sink", "electronic", "absorbed"), 0) for _ in range(nb)]
    cached: OrderedDict[float, tuple[Any, Any, int]] = OrderedDict()
    sampled_entry = gaussian(waist_ang=0.5) if entry is None else entry
    depths = np.asarray(config.depths_ang, dtype=float)
    # One metadata signature for restart checking; callback identity cannot be
    # hashed and is explicitly the caller's responsibility.
    signature = sha256()
    for values in (
        energies,
        ids,
        bounds,
        depths,
        np.asarray(config.shape),
        np.asarray(config.cells),
        np.asarray(config.axis),
        np.asarray([options.seed, options.n_electrons]),
        np.asarray(
            [
                config.step_ang,
                config.temperature_K,
                config.kinetic_energy_eV,
                config.oam_radius_ang,
                detector_sigma_eV,
                *config.tilt_rad,
            ]
        ),
        np.asarray([config.oam_string if config.oam_string is not None else -1]),
        np.asarray(pairs, dtype=int).reshape(-1, 2),
        np.asarray([] if photon_edges is None else photon_edges),
        np.asarray([options.sampling, config.mode, config.crystal, *config.mechanisms]),
    ):
        array = np.asarray(values)
        signature.update(str(array.shape).encode())
        signature.update(
            array.tobytes() if array.dtype.kind not in "USO" else repr(array.tolist()).encode()
        )
    fingerprint = signature.hexdigest()
    next_id = 0
    if resume_from is not None:
        cp = resume_from
        if (
            not isinstance(cp, AxialEnergyMCWFCheckpoint)
            or cp.signature != fingerprint
            or not 0 <= cp.processed_electrons <= options.n_electrons
        ):
            raise ValueError("checkpoint has incompatible assignment/configuration signature")
        if not np.array_equal(
            cp.bin_counts_processed, np.bincount(ids[: cp.processed_electrons], minlength=nb)
        ):
            raise ValueError("checkpoint contains inconsistent processed bin counts")
        if (
            cp.density_sums.shape != density_bins.shape
            or cp.survival_sums.shape != survival_bins.shape
        ):
            raise ValueError("checkpoint reduction shapes differ")
        density_bins[:] = cp.density_sums
        survival_bins[:] = cp.survival_sums
        for target, source in (
            (coh_bins, cp.coherence_sums),
            (oam_bins, cp.oam_sums),
            (photons_bins, cp.photon_sums),
        ):
            if (target is None) != (source is None):
                raise ValueError("checkpoint optional reduction layout differs")
            if target is not None:
                assert source is not None
                if target.shape != source.shape:
                    raise ValueError("checkpoint reduction shape differs")
                target[:] = source
        if cp.observable_sums is not None:
            additional_bins = np.array(cp.observable_sums, copy=True)
        if len(cp.jumps_by_bin) != nb:
            raise ValueError("checkpoint jump bins differ")
        jumps_bins = [dict(b) for b in cp.jumps_by_bin]
        next_id = cp.processed_electrons

    def emit_checkpoint(processed: int) -> None:
        def freeze(value):
            if value is None:
                return None
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            return copy

        snapshot = AxialEnergyMCWFCheckpoint(
            processed,
            fingerprint,
            freeze(np.bincount(ids[:processed], minlength=nb)),
            freeze(density_bins),
            freeze(survival_bins),
            freeze(coh_bins),
            freeze(oam_bins),
            freeze(additional_bins),
            freeze(photons_bins),
            tuple(dict(j) for j in jumps_bins),
        )
        if checkpoint_callback is not None:
            checkpoint_callback(snapshot)

    def get_solver(energy):
        if energy in cached:
            cached.move_to_end(energy)
            return cached[energy]
        solver = (
            AxialMCWF(
                axial_dissipator(
                    profile, energy, config.cells, config.shape, mechanisms=config.mechanisms
                ),
                energy,
                potential,
            )
            if solver_factory is None
            else solver_factory(energy, grid, potential)
        )
        if not np.isclose(float(solver.kinetic_energy_eV), energy, rtol=1e-12, atol=0):
            raise ValueError("solver_factory must construct the requested energy")
        if solver.grid.shape != grid.shape or solver.grid.extent_ang != grid.extent_ang:
            raise ValueError("all energy-specific solvers must use the same physical grid")
        level_observer = None
        if photon_edges is not None:
            if radiation_factory is None:
                from dataclasses import replace
                from importlib import import_module

                level_observer = import_module("pyrite.apps.axial_channeling")._level_observable(
                    replace(config, kinetic_energy_eV=energy)
                )
            else:
                level_observer = radiation_factory(energy, grid)
        level_size = 0
        if level_observer is not None:
            if not hasattr(level_observer, "levels") or not hasattr(
                level_observer.levels, "unpack"
            ):
                raise ValueError("radiation_factory must return a compatible level observable")
            sample_state = np.full(grid.shape, 1 / np.sqrt(grid.area_ang2), dtype=complex)
            level_size = int(np.asarray(level_observer(sample_state, (0.0, 0.0))).size)
            if nd * (npix + len(pairs) + level_size + 17) * 16 > options.storage_limit_bytes:
                raise ValueError("energy-dependent level history exceeds storage_limit_bytes")
        cached[energy] = (solver, level_observer, level_size)
        if len(cached) > options.max_cached_solvers:
            cached.popitem(last=False)
        return cached[energy]

    for start in range(next_id, options.n_electrons, options.chunk_size):
        for eid in range(start, min(options.n_electrons, start + options.chunk_size)):
            energy = float(energies[eid])
            bin_id = int(ids[eid])
            solver, level_observer, level_size = get_solver(energy)
            rng = _rng(options.seed, eid, _ENTRY_TAG)
            if entry_factory is None:
                current_entry, tilt_rad = sampled_entry, config.tilt_rad
            else:
                current_entry, tilt_rad = entry_factory(energy, eid, rng)
            if len(tilt_rad) != 2 or not np.all(np.isfinite(tilt_rad)):
                raise ValueError("entry tilt must be two finite radians")
            custom = None if observable_factory is None else observable_factory(energy)
            if custom is not None:
                # Bound the full single-electron history before requesting it.
                probe = np.ones(grid.shape, complex) / np.sqrt(grid.area_ang2)
                custom_size = np.asarray(custom(probe, (0.0, 0.0))).size
                single_history_bytes = nd * (npix + len(pairs) + custom_size + level_size + 17) * 16
                if single_history_bytes > options.storage_limit_bytes:
                    raise ValueError("custom observable history exceeds storage_limit_bytes")

            def observe(phi, tilt, *, custom=custom, level_observer=level_observer):
                base = np.asarray(np.abs(phi).reshape(-1) ** 2, float)
                output = [base.astype(complex)]
                if projection is not None:
                    output.append(np.asarray(projection(phi, tilt)).ravel())
                if pairs:
                    flat = phi.ravel()
                    coherence = np.array([flat[a] * flat[b].conjugate() for a, b in pairs])
                    output.append(coherence * np.exp(1j * (tilt[0] * dx + tilt[1] * dy)))
                if custom is not None:
                    output.append(np.asarray(custom(phi, tilt)).ravel())
                if level_observer is not None:
                    output.append(np.asarray(level_observer(phi, tilt)).ravel())
                return np.concatenate(output)

            # Unique seed for (global_id, stochastic trajectory) even when all
            # runs pass n_electrons=1 to the existing MCWF backend.
            stream_seed = int(mcwf_stream_key(options.seed ^ _TRAJECTORY_TAG, eid))
            electron = solver.run(
                current_entry,
                config.depths_ang,
                1,
                seed=stream_seed,
                step_ang=config.step_ang,
                tilt_rad=tuple(tilt_rad),
                observable=observe,
            )
            data = np.asarray(electron.observables)
            if data.ndim != 2 or data.shape[0] != nd or not np.all(np.isfinite(data)):
                raise ValueError("solver returned nonfinite or malformed observables")
            density_bins[bin_id] += data[:, :npix].real.reshape(nd, *config.shape)
            off = npix
            if oam_bins is not None:
                assert projection is not None
                n_ell = len(projection.ells)
                oam_bins[bin_id] += data[:, off : off + n_ell].real
                off += n_ell
            if pairs:
                assert coh_bins is not None
                coh_bins[bin_id] += data[:, off : off + len(pairs)]
                off += len(pairs)
            nlevel = level_size
            extra_stop = data.shape[1] - nlevel
            if custom is not None:
                extra = data[:, off:extra_stop]
                if additional_bins is None:
                    extra_bytes = (nb + 1) * extra.size * 16
                    if (nbytes + extra_bytes) * capacity_factor > options.storage_limit_bytes:
                        raise ValueError("custom observable would exceed storage_limit_bytes")
                    additional_bins = np.zeros((nb, *extra.shape), complex)
                    nbytes += extra_bytes
                if extra.shape != additional_bins.shape[1:]:
                    raise ValueError("observable_factory outputs must have a fixed common shape")
                additional_bins[bin_id] += extra
            elif extra_stop != off:
                raise ValueError("unexpected observable shape")
            if photon_edges is not None and level_observer is not None:
                assert photons_bins is not None
                photons_bins[bin_id] += _radiation_bins(
                    data[:, extra_stop:],
                    level_observer.levels,
                    level_observer.levels.catalog,
                    np.asarray(electron.depths_ang),
                    photon_edges,
                    detector_sigma_eV,
                )
            survival_bins[bin_id] += electron.surviving
            for kind, count in electron.jumps.items():
                jumps_bins[bin_id][kind] = jumps_bins[bin_id].get(kind, 0) + int(count)
            if (
                options.checkpoint_every
                and (eid + 1) % options.checkpoint_every == 0
                and eid + 1 < options.n_electrons
            ):
                emit_checkpoint(eid + 1)

    # Convert per-bin sums to per-assigned-incident-electron means. Empty bins
    # contain zeros (conditional observables undefined if no electrons landed).
    denom = np.maximum(counts, 1)
    density_bins /= denom[:, None, None, None]
    survival_bins /= denom[:, None]
    if coh_bins is not None:
        coh_bins /= denom[:, None, None]
    if oam_bins is not None:
        oam_bins /= denom[:, None, None]
    if additional_bins is not None:
        additional_bins /= denom.reshape((nb,) + (1,) * (additional_bins.ndim - 1))
    if photons_bins is not None:
        photons_bins /= denom[:, None, None]
    weights = counts / options.n_electrons
    jumps = {
        name: sum(item.get(name, 0) for item in jumps_bins)
        for name in {key for item in jumps_bins for key in item}
    }
    provenance = dict(
        backend="cpu",
        rng="counter-addressed Philox with separate tags",
        reduction_order="global electron ID ascending",
        n_electrons=options.n_electrons,
        seed=options.seed,
        sampling=options.sampling,
        chunk_size=options.chunk_size,
        max_cached_solvers=options.max_cached_solvers,
        distinct_energies=distinct_energies,
        max_distinct_energies=options.max_distinct_energies,
        resumed_from_electron_id=next_id,
        checkpoint_every=options.checkpoint_every,
        quadrature_method=nodes.method,
        excluded_probability=nodes.excluded_probability,
        work_limit=options.work_limit,
        storage_limit_bytes=options.storage_limit_bytes,
    )
    return AxialEnergyMCWFResult(
        energies,
        bounds,
        centers,
        counts,
        depths,
        grid,
        np.tensordot(weights, density_bins, axes=(0, 0)),
        density_bins,
        weights @ survival_bins,
        survival_bins,
        None if projection is None else projection.ells,
        None if oam_bins is None else np.tensordot(weights, oam_bins, axes=(0, 0)),
        oam_bins,
        pairs,
        None if coh_bins is None else np.tensordot(weights, coh_bins, axes=(0, 0)),
        coh_bins,
        None if additional_bins is None else np.tensordot(weights, additional_bins, axes=(0, 0)),
        additional_bins,
        jumps,
        tuple(jumps_bins),
        photon_edges,
        None if photons_bins is None else np.tensordot(weights, photons_bins[:, 0], axes=(0, 0)),
        None if photons_bins is None else photons_bins[:, 0],
        None if photons_bins is None else np.tensordot(weights, photons_bins[:, 1], axes=(0, 0)),
        None if photons_bins is None else np.tensordot(weights, photons_bins[:, 2], axes=(0, 0)),
        nodes.excluded_probability,
        provenance,
    )


__all__ = [
    "AxialEnergyMCWFOptions",
    "AxialEnergyMCWFResult",
    "AxialEnergyMCWFCheckpoint",
    "axial_energy_mcwf",
    "sample_electron_energies",
]
