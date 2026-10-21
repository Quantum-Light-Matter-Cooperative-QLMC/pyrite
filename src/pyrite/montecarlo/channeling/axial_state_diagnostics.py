"""Bounded axial finite-grid spectral and reference-point ensemble diagnostics.

These are measurement-only observables: they do not change MCWF propagation,
its RNG streams, or its dissipators. Populations are orthonormal *supercell*
Hamiltonian projections, not string-local line-catalog weights. Individual
Bloch carrier sectors have distinct Hamiltonians, H_q=T(k+q)+U.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics


MAX_DENSE_DIMENSION = 256
MAX_SPECTRAL_STATES = 64
MAX_CACHED_SECTORS = 64
HOST_SPECTRAL_CACHE_BYTES = 64 << 20
DEVICE_SPECTRAL_CACHE_BYTES = 32 << 20


def _xp(array):
    if hasattr(array, "__cuda_array_interface__"):
        import cupy

        return cupy
    return np


@dataclass(frozen=True)
class DiscreteSpectrum:
    """Euclidean-orthonormal FFT-grid eigenvectors: columns are eigenstates."""

    energies_eV: np.ndarray
    vectors: np.ndarray


def transverse_spectrum(grid, potential_eV, kinetic_energy_eV, tilt_inv_ang, count):
    """Diagonalize precisely the Hamiltonian used by the split-step FFT.

    Both the supercell boundary conditions and the trajectory's actual carrier
    q are included. A full eigensystem is permitted on small finite grids;
    larger grids use the lowest eigenpairs of a matrix-free Hermitian operator.
    The solver does *not* pretend these finite-supercell states are continuum
    scattering states or infinite-crystal Bloch bands.
    """
    n = int(np.prod(grid.shape))
    if not 1 <= count <= n or (n > MAX_DENSE_DIMENSION and count > MAX_SPECTRAL_STATES + 1):
        raise ValueError("spectral count exceeds the dense/sparse eigenproblem budget")
    if n > MAX_DENSE_DIMENSION and count >= n - 1:
        raise ValueError("full spectrum requires <= 256 grid points; reduce the grid")
    gamma, _ = longitudinal_kinematics(float(kinetic_energy_eV))
    qx, qy = tilt_inv_ang
    kinetic = (
        HBAR2_2M_EV_ANG2 / gamma * ((grid.kx[:, None] + qx) ** 2 + (grid.ky[None, :] + qy) ** 2)
    )
    potential = np.asarray(potential_eV, float)
    if potential.shape != grid.shape or not np.all(np.isfinite(potential)):
        raise ValueError("spectral potential must be finite with grid shape")

    def apply(v):
        state = v.reshape(grid.shape)
        return (np.fft.ifft2(kinetic * np.fft.fft2(state)) + potential * state).reshape(n)

    if n <= MAX_DENSE_DIMENSION:
        # Hermitize only to remove rounding-level numerical asymmetry.
        h = np.column_stack([apply(col) for col in np.eye(n)])
        h = (h + h.conj().T) / 2
        energies, vectors = eigh(h, subset_by_index=(0, count - 1), check_finite=False)
    else:
        h = LinearOperator((n, n), matvec=apply, dtype=np.complex128)
        energies, vectors = eigsh(h, k=count, which="SA", tol=1e-10)
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
    return DiscreteSpectrum(np.asarray(energies, float), np.asarray(vectors, complex))


class AxialSpectralObservable:
    """Orthonormal projectors on the discrete Hamiltonian's eigenstates.

    ``representation='states'`` requires a fixed carrier sector. For electronic
    kicks, use ``representation='energy_bins'``: each trajectory is projected in
    its own post-jump H_q basis, and probabilities are sorted by *that* state's
    transverse energy. Cached sectors are bounded (LRU) and never rounded.
    Truncated results always report the unrepresented surviving weight.
    """

    cuda_compatible = True

    def __init__(
        self,
        grid,
        potential_eV,
        kinetic_energy_eV,
        initial_tilt_inv_ang,
        *,
        count=16,
        cutoff_eV=None,
        representation="states",
        energy_edges_eV=None,
        degeneracy_tol_eV=1e-5,
    ):
        if representation not in ("states", "energy_bins"):
            raise ValueError("spectral representation must be states or energy_bins")
        capacity = int(np.prod(grid.shape))
        limit = capacity if capacity <= MAX_DENSE_DIMENSION else MAX_SPECTRAL_STATES
        if not isinstance(count, int) or count < 1 or count > limit:
            raise ValueError(f"spectral state count must be between 1 and {limit}")
        if not np.isfinite(degeneracy_tol_eV) or degeneracy_tol_eV < 0:
            raise ValueError("degeneracy tolerance must be finite and nonnegative")
        self.grid, self.potential = grid, np.asarray(potential_eV, float)
        self.energy_eV = float(kinetic_energy_eV)
        self.initial_tilt = tuple(float(x) for x in initial_tilt_inv_ang)
        self.count = count
        self.cutoff_eV = cutoff_eV
        self.representation = representation
        self.degeneracy_tol_eV = degeneracy_tol_eV
        n = int(np.prod(grid.shape))
        if count > n:
            raise ValueError("requested states exceed grid dimension")
        # Include the guard eigenvector and temporary eigensolver allocations.
        self._sector_bytes = (count + (count < n)) * n * 16
        if self._sector_bytes > HOST_SPECTRAL_CACHE_BYTES:
            raise ValueError("spectral eigenvectors exceed the 64 MiB per-solve cache budget")
        self._max_sectors = max(
            1, min(MAX_CACHED_SECTORS, HOST_SPECTRAL_CACHE_BYTES // self._sector_bytes)
        )
        self._max_device_sectors = max(
            1, min(MAX_CACHED_SECTORS, DEVICE_SPECTRAL_CACHE_BYTES // self._sector_bytes)
        )
        # The spare eigenpair checks that the final requested state does not
        # silently split a degenerate subspace at the truncation boundary.
        extra = 1 if count < n else 0
        self._cache = OrderedDict()
        while True:
            base = transverse_spectrum(
                grid, self.potential, self.energy_eV, self.initial_tilt, count + extra
            )
            if (
                not extra
                or base.energies_eV[count] - base.energies_eV[count - 1] > degeneracy_tol_eV
            ):
                break
            if count >= min(n, limit):
                raise ValueError(
                    "spectral count cuts through a degenerate manifold; "
                    "reduce count or increase the grid budget"
                )
            count += 1  # complete the degenerate subspace; never keep an arbitrary fragment
            extra = int(count < n)
        self.count = count
        self._cache[self.initial_tilt] = base
        self.energies_eV = base.energies_eV[:count].copy()
        if cutoff_eV is not None:
            if not np.isfinite(cutoff_eV):
                raise ValueError("spectral cutoff must be finite")
            if not np.any(self.energies_eV <= cutoff_eV):
                raise ValueError("cutoff excludes all requested eigenstates")
        self.barrier_eV = float(np.max(self.potential))
        if representation == "states":
            group_edges = [0]
            for i in range(1, count):
                if self.energies_eV[i] - self.energies_eV[i - 1] > degeneracy_tol_eV:
                    group_edges.append(i)
            group_edges.append(count)
            self.groups = [slice(a, b) for a, b in zip(group_edges[:-1], group_edges[1:])]
            self.labels = []
            for group in self.groups:
                energy = float(np.mean(self.energies_eV[group]))
                classification = "above barrier" if energy > self.barrier_eV else "below barrier"
                self.labels.append(
                    f"n={group.start}:{group.stop - 1} · {energy:.3g} eV "
                    f"(g={group.stop - group.start}; {classification})"
                )
            self.width = len(self.groups)
        else:
            if energy_edges_eV is None:
                raise ValueError("energy-bin representation requires explicit eV edges")
            edges = np.asarray(energy_edges_eV, float)
            if (
                edges.ndim != 1
                or edges.size < 2
                or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)
            ):
                raise ValueError("energy bins need strictly increasing finite edges")
            self.edges_eV = edges
            self.width = edges.size - 1
            self.labels = []
            for a, b in zip(edges[:-1], edges[1:]):
                category = (
                    "below barrier"
                    if b <= self.barrier_eV
                    else "above barrier"
                    if a >= self.barrier_eV
                    else "straddles barrier"
                )
                self.labels.append(f"[{a:.3g}, {b:.3g}) eV ({category})")
            self.groups = None
        self.recording_width = self.width + 1  # last channel: represented (including bins)
        self.recording_bytes = self.recording_width * 16
        self.scratch_bytes = self._sector_bytes * (2 + self._max_device_sectors)
        self._device_cache = OrderedDict()

    def _spectrum(self, q):
        if q not in self._cache:
            if self.representation != "energy_bins":
                raise ValueError(
                    "state-index population requires fixed Bloch/carrier sector; "
                    "use energy bins for electronic jumps"
                )
            n = int(np.prod(self.grid.shape))
            extra = 1 if self.count < n else 0
            self._cache[q] = transverse_spectrum(
                self.grid, self.potential, self.energy_eV, q, self.count + extra
            )
            if len(self._cache) > self._max_sectors:
                self._cache.popitem(last=False)
        self._cache.move_to_end(q)
        return self._cache[q]

    def __call__(self, phi, tilt):
        xp = _xp(phi)
        q = tuple(float(v) for v in tilt)
        spec = self._spectrum(q)
        # Discrete eigenvectors have sum |v_i|²=1, while psi is area normalized.
        if xp is np:
            basis = spec.vectors[:, : self.count]
        else:
            if q not in self._device_cache:
                self._device_cache[q] = xp.asarray(spec.vectors[:, : self.count])
                if len(self._device_cache) > self._max_device_sectors:
                    self._device_cache.popitem(last=False)
            self._device_cache.move_to_end(q)
            basis = self._device_cache[q]
        coeff = basis.conj().T @ phi.reshape(-1) * np.sqrt(self.grid.cell_area_ang2)
        pop = xp.abs(coeff) ** 2
        valid = np.ones(self.count, dtype=bool)
        if self.cutoff_eV is not None:
            valid &= spec.energies_eV[: self.count] <= self.cutoff_eV
        if self.representation == "states":
            values = xp.stack([xp.sum(pop[s] * xp.asarray(valid[s])) for s in self.groups])
        else:
            e = spec.energies_eV[: self.count]
            bins = np.searchsorted(self.edges_eV, e, side="right") - 1
            bins[e == self.edges_eV[-1]] = len(self.edges_eV) - 2
            values = xp.stack(
                [xp.sum(pop * xp.asarray(valid & (bins == i))) for i in range(self.width)]
            )
        represented = xp.sum(values)
        return xp.concatenate([values, represented[None]])


class ReferenceCoherenceObservable:
    """Physical spatial rho(r,r0), plus second moment for MC sampling SEM.

    The carrier factor changes under electronic kicks, so apply the trajectory's
    actual q. Never average bare MCWF amplitudes or stochastic phases.
    """

    cuda_compatible = True

    def __init__(self, grid, reference_index):
        self.grid = grid
        self.reference_index = tuple(int(i) for i in reference_index)
        if len(self.reference_index) != 2 or any(
            not 0 <= i < n for i, n in zip(self.reference_index, grid.shape)
        ):
            raise ValueError("reference pixel must be inside the spectral grid")
        self.delta_x = grid.x[:, None] - grid.x[self.reference_index[0]]
        self.delta_y = grid.y[None, :] - grid.y[self.reference_index[1]]
        self.recording_width = 2 * int(np.prod(grid.shape))
        self.recording_bytes = self.recording_width * 16
        self.scratch_bytes = 4 * int(np.prod(grid.shape)) * 16
        self._device_coordinates = None

    def __call__(self, phi, tilt):
        xp = _xp(phi)
        if self._device_coordinates is None or self._device_coordinates[0] is not xp:
            self._device_coordinates = (xp, xp.asarray(self.delta_x), xp.asarray(self.delta_y))
        _, dx, dy = self._device_coordinates
        cross = phi * xp.conj(phi[self.reference_index])
        phase = xp.exp(1j * (float(tilt[0]) * dx + float(tilt[1]) * dy))
        cross = cross * phase
        return xp.concatenate([cross.ravel(), xp.abs(cross).ravel() ** 2])


def choose_reference(grid, components):
    """Choose a nonzero-density reference from weighted normalized entry states."""
    density = np.zeros(grid.shape)
    for weight, phi in components:
        norm = grid.norm(phi)
        if norm > 0:
            density += weight * np.abs(phi) ** 2 / norm
    return tuple(int(i) for i in np.unravel_index(np.argmax(density), grid.shape))


def coherence_maps(
    numerator,
    second_moment,
    density,
    reference,
    electrons,
    *,
    min_density_fraction=1e-4,
    min_visibility=1e-3,
    sigma=2.0,
):
    """Return masked phase, visibility, and optional complex-estimator SEM.

    Mask low-density regions and stochastic correlation estimates whose
    magnitude is smaller than ``sigma`` standard errors. For coherent exact
    reductions, pass ``electrons=None`` (there is no sampling SEM).
    """
    rho = np.asarray(numerator)
    density = np.asarray(density, float)
    r0 = float(density[reference])
    denom = np.sqrt(np.maximum(density * r0, 0.0))
    g = np.divide(rho, denom, out=np.zeros_like(rho, complex), where=denom > 0)
    visibility = np.clip(np.abs(g), 0, 1)
    cutoff = max(float(density.max()), 0) * min_density_fraction
    visible = (density > cutoff) & (r0 > cutoff) & (visibility >= min_visibility)
    sem = None
    if electrons is not None and electrons > 1:
        # E|X|²-E|X|²bar includes extinct trajectories as zeros.
        variance = np.maximum(np.asarray(second_moment).real - np.abs(rho) ** 2, 0)
        sem = np.sqrt(variance / (electrons - 1))
        visible &= np.abs(rho) >= sigma * sem
    elif electrons is not None and electrons <= 1:
        # One trajectory cannot establish ensemble statistical significance.
        sem = np.full(density.shape, np.nan)
    phase = np.ma.masked_where(~visible, np.angle(g))
    return phase, np.ma.masked_where(~visible, visibility), sem
