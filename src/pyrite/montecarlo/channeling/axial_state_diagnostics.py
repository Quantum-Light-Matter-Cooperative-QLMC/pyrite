"""Bounded axial finite-grid spectral and reference-point ensemble diagnostics.

These are measurement-only observables: they do not change MCWF propagation,
its RNG streams, or its dissipators. Populations are orthonormal *supercell*
Hamiltonian projections, not string-local line-catalog weights. Individual
Bloch carrier sectors have distinct Hamiltonians, H_q=T(k+q)+U.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg
from threadpoolctl import threadpool_limits

from pyrite.materials.channeling.transverse import (
    HBAR2_2M_EV_ANG2,
    longitudinal_kinematics,
)

MAX_DENSE_DIMENSION = 256
MAX_SPECTRAL_STATES = 64
MAX_CACHED_SECTORS = 64
HOST_SPECTRAL_CACHE_BYTES = 64 << 20
DEVICE_SPECTRAL_CACHE_BYTES = 32 << 20
PENDING_SPECTRAL_BYTES = 16 << 20


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


def _warm_lobpcg(kinetic, potential, guess, count):
    """Lowest ``count`` eigenpairs by LOBPCG in the plane-wave basis, or ``None``.

    ``guess`` holds real-space columns already shifted to this sector. Only an
    explicit residual check accepts the result; otherwise ARPACK is used.
    """
    shape, n = potential.shape, potential.size
    t = kinetic.reshape(n)

    def apply(x):
        r = np.fft.ifft2(x.T.reshape(-1, *shape), norm="ortho") * potential
        return np.fft.fft2(r, norm="ortho").reshape(-1, n).T + t[:, None] * x

    x0 = np.fft.fft2(guess.T.reshape(-1, *shape), norm="ortho").reshape(-1, n).T[:, :count]
    if x0.shape[1] < count:
        fill = np.random.default_rng(0).standard_normal((n, 2 * (count - x0.shape[1])))
        x0 = np.hstack([x0, fill.view(complex)])
    precondition = (1.0 / (t + abs(potential.mean()) + 1.0))[:, None]

    def block(f):
        return lambda x: f(x.reshape(n, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            energies, vectors = lobpcg(
                block(apply),
                x0,
                M=block(lambda x: precondition * x),
                largest=False,
                tol=1e-9,
                maxiter=200,
            )
        except (Warning, np.linalg.LinAlgError, ValueError):
            return None
    residual = np.linalg.norm(apply(vectors) - vectors * energies, axis=0)
    if not np.all(residual <= 1e-9 * np.maximum(np.abs(energies), 1.0)):
        return None
    order = np.argsort(energies)
    vectors = np.fft.ifft2(vectors[:, order].T.reshape(-1, *shape), norm="ortho")
    return energies[order], vectors.reshape(-1, n).T


def _orthonormal_eigenpairs(apply, vectors):
    """Refine a sparse invariant subspace into orthonormal Hermitian eigenpairs.

    Complex ARPACK uses the general eigensolver, whose individually normalized
    vectors need not be mutually orthogonal inside a degenerate eigenspace.
    Projector probabilities require an orthonormal basis, even with tiny
    individual residuals. QR followed by Rayleigh--Ritz preserves the subspace.
    """
    basis, _ = np.linalg.qr(vectors)
    action = np.column_stack([apply(v) for v in basis.T])
    projected = basis.conj().T @ action
    energies, rotation = eigh((projected + projected.conj().T) / 2, check_finite=False)
    return energies, basis @ rotation


def transverse_spectrum(grid, potential_eV, kinetic_energy_eV, tilt_inv_ang, count, guess=None):
    """Diagonalize precisely the Hamiltonian used by the split-step FFT.

    Both the supercell boundary conditions and the trajectory's actual carrier
    q are included. A full eigensystem is permitted on small finite grids;
    larger grids use the lowest eigenpairs of a matrix-free Hermitian operator.
    The solver does *not* pretend these finite-supercell states are continuum
    scattering states or infinite-crystal Bloch bands.

    ``guess`` (real-space columns, shape ``(n, m)``) only warm-starts the
    matrix-free solve: LOBPCG pairs are kept only if every residual
    ``|H v - E v| <= 1e-9 max(|E|, 1)`` eV; otherwise ARPACK runs unseeded.
    BLAS runs single-threaded here: these small blocks oversubscribe cores.
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
        with threadpool_limits(1, user_api="blas"):
            solved = None if guess is None else _warm_lobpcg(kinetic, potential, guess, count)
            if solved is None:
                h = LinearOperator((n, n), matvec=apply, dtype=np.complex128)  # type: ignore
                solved = eigsh(h, k=count, which="SA", tol=1e-10)
            energies, vectors = _orthonormal_eigenpairs(apply, solved[1])
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
            self.groups = [
                slice(a, b) for a, b in zip(group_edges[:-1], group_edges[1:], strict=True)
            ]
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
            for a, b in zip(edges[:-1], edges[1:], strict=True):
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
        self._base_spectrum = base
        self._pending = []
        self._pending_bytes = 0
        if representation == "energy_bins":
            # Pending states and the bounded iterative solver coexist with caches.
            from ._axial_spectral_batch import SOLVER_BUDGET_BYTES

            self.scratch_bytes += PENDING_SPECTRAL_BYTES + SOLVER_BUDGET_BYTES

    def _guess(self, q):
        """Nearest cached sector's eigenvectors, carrier-shifted by a grid ``G``.

        ``H(q + G)`` is ``H(q)`` conjugated by ``exp(-i G r)`` up to FFT-grid
        wrap-around, so the shifted vectors are a close LOBPCG start.
        """
        dk = [2.0 * np.pi / e for e in self.grid.extent_ang]
        best = None
        for s, spec in self._cache.items():
            m = [round((a - b) / d) for a, b, d in zip(q, s, dk, strict=True)]
            miss = np.hypot(*(a - b - i * d for a, b, i, d in zip(q, s, m, dk, strict=True)))
            if best is None or miss < best[0]:
                best = (miss, spec.vectors, m)
        if best is None:
            return None
        _, vectors, (mx, my) = best
        phase = np.exp(
            -1j * (mx * dk[0] * self.grid.x[:, None] + my * dk[1] * self.grid.y[None, :])
        )
        return vectors * phase.reshape(-1, 1)

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
                self.grid, self.potential, self.energy_eV, q, self.count + extra, self._guess(q)
            )
            if len(self._cache) > self._max_sectors:
                self._cache.popitem(last=False)
        self._cache.move_to_end(q)
        return self._cache[q]

    def __call__(self, phi, tilt):
        q = tuple(float(v) for v in tilt)
        spec = self._spectrum(q)
        return self._project(phi, q, spec)

    def _project(self, phi, q, spec):
        xp = _xp(phi)
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
        scalar = phi.ndim == 2
        coeff = phi.reshape(-1, basis.shape[0]) @ basis.conj() * np.sqrt(self.grid.cell_area_ang2)
        pop = xp.abs(coeff) ** 2
        valid = np.ones(self.count, dtype=bool)
        if self.cutoff_eV is not None:
            valid &= spec.energies_eV[: self.count] <= self.cutoff_eV
        if self.representation == "states":
            assert self.groups is not None
            values = xp.stack(
                [xp.sum(pop[:, s] * xp.asarray(valid[s]), axis=-1) for s in self.groups],
                axis=-1,
            )
        else:
            e = spec.energies_eV[: self.count]
            bins = np.searchsorted(self.edges_eV, e, side="right") - 1
            bins[e == self.edges_eV[-1]] = len(self.edges_eV) - 2
            values = xp.stack(
                [xp.sum(pop * xp.asarray(valid & (bins == i)), axis=-1) for i in range(self.width)],
                axis=-1,
            )
        represented = xp.sum(values, axis=-1)
        result = xp.concatenate([values, represented[:, None]], axis=-1)
        return result[0] if scalar else result

    def defer_batch(self, states, tilts, destination):
        """Queue immutable states and a live accumulator view under a byte cap."""
        xp = _xp(states)
        # One transfer for all scalar carrier components, including device q.
        qs = xp.asarray(tilts)
        qs = qs if xp is np else xp.asnumpy(qs)
        if self.representation == "states":
            for q in dict.fromkeys(tuple(q) for q in qs):
                self._spectrum(q)  # retain the fixed-carrier refusal after a kick
            destination += self._project(states, self.initial_tilt, self._base_spectrum).sum(axis=0)
            return
        per_state = states[0].nbytes
        if per_state > PENDING_SPECTRAL_BYTES:
            # A single state must not silently exceed the queue's declared cap.
            self.flush()
            for state, q in zip(states, qs, strict=True):
                destination += self(state, q)
            return
        capacity = max(1, PENDING_SPECTRAL_BYTES // per_state)
        for start in range(0, len(states), capacity):
            chunk = states[start : start + capacity]
            if self._pending_bytes + chunk.nbytes > PENDING_SPECTRAL_BYTES:
                self.flush()
            self._pending.append((chunk.copy(), qs[start : start + capacity].copy(), destination))
            self._pending_bytes += chunk.nbytes

    def flush(self):
        """Resolve sectors in deterministic batches and fill queued accumulators."""
        if not self._pending:
            return
        from ._axial_spectral_batch import transverse_spectra_batch

        xp = _xp(self._pending[0][0])
        keys = list(dict.fromkeys(tuple(q) for _, qs, _ in self._pending for q in qs))
        for start in range(0, len(keys), self._max_sectors):
            group = keys[start : start + self._max_sectors]
            missing = [q for q in group if q not in self._cache]
            spectra = {}
            if missing and self.representation == "energy_bins" and xp is not np:
                solved = transverse_spectra_batch(
                    self.grid,
                    self.potential,
                    self.energy_eV,
                    missing,
                    self.count + int(self.count < np.prod(self.grid.shape)),
                    self._base_spectrum,
                    base_tilt=self.initial_tilt,
                    xp=xp,
                )
                spectra = dict(zip(missing, solved, strict=True))
            self._flush_sectors(group, spectra)
        self._pending.clear()
        self._pending_bytes = 0

    def _flush_sectors(self, keys, spectra):
        for q in keys:
            spec = spectra.pop(q, None)
            if spec is None:
                spec = self._spectrum(q)
            self._device_cache.pop(q, None)
            for states, qs, destination in self._pending:
                indices = np.flatnonzero(np.all(qs == q, axis=1))
                if indices.size:
                    destination += self._project(states[indices], q, spec).sum(axis=0)
            self._cache[q] = spec
            self._cache.move_to_end(q)
            if len(self._cache) > self._max_sectors:
                self._cache.popitem(last=False)


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
            not 0 <= i < n for i, n in zip(self.reference_index, grid.shape, strict=True)
        ):
            raise ValueError("reference pixel must be inside the spectral grid")
        self.delta_x = grid.x[:, None] - grid.x[self.reference_index[0]]
        self.delta_y = grid.y[None, :] - grid.y[self.reference_index[1]]
        self.recording_width = 2 * int(np.prod(grid.shape))
        self.recording_bytes = self.recording_width * 16
        self.scratch_bytes = 4 * int(np.prod(grid.shape)) * 16
        self._device_coordinates = None

    def sum_batch(self, states, tilts):
        """Sum physical reference correlations and their second moments."""
        xp = _xp(states)
        q = xp.asarray(tilts)[:, :, None, None]
        dx, dy = xp.asarray(self.delta_x), xp.asarray(self.delta_y)
        ref = states[:, self.reference_index[0], self.reference_index[1]]
        cross = states * ref.conj()[:, None, None]
        cross *= xp.exp(1j * (q[:, 0] * dx + q[:, 1] * dy))
        return xp.concatenate(
            [xp.sum(cross, axis=0).ravel(), xp.sum(xp.abs(cross) ** 2, axis=0).ravel()]
        )

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
