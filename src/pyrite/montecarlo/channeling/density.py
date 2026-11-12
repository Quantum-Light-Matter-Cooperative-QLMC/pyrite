"""Deterministic density-matrix depth evolution for planar channeling.

Integrates the Lindblad equation of :mod:`.operators` for the transverse
density matrix of a translation-averaged beam,

``d rho/dz = -(i / hbar v) (H_eff rho - rho H_eff^dagger) + J(rho)``,
``H_eff = H_perp - (i hbar v / 2) C(x, x)``, ``J(rho)(x, x') = C(x, x') rho(x, x')``.

**Representation.** The beam is averaged over its transverse entry point, so
``rho`` commutes with translation by one period and is block-diagonal in Bloch
``k``. On a supercell of ``n_cells`` periods the blocks sit at
``kappa_r = r g / n_cells`` (the kernel's grid). Each block is held in the
eigenbasis of ``H_perp(kappa_r)`` over plane-wave orders ``|h| <= H``, so the
coherent step is exact phases. The jump term is applied in position space,
where it is an elementwise product: blocks -> ``rho(x_a, x_b + j d)`` on a
cell grid of ``n_grid`` points by an FFT over ``r``, times the cell kernel,
and back. ``n_grid >= 3 H + 2`` keeps product momenta from aliasing into the
band. The depth step evolves only the eigenstate slots that hold a subspace
state in some block (the no-jump propagator vanishes elsewhere), mapping them
straight to the cell grid; the cost then follows ``|S|``, not ``N`` (#1125).

**Collapse rule (dechanneling).** Only the subspace ``S`` of states with
``E_perp < barrier + margin`` evolves. Jumps into in-basis states above the
cutoff (``F``) are removed as E_perp-resolved *free* flux; jump momenta outside
the plane-wave band and the kernel's kept band are *overflow* flux (they carry
the difference between the full loss ``tr(C(x,x) rho)`` and the in-band jump
trace). ``S``--``F`` coherences are dropped, and the no-jump propagator uses
``P_S H_eff P_S``: the loss operator's ``S``--``F`` elements, which would only
feed those coherences, are omitted. Entry components above the cutoff
are *entry* flux at ``z = 0``. Continuous energy loss changes ``gamma``; when it
drifts by more than a tolerance, the blocks are re-projected onto the new
eigenstates and the part leaving ``S`` is *reprojection* flux.

**Integrator.** Strang splitting: half a no-jump step ``exp(-i H_eff dz/2 hbar v)``,
the jump map ``1 + dz J + dz^2 J^2 / 2``, and a second half no-jump step. It is
second order in ``dz``. The flux is booked from the jump terms themselves, so
``trace + flux - 1 = O(dz^2)`` is a convergence check rather than an
identity.

Lengths in angstroms, energies in eV.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
from scipy import fft as sp_fft
from scipy.linalg import expm

from ...materials.channeling.potential import PlanarProfile
from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .bloch import _hamiltonian, longitudinal_kinematics
from .metrics import EnsembleMetrics, density_metrics
from .operators import ScatteringKernel
from .spatial import spatial_density

#: Declared calibration of the dechanneling cutoff: states more than this far
#: above the planar barrier [eV] are a sink (``cr-dechanneling-collapse``).
DECHANNELING_MARGIN_EV = 300.0
#: Provenance of :data:`DECHANNELING_MARGIN_EV`; carried into run identity once
#: channeling output is persisted (#507).
DECHANNELING_MARGIN_PROVENANCE = (
    "Calibrated on Kephart et al., Phys. Rev. B 40, 4249 (1989): Si(110) and Si(100) "
    "occupation lengths at 16.92 MeV, compared through the forward-detector observable with "
    "bound-weighted y-spreading (checks/kephart_occupation.py). With 300 eV, 5 of 6 Table I "
    "values including 54.5 MeV are within +-20%; Si(100) 54.5 MeV is 1.35x. Maintainer-declared "
    "2026-10-15 (#504)."
)


def _adjoint(matrices: np.ndarray) -> np.ndarray:
    return matrices.swapaxes(-1, -2).conj()


def _sandwich(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Batched ``A B A^dagger``."""
    return outer @ inner @ _adjoint(outer)


def _host(value: Any) -> np.ndarray:
    """``value`` as a NumPy array; device (CuPy) arrays are copied back."""
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _array_namespace(backend: str) -> Any:
    """Array module for :meth:`DensityMatrixSolver.propagate`'s ``backend``."""
    if backend == "cpu":
        return np
    if backend == "cuda":
        from ..._backend import select_cupy_backend

        return select_cupy_backend().xp
    raise ValueError("backend must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class DensityMatrixResult:
    """Depth history of a density-matrix channeling solve.

    Attributes
    ----------
    z_ang
        Recorded depths [angstrom], starting at 0.
    kinetic_energy_eV
        Electron kinetic energy at each recorded depth [eV].
    band_populations
        Population of the ``n``-th lowest state of every Bloch block, summed
        over blocks; shape ``(n_z, n_bands)``. Deep bands are flat in ``k``,
        so ``n`` labels one physical bound state.
    bound_population
        Population below the potential barrier at each depth.
    trace
        Total population in the evolving subspace ``S``.
    free_flux, overflow_flux, reprojection_flux
        Cumulative dechanneled population by channel at each depth.
    entry_flux
        Entry weight above the cutoff or outside the basis (``z = 0`` sink).
    free_flux_by_energy
        Cumulative free and reprojection flux binned on ``energy_edges_eV``,
        each deposit at the basis energies current when it happened. Weight
        outside the edges is clipped into the first or last bin.
    energy_edges_eV
        Transverse-energy bin edges of ``free_flux_by_energy`` [eV].
    barrier_eV, cutoff_eV
        Potential maximum and the subspace cutoff ``barrier + margin`` [eV].
    final_density
        Subspace density matrix at the final depth, per Bloch block in the
        eigenbasis of the final ``gamma``; shape ``(n_cells, N, N)``.
    spatial_x_ang, spatial_density_per_ang
        Optional cell coordinates [angstrom] and unconditioned ensemble density
        [angstrom**-1] at ``z_ang``, shape ``(n_z, n_x)``. Both are ``None``
        unless requested. The cell integral equals ``trace``.
    """

    z_ang: np.ndarray
    kinetic_energy_eV: np.ndarray
    band_populations: np.ndarray
    bound_population: np.ndarray
    trace: np.ndarray
    free_flux: np.ndarray
    overflow_flux: np.ndarray
    reprojection_flux: np.ndarray
    entry_flux: float
    free_flux_by_energy: np.ndarray
    energy_edges_eV: np.ndarray
    barrier_eV: float
    cutoff_eV: float
    final_density: np.ndarray
    spatial_x_ang: np.ndarray | None = None
    spatial_density_per_ang: np.ndarray | None = None

    metrics: EnsembleMetrics | None = None
    retained_band_populations: np.ndarray | None = None

    @property
    def total_flux(self) -> np.ndarray:
        """Cumulative dechanneled population, entry flux included."""
        return self.entry_flux + self.free_flux + self.overflow_flux + self.reprojection_flux

    @property
    def balance_defect(self) -> np.ndarray:
        """``trace + flux - 1``; ``O(dz^2)`` for the Strang integrator."""
        return self.trace + self.total_flux - 1.0


@dataclass
class _Basis:
    """Eigenbasis of every Bloch block at one Lorentz factor."""

    gamma: float
    energies: np.ndarray  # (n_cells, N)
    vectors: np.ndarray  # (n_cells, N, N); column n is state n in plane waves
    subspace: np.ndarray = field(init=False)  # (n_cells, N) bool, set by the solver
    operators: dict = field(default_factory=dict, repr=False)  # cached _JumpOperators


@dataclass(frozen=True)
class _JumpOperators:
    """Jump and loss maps restricted to eigenstate slots ``active`` of one basis.

    ``active`` must hold every subspace state of every block; ``rest`` is its
    complement, whose jump diagonals are free flux. ``to_grid[r] = W_r V_r[:, active]``
    takes a slot-restricted density straight to cell-grid samples. The array
    fields live on namespace ``xp``; :meth:`on` copies them to a device, while
    :meth:`restrict`/:meth:`expand` act on host densities.
    """

    active: np.ndarray  # (A,) eigenstate slots
    rest: np.ndarray  # (N - A,) the other slots
    keep: np.ndarray  # (n_cells, A) bool, subspace membership of each active slot
    to_grid: np.ndarray  # (n_cells, n_grid, A)
    from_grid: np.ndarray  # (n_cells, A, n_grid), adjoint of to_grid
    rest_grid: np.ndarray  # (n_cells, n_grid, N - A)
    vectors: np.ndarray  # (n_cells, N, A), plane-wave columns of the active slots
    loss: np.ndarray  # (n_cells, A, A), V^dagger C(x, x) V at unit kernel scale
    cell_kernel: np.ndarray  # (n_cells, n_grid, n_grid), the solver's unit-scale cell kernel
    wave_gram: np.ndarray  # (n_cells, n_grid, n_grid), (W W^dagger)^T per block
    xp: Any = np

    def on(self, xp: Any) -> _JumpOperators:
        """These maps with every array field copied to namespace ``xp``."""
        if xp is self.xp:
            return self
        arrays = {f.name: xp.asarray(getattr(self, f.name)) for f in fields(self) if f.name != "xp"}
        return replace(self, xp=xp, **arrays)

    def restrict(self, rho: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(rho[:, self.active][:, :, self.active])

    def expand(self, rho: np.ndarray) -> np.ndarray:
        size = self.active.size + self.rest.size
        full = np.zeros((rho.shape[0], size, size), dtype=complex)
        full[:, self.active[:, None], self.active[None, :]] = rho
        return full


class DensityMatrixSolver:
    """Planar-channeling density-matrix solver on one profile and kernel.

    Parameters
    ----------
    profile
        Planar continuum potential.
    kinetic_energy_eV
        Entry kinetic energy [eV].
    kernel
        Dissipator kernel (e.g. ``tds_kernel(...) + electronic_kernel(...)``) at
        the entry energy; its ``n_cells`` sets the Bloch grid and its
        ``max_order`` the plane-wave band.
    margin_eV
        Subspace cutoff above the barrier [eV]; states below
        ``barrier + margin`` evolve, the rest are a sink. Defaults to the
        declared calibration :data:`DECHANNELING_MARGIN_EV` (300 eV).
    n_grid
        Cell grid for the jump product; defaults to the smallest alias-free
        even size ``>= 3 H + 2``.

    Validation: cr-density-matrix-split-operator, cr-dechanneling-collapse
    """

    def __init__(
        self,
        profile: PlanarProfile,
        kinetic_energy_eV: float,
        kernel: ScatteringKernel,
        *,
        margin_eV: float = DECHANNELING_MARGIN_EV,
        n_grid: int | None = None,
    ) -> None:
        if not np.isclose(kernel.period_ang, profile.period_ang):
            raise ValueError("kernel and profile periods differ")
        self.profile = profile
        self.kernel = kernel
        self.n_cells = kernel.n_cells
        self.orders = kernel.orders
        self.max_order = kernel.max_order
        minimum = 3 * self.max_order + 2
        self.n_grid = int(n_grid) if n_grid is not None else minimum + minimum % 2
        if self.n_grid < minimum:
            raise ValueError(f"n_grid must be >= 3 * max_order + 2 = {minimum}")
        self.margin_eV = float(margin_eV)
        _, self.barrier_eV = profile.extrema()
        self.cutoff_eV = self.barrier_eV + self.margin_eV
        self.energy_eV = float(kinetic_energy_eV)
        self.gamma0, self.beta0 = longitudinal_kinematics(self.energy_eV)
        self.hbar_v0 = HBARC_EV_ANG * self.beta0
        if not np.isclose(self.hbar_v0, kernel.hbar_v_eV_ang, rtol=1e-9):
            raise ValueError("kernel was built for a different kinetic energy")
        g = profile.g_inv_ang
        self.kappa = np.arange(self.n_cells) * g / self.n_cells
        x = np.arange(self.n_grid) * profile.period_ang / self.n_grid
        waves = self.kappa[:, None] + self.orders[None, :] * g
        self._waves = np.exp(1j * waves[:, None, :] * x[None, :, None])  # (r, a, h)
        # (W W^dagger)^T per block: tr(W^dagger B W) = sum(B * gram) is the band trace.
        self._wave_gram = np.swapaxes(self._waves @ _adjoint(self._waves), -1, -2)
        self._cell_kernel0 = kernel.cell_kernel(self.n_grid)
        self._loss_toeplitz = self._toeplitz(kernel.diagonal_coefficients)
        self.basis = self._basis(self.gamma0)
        self._scale = 1.0

    # -- basis and operators -------------------------------------------------

    def restore_entry_basis(self) -> None:
        """Return the shared solver to its entry basis ``gamma0`` and ``_scale = 1``.

        :meth:`propagate` with stopping reprojects and leaves the basis at the
        exit-face ``gamma``. Anything that builds operators or catalogs from
        ``self.basis`` (:func:`.emission_lines`, MCWF) calls this first, so a
        cached solver scores identically whether or not transport ran on it.
        """
        if self.basis.gamma != self.gamma0 or self._scale != 1.0:
            self.basis, self._scale = self._basis(self.gamma0), 1.0

    def _toeplitz(self, coefficients: np.ndarray) -> np.ndarray:
        """Plane-wave matrix ``<h|f(x)|h'> = f_{h-h'}`` of a periodic function."""
        lag = self.orders[:, None] - self.orders[None, :]
        padded = np.zeros(2 * self.orders.size, dtype=complex)
        count = min(coefficients.size, padded.size)
        padded[:count] = coefficients[:count]
        return np.where(lag >= 0, padded[np.abs(lag)], np.conj(padded[np.abs(lag)]))

    def _basis(self, gamma: float) -> _Basis:
        energies, vectors = [], []
        for kappa in self.kappa:
            e, v = np.linalg.eigh(_hamiltonian(self.profile, gamma, kappa, self.orders))
            energies.append(e)
            vectors.append(v)
        basis = _Basis(gamma, np.array(energies), np.array(vectors))
        basis.subspace = basis.energies < self.cutoff_eV
        return basis

    def _propagators(self, dz: float, hbar_v: float) -> np.ndarray:
        """Half-step no-jump propagators ``exp(-i H_eff dz / (2 hbar v))`` per block."""
        size = self.orders.size
        out = np.zeros((self.n_cells, size, size), dtype=complex)
        for r in range(self.n_cells):
            keep = self.basis.subspace[r]
            vectors = self.basis.vectors[r][:, keep]
            loss = vectors.conj().T @ self._loss_toeplitz @ vectors * self._scale
            h_eff = np.diag(self.basis.energies[r][keep]) - 0.5j * hbar_v * loss
            out[r][np.ix_(keep, keep)] = expm(-0.5j * dz / hbar_v * h_eff)
        return out

    def _operators(self, *, restricted: bool) -> _JumpOperators:
        """Jump/loss maps of the current basis on every slot or only subspace slots.

        The restricted maps act on the slots holding a subspace state in some
        block; :meth:`propagate` evolves only those, since the no-jump propagator
        is zero outside ``S``. The cost per step then scales with ``|S|``, which the
        fixed cutoff holds near constant as the plane-wave band grows (#1125).
        """
        basis = self.basis
        key = "restricted" if restricted else "full"
        if key not in basis.operators:
            size = self.orders.size
            mask = basis.subspace.any(axis=0) if restricted else np.ones(size, dtype=bool)
            active, rest = np.flatnonzero(mask), np.flatnonzero(~mask)
            vectors = np.ascontiguousarray(basis.vectors[:, :, active])
            to_grid = self._waves @ vectors
            basis.operators[key] = _JumpOperators(
                active=active,
                rest=rest,
                keep=basis.subspace[:, active],
                to_grid=to_grid,
                from_grid=np.ascontiguousarray(_adjoint(to_grid)),
                rest_grid=self._waves @ basis.vectors[:, :, rest],
                vectors=vectors,
                loss=_adjoint(vectors) @ self._loss_toeplitz @ vectors,
                cell_kernel=self._cell_kernel0,
                wave_gram=self._wave_gram,
            )
        return basis.operators[key]

    def _loss_rate(self, ops: _JumpOperators, rho: Any) -> Any:
        """Loss rate on the slots of ``ops`` as a 0-d array on ``ops.xp`` (no sync)."""
        return ops.xp.einsum("rij,rji->", ops.loss, rho).real * self._scale

    def loss_rate(self, rho: np.ndarray) -> float:
        """Full loss ``tr(C(x, x) rho)`` [1/angstrom], every momentum transfer included."""
        return float(self._loss_rate(self._operators(restricted=False), rho))

    def _jump(self, ops: _JumpOperators, rho: Any) -> tuple[Any, Any, Any]:
        """:meth:`jump` on the slots of ``ops``; ``rho`` and ``J_S`` are slot-restricted.

        Runs on ``ops.xp``; ``band_trace`` is a 0-d array there, so a device step
        never synchronises with the host.

        Validation: cr-density-matrix-split-operator
        """
        xp = ops.xp
        # Everything below is linear in ``back``; the kernel scale and the grid
        # normalisation are applied to the small outputs, not the grid arrays.
        factor = self._scale / float(self.n_grid) ** 2
        grid = ops.to_grid @ rho @ ops.from_grid
        if xp is np:
            # ``scipy.fft`` in place; callers may raise its thread count with
            # ``scipy.fft.set_workers``.
            product = sp_fft.fft(grid, axis=0, overwrite_x=True)
            product *= ops.cell_kernel
            back = sp_fft.ifft(product, axis=0, overwrite_x=True)
        else:
            back = xp.fft.ifft(xp.fft.fft(grid, axis=0) * ops.cell_kernel, axis=0)
        eigen = ops.from_grid @ back @ ops.to_grid * factor
        diagonal = xp.einsum("rnn->rn", eigen).real
        free = xp.zeros((self.n_cells, self.orders.size))
        free[:, ops.active] = xp.where(ops.keep, 0.0, diagonal)
        if ops.rest.size:
            outgoing = back @ ops.rest_grid
            free[:, ops.rest] = (
                xp.einsum("ran,ran->rn", ops.rest_grid.conj(), outgoing).real * factor
            )
        inside = eigen * (ops.keep[:, :, None] & ops.keep[:, None, :])
        band_trace = xp.einsum("rab,rab->", back, ops.wave_gram).real * factor
        return inside, free, band_trace

    def jump(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply ``J(rho) = C(x, x') rho(x, x')`` and project it onto the basis.

        Returns ``(J_S, free, band_trace)``: the subspace block of ``J``
        (eigenbasis, zero outside ``S``), the diagonal of ``J`` on in-basis
        states above the cutoff, shape ``(n_cells, N)``, and ``tr J`` within the
        plane-wave band.

        Validation: cr-density-matrix-split-operator
        """
        inside, free, band_trace = self._jump(self._operators(restricted=False), rho)
        return inside, free, float(band_trace)

    # -- states ----------------------------------------------------------------

    def entry_lattice(
        self, incidence_rad: float = 0.0, divergence_rad: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Entry grid momenta ``l`` (``k_x = l g / n_cells``) and their weights.

        Validation: cr-dechanneling-collapse
        """
        p_inv_ang = self.gamma0 * self.beta0 * M_E_EV / HBARC_EV_ANG
        step = self.profile.g_inv_ang / self.n_cells
        centre = p_inv_ang * float(incidence_rad) / step
        if divergence_rad > 0.0:
            sigma = p_inv_ang * float(divergence_rad) / step
            if sigma < 1.0:
                raise ValueError("divergence is unresolved by the Bloch grid; raise n_cells")
            span = int(np.ceil(8.0 * sigma))
            lattice = np.arange(int(np.floor(centre)) - span, int(np.ceil(centre)) + span + 1)
            weights = np.exp(-0.5 * ((lattice - centre) / sigma) ** 2)
            return lattice, weights / weights.sum()
        return np.array([int(np.round(centre))]), np.ones(1)

    def entry_state(
        self, incidence_rad: float = 0.0, divergence_rad: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Divergence-averaged entry density matrix on the Bloch grid.

        A plane wave at angle ``psi`` has ``k_x = p psi / hbar``; on the grid
        ``k_x = l g / n_cells`` it is plane-wave order ``nu`` of block ``r``,
        with ``l = r + nu n_cells``. A Gaussian divergence weights the grid
        momenta by ``exp(-(k_x - p psi_0/hbar)^2 / 2 (p sigma/hbar)^2)``. With no
        divergence the beam is snapped to the nearest grid momentum. Pass the
        refracted ``effective_rad`` of
        :func:`~pyrite.montecarlo.channeling.entry.plane_incidence`.

        Returns ``(rho_S, free, outside)``: the subspace block, the diagonal
        weight on in-basis states above the cutoff, and the weight outside the
        plane-wave band.

        Validation: cr-dechanneling-collapse
        """
        lattice, weights = self.entry_lattice(incidence_rad, divergence_rad)
        residue = lattice % self.n_cells
        order = (lattice - residue) // self.n_cells
        inband = np.abs(order) <= self.max_order
        size = self.orders.size
        plane = np.zeros((self.n_cells, size, size), dtype=complex)
        plane[residue[inband], order[inband] + self.max_order, order[inband] + self.max_order] = (
            weights[inband]
        )
        return self.entry_from_plane(plane, float(weights[~inband].sum()))

    def entry_from_plane(
        self, plane: np.ndarray, outside: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Collapse a plane-wave-basis entry density onto the evolving subspace.

        ``plane[r, i, j]`` is block ``kappa_r`` over plane-wave orders (e.g. from
        :func:`~.states.planar_entry_density`); returns ``(rho_S, free,
        outside)`` like :meth:`entry_state`.

        Validation: cr-dechanneling-collapse, cr-entry-state-projection
        """
        plane = np.asarray(plane, dtype=complex)
        if plane.shape != (self.n_cells, self.orders.size, self.orders.size):
            raise ValueError("plane must have shape (n_cells, N, N) of this solver")
        eigen = _sandwich(_adjoint(self.basis.vectors), plane)
        keep = self.basis.subspace
        free = np.where(keep, 0.0, np.einsum("rnn->rn", eigen).real)
        rho = eigen * (keep[:, :, None] & keep[:, None, :])
        return rho, free, float(outside)

    def rate_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Golden-rule rates of the rate-equation (Pauli) limit on ``S``.

        ``W[(r', n'), (r, n)] = J(|r n><r n|)_{(r'n'),(r'n')}`` and the loss
        ``Gamma_(r,n) = <r n|C(x, x)|r n>`` [1/angstrom], flattened over the
        ``S`` states in block-major order. When coherences dephase faster than
        scattering acts, populations obey ``dP/dz = W P - Gamma P``.

        Validation: cr-density-matrix-split-operator
        """
        index = np.argwhere(self.basis.subspace)
        size = self.orders.size
        rates = np.zeros((index.shape[0], index.shape[0]))
        loss = np.zeros(index.shape[0])
        for column, (r, n) in enumerate(index):
            rho = np.zeros((self.n_cells, size, size), dtype=complex)
            rho[r, n, n] = 1.0
            inside, _, _ = self.jump(rho)
            rates[:, column] = np.einsum("rnn->rn", inside).real[self.basis.subspace]
            loss[column] = self.loss_rate(rho)
        return rates, loss

    def band_populations(self, rho: np.ndarray) -> np.ndarray:
        """Populations of the ``n``-th lowest state, summed over Bloch blocks."""
        return np.einsum("rnn->n", rho).real

    def bound_population(self, rho: np.ndarray) -> float:
        """Population of states below the potential barrier."""
        diagonal = np.einsum("rnn->rn", rho).real
        return float(diagonal[self.basis.energies < self.barrier_eV].sum())

    def _reproject(self, rho: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """Move ``rho`` onto the eigenbasis at ``gamma``; return it and the free weight."""
        old = self.basis
        plane = _sandwich(old.vectors, rho)
        self.basis = self._basis(gamma)
        eigen = _sandwich(_adjoint(self.basis.vectors), plane)
        keep = self.basis.subspace
        free = np.where(keep, 0.0, np.einsum("rnn->rn", eigen).real)
        return eigen * (keep[:, :, None] & keep[:, None, :]), free

    # -- depth evolution -----------------------------------------------------------

    def _strang_step(
        self, ops: _JumpOperators, half: Any, rho: Any, dz: float
    ) -> tuple[Any, Any, Any, Any]:
        """One Strang step on ``ops.xp``: ``(rho, source, free, overflow)``.

        ``source`` is the step's incoming jump density in the slot basis before
        the kernel scale; ``free`` is ``(n_cells, N)`` and ``overflow`` a 0-d
        array, both still on ``ops.xp``.

        Validation: cr-density-matrix-split-operator
        """
        rho = _sandwich(half, rho)
        first, free1, band1 = self._jump(ops, rho)
        second, free2, band2 = self._jump(ops, first)
        loss1, loss2 = self._loss_rate(ops, rho), self._loss_rate(ops, first)
        source = dz * rho + 0.5 * dz**2 * first
        rho = rho + dz * first + 0.5 * dz**2 * second
        free = dz * free1 + 0.5 * dz**2 * free2
        overflow = dz * (loss1 - band1) + 0.5 * dz**2 * (loss2 - band2)
        rho = _sandwich(half, rho)
        return 0.5 * (rho + _adjoint(rho)), source, free, overflow

    def propagate(
        self,
        thickness_ang: float,
        dz_ang: float,
        *,
        incidence_rad: float = 0.0,
        divergence_rad: float = 0.0,
        stopping_eV_per_ang: float = 0.0,
        reprojection_tolerance: float = 1e-4,
        record_every: int = 1,
        energy_edges_eV: np.ndarray | None = None,
        initial: tuple[np.ndarray, float] | None = None,
        spatial_x_ang: np.ndarray | None = None,
        record_metrics: bool = False,
        jump_source_observer: Callable[[np.ndarray], None] | None = None,
        backend: Literal["cpu", "cuda"] = "cpu",
    ) -> DensityMatrixResult:
        """Integrate the entry state to ``thickness_ang`` with steps ``dz_ang``.

        ``stopping_eV_per_ang`` is a constant continuous energy loss. The basis
        is re-projected whenever ``gamma`` drifts by ``reprojection_tolerance``
        (relative) and the kernel is rescaled by ``(beta_0 / beta)^2``, exact
        for TDS and the leading term for electronic excitation. ``initial``
        replaces the entry state by ``(rho_S, entry_flux)`` in the entry
        eigenbasis.

        ``spatial_x_ang`` opts into cell-density samples at ``record_every``
        steps, including entrance and exit. Reconstruction uses the current
        basis after any energy reprojection; no matrix history is retained.
        Storage is ``O(n_recorded * n_x)``; default integration is unchanged.
        ``record_metrics`` additionally records exact retained-ensemble purity,
        instantaneous Bloch-basis squared coherence and transverse H moments,
        plus all retained band populations. Zero-trace conditional metrics are NaN.

        Validation: cr-quantum-ensemble-metrics

        ``jump_source_observer`` optionally receives each step's integrated
        incoming jump density in the plane basis, including the kernel's
        energy-rescaling factor. Its quadrature is exactly the one used to
        book free/overflow flux: ``dz rho_half + dz² J_S(rho_half)/2``.
        Summing these arrays permits a linear differential-kernel diagnostic
        without retaining a density history. The callback receives a new array.

        ``backend="cuda"`` runs the depth step on the CuPy device (CUDA or ROCm;
        never a silent CPU fallback) in fp64. The entry state, recording,
        reprojection and the result stay on the host; the density crosses to
        the host only on record and reprojection steps, and on every step when
        ``jump_source_observer`` is set. Results agree with ``"cpu"`` to
        rounding (#1125).

        Validation: cr-density-matrix-split-operator, cr-dechanneling-collapse,
        cr-planar-overflow-spectrum
        """
        xp = _array_namespace(backend)
        if thickness_ang <= 0.0 or dz_ang <= 0.0:
            raise ValueError("thickness_ang and dz_ang must be positive")
        if record_every < 1 or int(record_every) != record_every:
            raise ValueError("record_every must be a positive integer")
        spatial_x = (
            None if spatial_x_ang is None else np.array(spatial_x_ang, dtype=float, copy=True)
        )
        metric_history = []
        spatial_history = [] if spatial_x is not None else None
        steps = max(1, int(round(thickness_ang / dz_ang)))
        dz = thickness_ang / steps
        self.restore_entry_basis()
        if initial is None:
            rho, free, outside = self.entry_state(incidence_rad, divergence_rad)
            entry = float(free.sum()) + outside
        else:
            rho, entry = np.array(initial[0], dtype=complex), float(initial[1])
        if energy_edges_eV is None:
            top = float(self.basis.energies.max())
            energy_edges_eV = np.linspace(self.cutoff_eV, top, 65)
        edges = np.asarray(energy_edges_eV, dtype=float)
        by_energy = np.zeros(edges.size - 1)

        def bin_free(weights: np.ndarray) -> None:
            np.add.at(
                by_energy,
                np.clip(
                    np.searchsorted(edges, self.basis.energies, "right") - 1, 0, by_energy.size - 1
                ),
                weights,
            )

        energy, hbar_v = self.energy_eV, self.hbar_v0
        gamma_basis = self.gamma0
        half = self._propagators(dz, hbar_v)
        totals = {"free": 0.0, "overflow": 0.0, "reprojection": 0.0}
        history: dict[str, list] = {
            key: []
            for key in ("z", "E", "bands", "bound", "trace", "free", "overflow", "reprojection")
        }

        def record(z: float, rho: np.ndarray) -> None:
            if record_metrics:
                metric_history.append(density_metrics(rho, self.basis.energies))
            if spatial_history is not None and spatial_x is not None:
                spatial_history.append(
                    spatial_density(
                        spatial_x,
                        rho,
                        self.basis.vectors,
                        self.orders,
                        self.kappa,
                        self.profile.period_ang,
                    )
                )
            history["z"].append(z)
            history["E"].append(energy)
            history["bands"].append(self.band_populations(rho))
            history["bound"].append(self.bound_population(rho))
            history["trace"].append(float(np.einsum("rnn->", rho).real))
            for key in ("free", "overflow", "reprojection"):
                history[key].append(totals[key])

        # Free and overflow flux accumulate on ``xp`` and reach ``totals`` (and the
        # E_perp bins of the basis they were booked in) only when flushed.
        pending = {"free": xp.zeros((self.n_cells, self.orders.size)), "overflow": 0.0}

        def flush() -> None:
            weights = _host(pending["free"])
            totals["free"] += float(weights.sum())
            totals["overflow"] += float(pending["overflow"])
            bin_free(weights)
            pending["free"], pending["overflow"] = xp.zeros_like(pending["free"]), 0.0

        record(0.0, rho)
        # The no-jump propagator is zero outside S, so the step runs on the slots
        # holding a subspace state; ``rho`` stays restricted to them until output.
        host_ops = self._operators(restricted=True)
        ops = host_ops.on(xp)
        half, rho = xp.asarray(host_ops.restrict(half)), xp.asarray(host_ops.restrict(rho))
        for step in range(1, steps + 1):
            rho, source, free_step, overflow_step = self._strang_step(ops, half, rho, dz)
            if jump_source_observer is not None:
                jump_source_observer(_sandwich(host_ops.vectors, _host(source)) * self._scale)
            pending["free"] = pending["free"] + free_step
            pending["overflow"] = pending["overflow"] + overflow_step
            if stopping_eV_per_ang:
                energy = self.energy_eV - stopping_eV_per_ang * step * dz
                gamma, beta = longitudinal_kinematics(energy)
                if abs(gamma / gamma_basis - 1.0) > reprojection_tolerance:
                    flush()
                    before = float(xp.einsum("rnn->", rho).real)
                    full, lost = self._reproject(host_ops.expand(_host(rho)), gamma)
                    bin_free(lost)
                    totals["reprojection"] += before - float(np.einsum("rnn->", full).real)
                    gamma_basis, hbar_v = gamma, HBARC_EV_ANG * beta
                    self._scale = (self.beta0 / beta) ** 2
                    host_ops = self._operators(restricted=True)
                    ops = host_ops.on(xp)
                    half = xp.asarray(host_ops.restrict(self._propagators(dz, hbar_v)))
                    rho = xp.asarray(host_ops.restrict(full))
            if step % record_every == 0 or step == steps:
                flush()
                record(step * dz, host_ops.expand(_host(rho)))
        rho = host_ops.expand(_host(rho))
        bands = np.zeros((len(history["bands"]), self.orders.size))
        for row, populations in enumerate(history["bands"]):
            bands[row, : populations.size] = populations
        width = int(np.max(np.count_nonzero(self.basis.energies < self.barrier_eV, axis=1)))
        return DensityMatrixResult(
            z_ang=np.asarray(history["z"]),
            kinetic_energy_eV=np.asarray(history["E"]),
            band_populations=bands[:, : max(width, 1)],
            bound_population=np.asarray(history["bound"]),
            trace=np.asarray(history["trace"]),
            free_flux=np.asarray(history["free"]),
            overflow_flux=np.asarray(history["overflow"]),
            reprojection_flux=np.asarray(history["reprojection"]),
            entry_flux=entry,
            free_flux_by_energy=by_energy,
            energy_edges_eV=edges,
            barrier_eV=self.barrier_eV,
            cutoff_eV=self.cutoff_eV,
            final_density=rho,
            metrics=None
            if not record_metrics
            else EnsembleMetrics(
                purity=np.asarray(metric_history)[:, 0],
                coherence_squared=np.asarray(metric_history)[:, 1],
                energy_mean_eV=np.asarray(metric_history)[:, 2],
                energy_std_eV=np.asarray(metric_history)[:, 3],
                purity_probe_sem=np.zeros(len(metric_history)),
                basis="instantaneous Bloch eigenbasis",
            ),
            retained_band_populations=bands if record_metrics else None,
            spatial_x_ang=spatial_x,
            spatial_density_per_ang=None
            if spatial_history is None
            else np.asarray(spatial_history),
        )
