"""Out-of-plane-resolved planar exit distribution (#1109).

The planar density solver integrates the out-of-plane transfer ``q_y`` out of
its kernels, so :func:`~.exit_angular.open_exit_distribution` returns the
``theta_x`` marginal. A film records the two-dimensional pattern, where a spot
at ``(m lambda/d, 0)`` sits on a background that the out-of-plane kicks also
spread in ``theta_y``. Spot visibility is a property of the 2D density, not of
the marginal.

**Out-of-plane label.** ``H_perp`` does not depend on ``kappa_y``, and every
jump adds its ``q_y`` to it (``cr-planar-fiber-unraveling``). The
``kappa_y``-resolved state therefore obeys
``d rho_k/dz = L0 rho_k + int d q_y J_{q_y}(rho_{k - q_y})``, with ``L0`` the
no-jump part. Fourier transforming ``kappa_y -> eta`` gives, for each
``eta``, the planar master equation with the jump weights multiplied by
``cos(q_y eta)`` (the kernels are even in ``q_y``) and the loss unchanged:
the ``out_of_plane_ang`` option of :func:`~.operators.tds_kernel` and
:func:`~.operators.electronic_kernel`. A Gaussian out-of-plane divergence
``sigma_y`` multiplies the transform by ``exp(-(p sigma_y eta)^2 / 2)``, and

``dP/dtheta_y (theta_x, theta_y) = (p / pi) int_0^inf d eta
cos(p theta_y eta) exp(-(p sigma_y eta)^2 / 2) P~(theta_x, eta)``.

``eta = 0`` is the marginal. The ``eta`` integral uses composite
Gauss--Legendre panels, fine near 0, where ``P~`` falls on the scale of the
largest frequent kicks.

The cosine transform supports thermal covariances with ``S_xy = 0``.
Nonzero cross covariances also need an odd sine transform and are refused.

**Batched integrator.** :func:`propagate_batch` advances one entry state
under many cell kernels with the Strang step of
:meth:`~.density.DensityMatrixSolver.propagate` (no stopping, so no
reprojection). It is written for an array module ``xp`` (NumPy or CuPy), so
the same code runs on the CPU and on a CUDA device.

Lengths in angstroms, energies in eV, angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np

from ...materials.channeling.potential import PlanarProfile
from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .bloch import longitudinal_kinematics
from .density import DensityMatrixSolver
from .exit_angular import ExitAngularDistribution
from .operators import DEFAULT_CELLS, ScatteringKernel, electronic_kernel, tds_kernel

#: Plane-wave orders kept by default: the Gridnev patterns need |m| <= 3.
DEFAULT_EXIT_ORDERS = 8
#: Panel-width scales [angstrom] on [0, 1), [1, 4) and [4, eta_max].
DEFAULT_ETA_STEPS = (0.05, 0.2, 1.0)
#: ``eta_max = ETA_SPAN / (p sigma_y)``: the Gaussian factor is e^-12.5 there.
ETA_SPAN = 5.0
MECHANISMS = ("tds", "electronic")


def eta_nodes(
    divergence_y_rad: float,
    kinetic_energy_eV: float,
    steps_ang: tuple[float, float, float] = DEFAULT_ETA_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``eta`` nodes [angstrom] and weights for ``int_0^eta_max d eta``.

    Composite 4-point Gauss--Legendre panels of width ``4 step`` on
    ``[0, 1]``, ``[1, 4]`` and ``[4, eta_max]``; the last step is capped at a
    quarter of the Gaussian width ``1 / (p sigma_y)``. Node 0 is ``eta = 0``
    with zero weight: it carries the marginal.

    Validation: cr-planar-exit-out-of-plane
    """
    if divergence_y_rad <= 0.0:
        raise ValueError("divergence_y_rad must be positive")
    fine, mid, coarse = (float(step) for step in steps_ang)
    if min(fine, mid, coarse) <= 0.0:
        raise ValueError("eta steps must be positive")
    width = 1.0 / (_momentum_inv_ang(kinetic_energy_eV) * divergence_y_rad)
    top = max(ETA_SPAN * width, 4.0)
    coarse = min(coarse, 0.25 * width)
    edges = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, max(1, int(np.ceil(0.25 / fine))) + 1),
                np.linspace(1.0, 4.0, max(1, int(np.ceil(0.75 / mid))) + 1),
                np.linspace(4.0, top, max(1, int(np.ceil((top - 4.0) / (4.0 * coarse)))) + 1),
            ]
        )
    )
    x, w = np.polynomial.legendre.leggauss(4)
    half = 0.5 * np.diff(edges)[:, None]
    nodes = (edges[:-1, None] + half * (x + 1.0)).ravel()
    weights = (half * w).ravel()
    return np.concatenate([[0.0], nodes]), np.concatenate([[0.0], weights])


def out_of_plane_kernel(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    eta_ang: float,
    *,
    mechanisms: tuple[str, ...] = MECHANISMS,
    n_cells: int = DEFAULT_CELLS,
    max_order: int = DEFAULT_EXIT_ORDERS,
) -> ScatteringKernel:
    """Sum of the requested kernels at out-of-plane transform ``eta``.

    Validation: cr-planar-exit-out-of-plane
    """
    builders = {"tds": tds_kernel, "electronic": electronic_kernel}
    unknown = set(mechanisms) - set(builders)
    if not mechanisms or unknown:
        raise ValueError(f"mechanisms must be a non-empty subset of {MECHANISMS}")
    parts = [
        builders[name](
            profile,
            kinetic_energy_eV,
            n_cells=n_cells,
            max_order=max_order,
            out_of_plane_ang=eta_ang,
        )
        for name in mechanisms
    ]
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def propagate_batch(
    solver: DensityMatrixSolver,
    cell_kernels,
    rho,
    thickness_ang,
    dz_ang: float,
    *,
    xp: ModuleType = np,
) -> np.ndarray:
    """Integrate one entry state under a batch of cell kernels.

    ``cell_kernels`` has shape ``(B, n_cells, n_grid, n_grid)``
    (:meth:`~.operators.ScatteringKernel.cell_kernel` at ``solver.n_grid``);
    the no-jump propagator is the solver's, so every kernel must share its loss
    diagonal. ``rho`` is the solver entry state ``(n_cells, N, N)``.
    ``thickness_ang`` is one depth or an increasing sequence; each must be a
    multiple of the step. Returns the plane-wave diagonal weights
    ``(n_depths, B, n_cells, N)`` as NumPy, ordered like
    :func:`~.exit_angular.density_exit_distribution`.

    Evolution omits eigenstate slots outside the retained subspace in every
    residue; the output still covers the full plane-wave band.

    Validation: cr-planar-exit-out-of-plane, cr-density-matrix-split-operator
    """
    depths = np.atleast_1d(np.asarray(thickness_ang, dtype=float))
    if dz_ang <= 0.0 or np.any(depths <= 0.0) or np.any(np.diff(depths) <= 0.0):
        raise ValueError("dz_ang and depths must be positive and depths increasing")
    marks = np.rint(depths / dz_ang).astype(int)
    if not np.allclose(marks * dz_ang, depths, rtol=1e-9, atol=1e-9):
        raise ValueError("every depth must be a multiple of dz_ang")
    if solver.basis.gamma != solver.gamma0 or solver._scale != 1.0:
        raise ValueError("solver must be at its entry energy")
    dz = float(dz_ang)
    # After the first half-step, states and jump outputs lie in the solver's
    # retained subspace. Omit eigenstate slots inactive in every residue;
    # keep the per-residue mask for slots retained in only some residues.
    active = np.flatnonzero(solver.basis.subspace.any(axis=0))
    half = xp.ascontiguousarray(
        xp.asarray(solver._propagators(dz, solver.hbar_v0)[:, active][:, :, active])
    )
    vectors = xp.ascontiguousarray(xp.asarray(solver.basis.vectors[:, :, active]))
    waves = xp.asarray(solver._waves)
    keep = xp.asarray(solver.basis.subspace[:, active])
    mask = keep[:, :, None] & keep[:, None, :]
    kernels = xp.asarray(cell_kernels)
    if kernels.ndim != 4 or kernels.shape[1:] != (solver.n_cells, solver.n_grid, solver.n_grid):
        raise ValueError("cell_kernels must have shape (B, n_cells, n_grid, n_grid)")
    if not active.size:
        return np.zeros((depths.size, kernels.shape[0], solver.n_cells, solver.orders.size))
    to_grid = waves @ vectors  # (r, a, n)
    from_grid = xp.ascontiguousarray(xp.conj(xp.swapaxes(to_grid, -1, -2)))
    half_adjoint = xp.ascontiguousarray(xp.conj(xp.swapaxes(half, -1, -2)))
    vectors_adjoint = xp.ascontiguousarray(xp.conj(xp.swapaxes(vectors, -1, -2)))
    norm = float(solver.n_grid) ** 2

    def sandwich(outer, inner, adjoint):
        return outer @ inner @ adjoint

    def jump(state):
        grid = sandwich(to_grid, state, from_grid)
        back = xp.fft.ifft(xp.fft.fft(grid, axis=1) * kernels, axis=1)
        eigen = from_grid @ back @ to_grid / norm
        return eigen * mask

    retained = xp.asarray(rho)[:, active][:, :, active]
    state = xp.broadcast_to(retained, (kernels.shape[0],) + retained.shape).copy()
    out = []
    for step in range(1, int(marks[-1]) + 1):
        state = sandwich(half, state, half_adjoint)
        first = jump(state)
        state = state + dz * first + 0.5 * dz**2 * jump(first)
        state = sandwich(half, state, half_adjoint)
        state = 0.5 * (state + xp.conj(xp.swapaxes(state, -1, -2)))
        if step in marks:
            plane = sandwich(vectors, state, vectors_adjoint)
            weights = xp.real(xp.diagonal(plane, axis1=-2, axis2=-1))
            out.append(weights.get() if hasattr(weights, "get") else np.asarray(weights))
    return np.stack(out)


@dataclass(frozen=True)
class ExitAngularCut:
    """Exit density along ``theta_x`` at fixed ``theta_y``.

    Attributes
    ----------
    angles_rad
        ``theta_x`` of each component [rad] (the solver's Bloch grid).
    density_per_rad
        ``dP/dtheta_y`` carried by each ``theta_x`` component at
        ``theta_y_rad`` [1/rad].
    theta_y_rad
        Out-of-plane angle of the cut [rad].
    marginal
        The ``theta_y``-integrated distribution (the ``eta = 0`` solve).

    Validation: cr-planar-exit-out-of-plane
    """

    angles_rad: np.ndarray
    density_per_rad: np.ndarray
    theta_y_rad: float
    marginal: ExitAngularDistribution

    @property
    def order_spacing_rad(self) -> float:
        """Diffraction-comb spacing ``lambda / d`` [rad]."""
        return self.marginal.order_spacing_rad

    def local_contrast(self, order: int, inner: float = 0.15, outer: float = 0.35) -> float:
        """Cut at order ``m`` over its mean on ``inner < |theta_x/s - m| < outer``.

        Validation: cr-planar-exit-out-of-plane
        """
        return local_contrast(
            self.angles_rad, self.density_per_rad, self.order_spacing_rad, order, inner, outer
        )


def local_contrast(angles_rad, weights, spacing_rad, order, inner=0.15, outer=0.35) -> float:
    """Grid value at ``theta = m s`` over the mean of grid values in the ring.

    The components sit on a uniform grid, so per-component weights are
    proportional to densities and no binning is needed.

    Validation: cr-planar-exit-out-of-plane
    """
    if not 0.0 <= inner < outer <= 0.5:
        raise ValueError("need 0 <= inner < outer <= 0.5")
    phase = np.asarray(angles_rad, dtype=float) / spacing_rad - order
    weights = np.asarray(weights, dtype=float)
    centre = weights[np.argmin(np.abs(phase))]
    ring = (np.abs(phase) > inner) & (np.abs(phase) < outer)
    if not ring.any():
        raise ValueError("ring holds no grid point; raise n_cells")
    return float(centre / weights[ring].mean())


def exit_cut(
    solver: DensityMatrixSolver,
    weights: np.ndarray,
    eta_ang: np.ndarray,
    eta_weights: np.ndarray,
    divergence_y_rad: float,
    theta_y_rad: float = 0.0,
) -> ExitAngularCut:
    """Assemble the cut from per-``eta`` diagonals ``weights`` ``(B, n_cells, N)``.

    ``eta_ang[0]`` must be 0 (the marginal; its weight is ignored). The marginal's unresolved weight
    is ``1 - total``, which includes the integrator's balance defect.

    Validation: cr-planar-exit-out-of-plane
    """
    eta = np.asarray(eta_ang, dtype=float)
    if eta[0] != 0.0 or weights.shape[0] != eta.size:
        raise ValueError("weights must follow eta_ang, starting at eta = 0")
    p = _momentum_inv_ang(solver.energy_eV)
    factor = (
        np.asarray(eta_weights)
        * np.cos(p * theta_y_rad * eta)
        * np.exp(-0.5 * (p * divergence_y_rad * eta) ** 2)
    )
    density = p / np.pi * np.tensordot(factor, weights, axes=1)
    g = solver.profile.g_inv_ang
    angles = ((solver.kappa[:, None] + solver.orders[None, :] * g) / p).ravel()
    marginal = ExitAngularDistribution(
        angles_rad=angles,
        weights=weights[0].ravel(),
        unresolved=float(1.0 - weights[0].sum()),
        order_spacing_rad=g / p,
    )
    return ExitAngularCut(marginal.angles_rad, density.ravel(), float(theta_y_rad), marginal)


def out_of_plane_exit_cut(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    thickness_ang: float,
    dz_ang: float,
    *,
    divergence_rad: float,
    divergence_y_rad: float | None = None,
    theta_y_rad: float = 0.0,
    mechanisms: tuple[str, ...] = MECHANISMS,
    n_cells: int = DEFAULT_CELLS,
    max_order: int = DEFAULT_EXIT_ORDERS,
    eta_steps_ang: tuple[float, float, float] = DEFAULT_ETA_STEPS,
    xp: ModuleType = np,
) -> ExitAngularCut:
    """Out-of-plane-resolved exit cut at normal incidence.

    ``divergence_rad`` is the in-plane Gaussian divergence (resolved on the
    Bloch grid); ``divergence_y_rad`` defaults to it. Builds one kernel per
    ``eta`` node on the CPU and integrates them as one batch on ``xp``.

    Validation: cr-planar-exit-out-of-plane
    """
    sigma_y = divergence_rad if divergence_y_rad is None else divergence_y_rad
    eta, eta_w = eta_nodes(sigma_y, kinetic_energy_eV, eta_steps_ang)
    base = out_of_plane_kernel(
        profile, kinetic_energy_eV, 0.0, mechanisms=mechanisms, n_cells=n_cells, max_order=max_order
    )
    solver = DensityMatrixSolver(profile, kinetic_energy_eV, base)
    kernels = np.stack(
        [solver._cell_kernel0]
        + [
            out_of_plane_kernel(
                profile,
                kinetic_energy_eV,
                value,
                mechanisms=mechanisms,
                n_cells=n_cells,
                max_order=max_order,
            ).cell_kernel(solver.n_grid)
            for value in eta[1:]
        ]
    )
    rho, _, _ = solver.entry_state(0.0, divergence_rad)
    weights = propagate_batch(solver, kernels, rho, thickness_ang, dz_ang, xp=xp)[-1]
    return exit_cut(solver, weights, eta, eta_w, sigma_y, theta_y_rad)


def _momentum_inv_ang(kinetic_energy_eV: float) -> float:
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    return gamma * beta * M_E_EV / HBARC_EV_ANG
