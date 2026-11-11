"""Angular distribution of electrons leaving a planar-channeling crystal.

The transverse wavefunction at the exit face ``z = L`` decomposes into plane
waves; transverse momentum is conserved across the face, so each component
``k_x`` leaves at the angle ``theta_x = k_x / p``. The far-field distribution is
the transverse momentum distribution at ``L`` (Shul'ga & Shulga, Phys. Lett. B
769, 141 (2017), Eqs. 13--14; Shulga et al., NIMB 402, 16 (2017), Eqs. 6--9):

``dw/dtheta_x = (p / 2 pi) |int dx e^{-i q_x x} u(x, L)|^2``.

That form is per unit transverse length; the functions here return
probabilities per incident electron.

**Coherent limit.** For a tilted plane wave ``k_x = p psi`` the state lives in
the single Bloch block ``k = k_x mod g`` and enters on plane-wave order ``m0``.
With ``psi_n = sum_m c_{mn} |k + m g>`` the exit amplitude on order ``m`` is

``a_m(L) = sum_n c_{mn} exp(-i E_n L / hbar v) conj(c_{m0 n})``,

and ``|a_m|^2`` is the weight at ``theta = (k + m g) / p``. Diffraction therefore
puts the beam on the comb ``psi + m lambda/d``. A Gaussian divergence is an
incoherent average over ``psi``.

**Open system.** For the translation-averaged density matrix of
:class:`~.density.DensityMatrixSolver` the blocks are Bloch-diagonal, so the
momentum distribution is the plane-wave diagonal ``<k|rho|k>`` of each block.
It contains the coherent part and the incoherent part redistributed by TDS and
electronic jumps. Population the solver has removed (free, overflow,
reprojection and entry flux) is not angle-resolved and is reported as
``unresolved``.

**Classical reference.** The deflection function ``theta(x0)`` of
``d^2x/dz^2 = -U'(x) / (p v)`` (Shulga et al. 2017, Eqs. 1--5) averaged
uniformly over the impact parameter ``x0`` gives the classical (rainbow)
distribution, the ``hbar -> 0`` limit at fixed ``L``.

Lengths in angstroms, energies in eV, angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.channeling.potential import PlanarProfile
from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .bloch import DEFAULT_PLANE_WAVES, longitudinal_kinematics, transverse_states
from .density import DensityMatrixSolver, _sandwich

#: Incidence nodes per divergence average: the comb lines are spread over
#: ``+-DIVERGENCE_SPAN`` standard deviations at this many uniform midpoints.
DEFAULT_INCIDENCE_NODES = 241
DIVERGENCE_SPAN = 4.0


def _momentum_inv_ang(kinetic_energy_eV: float) -> tuple[float, float]:
    """Return ``(p / hbar [1/angstrom], hbar v [eV angstrom])``."""
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    return gamma * beta * M_E_EV / HBARC_EV_ANG, HBARC_EV_ANG * beta


@dataclass(frozen=True)
class ExitAngularDistribution:
    """Exit angles as weighted point masses.

    Attributes
    ----------
    angles_rad
        Exit angle ``theta_x = k_x / p`` of each component [rad].
    weights
        Probability of each component; ``weights.sum() + unresolved = 1``.
    unresolved
        Weight without an exit angle: outside the plane-wave band, or removed
        by the density solver's dechanneling sink.
    order_spacing_rad
        Diffraction-comb spacing ``g / p = lambda / d`` [rad].

    Validation: cr-planar-exit-angular
    """

    angles_rad: np.ndarray
    weights: np.ndarray
    unresolved: float
    order_spacing_rad: float

    @property
    def total(self) -> float:
        """Resolved probability ``sum(weights)``."""
        return float(self.weights.sum())

    def moments(self) -> tuple[float, float]:
        """Return the resolved ``(mean, rms about zero)`` exit angle [rad]."""
        total = self.total
        mean = float(self.weights @ self.angles_rad) / total
        rms = float(np.sqrt(self.weights @ self.angles_rad**2 / total))
        return mean, rms

    def histogram(self, edges_rad) -> np.ndarray:
        """Return the density per radian on ``edges_rad`` (unit-normalized total)."""
        edges = np.asarray(edges_rad, dtype=float)
        counts, _ = np.histogram(self.angles_rad, bins=edges, weights=self.weights)
        return counts / np.diff(edges)

    def smoothed(self, theta_rad, resolution_rad: float) -> np.ndarray:
        """Return the density per radian convolved with a Gaussian resolution."""
        if resolution_rad <= 0.0:
            raise ValueError("resolution_rad must be positive")
        theta = np.asarray(theta_rad, dtype=float)
        offset = (np.multiply.outer(theta, np.ones_like(self.angles_rad)) - self.angles_rad) / (
            resolution_rad
        )
        kernel = np.exp(-0.5 * offset**2) / (np.sqrt(2.0 * np.pi) * resolution_rad)
        return kernel @ self.weights

    def order_fraction(self, window: float = 0.25) -> float:
        """Resolved fraction within ``+-window`` comb spacings of a diffraction order."""
        phase = self.angles_rad / self.order_spacing_rad
        near = np.abs(phase - np.round(phase)) <= window
        return float(self.weights[near].sum() / self.total)


def incidence_nodes(
    incidence_rad: float, divergence_rad: float, n_nodes: int = DEFAULT_INCIDENCE_NODES
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform midpoint nodes over ``+-4 sigma`` with normalized Gaussian weights.

    The comb lines of :func:`coherent_exit_distribution` inherit this spacing,
    so histogram bins should be wider than ``8 sigma / n_nodes``.

    Validation: cr-planar-exit-angular
    """
    if divergence_rad <= 0.0:
        return np.array([float(incidence_rad)]), np.ones(1)
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2 for a divergent beam")
    half = DIVERGENCE_SPAN * divergence_rad
    step = 2.0 * half / n_nodes
    angles = float(incidence_rad) - half + step * (np.arange(n_nodes) + 0.5)
    weights = np.exp(-0.5 * ((angles - incidence_rad) / divergence_rad) ** 2)
    return angles, weights / weights.sum()


def coherent_exit_distribution(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    thickness_ang: float,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    n_incidence: int = DEFAULT_INCIDENCE_NODES,
    n_plane_waves: int = DEFAULT_PLANE_WAVES,
) -> ExitAngularDistribution:
    """Return the coherent (scattering-free) exit distribution of a planar crystal.

    Each incidence angle ``psi`` is a plane wave ``k_x = p psi`` that evolves
    exactly in its Bloch block; the exit weights ``|a_m(L)|^2`` sit at
    ``(k + m g) / p``. Unitarity makes ``sum_m |a_m|^2 = 1`` whenever the entry
    order lies in the basis; entries outside it are ``unresolved``. A Gaussian
    divergence is integrated incoherently on :func:`incidence_nodes`, so the
    comb lines acquire the incident spread. Surface refraction is not applied:
    pass the refracted ``effective_rad`` of
    :func:`~pyrite.montecarlo.channeling.entry.plane_incidence`.

    Assumptions: continuum planar potential (no thermal or electronic
    scattering, no energy loss), transverse kinetic mass ``gamma m``, small
    angles. This is the Shul'ga & Shulga (2017) amplitude.

    Validation: cr-planar-exit-angular
    """
    if thickness_ang < 0.0:
        raise ValueError("thickness_ang must be non-negative")
    gamma, _ = longitudinal_kinematics(kinetic_energy_eV)
    p_inv_ang, hbar_v = _momentum_inv_ang(kinetic_energy_eV)
    g = profile.g_inv_ang
    psi, psi_weights = incidence_nodes(incidence_rad, divergence_rad, n_incidence)
    angles, weights = [], []
    unresolved = 0.0
    for angle, weight in zip(psi, psi_weights, strict=True):
        k_x = p_inv_ang * angle
        states = transverse_states(profile, gamma, k_x, n_plane_waves)
        m0 = int(np.round((k_x - states.k_inv_ang) / g))
        if abs(m0) > n_plane_waves:
            unresolved += weight
            continue
        c = states.coefficients
        phase = np.exp(-1j * states.energies_eV * thickness_ang / hbar_v)
        amplitude = c @ (phase * np.conj(c[m0 + n_plane_waves, :]))
        angles.append((states.k_inv_ang + states.orders * g) / p_inv_ang)
        weights.append(weight * np.abs(amplitude) ** 2)
    if not angles:
        return ExitAngularDistribution(np.zeros(0), np.zeros(0), unresolved, g / p_inv_ang)
    return ExitAngularDistribution(
        angles_rad=np.concatenate(angles),
        weights=np.concatenate(weights),
        unresolved=float(unresolved),
        order_spacing_rad=g / p_inv_ang,
    )


def density_exit_distribution(
    solver: DensityMatrixSolver, rho: np.ndarray, unresolved: float = 0.0
) -> ExitAngularDistribution:
    """Return the exit distribution of a Bloch-blocked density matrix.

    ``rho`` is a solver state (e.g. ``DensityMatrixResult.final_density``) in
    the eigenbasis of ``solver.basis``. The weight at ``(kappa_r + h g) / p`` is
    the plane-wave diagonal ``(V_r rho_r V_r^dagger)_{hh}``, with ``p`` at the
    basis Lorentz factor. Pass the solver's cumulative dechanneled flux as
    ``unresolved`` so that resolved plus unresolved weight sums to one up to
    the integrator's balance defect.

    Validation: cr-planar-exit-angular
    """
    rho = np.asarray(rho, dtype=complex)
    plane = _sandwich(solver.basis.vectors, rho)
    weights = np.einsum("rhh->rh", plane).real
    gamma = solver.basis.gamma
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    g = solver.profile.g_inv_ang
    k = solver.kappa[:, None] + solver.orders[None, :] * g
    return ExitAngularDistribution(
        angles_rad=(k / p_inv_ang).ravel(),
        weights=weights.ravel(),
        unresolved=float(unresolved),
        order_spacing_rad=g / p_inv_ang,
    )


def open_exit_distribution(
    solver: DensityMatrixSolver,
    thickness_ang: float,
    dz_ang: float,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    **propagate_kwargs,
) -> ExitAngularDistribution:
    """Propagate the divergent entry state to ``thickness_ang`` and return its exit distribution.

    Thermal and electronic scattering act through the solver's kernel; the
    divergence is resolved on its Bloch grid (``k_x = l g / n_cells``), so the
    angular step is ``lambda / (d n_cells)``. Extra keywords go to
    :meth:`~.density.DensityMatrixSolver.propagate`.

    Validation: cr-planar-exit-angular
    """
    result = solver.propagate(
        thickness_ang,
        dz_ang,
        incidence_rad=incidence_rad,
        divergence_rad=divergence_rad,
        **propagate_kwargs,
    )
    return density_exit_distribution(solver, result.final_density, float(result.total_flux[-1]))


def classical_exit_angles(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    thickness_ang: float,
    *,
    incidence_rad=0.0,
    n_impact: int = 400,
    dz_ang: float = 1.0,
    n_table: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the classical deflection function ``theta(x0)`` at depth ``thickness_ang``.

    Integrates ``d^2x/dz^2 = -U'(x) / (p v)`` from ``x0`` on ``n_impact``
    uniform midpoints of one period with ``dx/dz = psi`` for each incidence
    angle ``psi``, by velocity Verlet (symplectic, second order in ``dz``)
    with ``U'`` from its Fourier series tabulated on ``n_table`` periodic
    points. Returns ``(x0_ang, theta_rad)``, ``theta`` shaped
    ``incidence.shape + (n_impact,)``. A uniform average over ``x0`` (and the
    divergence) is the classical distribution, including rainbow peaks.

    Validation: cr-planar-exit-angular
    """
    if thickness_ang < 0.0 or dz_ang <= 0.0:
        raise ValueError("thickness_ang must be non-negative and dz_ang positive")
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    pv = gamma * M_E_EV * beta**2
    d, g = profile.period_ang, profile.g_inv_ang
    m = np.arange(1, profile.potential_eV.size)
    grid = np.arange(n_table) * d / n_table
    slope = (
        2.0
        * (
            np.exp(1j * g * np.multiply.outer(grid, m)) @ (1j * g * m * profile.potential_eV[1:])
        ).real
    )

    def accel(x):
        return -np.interp(x, grid, slope, period=d) / pv

    x0 = d * (np.arange(n_impact) + 0.5) / n_impact
    psi = np.asarray(incidence_rad, dtype=float)
    x = np.broadcast_to(x0, psi.shape + (n_impact,)).copy()
    theta = np.broadcast_to(psi[..., None], x.shape).copy()
    steps = int(np.ceil(thickness_ang / dz_ang))
    if steps:
        dz = thickness_ang / steps
        a = accel(x)
        for _ in range(steps):
            theta += 0.5 * dz * a
            x += dz * theta
            a = accel(x)
            theta += 0.5 * dz * a
    return x0, theta
