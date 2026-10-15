"""Transverse quantum states of a planar-channeled electron.

Dropping terms of order ``(U/E)^2`` and the spin coupling from the squared
Dirac equation separates the longitudinal motion; the transverse
wavefunction then obeys

``[-(hbar^2 / (2 gamma m)) d^2/dx^2 + U(x)] psi = E_perp psi``

(Andersen, Bonderup, Laegsgaard & Sorensen 1983). For a periodic ``U`` the
solutions are Bloch waves ``psi_k(x) = d^{-1/2} sum_m c_m exp(i (k + m g) x)``;
the plane-wave (many-beam) matrix is

``H_{mm'} = (hbar^2 / (2 gamma m)) (k + m g)^2 delta_{mm'} + U_{m - m'}``.

Energies are relative to the mean inner potential (``U_0 = 0``). A state is
bound when its energy lies below the potential maximum (the barrier).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..crystal import HBARC_EV_ANG, M_E_EV
from .potential import PlanarProfile

# hbar^2 / (2 m_e) [eV angstrom^2]
HBAR2_2M_EV_ANG2 = HBARC_EV_ANG**2 / (2.0 * M_E_EV)
DEFAULT_PLANE_WAVES = 32


@dataclass(frozen=True)
class TransverseStates:
    """Bloch eigenstates of one planar profile at one Lorentz factor and ``k``.

    Attributes
    ----------
    profile
        The planar potential.
    gamma
        Lorentz factor of the longitudinal motion.
    k_inv_ang
        Bloch wavevector [1/angstrom], in ``(-g/2, g/2]``.
    orders
        Plane-wave orders ``m = -M..M``.
    energies_eV
        Ascending transverse energies [eV].
    coefficients
        Plane-wave amplitudes ``c_m``; column ``n`` is state ``n`` (unit norm).
    barrier_eV
        Maximum of ``U(x)`` [eV]; states below it are bound.
    """

    profile: PlanarProfile
    gamma: float
    k_inv_ang: float
    orders: np.ndarray
    energies_eV: np.ndarray
    coefficients: np.ndarray
    barrier_eV: float

    @property
    def bound(self) -> np.ndarray:
        """Boolean mask of states below the potential barrier."""
        return self.energies_eV < self.barrier_eV

    @property
    def n_bound(self) -> int:
        """Number of bound states at this ``k``."""
        return int(self.bound.sum())

    def wavefunction(self, x_ang, state: int):
        """Return ``psi_n(x)`` normalized to unit probability per period.

        Validation: cr-transverse-eigenproblem
        """
        x = np.asarray(x_ang, dtype=float)
        wave = self.k_inv_ang + self.orders * self.profile.g_inv_ang
        basis = np.exp(1j * np.multiply.outer(x, wave)) / np.sqrt(self.profile.period_ang)
        return basis @ self.coefficients[:, state]

    def parity(self, state: int, atol: float = 1e-8) -> int:
        """Return ``+1``/``-1`` for even/odd states at ``k = 0``, else ``0``.

        Parity is defined about ``x = 0`` and exists only for a potential that
        is symmetric there and for ``k = 0``.
        """
        if self.k_inv_ang != 0.0:
            return 0
        c = self.coefficients[:, state]
        if np.allclose(c, c[::-1], atol=atol):
            return 1
        if np.allclose(c, -c[::-1], atol=atol):
            return -1
        return 0

    def dipole_ang(self, final: int, initial: int) -> complex:
        """Return the transverse dipole matrix element ``<f|x|i>`` [angstrom].

        Uses ``[H, x] = -i hbar p / (gamma m)``, so ``x_fi = -i (hbar^2 /
        (gamma m)) sum_m c_f* (k + m g) c_i / (E_f - E_i)``. It is exact for
        Bloch states at the same ``k`` and avoids the ill-defined position
        operator on periodic functions. The phase follows the arbitrary
        eigenvector gauge; rates use ``|x_fi|^2``.

        Validation: cr-dipole-matrix-element
        """
        delta = self.energies_eV[final] - self.energies_eV[initial]
        if final == initial or abs(delta) < 1e-12:
            raise ValueError("dipole matrix element needs distinct, nondegenerate states")
        wave = self.k_inv_ang + self.orders * self.profile.g_inv_ang
        momentum = np.vdot(self.coefficients[:, final], wave * self.coefficients[:, initial])
        return complex(-1j * 2.0 * HBAR2_2M_EV_ANG2 / self.gamma * momentum / delta)


def _hamiltonian(profile: PlanarProfile, gamma: float, k: float, orders: np.ndarray):
    g = profile.g_inv_ang
    kinetic = HBAR2_2M_EV_ANG2 / gamma * (k + orders * g) ** 2
    size = orders.size
    coeff = profile.potential_eV
    # U_{m-m'} with U_{-j} = conj(U_j); harmonics beyond the profile are zero.
    lag = orders[:, None] - orders[None, :]
    padded = np.zeros(2 * size, dtype=complex)
    count = min(coeff.size, padded.size)
    padded[:count] = coeff[:count]
    u = np.where(lag >= 0, padded[np.abs(lag)], np.conj(padded[np.abs(lag)]))
    return u + np.diag(kinetic)


def transverse_states(
    profile: PlanarProfile,
    gamma: float,
    k_inv_ang: float = 0.0,
    n_plane_waves: int = DEFAULT_PLANE_WAVES,
) -> TransverseStates:
    """Solve the transverse Bloch eigenproblem for one ``k``.

    Parameters
    ----------
    profile
        Planar continuum potential.
    gamma
        Lorentz factor; the transverse kinetic term uses mass ``gamma m``.
    k_inv_ang
        Bloch wavevector [1/angstrom]; folded into ``(-g/2, g/2]``.
    n_plane_waves
        Highest plane-wave order ``M``; the basis has ``2M + 1`` waves.

    Returns
    -------
    TransverseStates
        Ascending eigenstates with their coefficients and barrier.

    Validation: cr-transverse-eigenproblem
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    g = profile.g_inv_ang
    k = float(k_inv_ang) - g * np.round(float(k_inv_ang) / g)
    if np.isclose(k, -0.5 * g):
        k = 0.5 * g
    orders = np.arange(-n_plane_waves, n_plane_waves + 1)
    energies, vectors = np.linalg.eigh(_hamiltonian(profile, gamma, k, orders))
    _, barrier = profile.extrema()
    return TransverseStates(
        profile=profile,
        gamma=float(gamma),
        k_inv_ang=k,
        orders=orders,
        energies_eV=energies,
        coefficients=vectors,
        barrier_eV=barrier,
    )


def longitudinal_kinematics(kinetic_energy_eV: float) -> tuple[float, float]:
    """Return ``(gamma, beta)`` for an electron of the given kinetic energy."""
    gamma = 1.0 + kinetic_energy_eV / M_E_EV
    return gamma, float(np.sqrt(1.0 - 1.0 / gamma**2))


def entry_populations(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    incidence_rad,
    *,
    divergence_rad: float = 0.0,
    n_quadrature: int = 24,
    n_plane_waves: int = DEFAULT_PLANE_WAVES,
) -> np.ndarray:
    """Return the entry populations of transverse states for a tilted plane wave.

    An electron entering at angle ``psi`` to the plane carries transverse
    wavevector ``k_x = p psi / hbar``. Its projection onto the Bloch states at
    the folded ``k`` selects the single plane-wave order ``m0`` with
    ``k + m0 g = k_x``, so ``P_n = |c_n(m0)|^2`` and ``sum_n P_n = 1`` for a
    complete basis (Andersen et al. 1983; Azadegan 2013, Eq. 17). Averaging over
    the transverse entry point is implicit in the Bloch normalization. A
    Gaussian beam divergence ``sigma`` is integrated by Gauss--Hermite
    quadrature over ``psi``. Surface refraction is not applied here: pass the
    ``effective_rad`` of :func:`~pyrite.materials.channeling.entry.plane_incidence`,
    which carries the refracted transverse wavevector. Plane-wave orders beyond
    ``n_plane_waves`` contribute nothing, so ``sum_n P_n < 1`` when
    ``|k_x| > (M + 1/2) g``; raise ``n_plane_waves`` for large angles.

    The band index ``n`` labels states in energy order at each ``k``; deeply
    bound bands are flat, so it identifies the same physical state.

    Parameters
    ----------
    profile
        Planar continuum potential.
    kinetic_energy_eV
        Electron kinetic energy [eV].
    incidence_rad
        Scalar or array mean incidence angle to the plane [rad].
    divergence_rad
        RMS Gaussian divergence in the plane-normal direction [rad].
    n_quadrature
        Gauss--Hermite nodes for the divergence average.
    n_plane_waves
        Plane-wave order ``M`` of each eigensolve.

    Returns
    -------
    numpy.ndarray
        Populations shaped ``incidence.shape + (2M + 1,)``, in energy order.

    Validation: cr-initial-occupation
    """
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    incidence = np.asarray(incidence_rad, dtype=float)
    if divergence_rad > 0.0:
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
        angles = incidence[..., None] + divergence_rad * nodes
        weights = weights / weights.sum()
    else:
        angles = incidence[..., None]
        weights = np.ones(1)
    out = np.zeros(incidence.shape + (2 * n_plane_waves + 1,))
    g = profile.g_inv_ang
    for index in np.ndindex(angles.shape):
        k_x = p_inv_ang * angles[index]
        m0 = int(np.round(k_x / g))
        if abs(m0) > n_plane_waves:
            continue
        states = transverse_states(profile, gamma, k_x, n_plane_waves)
        # The fold can move k to +g/2, which shifts the matching order by one.
        m_match = int(np.round((k_x - states.k_inv_ang) / g))
        row = np.abs(states.coefficients[m_match + n_plane_waves, :]) ** 2
        out[index[:-1]] += weights[index[-1]] * row
    return out


def doppler_denominator(kinetic_energy_eV, theta_rad=0.0):
    """Return ``1 - beta cos(theta)`` in cancellation-free form.

    Uses ``1 / (gamma^2 (1 + beta)) + 2 beta sin^2(theta / 2)``, algebraically
    equal to ``1 - beta cos theta`` and exact at forward angles for any
    ``gamma``. It is the same expression as the PXR/CBS vacuum resonance in
    :func:`pyrite.materials.emission.photon_energies`.

    Validation: cr-doppler-line-energy
    """
    energy = np.asarray(kinetic_energy_eV, dtype=float)
    inverse_gamma = M_E_EV / (M_E_EV + energy)
    beta = np.sqrt(1.0 - inverse_gamma**2)
    return inverse_gamma**2 / (1.0 + beta) + 2.0 * beta * np.sin(np.asarray(theta_rad) / 2) ** 2


def line_energy_eV(delta_transverse_eV, kinetic_energy_eV, theta_rad=0.0):
    """Return the lab photon energy of a transverse transition.

    ``hbar omega = Delta E_perp / (1 - beta cos theta)``, where ``theta`` is the
    photon angle to the electron velocity: the Doppler denominator ``1 - n . v``
    shared with the PXR/CBS resonance (``line-energy-dispersion``). Forward
    emission gives ``gamma^2 (1 + beta) Delta E_perp``. Photon recoil is
    neglected; its relative correction is ``O(hbar omega / E)``, below
    ``2.5e-3`` at 10 MeV. ``beta`` is the total speed; the longitudinal speed
    differs by ``O(E_perp / (p v)) < 1e-5``.

    Validation: cr-doppler-line-energy
    """
    return np.asarray(delta_transverse_eV, dtype=float) / doppler_denominator(
        kinetic_energy_eV, theta_rad
    )
