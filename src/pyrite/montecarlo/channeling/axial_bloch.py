"""2D transverse Bloch states of an axially channeled electron (experimental, #508).

The paraxial transverse Hamiltonian ``-(hbar^2 / 2 gamma m) nabla_perp^2 + U(r)``
of an axial string potential (:mod:`pyrite.materials.channeling.axial`) has
Bloch solutions ``psi_k(r) = A^{-1/2} sum_G c_G exp(i (k + G) . r)``;
:func:`axial_states` diagonalizes the plane-wave matrix as
:func:`pyrite.montecarlo.channeling.bloch.transverse_states` does in 1D. The
solve depends on the beam Lorentz factor, so it lives here rather than with
the crystal-only potential (#1147).

Lengths in angstroms, energies in eV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.channeling.axial import AxialProfile
from ...materials.crystal import HBAR2_2M_EV_ANG2


@dataclass(frozen=True)
class AxialStates:
    """Plane-wave eigenstates of one axial profile at one Bloch ``k``.

    Attributes
    ----------
    profile
        The string potential.
    gamma
        Lorentz factor (transverse mass ``gamma m``).
    k_inv_ang
        Bloch wavevector ``(k_x, k_y)`` [1/angstrom].
    orders
        Plane-wave orders ``(m, n)`` of the basis, shape ``(n_basis, 2)``.
    energies_eV
        Ascending transverse energies [eV].
    coefficients
        Eigenvectors as columns, shape ``(n_basis, n_basis)``.
    """

    profile: AxialProfile
    gamma: float
    k_inv_ang: tuple[float, float]
    orders: np.ndarray
    energies_eV: np.ndarray
    coefficients: np.ndarray

    def wavefunction(self, x_ang, y_ang, state: int) -> np.ndarray:
        """Return ``psi(x, y)`` of one state, normalized over one cell."""
        gx, gy = self.profile.g_inv_ang
        kx = self.k_inv_ang[0] + self.orders[:, 0] * gx
        ky = self.k_inv_ang[1] + self.orders[:, 1] * gy
        ex = np.exp(1j * np.multiply.outer(np.asarray(x_ang, float), kx))
        ey = np.exp(1j * np.multiply.outer(np.asarray(y_ang, float), ky))
        c = self.coefficients[:, state] / np.sqrt(self.profile.area_ang2)
        return np.einsum("ib,b,jb->ij", ex, c, ey)


def axial_states(
    profile: AxialProfile,
    gamma: float,
    k_inv_ang: tuple[float, float] = (0.0, 0.0),
    max_order: tuple[int, int] | int = 8,
) -> AxialStates:
    """Solve the 2D transverse Bloch eigenproblem at one ``k``.

    Parameters
    ----------
    profile
        Axial string potential.
    gamma
        Lorentz factor; the kinetic term uses mass ``gamma m``.
    k_inv_ang
        Bloch wavevector ``(k_x, k_y)`` [1/angstrom].
    max_order
        Highest plane-wave order per axis; the basis has
        ``(2 M_x + 1)(2 M_y + 1)`` waves.

    Returns
    -------
    AxialStates
        Ascending eigenstates.

    Validation: cr-axial-2d-eigenproblem
    """
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    if isinstance(max_order, int):
        mx = my = max_order
    else:
        mx, my = (int(v) for v in max_order)
    om, on = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
    orders = np.stack([om.ravel(), on.ravel()], 1)
    gx, gy = profile.g_inv_ang
    kx = k_inv_ang[0] + orders[:, 0] * gx
    ky = k_inv_ang[1] + orders[:, 1] * gy
    pm, pn = profile.max_order
    dm = orders[:, None, 0] - orders[None, :, 0]
    dn = orders[:, None, 1] - orders[None, :, 1]
    inside = (np.abs(dm) <= pm) & (np.abs(dn) <= pn)
    u = np.where(
        inside, profile.potential_eV[np.clip(dm + pm, 0, 2 * pm), np.clip(dn + pn, 0, 2 * pn)], 0.0
    )
    hamiltonian = u + np.diag(HBAR2_2M_EV_ANG2 / gamma * (kx**2 + ky**2))
    energies, vectors = np.linalg.eigh(hamiltonian)
    return AxialStates(
        profile=profile,
        gamma=float(gamma),
        k_inv_ang=(float(k_inv_ang[0]), float(k_inv_ang[1])),
        orders=orders,
        energies_eV=energies,
        coefficients=vectors,
    )


__all__ = ["AxialStates", "axial_states"]
