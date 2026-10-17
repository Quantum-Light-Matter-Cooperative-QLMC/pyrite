"""Orbital-angular-momentum projection about an atomic string (experimental, #508).

A vortex probe couples to the axial bound states of one string only when it
is centred on that string (proposal §9), so ``L_z`` is measured about a
chosen centre ``c``, not the beam axis. Inside a disk ``|r - c| <= R`` the
state splits into cylindrical harmonics,

``psi(c + rho e_theta) = sum_l a_l(rho) e^{i l theta} / sqrt(2 pi)``,

``a_l(rho) = (2 pi)^{-1/2} int_0^{2 pi} psi(c + rho e_theta) e^{-i l theta} d theta``,

and the disk weight carried by ``l`` is ``P_l = int_0^R |a_l|^2 rho d rho``.
``sum_l P_l = int_disk |psi|^2`` and ``<L_z>_disk = hbar sum_l l P_l``
(``L_z = -i hbar d/d theta`` is diagonal in ``l``). ``P_l`` is the
expectation of a projector, hence linear in the density matrix, so the
ensemble average of an MCWF run is the Lindblad ``P_l``.

``psi = e^{i k_t . r} phi`` is evaluated exactly at the polar nodes: the
periodic ``phi`` by its band-limited Fourier series (an even grid's Nyquist
term split symmetrically, so a real ``phi`` stays real) and the carrier tilt
analytically. The azimuthal integral is a DFT on ``n_theta`` nodes, chosen
above twice the largest ``|l|`` present (``|k + k_t|_max R``), so ``P_l`` is
alias-free; the radial integral is Gauss–Legendre. A plane wave of any tilt
gives ``P_l = P_{-l}`` (Jacobi–Anger), hence ``<L_z> = 0`` about every
centre.

Lengths in angstroms, momenta in 1/angstrom; ``L_z`` in units of ``hbar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .realspace import RealSpaceGrid

#: Azimuthal nodes added above twice the largest occupied ``|l|``.
_THETA_MARGIN = 16


@dataclass(frozen=True)
class CylindricalProjection:
    """``P_l`` of a grid state inside a disk about ``center_ang``.

    Callable as an :class:`.axial_mcwf.AxialMCWF` observable:
    ``projection(phi, tilt_inv_ang)`` returns ``P_l`` for
    ``l = -max_ell .. max_ell`` (:attr:`ells`) of ``phi`` as given (the MCWF
    driver passes normalized states).

    Parameters
    ----------
    grid
        Periodic grid of the state.
    center_ang
        Disk centre ``(x, y)`` [angstrom], e.g. an
        :class:`~pyrite.materials.channeling.axial.AtomicString` position.
    radius_ang
        Disk radius ``R`` [angstrom]; ``2R`` must fit in the cell.
    max_ell
        Largest reported ``|l|``; weight beyond it is in :meth:`disk_weight`
        but not in the returned ``P_l``.
    n_radial
        Gauss–Legendre nodes in ``rho``; resolve the occupied band,
        ``n_radial >~ k_occ R`` (the default is exact to 1e-12 for smooth
        probes but ~1e-3 for white noise at ``k R ~ 100``).

    Validation: cr-vortex-string-coupling
    """

    grid: RealSpaceGrid
    center_ang: tuple[float, float]
    radius_ang: float
    max_ell: int = 8
    n_radial: int = 48
    _rho: np.ndarray = field(init=False, repr=False)
    _rho_weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.radius_ang <= 0.5 * min(self.grid.extent_ang):
            raise ValueError("radius_ang must be positive and 2R must fit in the cell")
        if self.max_ell < 0 or self.n_radial < 2:
            raise ValueError("max_ell must be >= 0 and n_radial >= 2")
        nodes, weights = np.polynomial.legendre.leggauss(self.n_radial)
        rho = 0.5 * self.radius_ang * (nodes + 1.0)
        object.__setattr__(self, "_rho", rho)
        object.__setattr__(self, "_rho_weight", 0.5 * self.radius_ang * weights * rho)

    @property
    def ells(self) -> np.ndarray:
        """Reported angular momenta ``-max_ell .. max_ell``."""
        return np.arange(-self.max_ell, self.max_ell + 1)

    def _theta_nodes(self, tilt) -> int:
        k_max = np.hypot(
            np.abs(self.grid.kx).max() + abs(tilt[0]), np.abs(self.grid.ky).max() + abs(tilt[1])
        )
        need = 2 * (int(np.ceil(k_max * self.radius_ang)) + self.max_ell) + _THETA_MARGIN
        return 1 << int(np.ceil(np.log2(need)))

    def harmonics(self, phi: np.ndarray, tilt_inv_ang=(0.0, 0.0)) -> np.ndarray:
        """Return ``a_l(rho_j)`` for all ``n_theta`` harmonics, shape ``(n_radial, n_theta)``.

        Column ``l mod n_theta`` holds ``a_l``; ``phi`` is used as given (not
        renormalized).
        """
        phi = np.asarray(phi, dtype=complex)
        if phi.shape != self.grid.shape:
            raise ValueError(f"phi must have the grid shape {self.grid.shape}")
        n_theta = self._theta_nodes(tilt_inv_ang)
        theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
        x = self.center_ang[0] + self._rho[:, None] * np.cos(theta)
        y = self.center_ang[1] + self._rho[:, None] * np.sin(theta)
        # Exact band-limited interpolation: phi(r) = (1/N) sum_k phi~_k e^{i k.r}.
        spectrum = np.fft.fft2(phi) / phi.size
        ey = _fourier_basis(self.grid.ky, y)
        partial = np.einsum("ab,rtb->rta", spectrum, ey)
        ex = _fourier_basis(self.grid.kx, x)
        values = np.einsum("rta,rta->rt", ex, partial)
        values *= np.exp(1j * (tilt_inv_ang[0] * x + tilt_inv_ang[1] * y))
        return np.fft.fft(values, axis=1) * (np.sqrt(2.0 * np.pi) / n_theta)

    def disk_weight(self, phi: np.ndarray, tilt_inv_ang=(0.0, 0.0)) -> float:
        """Return ``int_disk |psi|^2`` (all ``l``) of ``phi`` as given."""
        a = self.harmonics(phi, tilt_inv_ang)
        return float(self._rho_weight @ np.sum(np.abs(a) ** 2, axis=1))

    def project(self, phi: np.ndarray, tilt_inv_ang=(0.0, 0.0)) -> np.ndarray:
        """Return ``P_l`` for :attr:`ells` of ``phi`` as given (not renormalized)."""
        a = self.harmonics(phi, tilt_inv_ang)
        columns = np.mod(self.ells, a.shape[1])
        return self._rho_weight @ (np.abs(a[:, columns]) ** 2)

    def __call__(self, phi: np.ndarray, tilt_inv_ang) -> np.ndarray:
        return self.project(phi, tilt_inv_ang)


def _fourier_basis(k: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """``e^{i k x}`` at off-grid ``x``; an even grid's Nyquist term is split as ``cos``.

    ``fftfreq`` keeps only ``-N/2``; on the nodes ``e^{+-i pi n}`` agree, but
    off them the one-sided term is neither real for real ``phi`` nor
    rotation-covariant. The symmetric split is the real band-limited
    interpolant.
    """
    basis = np.exp(1j * k[None, None, :] * coordinate[..., None])
    if k.size % 2 == 0:
        nyquist = k.size // 2
        basis[..., nyquist] = np.cos(k[nyquist] * coordinate)
    return basis


def angular_momentum(populations: np.ndarray, ells: np.ndarray) -> np.ndarray:
    """Return ``<L_z> = sum_l l P_l`` [hbar] over the last axis of ``populations``.

    Validation: cr-vortex-string-coupling
    """
    return np.asarray(populations) @ np.asarray(ells, dtype=float)


__all__ = ["CylindricalProjection", "angular_momentum"]
