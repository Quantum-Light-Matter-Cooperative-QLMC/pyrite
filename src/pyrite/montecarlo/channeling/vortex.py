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

**Site symmetry.** ``l`` labels a crystal state only where the projected
potential is ``C_m``-symmetric about ``c`` with ``m >= 3`` (``l mod m`` is
conserved, so ``l = +1`` and ``-1`` stay distinct), and only for states below
the inter-string saddle. On Si ⟨110⟩ the strings form close pairs: no string
site or pair centre has ``C_m``, ``m >= 3``, and above the saddle the motion
is chaotic (Shul'ga, Syshchenko, Tarnovsky & Isupov, NIMB 370, 1 (2016)).
Given the ``profile``, :class:`CylindricalProjection` refuses such a centre
unless ``symmetry="accept-asymmetric"`` acknowledges that ``P_l`` is then not a
quantum-number distribution.

Lengths in angstroms, momenta in 1/angstrom; ``L_z`` in units of ``hbar``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ...materials.channeling.axial import AxialProfile
from .realspace import RealSpaceGrid

#: Azimuthal nodes added above twice the largest occupied ``|l|``.
_THETA_MARGIN = 16
#: Smallest site rotation order for which ``l`` labels states.
MIN_ELL_SITE_ORDER = 3
#: Relative ``C_m`` residual (of the potential's full range) accepted as symmetric.
SITE_SYMMETRY_TOL = 1e-6
_SITE_ORDERS = (6, 4, 3, 2)
_SITE_THETA = 48  # divisible by every order in _SITE_ORDERS

SiteSymmetry = Literal["required", "accept-asymmetric"]


def site_rotation_order(
    profile: AxialProfile, center_ang, radius_ang: float, *, tol: float = SITE_SYMMETRY_TOL
) -> int:
    """Largest ``m`` in (6, 4, 3, 2) with ``U`` ``C_m``-symmetric about ``center_ang``, else 1.

    ``U`` is the profile's Fourier series, sampled exactly on 6 rings over
    ``(0, radius_ang]`` at 48 azimuths; ``C_m`` holds when the largest change
    under a ``2 pi / m`` rotation is at most ``tol`` times ``max U - min U``.

    Validation: cr-vortex-string-coupling
    """
    theta = 2.0 * np.pi * np.arange(_SITE_THETA) / _SITE_THETA
    rho = np.linspace(radius_ang / 6.0, radius_ang, 6)
    x = center_ang[0] + np.multiply.outer(rho, np.cos(theta)).ravel()
    y = center_ang[1] + np.multiply.outer(rho, np.sin(theta)).ravel()
    coeff = profile.potential_eV
    m_max, n_max = (n // 2 for n in coeff.shape)
    gx, gy = profile.g_inv_ang
    ex = np.exp(1j * gx * np.multiply.outer(x, np.arange(-m_max, m_max + 1)))
    ey = np.exp(1j * gy * np.multiply.outer(y, np.arange(-n_max, n_max + 1)))
    u = np.einsum("pm,mn,pn->p", ex, coeff, ey).real.reshape(rho.size, _SITE_THETA)
    low, high = profile.extrema()
    for m in _SITE_ORDERS:
        if np.abs(np.roll(u, -_SITE_THETA // m, axis=1) - u).max() <= tol * (high - low):
            return m
    return 1


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
    profile
        Crystal profile of the state (keyword-only). When given, the centre's
        :func:`site_rotation_order` over the disk must be at least
        ``MIN_ELL_SITE_ORDER``; omit it only for geometry-only use (free
        vortex probes, analytic checks).
    symmetry
        ``"required"`` (default) raises ``ValueError`` below that order;
        ``"accept-asymmetric"`` projects anyway with a ``UserWarning``,
        acknowledging that ``P_l`` then mixes ``l`` and is not a label.
    site_order
        The computed rotation order, or ``None`` without a profile.

    Validation: cr-vortex-string-coupling
    """

    grid: RealSpaceGrid
    center_ang: tuple[float, float]
    radius_ang: float
    max_ell: int = 8
    n_radial: int = 48
    profile: AxialProfile | None = field(default=None, kw_only=True, repr=False)
    symmetry: SiteSymmetry = field(default="required", kw_only=True)
    site_order: int | None = field(init=False, default=None)
    _rho: np.ndarray = field(init=False, repr=False)
    _rho_weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.radius_ang <= 0.5 * min(self.grid.extent_ang):
            raise ValueError("radius_ang must be positive and 2R must fit in the cell")
        if self.max_ell < 0 or self.n_radial < 2:
            raise ValueError("max_ell must be >= 0 and n_radial >= 2")
        if self.symmetry not in ("required", "accept-asymmetric"):
            raise ValueError("symmetry must be 'required' or 'accept-asymmetric'")
        if self.profile is not None:
            self._check_site(self.profile)
        nodes, weights = np.polynomial.legendre.leggauss(self.n_radial)
        rho = 0.5 * self.radius_ang * (nodes + 1.0)
        object.__setattr__(self, "_rho", rho)
        object.__setattr__(self, "_rho_weight", 0.5 * self.radius_ang * weights * rho)

    def _check_site(self, profile: AxialProfile) -> None:
        order = site_rotation_order(profile, self.center_ang, self.radius_ang)
        object.__setattr__(self, "site_order", order)
        if order >= MIN_ELL_SITE_ORDER:
            return
        message = (
            f"centre {tuple(self.center_ang)} of {profile.crystal} "
            f"{profile.uvw} has site rotation order C_{order} < C_{MIN_ELL_SITE_ORDER}"
            " over the disk, so l is not an approximate quantum number there "
            "(paired strings, e.g. Si <110>; Shul'ga et al. 2016)"
        )
        if self.symmetry == "required":
            raise ValueError(
                f"{message}; pass symmetry='accept-asymmetric' to project P_l anyway "
                "and accept that it is not an angular-momentum distribution"
            )
        warnings.warn(f"{message}; P_l is reported as a mixed-l projection", stacklevel=3)

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


__all__ = ["CylindricalProjection", "angular_momentum", "site_rotation_order"]
