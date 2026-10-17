"""Planar continuum potential and densities for electron channeling.

Averaging the crystal potential over a crystal plane keeps only the Fourier
components whose reciprocal vectors are normal to it. For a plane with
coprime Miller indices ``(hkl)`` and harmonics ``n (hkl)``, the electron's
potential energy is

``U(x) = sum_{n != 0} U_n exp(i n g x)``, with
``U_n = -(4 pi e^2 / (V g_n^2)) sum_j (Z_j - f_j(g_n)) exp(-i 2 pi n hkl . R_j)
exp(-B_j s_n^2)``,

where ``g_n = n |g_hkl|``, ``s_n = g_n / (4 pi)``, ``f_j`` is the X-ray form
factor and ``B_j`` the site Debye--Waller factor. The bracket is the
Mott--Bethe electron scattering factor, ``f^e = (Z - f) / (8 pi^2 a0 s^2)``,
written in the potential normalization (Andersen et al. 1983; the CBS
coupling of :func:`pyrite.materials.crystal.U_g`, with the opposite sign
because the electron charge is ``-e``). Wells therefore sit **on** the
atomic planes. ``U_0`` (the mean inner potential) is set to zero: it shifts all
transverse levels equally and enters only surface refraction.

Only the energy-independent form factor ``f0`` enters; the anomalous X-ray
terms describe photon absorption, not the electrostatic potential.

``x`` is the coordinate along the reciprocal plane normal, measured from the
unit-cell origin, in angstroms.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd

import numpy as np

from ..atomic import Z_TABLE, cromer_mann_f0
from ..crystal import CRYSTALS, E2_EV_ANG, _direct_lattice_vectors, reciprocal_g_vector
from .thermal import ChannelingDataError, UniaxialThermalSite, thermal_site

# Waasmaier--Kirfel f0 fits are valid to s = sin(theta)/lambda = 6 / angstrom.
F0_S_MAX_INV_ANG = 6.0
_EXTINCTION_RTOL = 1e-9


@dataclass(frozen=True)
class PlanarProfile:
    """Fourier description of a planar continuum potential.

    Attributes
    ----------
    crystal
        Catalog crystal key, or a descriptive label for synthetic profiles.
    hkl
        Coprime Miller indices of the plane family.
    period_ang
        True period of the planar potential [angstrom]: the coprime plane
        spacing divided by the greatest common divisor of non-extinct harmonics.
    potential_eV
        Complex Fourier coefficients ``U_m`` [eV] of the electron potential
        energy for ``m = 0..M`` in units of ``2 pi / period_ang``. ``U_0 = 0``;
        ``U_{-m} = conj(U_m)``.
    nuclear_density
        Fourier coefficients of the thermally smeared nuclear number density
        [1/angstrom^3]; element 0 is the mean atom density.
    electron_density
        Fourier coefficients of the thermally smeared atomic electron density
        [electrons/angstrom^3]; element 0 is the mean electron density.
    temperature_K
        Crystal temperature [K] of the thermal smearing.
    b_factors_ang2
        Site Debye--Waller ``B`` per element at ``temperature_K``. When
        ``thermal_projected`` is true it is ``8 pi^2 n.U.n`` along the plane
        normal ``n`` only, not an isotropic ``B``.
    thermal_projected
        True when any element's ``B`` is the plane-normal projection of an
        anisotropic displacement; transverse scattering then needs
        ``covariances_ang2`` and a beam direction.
    covariances_ang2
        Per-element Cartesian displacement covariance ``U`` [angstrom^2] in the
        frame of the catalog direct-lattice vectors; ``None`` when every
        element is isotropic (``U = B / 8 pi^2 I``).
    normal
        Unit reciprocal plane normal ``g / |g|`` in that frame: the direction
        of ``x``.
    """

    crystal: str
    hkl: tuple[int, int, int]
    period_ang: float
    potential_eV: np.ndarray
    nuclear_density: np.ndarray
    electron_density: np.ndarray
    temperature_K: float
    b_factors_ang2: dict[str, float]
    thermal_projected: bool = False
    covariances_ang2: dict[str, np.ndarray] | None = None
    normal: np.ndarray | None = None

    @property
    def g_inv_ang(self) -> float:
        """Fundamental reciprocal period ``2 pi / period_ang`` [1/angstrom]."""
        return 2.0 * np.pi / self.period_ang

    def potential(self, x_ang):
        """Return the real electron potential energy ``U(x)`` [eV].

        Validation: cr-planar-continuum-potential
        """
        return _synthesize(self.potential_eV, self.g_inv_ang, x_ang)

    def nuclear(self, x_ang):
        """Return the nuclear number density ``n(x)`` [1/angstrom^3]."""
        return _synthesize(self.nuclear_density, self.g_inv_ang, x_ang)

    def electrons(self, x_ang):
        """Return the atomic electron density ``rho_e(x)`` [1/angstrom^3]."""
        return _synthesize(self.electron_density, self.g_inv_ang, x_ang)

    def extrema(self, n_grid: int = 2048) -> tuple[float, float]:
        """Return ``(min U, max U)`` [eV] over one period.

        The difference is the planar well depth; the maximum is the barrier
        that separates bound from free transverse states.
        """
        x = np.linspace(0.0, self.period_ang, n_grid, endpoint=False)
        u = self.potential(x)
        return float(u.min()), float(u.max())


def _synthesize(coefficients: np.ndarray, g: float, x_ang):
    x = np.asarray(x_ang, dtype=float)
    m = np.arange(1, coefficients.size)
    phase = np.exp(1j * g * np.multiply.outer(x, m))
    value = coefficients[0].real + 2.0 * (phase @ coefficients[1:]).real
    return value if np.ndim(x_ang) else float(value)


def _coprime(hkl) -> tuple[int, int, int]:
    if len(hkl) != 3 or any(int(v) != v for v in hkl):
        raise ValueError("hkl must be three integers")
    indices = tuple(int(v) for v in hkl)
    if indices == (0, 0, 0):
        raise ValueError("hkl must be nonzero")
    divisor = reduce(gcd, (abs(v) for v in indices))
    h, k, idx_l = (v // divisor for v in indices)
    return h, k, idx_l


def planar_profile(
    crystal: str,
    hkl,
    temperature_K: float = 293.15,
    *,
    s_max_inv_ang: float = F0_S_MAX_INV_ANG,
) -> PlanarProfile:
    """Return the thermally averaged planar continuum profile of a catalog crystal.

    Parameters
    ----------
    crystal
        Catalog crystal key with channeling thermal data (Si, diamond).
    hkl
        Miller indices of the plane family; reduced to coprime form. Extinct
        harmonics are detected numerically and set the true period.
    temperature_K
        Crystal temperature [K].
    s_max_inv_ang
        Highest ``s = g / (4 pi)`` kept [1/angstrom]; defaults to the validity
        limit of the form-factor fits.

    Returns
    -------
    PlanarProfile
        Potential and density Fourier coefficients on the true period.

    Raises
    ------
    ChannelingDataError
        If any basis element lacks channeling thermal data.

    Validation: cr-planar-continuum-potential, cr-mott-bethe-factor
    """
    info = CRYSTALS[crystal]
    basis = info["basis"]
    volume = float(info["V_cell"])
    hkl_c = _coprime(hkl)
    g_vec, g_prim = reciprocal_g_vector(hkl_c, info["lattice"])
    elements = sorted({el for el, _ in basis})
    b_factors, covariances = _normal_b_factors(
        crystal, info["lattice"], g_vec, elements, temperature_K
    )

    n_max = max(1, int(np.floor(4.0 * np.pi * s_max_inv_ang / g_prim)))
    n = np.arange(n_max + 1)
    g_n = n * g_prim
    s_n = g_n / (4.0 * np.pi)
    positions = np.array([r for _, r in basis], dtype=float)
    phases = np.exp(-2j * np.pi * np.outer(n, positions @ np.asarray(hkl_c, float)))

    potential = np.zeros(n.size, dtype=complex)
    nuclear = np.zeros(n.size, dtype=complex)
    electrons = np.zeros(n.size, dtype=complex)
    for el in elements:
        mask = np.array([e == el for e, _ in basis])
        site_sum = phases[:, mask].sum(axis=1) * np.exp(-b_factors[el] * s_n**2)
        f0 = np.empty(n.size)
        f0[0] = Z_TABLE[el]
        f0[1:] = cromer_mann_f0(el, g_n[1:])
        nuclear += site_sum
        electrons += site_sum * f0
        with np.errstate(divide="ignore", invalid="ignore"):
            potential[1:] += (Z_TABLE[el] - f0[1:]) / g_n[1:] ** 2 * site_sum[1:]
    potential *= -4.0 * np.pi * E2_EV_ANG / volume
    nuclear /= volume
    electrons /= volume

    scale = max(np.abs(potential[1:]).max(), 1e-300)
    live = [int(m) for m in n[1:] if abs(potential[m]) > _EXTINCTION_RTOL * scale]
    step = reduce(gcd, live) if live else 1
    keep = n[::step]
    return PlanarProfile(
        crystal=crystal,
        hkl=hkl_c,
        period_ang=float(2.0 * np.pi / (step * g_prim)),
        potential_eV=potential[keep],
        nuclear_density=nuclear[keep],
        electron_density=electrons[keep],
        temperature_K=float(temperature_K),
        b_factors_ang2=b_factors,
        thermal_projected=covariances is not None,
        covariances_ang2=covariances,
        normal=g_vec / g_prim,
    )


def _is_hexagonal(lattice) -> bool:
    if lattice["system"] == "hexagonal":
        return True
    angles = (lattice.get("alpha"), lattice.get("beta"), lattice.get("gamma"))
    return (
        lattice["system"] == "general"
        and np.isclose(lattice["a"], lattice["b"])
        and np.allclose(angles, (90.0, 90.0, 120.0))
    )


def _normal_b_factors(crystal, lattice, g_vec, elements, temperature_K):
    """Return per-element ``B`` along the plane normal and, if any is anisotropic, every ``U``.

    The Debye--Waller factor of a Fourier component ``g`` is
    ``exp(-g.U.g / 2) = exp(-B_n s^2)`` with ``B_n = 8 pi^2 n.U.n`` and
    ``n = g / |g|``, so a uniaxial site enters the planar sum exactly through
    its plane-normal projection. The covariances (``None`` when all elements
    are isotropic) feed the transverse scattering kernels.

    Validation: cr-uniaxial-thermal-projection
    """
    b_factors: dict[str, float] = {}
    covariances: dict[str, np.ndarray] = {}
    c_axis = None
    for el in elements:
        site = thermal_site(crystal, el)
        if not isinstance(site, UniaxialThermalSite):
            b_factors[el] = site.b_factor(temperature_K)
            covariances[el] = b_factors[el] / (8.0 * np.pi**2) * np.eye(3)
            continue
        if c_axis is None:
            if not _is_hexagonal(lattice):
                raise ChannelingDataError(f"{crystal}: uniaxial thermal rows need a hexagonal cell")
            c_axis = _direct_lattice_vectors(lattice)[2]
            cos2_c = float(
                np.dot(g_vec, c_axis) ** 2 / (np.dot(g_vec, g_vec) * np.dot(c_axis, c_axis))
            )
        b_factors[el] = site.b_factor_along(cos2_c, temperature_K)
        covariances[el] = site.covariance(c_axis, temperature_K)
    return b_factors, (covariances if c_axis is not None else None)
