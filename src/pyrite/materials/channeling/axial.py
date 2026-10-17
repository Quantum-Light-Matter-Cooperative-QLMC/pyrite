"""Axial (string) continuum potential and 2D transverse states (experimental, #508).

Averaging the crystal potential along a lattice direction ``[uvw]`` keeps the
Fourier components whose reciprocal vectors are perpendicular to it (the zone
law ``h u + k v + l w = 0``). The electron's transverse potential energy is

``U(r) = sum_{g != 0} U_g exp(i g . r)``, with
``U_g = -(4 pi e^2 / (V g^2)) sum_j (Z_j - f_j(g)) exp(-i G . R_j) exp(-g . U_j g / 2)``,

the planar formula of :mod:`.potential` with ``g`` running over the 2D zone
lattice instead of one line. ``U_j`` is the site's mean-square displacement
tensor [angstrom^2]; the isotropic default ``u^2 I`` with ``u^2 = B / 8 pi^2``
reproduces the planar ``exp(-B s^2)``. Wells sit on the atomic strings.
``U_0 = 0`` as in the planar case.

**Frame.** ``z`` is along ``[uvw]``. ``x`` is along the shortest non-extinct
zone vector and ``y = z x x``. The transverse cell is the rectangle
``period_ang = (L_x, L_y)`` whose reciprocal grid ``(2 pi m / L_x, 2 pi n / L_y)``
holds every zone vector; centred projections (two strings per rectangle)
are represented by their zero coefficients. Oblique projections, which have
no rectangular cell, are rejected.

**Planar limit.** Averaging ``U(x, y)`` over ``y`` keeps ``g`` along ``x``:
the planar profile of the zone plane normal to ``x``. Tests pin it.

**Transverse states.** The paraxial transverse Hamiltonian
``-(hbar^2 / 2 gamma m) nabla_perp^2 + U(r)`` has Bloch solutions
``psi_k(r) = A^{-1/2} sum_G c_G exp(i (k + G) . r)``; :func:`axial_states`
diagonalizes the plane-wave matrix as :func:`.transverse_states` does in 1D.

Lengths in angstroms, energies in eV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..atomic import Z_TABLE, cromer_mann_f0
from ..crystal import CRYSTALS, E2_EV_ANG, _direct_lattice_vectors, _reciprocal_basis
from .potential import _EXTINCTION_RTOL, F0_S_MAX_INV_ANG, _coprime
from .thermal import thermal_b_factor
from .transverse import HBAR2_2M_EV_ANG2

#: Relative tolerance for matching projected lengths to the rectangular grid.
_GRID_RTOL = 1e-7
#: Real-space tolerance [angstrom] for merging projected atom positions.
_STRING_TOL_ANG = 1e-6


@dataclass(frozen=True)
class AtomicString:
    """One atomic string of the projected cell.

    Attributes
    ----------
    position_ang
        ``(x, y)`` of the string inside the rectangular cell [angstrom].
    linear_density
        Atoms per angstrom along the axis, per element.
    """

    position_ang: tuple[float, float]
    linear_density: dict[str, float]


@dataclass(frozen=True)
class AxialProfile:
    """Fourier description of an axial (string) continuum potential.

    Attributes
    ----------
    crystal
        Catalog crystal key.
    uvw
        Coprime direction indices of the axis.
    axes
        Cartesian unit vectors ``(x, y, z)`` of the transverse frame (rows).
    period_ang
        Rectangular cell ``(L_x, L_y)`` [angstrom].
    potential_eV
        Complex coefficients ``U[m, n]`` [eV] at ``g = (2 pi m / L_x, 2 pi n / L_y)``,
        indexed ``[m + M, n + N]`` with ``max_order = (M, N)``; ``U[0, 0] = 0``.
    nuclear_density, electron_density
        Coefficients of the thermally smeared nuclear and atomic-electron number
        densities [1/angstrom^3]; ``[M, N]`` is the mean density.
    max_order
        Highest harmonic ``(M, N)`` kept along ``x`` and ``y``.
    strings
        Distinct atomic strings in one cell.
    temperature_K
        Crystal temperature [K].
    displacements_ang2
        Mean-square displacement tensor per basis site [angstrom^2].
    """

    crystal: str
    uvw: tuple[int, int, int]
    axes: np.ndarray
    period_ang: tuple[float, float]
    potential_eV: np.ndarray
    nuclear_density: np.ndarray
    electron_density: np.ndarray
    max_order: tuple[int, int]
    strings: tuple[AtomicString, ...]
    temperature_K: float
    displacements_ang2: tuple[np.ndarray, ...]

    @property
    def g_inv_ang(self) -> tuple[float, float]:
        """Fundamental reciprocal periods ``(2 pi / L_x, 2 pi / L_y)`` [1/angstrom]."""
        return 2.0 * np.pi / self.period_ang[0], 2.0 * np.pi / self.period_ang[1]

    @property
    def area_ang2(self) -> float:
        """Cell area ``L_x L_y`` [angstrom^2]."""
        return self.period_ang[0] * self.period_ang[1]

    def potential(self, x_ang, y_ang):
        """Return ``U(x, y)`` [eV] on the tensor grid ``x`` x ``y``.

        Validation: cr-axial-2d-eigenproblem
        """
        return _synthesize(self.potential_eV, self.g_inv_ang, x_ang, y_ang)

    def nuclear(self, x_ang, y_ang):
        """Return the nuclear number density [1/angstrom^3] on ``x`` x ``y``."""
        return _synthesize(self.nuclear_density, self.g_inv_ang, x_ang, y_ang)

    def electrons(self, x_ang, y_ang):
        """Return the atomic electron density [1/angstrom^3] on ``x`` x ``y``."""
        return _synthesize(self.electron_density, self.g_inv_ang, x_ang, y_ang)

    def on_grid(self, cells: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
        """Return ``U`` on a ``cells[0] x cells[1]`` supercell sampled at ``shape`` points.

        Harmonics above the grid's Nyquist limit are dropped, so the samples are
        the band-limited potential (no aliasing). Point ``(i, j)`` sits at
        ``(i L_x cells[0] / shape[0], j L_y cells[1] / shape[1])``.

        Validation: cr-axial-2d-eigenproblem
        """
        return _on_grid(self.potential_eV, cells, shape)

    def extrema(self, n_grid: int = 256) -> tuple[float, float]:
        """Return ``(min U, max U)`` [eV] over one cell."""
        u = self.on_grid((1, 1), (n_grid, n_grid))
        return float(u.min()), float(u.max())


def _synthesize(coefficients: np.ndarray, g: tuple[float, float], x_ang, y_ang):
    x = np.asarray(x_ang, dtype=float)
    y = np.asarray(y_ang, dtype=float)
    m_max, n_max = (s // 2 for s in coefficients.shape)
    ex = np.exp(1j * g[0] * np.multiply.outer(x, np.arange(-m_max, m_max + 1)))
    ey = np.exp(1j * g[1] * np.multiply.outer(y, np.arange(-n_max, n_max + 1)))
    return np.einsum("im,mn,jn->ij", ex, coefficients, ey).real


def _on_grid(coefficients: np.ndarray, cells, shape) -> np.ndarray:
    cx, cy = (int(c) for c in cells)
    nx, ny = (int(s) for s in shape)
    if min(cx, cy, nx, ny) < 1:
        raise ValueError("cells and shape must be positive")
    m_max, n_max = (s // 2 for s in coefficients.shape)
    spectrum = np.zeros((nx, ny), dtype=complex)
    for m in range(-m_max, m_max + 1):
        if 2 * abs(m * cx) >= nx:
            continue
        for n in range(-n_max, n_max + 1):
            if 2 * abs(n * cy) >= ny:
                continue
            spectrum[(m * cx) % nx, (n * cy) % ny] = coefficients[m + m_max, n + n_max]
    return np.fft.ifft2(spectrum).real * (nx * ny)


def _grid_step(values: np.ndarray) -> float | None:
    """Return the 1D lattice spacing holding every nonzero ``values``, or ``None``."""
    values = np.abs(values[np.abs(values) > _GRID_RTOL * np.abs(values).max(initial=0.0)])
    if values.size == 0:
        return None
    smallest = values.min()
    for divisor in range(1, 7):
        step = smallest / divisor
        ratio = values / step
        if np.all(np.abs(ratio - np.rint(ratio)) < 1e-6 * ratio.max()):
            return float(step)
    return None


def _rectangular_cell(g_vec, g_mag, live, axis):
    """Pick ``x`` among short zone directions so the rectangular cell is smallest."""
    best = None
    order = np.argsort(np.where(live, g_mag, np.inf))
    tried: list[np.ndarray] = []
    for idx in order[: min(64, int(live.sum()))]:
        x_hat = g_vec[idx] / g_mag[idx]
        if any(abs(abs(x_hat @ t) - 1.0) < 1e-9 for t in tried):
            continue
        tried.append(x_hat)
        y_hat = np.cross(axis, x_hat)
        dgx = _grid_step(g_vec[live] @ x_hat)
        dgy = _grid_step(g_vec[live] @ y_hat)
        if dgx is None or dgy is None:
            continue
        area = (2.0 * np.pi) ** 2 / (dgx * dgy)
        if best is None or area < best[0] * (1.0 - 1e-9):
            best = (area, x_hat, y_hat, dgx, dgy)
    return best


def _displacement_tensors(basis, temperature_K, crystal, displacements) -> list[np.ndarray]:
    def tensor(value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(3)
        if arr.shape != (3, 3) or not np.allclose(arr, arr.T):
            raise ValueError("a displacement tensor must be a symmetric 3x3 array")
        return arr

    if displacements is None:
        cache = {}
        out = []
        for el, _ in basis:
            if el not in cache:
                cache[el] = thermal_b_factor(crystal, el, temperature_K) / (8.0 * np.pi**2)
            out.append(cache[el] * np.eye(3))
        return out
    if isinstance(displacements, Mapping):
        return [tensor(displacements[el]) for el, _ in basis]
    if len(displacements) != len(basis):
        raise ValueError(f"need one displacement per basis site ({len(basis)})")
    return [tensor(v) for v in displacements]


def axial_profile(
    crystal: str,
    uvw,
    temperature_K: float = 293.15,
    *,
    s_max_inv_ang: float = F0_S_MAX_INV_ANG,
    displacements_ang2: Mapping[str, object] | Sequence[object] | None = None,
) -> AxialProfile:
    """Return the thermally averaged string potential of a catalog crystal axis.

    Parameters
    ----------
    crystal
        Catalog crystal key with channeling thermal data (``silicon``, ``diamond``).
    uvw
        Direction indices of the axis in the conventional cell; reduced to
        coprime form.
    temperature_K
        Crystal temperature [K] for the catalog Debye--Waller factors.
    s_max_inv_ang
        Highest ``s = |g| / (4 pi)`` kept [1/angstrom].
    displacements_ang2
        Optional override of the thermal displacements: a mapping element ->
        value or one value per basis site, each a scalar ``<u^2>`` per axis or a
        symmetric 3x3 Cartesian tensor [angstrom^2]. The catalog default is
        isotropic, ``B / 8 pi^2`` per element.

    Returns
    -------
    AxialProfile
        Potential and density coefficients on the rectangular projected cell.

    Raises
    ------
    ValueError
        If the projected lattice has no rectangular cell.

    Validation: cr-axial-2d-eigenproblem
    """
    info = CRYSTALS[crystal]
    basis = info["basis"]
    volume = float(info["V_cell"])
    uvw_c = _coprime(uvw)
    direct = np.array(_direct_lattice_vectors(info["lattice"]), dtype=float)
    recip = np.asarray(_reciprocal_basis(info["lattice"]), dtype=float)
    axis = np.asarray(uvw_c, float) @ direct
    axis /= np.linalg.norm(axis)
    u2 = _displacement_tensors(basis, temperature_K, crystal, displacements_ang2)

    # Zone vectors G = h b1 + k b2 + l b3 with h u + k v + l w = 0, |G| <= g_max.
    g_max = 4.0 * np.pi * s_max_inv_ang
    h_max = int(np.ceil(g_max / np.linalg.norm(recip, axis=1).min())) + 1
    span = np.arange(-h_max, h_max + 1)
    hkl = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[(hkl @ np.asarray(uvw_c)) == 0]
    g_vec = hkl @ recip
    g_mag = np.linalg.norm(g_vec, axis=1)
    keep = (g_mag <= g_max) & (g_mag > 0)
    hkl, g_vec, g_mag = hkl[keep], g_vec[keep], g_mag[keep]

    positions = np.array([r for _, r in basis], dtype=float)
    phases = np.exp(-2j * np.pi * hkl @ positions.T)
    debye = np.exp(-0.5 * np.stack([np.einsum("gi,ij,gj->g", g_vec, t, g_vec) for t in u2], 1))
    structure = phases * debye
    elements = [el for el, _ in basis]
    f0 = {el: cromer_mann_f0(el, g_mag) for el in set(elements)}
    z = np.array([Z_TABLE[el] for el in elements], dtype=float)
    f_sites = np.stack([f0[el] for el in elements], 1)
    potential = -4.0 * np.pi * E2_EV_ANG / volume * ((z - f_sites) * structure).sum(1) / g_mag**2
    nuclear = structure.sum(1) / volume
    electrons = (f_sites * structure).sum(1) / volume

    live = np.abs(potential) > _EXTINCTION_RTOL * np.abs(potential).max()
    if not live.any():
        raise ValueError(f"axis {uvw_c} has no non-extinct zone reflections")
    cell = _rectangular_cell(g_vec, g_mag, live, axis)
    if cell is None:
        raise ValueError(f"axis {uvw_c} projects to an oblique lattice; only rectangular cells")
    _, x_hat, y_hat, dgx, dgy = cell
    gx, gy = g_vec @ x_hat, g_vec @ y_hat
    m = np.rint(gx / dgx).astype(int)
    n = np.rint(gy / dgy).astype(int)
    off_grid = (np.abs(gx - m * dgx) > 1e-6 * dgx) | (np.abs(gy - n * dgy) > 1e-6 * dgy)
    if np.any(off_grid & live):
        raise ValueError(f"axis {uvw_c} projects to an oblique lattice; only rectangular cells")
    m_max, n_max = int(np.abs(m).max()), int(np.abs(n).max())
    dense = [np.zeros((2 * m_max + 1, 2 * n_max + 1), dtype=complex) for _ in range(3)]
    on = ~off_grid
    for arr, values in zip(dense, (potential, nuclear, electrons), strict=True):
        arr[m[on] + m_max, n[on] + n_max] = values[on]
    dense[1][m_max, n_max] = len(basis) / volume
    dense[2][m_max, n_max] = z.sum() / volume

    period = (2.0 * np.pi / dgx, 2.0 * np.pi / dgy)
    return AxialProfile(
        crystal=crystal,
        uvw=uvw_c,
        axes=np.stack([x_hat, y_hat, axis]),
        period_ang=period,
        potential_eV=dense[0],
        nuclear_density=dense[1],
        electron_density=dense[2],
        max_order=(m_max, n_max),
        strings=_strings(basis, direct, x_hat, y_hat, period, volume),
        temperature_K=float(temperature_K),
        displacements_ang2=tuple(u2),
    )


def _strings(basis, direct, x_hat, y_hat, period, volume) -> tuple[AtomicString, ...]:
    """Merge projected basis sites and their lattice-translation images into strings."""
    lx, ly = period

    def fold(points: np.ndarray) -> np.ndarray:
        folded = np.stack([(points @ x_hat) % lx, (points @ y_hat) % ly], -1)
        folded[np.isclose(folded[:, 0], lx, atol=_STRING_TOL_ANG), 0] = 0.0
        folded[np.isclose(folded[:, 1], ly, atol=_STRING_TOL_ANG), 1] = 0.0
        return folded

    def unique(points: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for p in points:
            if not any(_same(p, q, period) for q in out):
                out.append(p)
        return out

    span = np.arange(-4, 5)
    shifts = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3) @ direct
    centring = unique(fold(shifts))
    area = lx * ly
    weight = area / (volume * len(centring))
    strings: list[tuple[np.ndarray, dict[str, float]]] = []
    for el, frac in basis:
        site = np.asarray(frac, float) @ direct
        for c in centring:
            p = fold(site[None, :])[0] + c
            p = np.array([p[0] % lx, p[1] % ly])
            for q, dens in strings:
                if _same(p, q, period):
                    dens[el] = dens.get(el, 0.0) + weight
                    break
            else:
                strings.append((p, {el: weight}))
    return tuple(AtomicString((float(p[0]), float(p[1])), dens) for p, dens in strings)


def _same(p, q, period) -> bool:
    d = np.asarray(p) - np.asarray(q)
    d -= np.rint(d / np.asarray(period)) * np.asarray(period)
    return bool(np.all(np.abs(d) < 1e-5))


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


__all__ = ["AtomicString", "AxialProfile", "AxialStates", "axial_profile", "axial_states"]
