"""Helicity- and angular-momentum-resolved axial channeling radiation (experimental, #508).

Bound axial states of one string are labelled by an approximate angular
momentum ``l`` (proposal §9). A dipole transition ``i -> f`` emits a photon
whose total angular momentum along the electron velocity ``z`` is
``J_z = l_i - l_f = +-1``; the boost along ``z`` preserves ``J_z``. With the
transverse dipole ``d = <f| r_perp |i>`` the two components are

``A_+ = e_+^* . d = (d_x - i d_y) / sqrt 2``  (``J_z = +1``, lowers ``l``),
``A_- = e_-^* . d = (d_x + i d_y) / sqrt 2``  (``J_z = -1``),

(``e_+- = (x +- i y)/sqrt 2``; the emitting dipole ``<r>(t) ∝ d e^{-i omega t}``
rotates counter-clockwise for ``A_+``). For an initial density ``rho`` in the
(degenerate) initial level the line's polarization matrix is
``T_ab = sum_f (A_a rho A_b^dagger)_ff`` (``a, b = +-``), linear in ``rho``.

**On axis** the photon helicity equals ``J_z``, so the Stokes parameters are
``I = T_++ + T_--`` and ``V = T_++ - T_--`` (``V > 0``: positive helicity);
in Cartesian form ``Q = T_xx - T_yy``, ``U = 2 Re T_xy``, ``V = 2 Im T_xy``
with ``T_xy = sum conj(d_x) d_y``.
A real (planar) dipole has ``|A_+| = |A_-|``, so ``V = 0``.

**Off axis.** Each ``J_z = m`` component is a rest-frame ``J = 1`` dipole.
Its photons at rest-frame polar angle ``theta'`` have helicity ``Lambda``
with probability density ``(3/8 pi) |d^1_{m Lambda}(theta')|^2`` per sr
(Wigner small-d; ``|d^1_{11}|^2 = ((1 + cos)/2)^2``,
``|d^1_{1,-1}|^2 = ((1 - cos)/2)^2``); helicity and photon number are
invariant and ``cos theta' = (cos theta - beta)/(1 - beta cos theta)``. A
photon of fixed ``J_z = m`` and helicity ``Lambda`` travelling at polar angle
``theta`` (lab) carries spin ``S_z = Lambda cos theta`` and orbital
``L_z = m - Lambda cos theta`` about ``z`` (twisted-photon decomposition;
Korotchenko & Kunashenko, arXiv:2403.01930; Knyazev & Serbo, Phys. Usp. 61,
449 (2018)). Over the full sphere the two helicities are equally likely, so
net helicity is an aperture effect; ``J_z`` is not.

**Levels.** Lines use the ``k = 0`` single-cell eigenstates of
:func:`~pyrite.materials.channeling.axial.axial_states`. Only flat bands
(bandwidth over the Brillouin zone below ``max_bandwidth_eV``) are kept; their
on-column orbitals are the ``k = 0`` Bloch functions cut to the string's
Wigner–Seitz cell (tight binding). Dipoles use the exact velocity form
``<f|r|i> = -i (hbar^2/gamma m) <f|k|i> / (E_f - E_i)``.

Lengths in angstroms, energies in eV, angular momenta in hbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...materials.channeling.axial import AxialProfile, AxialStates, axial_states
from ...materials.channeling.transverse import (
    HBAR2_2M_EV_ANG2,
    doppler_denominator,
    longitudinal_kinematics,
)
from ...materials.crystal import M_E_EV
from .dipole import dipole_rate_per_ang
from .realspace import RealSpaceGrid

#: Default flat-band limit for a level to enter the line catalog [eV].
DEFAULT_MAX_BANDWIDTH_EV = 0.1


@dataclass(frozen=True)
class AxialLevel:
    """A (possibly degenerate) bound level of the string.

    Attributes
    ----------
    energy_eV
        Mean transverse energy at ``k = 0`` [eV].
    states
        Indices of the ``k = 0`` eigenstates spanning the level.
    bandwidth_eV
        Spread of the level over the zone centre, edge and corner [eV].
    """

    energy_eV: float
    states: tuple[int, ...]
    bandwidth_eV: float


@dataclass(frozen=True)
class AxialLine:
    """Dipole line between two levels.

    Attributes
    ----------
    initial, final
        Level indices (``E_initial > E_final``).
    delta_eV
        Transverse energy released ``E_i - E_f`` [eV].
    dipole_ang
        ``<f| r_a |i>`` [angstrom], shape ``(2, n_final, n_initial)`` for
        ``a = x, y``.
    """

    initial: int
    final: int
    delta_eV: float
    dipole_ang: np.ndarray

    @property
    def circular_ang(self) -> np.ndarray:
        """``(A_+, A_-)`` [angstrom], shape ``(2, n_final, n_initial)``."""
        dx, dy = self.dipole_ang
        return np.stack([(dx - 1j * dy), (dx + 1j * dy)]) / np.sqrt(2.0)

    def polarization(self, rho_initial: np.ndarray) -> np.ndarray:
        """Return ``T_ab = sum_f (A_a rho A_b^dagger)_ff`` [angstrom^2], ``a, b = +, -``.

        Validation: cr-axial-helicity-emission
        """
        a = self.circular_ang
        rho = np.asarray(rho_initial, dtype=complex)
        return np.einsum("afi,ij,bfj->ab", a, rho, a.conj())


@dataclass(frozen=True)
class AxialLineCatalog:
    """Flat bound levels and dipole lines of one string at one energy.

    Validation: cr-axial-helicity-emission
    """

    profile: AxialProfile
    kinetic_energy_eV: float
    gamma: float
    states: AxialStates
    levels: tuple[AxialLevel, ...]
    lines: tuple[AxialLine, ...]
    _string_ang: tuple[float, float] = field(default=(0.0, 0.0), repr=False)

    def photon_energy_eV(self, line: AxialLine, theta_rad=0.0):
        """Lab photon energy ``Delta E / (1 - beta cos theta)`` [eV]."""
        return line.delta_eV / doppler_denominator(self.kinetic_energy_eV, theta_rad)

    def yield_per_ang(self, line: AxialLine, rho_initial: np.ndarray) -> float:
        """Photons per angstrom from ``rho_initial`` (trace = level population)."""
        strength = float(np.trace(line.polarization(rho_initial)).real)
        return float(dipole_rate_per_ang(line.delta_eV, np.sqrt(strength), self.gamma))

    def orbitals(self, grid: RealSpaceGrid, center_ang) -> list[np.ndarray]:
        """On-column orbitals of every level on ``grid`` about the string at ``center_ang``.

        One array ``(n_states, nx, ny)`` per level: the ``k = 0`` Bloch
        functions cut to the square Wigner–Seitz cell of the string (one
        string per cell is required).
        """
        if len(self.profile.strings) != 1:
            raise NotImplementedError("on-column orbitals need one string per cell")
        lx, ly = self.profile.period_ang
        x = grid.x - center_ang[0]
        y = grid.y - center_ang[1]
        inside = (np.abs(x)[:, None] <= lx / 2) & (np.abs(y)[None, :] <= ly / 2)
        sx, sy = self._string_ang
        return [
            np.stack(
                [
                    np.where(inside, self.states.wavefunction(x + sx, y + sy, s), 0.0)
                    for s in level.states
                ]
            )
            for level in self.levels
        ]


def axial_line_catalog(
    profile: AxialProfile,
    kinetic_energy_eV: float,
    *,
    max_order: int = 12,
    max_bandwidth_eV: float = DEFAULT_MAX_BANDWIDTH_EV,
    degeneracy_tol_eV: float = 1e-6,
) -> AxialLineCatalog:
    """Return the flat bound levels of ``profile`` and every dipole line between them.

    Validation: cr-axial-helicity-emission
    """
    gamma, _ = longitudinal_kinematics(float(kinetic_energy_eV))
    states = axial_states(profile, gamma, max_order=max_order)
    gx, gy = profile.g_inv_ang
    edges = [(0.5 * gx, 0.0), (0.5 * gx, 0.5 * gy)]
    barrier = profile.extrema()[1]
    bound = int(np.searchsorted(states.energies_eV, barrier))
    others = [
        axial_states(profile, gamma, k, max_order=max_order).energies_eV[:bound] for k in edges
    ]
    levels: list[AxialLevel] = []
    start = 0
    while start < bound:
        stop = start + 1
        while (
            stop < bound
            and states.energies_eV[stop] - states.energies_eV[start] <= degeneracy_tol_eV
        ):
            stop += 1
        block = [states.energies_eV[start:stop], *(e[start:stop] for e in others)]
        spread = float(np.ptp(np.concatenate(block)))
        if spread > max_bandwidth_eV:
            break  # higher levels are wider still
        levels.append(
            AxialLevel(
                float(states.energies_eV[start:stop].mean()), tuple(range(start, stop)), spread
            )
        )
        start = stop
    # Velocity form: <f|r|i> = -i (hbar^2 / gamma m) <f|k|i> / (E_f - E_i).
    k = np.stack([states.orders[:, 0] * gx, states.orders[:, 1] * gy])
    c = states.coefficients
    lines = []
    for i, upper in enumerate(levels):
        for f, lower in enumerate(levels[:i]):
            ci = c[:, list(upper.states)]
            cf = c[:, list(lower.states)]
            momentum = np.einsum("bf,ab,bi->afi", cf.conj(), k, ci)
            delta = upper.energy_eV - lower.energy_eV
            dipole = -1j * (2.0 * HBAR2_2M_EV_ANG2 / gamma) * momentum / (-delta)
            lines.append(AxialLine(i, f, float(delta), dipole))
    string = profile.strings[0].position_ang if profile.strings else (0.0, 0.0)
    return AxialLineCatalog(
        profile=profile,
        kinetic_energy_eV=float(kinetic_energy_eV),
        gamma=float(gamma),
        states=states,
        levels=tuple(levels),
        lines=tuple(lines),
        _string_ang=(float(string[0]), float(string[1])),
    )


# ---------------------------------------------------------------------------
# Polarization and photon angular momentum
# ---------------------------------------------------------------------------


def axis_stokes(polarization: np.ndarray) -> np.ndarray:
    """Return on-axis ``(I, Q, U, V)`` of a line from its ``T_ab`` (``a, b = +, -``).

    ``Q, U`` refer to the crystal ``x, y`` axes; ``V > 0`` is positive helicity.

    Validation: cr-axial-helicity-emission
    """
    t = np.asarray(polarization, dtype=complex)
    # Back to Cartesian: d_x = (A_+ + A_-)/sqrt2, d_y = i (A_+ - A_-)/sqrt2.
    u = np.array([[1.0, 1.0], [1j, -1j]]) / np.sqrt(2.0)
    cart = u @ t @ u.conj().T  # cart[a, b] = sum d_a d_b^*
    intensity = (cart[0, 0] + cart[1, 1]).real
    q = (cart[0, 0] - cart[1, 1]).real
    u_stokes = 2.0 * cart[1, 0].real
    v = 2.0 * cart[1, 0].imag
    return np.array([intensity, q, u_stokes, v])


def helicity_pattern(theta_rad, gamma: float, m: int, helicity: int):
    """Lab angular density of ``J_z = m`` dipole photons of one helicity [1/sr].

    ``(3/8 pi) |d^1_{m Lambda}(theta')|^2 / (gamma^2 (1 - beta cos theta)^2)``;
    the two helicities sum to the azimuth-averaged boosted dipole pattern and
    integrate to 1 over the sphere.

    Validation: cr-axial-helicity-emission
    """
    if abs(m) != 1 or abs(helicity) != 1:
        raise ValueError("dipole photons have m, helicity = +-1")
    gamma = float(gamma)
    denom = doppler_denominator((gamma - 1.0) * M_E_EV, theta_rad)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    cos_rest = (np.cos(theta_rad) - beta) / denom
    d = 0.5 * (1.0 + m * helicity * cos_rest)
    return 3.0 / (8.0 * np.pi) * d**2 / (gamma * denom) ** 2


@lru_cache(maxsize=8)
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def photon_angular_momentum(
    polarization: np.ndarray, gamma: float, theta_max_rad: float = np.pi, n_theta: int = 256
) -> dict[str, float]:
    """Photon-number-weighted angular momentum of a line inside a lab cone ``theta <= theta_max``.

    Returns ``fraction`` (photons in the cone per emitted photon), ``helicity``
    (mean ``Lambda``), ``J_z``, ``S_z = <Lambda cos theta>`` and
    ``L_z = J_z - S_z`` per photon in the cone [hbar].

    Validation: cr-axial-helicity-emission
    """
    t = np.asarray(polarization, dtype=complex)
    weights = {1: float(t[0, 0].real), -1: float(t[1, 1].real)}
    total = weights[1] + weights[-1]
    if not total > 0.0:
        raise ValueError("line has no emission")
    # Quadrature in the rest-frame cosine (smooth); map the cone edge by aberration.
    beta = np.sqrt(1.0 - 1.0 / float(gamma) ** 2)
    c_lab = np.cos(theta_max_rad)
    c_edge = (c_lab - beta) / (1.0 - beta * c_lab)
    nodes, w = _gauss_legendre(n_theta)
    c_rest = 0.5 * (1.0 - c_edge) * nodes + 0.5 * (1.0 + c_edge)
    w = 0.5 * (1.0 - c_edge) * w
    cos_lab = (c_rest + beta) / (1.0 + beta * c_rest)
    number = helicity = spin = momentum = 0.0
    for m, weight in weights.items():
        for lam in (1, -1):
            # (3/8pi) |d|^2 dOmega' -> (3/4) |d|^2 dcos' after the azimuth.
            density = 0.75 * (0.5 * (1.0 + m * lam * c_rest)) ** 2 * w * weight / total
            n = float(density.sum())
            number += n
            helicity += lam * n
            spin += lam * float(np.sum(density * cos_lab))
            momentum += m * n
    j_z = momentum / number
    s_z = spin / number
    return {
        "fraction": number,
        "helicity": helicity / number,
        "J_z": j_z,
        "S_z": s_z,
        "L_z": j_z - s_z,
    }


# ---------------------------------------------------------------------------
# Level densities from grid states (MCWF observable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelDensity:
    """Observable: on-column level density matrices of a grid state.

    ``density(phi, tilt)`` returns the concatenated, flattened
    ``rho_L = a a^dagger`` (``a_s = <orbital_s | psi>``) of every level, linear
    in the state's density matrix, so an :class:`.axial_mcwf.AxialMCWF`
    ensemble mean is the Lindblad level density. :meth:`unpack` splits it.

    Validation: cr-axial-helicity-emission
    """

    cuda_compatible = True

    catalog: AxialLineCatalog
    grid: RealSpaceGrid
    center_ang: tuple[float, float]
    _orbitals: list[np.ndarray] = field(init=False, repr=False)
    _cuda_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_orbitals", self.catalog.orbitals(self.grid, self.center_ang))

    def __call__(self, phi: np.ndarray, tilt_inv_ang=(0.0, 0.0)) -> np.ndarray:
        xp = np
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as xp

            # Copy static projection orbitals to VRAM only once per observable.
            if self._cuda_cache is None:
                object.__setattr__(
                    self,
                    "_cuda_cache",
                    (
                        xp.asarray(self.grid.x)[:, None],
                        xp.asarray(self.grid.y)[None, :],
                        tuple(xp.asarray(orbitals) for orbitals in self._orbitals),
                    ),
                )
            assert self._cuda_cache is not None
            x, y, all_orbitals = self._cuda_cache
        else:
            x = self.grid.x[:, None]
            y = self.grid.y[None, :]
            all_orbitals = self._orbitals
        psi = xp.exp(1j * (tilt_inv_ang[0] * x + tilt_inv_ang[1] * y)) * phi
        blocks = []
        for orbitals in all_orbitals:
            amplitude = xp.einsum("sij,ij->s", orbitals.conj(), psi) * self.grid.cell_area_ang2
            blocks.append(xp.outer(amplitude, amplitude.conj()).ravel())
        return xp.concatenate(blocks)

    def sum_batch(self, states, tilts):
        """Reduce orbital density matrices across live electrons on device."""
        xp = np
        if hasattr(states, "__cuda_array_interface__"):
            import cupy as xp
        self(states[0], tilts[0])  # populate immutable device orbitals once
        if xp is np:
            x, y, orbitals = self.grid.x[:, None], self.grid.y[None, :], self._orbitals
        else:
            assert self._cuda_cache is not None
            x, y, orbitals = self._cuda_cache
        q = xp.asarray(tilts)[:, :, None, None]
        psi = states * xp.exp(1j * (q[:, 0] * x + q[:, 1] * y))
        blocks = []
        for basis in orbitals:
            amplitude = xp.einsum("sij,bij->bs", basis.conj(), psi) * self.grid.cell_area_ang2
            blocks.append((amplitude.T @ amplitude.conj()).ravel())
        return xp.concatenate(blocks)

    def unpack(self, values: np.ndarray) -> list[np.ndarray]:
        """Split an observable vector into per-level density matrices."""
        out, start = [], 0
        for level in self.catalog.levels:
            n = len(level.states)
            out.append(np.asarray(values[start : start + n * n]).reshape(n, n))
            start += n * n
        return out


__all__ = [
    "AxialLevel",
    "AxialLine",
    "AxialLineCatalog",
    "LevelDensity",
    "axial_line_catalog",
    "axis_stokes",
    "helicity_pattern",
    "photon_angular_momentum",
]
