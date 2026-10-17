"""Lindblad scattering kernels for planar channeling.

A planar-channeled electron's transverse density matrix ``rho(x, x')`` obeys,
per unit depth ``z`` (paraxial, ``i hbar v d/dz psi = H_perp psi``),

``d rho/dz = -(i / hbar v) [H_perp, rho]
+ [C(x, x') - C(x, x)/2 - C(x', x')/2] rho(x, x')``,

where the kernel ``C(x, x') = sum_k L_k(x) L_k(x')*`` collects every incoherent
jump operator. Its diagonal is the local loss rate, so the imaginary
(absorptive) potential is ``W(x) = (hbar v / 2) C(x, x)``. This module builds
``C`` for two mechanisms.

**Thermal diffuse scattering (TDS).** Each atom adds a phase
``phi(x, y) = -(1/hbar v) V_proj``. With independent Gaussian displacements
(Einstein model) with transverse covariance ``S`` in the frame ``x`` (plane
normal), ``y`` (in the plane, normal to the beam), the incoherent part of the
transmission is the phase covariance, ``C(x, x') = Cov[phi(x), phi(x')]`` per
unit depth. In Fourier form, with ``q = (p+G, q_y)`` and ``q' = (p, q_y)``,

``C(x, x') = (1/(hbar v)^2) sum_G n_G e^{iGx} int dp dq_y/(2 pi)^2
v(q) v(q') e^{ip(x-x')} [e^{-S_xx G^2/2} - e^{-(q.S.q + q'.S.q')/2}]``,

with ``v(q) = -4 pi e^2 (Z - f0(q)) / q^2`` (Mott--Bethe) and ``n_G`` the
static site density of each species. An isotropic site has ``S = a^2 I``,
``a^2 = B / 8 pi^2``. Displacement along the beam does not change the
projected potential, so only the transverse block of the site covariance
``U`` enters; ``S_xy`` couples ``q_y`` to ``p``. ``G = 0`` is the familiar
``|f^e|^2 (1 - e^{-2 B s^2})`` TDS cross section (Pennycook & Jesson 1991)
times the mean density. The diagonal is the Hall--Hirsch absorptive potential
(Hall & Hirsch 1965; Bird & King 1990), which is the open-system form of the
Yoshioka (1957) optical potential. ``C`` is a covariance, so it is Hermitian
and positive semidefinite.

**Electronic excitation.** Local-density approximation with localization
length ``1/|q_perp|`` (maintainer decision, #504):
``C_el(x, x') = int d^2q_perp w(q) sqrt(rho_q(x) rho_q(x')) e^{i q_x (x-x')}``,
where ``rho_q`` is the electron density Gaussian-smoothed with
``sigma = 1/|q_perp|`` and ``w`` is the per-electron PENELOPE-2024 shell GOS
cross section resolved in ``q_perp`` (distant longitudinal plus close
Møller). It is uniform as ``q -> 0``, so soft collisions cannot decohere
spurious position differences, and local as ``q -> inf``. The distant
transverse term transfers ``q_perp ~ 0`` with a uniform weight, which is a
null Lindblad term, so it is dropped.

**Representation.** A translation-averaged beam keeps ``rho`` block-diagonal
in Bloch ``k``. On a supercell of ``n_cells`` periods, Bloch momenta are
``kappa_r + h g`` with ``kappa_r = r g / n_cells``. Because ``C`` is invariant
under a common shift by one period, its momentum matrix couples only momenta
in the same residue ``r``, so it is stored as
``momentum_kernel[r, i, j] = C(kappa_r + h_i g, kappa_r + h_j g)`` with the
``p`` integral binned on ``g / n_cells``. Both momenta are restricted to
orders ``|h| <= max_order``: the projection ``P C P`` stays Hermitian and positive
semidefinite. Jumps that leave that band cannot land in a basis of
``max_order`` plane waves; solvers count them through the full diagonal
``C(x, x)``, which keeps every momentum transfer.

Lengths in angstroms, energies in eV; kernels in 1/angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.special import ive, j0, jv

from ...materials.atomic import Z_TABLE, cromer_mann_f0
from ...materials.channeling.potential import F0_S_MAX_INV_ANG, PlanarProfile
from ...materials.channeling.thermal import ChannelingDataError
from ...materials.channeling.transverse import longitudinal_kinematics
from ...materials.crystal import (
    CRYSTALS,
    E2_EV_ANG,
    HBARC_EV_ANG,
    M_E_EV,
    _direct_lattice_vectors,
    reciprocal_g_vector,
)
from ..transport.inelastic import _qmin_ev
from ..transport.shell_gos import _PREF_EV_CM2, _triangle_moments, max_energy_loss_eV
from ..transport.shell_oscillators import MaterialShellOscillators
from ..transport.shell_rates import catalog_shell_oscillators

DEFAULT_CELLS = 32
DEFAULT_MAX_ORDER = 24
_ANG2_PER_CM2 = 1e16
# Below this |q| the Mott--Bethe ratio (Z - f0)/q^2 is held at its value
# here; the omitted O(q^2 <r^4>) curvature is < 1e-5 relative.
_Q_SMALL_INV_ANG = 0.02
# Upper |q| of every q integral. The TDS tail beyond it carries a relative
# share ~ 1/(a Q)^2 < 1e-5; the electronic close tail ~ (q_typ / Q)^2.
_Q_MAX_INV_ANG = 1.0e4
_GL = {n: np.polynomial.legendre.leggauss(n) for n in (2, 3, 8, 24)}


def _panels(edges, order: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss--Legendre nodes and weights over consecutive ``edges``."""
    x, w = _GL[order]
    edges = np.asarray(edges, dtype=float)
    lo, hi = edges[:-1, None], edges[1:, None]
    half = 0.5 * (hi - lo)
    return (lo + half * (x + 1.0)).ravel(), (half * w).ravel()


def _geometric_edges(start: float, stop: float, ratio: float = 2.0) -> np.ndarray:
    """``0, start, start*ratio, ..., stop``: panels that resolve a scale ``start``."""
    count = max(1, int(np.ceil(np.log(stop / start) / np.log(ratio))))
    return np.concatenate([[0.0], start * ratio ** np.arange(count), [stop]])


# ---------------------------------------------------------------------------
# Thermal diffuse scattering
# ---------------------------------------------------------------------------


@cache
def _potential_table(element: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(log q, v(q))`` [eV angstrom^3] for one element's Mott--Bethe potential.

    ``f0`` is rescaled by ``Z / f0(0)`` so ``Z - f0 -> 0`` at ``q = 0``: the
    Waasmaier--Kirfel fits miss ``Z`` by ~1e-4 (Si: 13.9989), which would
    otherwise give ``v ~ 1/q^2`` and a divergent TDS integral. Beyond the fit
    range ``s > 6 / angstrom``, ``f0`` decays as ``s^-4`` (the hydrogenic tail).
    """
    z = float(Z_TABLE[element])
    q = np.geomspace(_Q_SMALL_INV_ANG, 2.0 * _Q_MAX_INV_ANG, 6000)
    q_fit = 4.0 * np.pi * F0_S_MAX_INV_ANG
    scale = z / float(cromer_mann_f0(element, np.array([0.0]))[0])
    f0 = scale * cromer_mann_f0(element, np.minimum(q, q_fit))
    f0 = np.where(q > q_fit, f0 * (q_fit / q) ** 4, f0)
    return np.log(q), -4.0 * np.pi * E2_EV_ANG * (z - f0) / q**2


def _potential(element: str, q) -> np.ndarray:
    log_q, values = _potential_table(element)
    q = np.maximum(np.asarray(q, dtype=float), _Q_SMALL_INV_ANG)
    return np.interp(np.log(q), log_q, values)


def _site_densities(profile: PlanarProfile, max_harmonic: int) -> dict[str, np.ndarray]:
    """Static per-element site-density coefficients ``n_h`` [1/angstrom^3].

    Index ``h + max_harmonic`` holds harmonic ``h`` of the true period, in the
    phase convention of :func:`~pyrite.materials.channeling.potential.planar_profile`.
    Multiplying by ``exp(-B s_h^2)`` and summing elements gives
    ``profile.nuclear_density``.
    """
    if profile.crystal not in CRYSTALS:
        raise ValueError(f"profile crystal {profile.crystal!r} is not a catalog crystal")
    info = CRYSTALS[profile.crystal]
    _, g_coprime = reciprocal_g_vector(profile.hkl, info["lattice"])
    step = int(round(profile.g_inv_ang / g_coprime))
    harmonics = np.arange(-max_harmonic, max_harmonic + 1) * step
    out: dict[str, np.ndarray] = {}
    for element, position in info["basis"]:
        phase = np.exp(-2j * np.pi * harmonics * float(np.dot(profile.hkl, position)))
        out[element] = out.get(element, 0.0) + phase / float(info["V_cell"])
    return out


Covariance = tuple[float, float, float]


def _transverse_covariances(profile: PlanarProfile, beam_uvw=None) -> dict[str, Covariance]:
    """Per-element transverse displacement covariance ``(S_xx, S_xy, S_yy)`` [angstrom^2].

    ``x`` is the plane normal and ``y = x cross b`` with ``b`` the in-plane
    projection of the beam direction ``beam_uvw`` (direct-lattice indices of
    the catalog cell). ``S`` is the ``(x, y)`` block of the site covariance
    ``U``. Isotropic profiles give ``S = (B / 8 pi^2) I`` for any beam. An
    anisotropic profile needs ``beam_uvw`` unless ``x`` is an eigenvector of
    every ``U`` and the in-plane block is isotropic (basal planes of a
    uniaxial crystal).

    Validation: cr-lindblad-jump-operators
    """
    if profile.covariances_ang2 is None:
        if profile.thermal_projected:
            raise ChannelingDataError(f"{profile.crystal}: projected profile carries no covariance")
        return {
            el: (b / (8.0 * np.pi**2), 0.0, b / (8.0 * np.pi**2))
            for el, b in profile.b_factors_ang2.items()
        }
    n = np.asarray(profile.normal, dtype=float)
    plane = np.eye(3) - np.outer(n, n)
    if beam_uvw is None:
        for el, u in profile.covariances_ang2.items():
            block = plane @ u @ plane
            tol = 1e-12 * np.trace(u)
            if np.linalg.norm(plane @ u @ n) > tol or not np.allclose(
                block, 0.5 * np.trace(block) * plane, atol=tol
            ):
                raise ChannelingDataError(
                    f"{profile.crystal}{profile.hkl}: transverse displacement of {el} depends "
                    "on the beam azimuth; pass beam_uvw"
                )
        y = np.linalg.svd(plane)[0][:, 0]
    else:
        uvw = np.asarray(beam_uvw, dtype=float)
        if uvw.shape != (3,) or not np.all(np.isfinite(uvw)):
            raise ValueError("beam_uvw must be three finite numbers")
        beam = uvw @ np.array(_direct_lattice_vectors(CRYSTALS[profile.crystal]["lattice"]))
        along = plane @ beam
        if np.linalg.norm(along) <= 1e-9 * np.linalg.norm(beam):
            raise ValueError("beam_uvw must not be parallel to the plane normal")
        y = np.cross(n, along / np.linalg.norm(along))
    out: dict[str, Covariance] = {}
    for el, u in profile.covariances_ang2.items():
        sxx, sxy, syy = float(n @ u @ n), float(n @ u @ y), float(y @ u @ y)
        if sxx <= 0.0 or sxx * syy - sxy**2 <= 0.0:
            raise ChannelingDataError(
                f"{profile.crystal}.{el}: covariance is not positive definite"
            )
        out[el] = (sxx, sxy, syy)
    return out


def _min_eigenvalue(cov: Covariance) -> float:
    sxx, sxy, syy = cov
    return 0.5 * (sxx + syy) - float(np.hypot(0.5 * (sxx - syy), sxy))


# ---------------------------------------------------------------------------
# Electronic excitation
# ---------------------------------------------------------------------------


def _recoil_wavenumber(recoil_eV):
    """``q`` [1/angstrom] of recoil energy ``Q``: ``(hbar c q)^2 = Q (Q + 2 m c^2)``."""
    recoil = np.asarray(recoil_eV, dtype=float)
    return np.sqrt(recoil * (recoil + 2.0 * M_E_EV)) / HBARC_EV_ANG


@dataclass(frozen=True)
class ElectronicChannels:
    """Per-electron, ``q_perp``-resolved PENELOPE-2024 shell GOS at one energy.

    Attributes
    ----------
    energy_eV
        Electron kinetic energy [eV].
    electrons_per_formula
        Electrons per formula unit; cross sections are divided by it.
    distant_ang2
        Distant longitudinal strength ``pref f_k int p_dis / W dW`` per
        electron [angstrom^2], one entry per oscillator.
    distant_qmin_inv_ang, distant_qmax_inv_ang
        Zero-angle and Bethe-ridge momentum transfers ``q(Q_-)``, ``q(Q'_k)``.
    close_ang2_eV
        Close-collision strength ``pref f_k`` per electron [angstrom^2 eV].
    close_lower_eV, close_upper_eV
        Close-loss range ``[Q_k, W_max]`` per oscillator.
    close_prime_eV
        ``E' = E + U_k`` of the Møller factor.
    """

    energy_eV: float
    electrons_per_formula: float
    distant_ang2: np.ndarray
    distant_qmin_inv_ang: np.ndarray
    distant_qmax_inv_ang: np.ndarray
    close_ang2_eV: np.ndarray
    close_lower_eV: np.ndarray
    close_upper_eV: np.ndarray
    close_prime_eV: np.ndarray

    def distant(self, q_perp):
        """Distant longitudinal ``dsigma/d^2q_perp`` per electron [angstrom^4].

        From PENELOPE-2024 Eq. 3.104's ``dQ / (Q (1 + Q/2mc^2))`` with
        ``(hbar c q)^2 = Q(Q + 2mc^2)`` and ``q^2 = q_perp^2 + q_min^2``:
        ``(pref f_k / W) / (pi q^2 (1 + Q/mc^2))`` for ``q <= q(Q'_k)``.

        Validation: cr-lindblad-jump-operators
        """
        q_perp = np.asarray(q_perp, dtype=float)[..., None]
        q2 = q_perp**2 + self.distant_qmin_inv_ang**2
        recoil = np.sqrt(HBARC_EV_ANG**2 * q2 + M_E_EV**2) - M_E_EV
        value = self.distant_ang2 / (np.pi * q2 * (1.0 + recoil / M_E_EV))
        return np.where(q2 <= self.distant_qmax_inv_ang**2, value, 0.0).sum(axis=-1)

    def close_loss(self, loss_eV):
        """Close Møller ``dsigma/dW`` per electron [angstrom^2 / eV], summed over shells.

        Validation: cr-lindblad-jump-operators
        """
        w = np.asarray(loss_eV, dtype=float)[..., None]
        a = (self.energy_eV / (self.energy_eV + M_E_EV)) ** 2
        prime = self.close_prime_eV
        ratio = w / (prime - w)
        factor = 1.0 + ratio**2 - (1.0 - a) * ratio + a * (w / prime) ** 2
        inside = (w >= self.close_lower_eV) & (w <= self.close_upper_eV)
        return np.where(inside, self.close_ang2_eV * factor / w**2, 0.0).sum(axis=-1)

    def close_transfer(self, loss_eV):
        """Free-electron ``q_perp`` [1/angstrom] of a close loss: ``(hbar c q)^2 = 2mc^2 W (1 - W/E)``."""
        w = np.asarray(loss_eV, dtype=float)
        return np.sqrt(2.0 * M_E_EV * w * (1.0 - w / self.energy_eV)) / HBARC_EV_ANG

    def close(self, q_perp):
        """Close ``dsigma/d^2q_perp`` per electron [angstrom^4] on the ``W <= E/2`` branch.

        Binary kinematics give ``(hbar c q_perp)^2 = 2 m c^2 W (1 - W/E)``, which
        increases up to ``W = E/2``; losses above ``E/2`` (only possible for
        ``U_k > 0``) carry a share ``< U_k/E`` and appear only in the diagonal.

        Validation: cr-lindblad-jump-operators
        """
        q = np.asarray(q_perp, dtype=float)
        x = 2.0 * (HBARC_EV_ANG * q) ** 2 / (M_E_EV * self.energy_eV)
        valid = x < 1.0
        root = np.sqrt(np.where(valid, 1.0 - x, 1.0))
        loss = (HBARC_EV_ANG * q) ** 2 / (M_E_EV * (1.0 + root))
        jacobian = HBARC_EV_ANG**2 / (2.0 * M_E_EV * root)  # dW / dq_perp^2
        return np.where(valid, self.close_loss(loss) * jacobian / np.pi, 0.0)

    def weight(self, q_perp):
        """Total ``dsigma/d^2q_perp`` per electron [angstrom^4]."""
        return self.distant(q_perp) + self.close(q_perp)


def electronic_channels(
    oscillators: MaterialShellOscillators, kinetic_energy_eV: float
) -> ElectronicChannels:
    """Resolve PENELOPE-2024 shell GOS channels in transverse momentum.

    Uses the same oscillator kinematics as
    :func:`~pyrite.montecarlo.transport.shell_gos.shell_gos_moments`
    (near-threshold ``W'_k``, ``Q'_k``; Eqs. 3.76--3.80, 3.83, 3.96), so the
    ``q_perp`` integrals of :meth:`ElectronicChannels.distant` and
    :meth:`ElectronicChannels.close_loss` equal its zeroth moments per electron.

    Validation: cr-lindblad-jump-operators
    """
    energy = float(kinetic_energy_eV)
    _, beta = longitudinal_kinematics(energy)
    pref = _PREF_EV_CM2 / beta**2 * _ANG2_PER_CM2 / oscillators.electrons_per_formula
    rows = []
    for osc in oscillators.oscillators:
        u, w, f = osc.ionization_energy_eV, osc.resonance_energy_eV, osc.strength
        w_max = max_energy_loss_eV(energy, u, "electron")
        if u > 0.0 and energy <= u:
            # Closed channel: zero strength and an empty close range.
            rows.append((0.0, 1.0, 0.0, 0.0, u, u, energy + u))
            continue
        if u > 0.0:
            w_dis_full = 3.0 * w - 2.0 * u
            if energy > w_dis_full:
                w_mod, q_mod = w, u
            else:
                w_mod, q_mod = (energy + 2.0 * u) / 3.0, u * energy / w_dis_full
            loss0 = _triangle_moments(u, 3.0 * w_mod - 2.0 * u, u, w_max)[0]
        else:
            w_mod, q_mod = w, w
            loss0 = 1.0 / w if w < w_max else 0.0
        strength, q_lo, q_hi = 0.0, 1.0, 0.0
        if w_mod < energy and loss0 > 0.0:
            q_minus = float(_qmin_ev(energy, w_mod))
            if q_minus < q_mod:
                strength = pref * f * loss0
                q_lo = float(_recoil_wavenumber(q_minus))
                q_hi = float(_recoil_wavenumber(q_mod))
        lower = u if u > 0.0 else q_mod
        rows.append((strength, q_lo, q_hi, pref * f, lower, w_max, energy + u))
    dist, qmin, qmax, close, lower, upper, prime = (np.asarray(c) for c in zip(*rows, strict=True))
    return ElectronicChannels(
        energy_eV=energy,
        electrons_per_formula=float(oscillators.electrons_per_formula),
        distant_ang2=dist,
        distant_qmin_inv_ang=qmin,
        distant_qmax_inv_ang=qmax,
        close_ang2_eV=close,
        close_lower_eV=lower,
        close_upper_eV=upper,
        close_prime_eV=prime,
    )


@dataclass(frozen=True)
class _RootDensity:
    """Fourier coefficients of ``sqrt(rho_q)`` tabulated on ``log q``."""

    log_q: np.ndarray
    orders: np.ndarray
    coefficients: np.ndarray  # (n_q, n_orders)

    def hat_weights(self, q):
        """Linear-in-``log q`` interpolation ``(index, weight)`` pairs, clamped at the ends."""
        x = np.clip(np.log(np.maximum(q, 1e-300)), self.log_q[0], self.log_q[-1])
        upper = np.clip(np.searchsorted(self.log_q, x), 1, self.log_q.size - 1)
        frac = (x - self.log_q[upper - 1]) / (self.log_q[upper] - self.log_q[upper - 1])
        return upper - 1, 1.0 - frac, upper, frac

    def autocorrelation(self, q, g: float, delta_ang):
        """``<s_q(x) s_q(x + delta)>_x`` for ``s_q = sqrt(rho_q)``; shape ``q.shape + delta.shape``."""
        lo, w_lo, hi, w_hi = self.hat_weights(q)
        power = np.abs(self.coefficients) ** 2
        power = w_lo[:, None] * power[lo] + w_hi[:, None] * power[hi]
        cosine = np.cos(np.multiply.outer(self.orders * g, np.asarray(delta_ang, dtype=float)))
        return power @ cosine


def _root_density(profile: PlanarProfile, max_order: int) -> _RootDensity:
    """Tabulate ``sqrt(rho_e * Gaussian(1/q))`` harmonics ``|h| <= max_order``.

    ``rho_q`` keeps harmonics ``rho_G exp(-G^2 / 2 q^2)``: uniform for
    ``q << g`` and the local density for ``q`` above its highest harmonic.
    """
    g = profile.g_inv_ang
    rho = profile.electron_density
    n_x = max(1024, 8 * max_order)
    x = np.arange(n_x) * profile.period_ang / n_x
    log_q = np.linspace(np.log(0.02 * g), np.log(4.0 * g * max(rho.size, max_order)), 160)
    harmonics = np.arange(rho.size) * g
    smooth = np.exp(-0.5 * (harmonics[None, :] / np.exp(log_q)[:, None]) ** 2) * rho
    phase = np.exp(1j * np.multiply.outer(x, harmonics[1:]))
    density = smooth[:, :1].real + 2.0 * (smooth[:, 1:] @ phase.T).real
    root = np.sqrt(np.maximum(density, 0.0))
    spectrum = np.fft.fft(root, axis=1) / n_x
    orders = np.arange(-max_order, max_order + 1)
    return _RootDensity(log_q, orders, spectrum[:, orders % n_x])


# ---------------------------------------------------------------------------
# Kernel container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatteringKernel:
    """Dissipator kernel ``C(x, x')`` of one or more scattering mechanisms.

    Attributes
    ----------
    label
        Mechanism name(s), joined by ``+`` for sums.
    period_ang
        True planar period ``d`` [angstrom].
    n_cells
        Supercell periods; Bloch momenta ``kappa_r = r g / n_cells``.
    orders
        Plane-wave orders ``h = -H..H`` of the kept momentum band.
    momentum_kernel
        ``C(kappa_r + h_i g, kappa_r + h_j g)`` [1/angstrom], shape
        ``(n_cells, 2H + 1, 2H + 1)``, Hermitian and positive semidefinite per
        ``r``. ``C(x, x') = sum_{r,i,j} K[r,i,j] e^{i k_i x} e^{-i k_j x'}``.
    diagonal_coefficients
        Fourier coefficients ``D_m`` [1/angstrom], ``m >= 0``, of the full
        ``C(x, x)`` with every momentum transfer; ``D_{-m} = conj(D_m)``.
    hbar_v_eV_ang
        ``hbar v`` [eV angstrom] of the electron energy.
    """

    label: str
    period_ang: float
    n_cells: int
    orders: np.ndarray
    momentum_kernel: np.ndarray
    diagonal_coefficients: np.ndarray
    hbar_v_eV_ang: float

    @property
    def g_inv_ang(self) -> float:
        """Reciprocal period ``2 pi / d`` [1/angstrom]."""
        return 2.0 * np.pi / self.period_ang

    @property
    def max_order(self) -> int:
        """Highest plane-wave order ``H`` of the kept band."""
        return int(self.orders[-1])

    def wavevectors(self) -> np.ndarray:
        """Bloch momenta ``kappa_r + h g`` [1/angstrom], shape ``(n_cells, 2H + 1)``."""
        kappa = np.arange(self.n_cells) * self.g_inv_ang / self.n_cells
        return kappa[:, None] + self.orders[None, :] * self.g_inv_ang

    def __add__(self, other: ScatteringKernel) -> ScatteringKernel:
        same = (
            np.isclose(self.period_ang, other.period_ang)
            and self.n_cells == other.n_cells
            and np.array_equal(self.orders, other.orders)
            and np.isclose(self.hbar_v_eV_ang, other.hbar_v_eV_ang)
        )
        if not same:
            raise ValueError("kernels must share period, cells, orders and energy")
        size = max(self.diagonal_coefficients.size, other.diagonal_coefficients.size)
        diagonal = np.zeros(size, dtype=complex)
        diagonal[: self.diagonal_coefficients.size] += self.diagonal_coefficients
        diagonal[: other.diagonal_coefficients.size] += other.diagonal_coefficients
        return ScatteringKernel(
            label=f"{self.label}+{other.label}",
            period_ang=self.period_ang,
            n_cells=self.n_cells,
            orders=self.orders,
            momentum_kernel=self.momentum_kernel + other.momentum_kernel,
            diagonal_coefficients=diagonal,
            hbar_v_eV_ang=self.hbar_v_eV_ang,
        )

    def diagonal(self, x_ang):
        """Full loss rate ``C(x, x)`` [1/angstrom], every momentum transfer included.

        Validation: cr-lindblad-jump-operators
        """
        x = np.asarray(x_ang, dtype=float)
        m = np.arange(1, self.diagonal_coefficients.size)
        phase = np.exp(1j * self.g_inv_ang * np.multiply.outer(x, m))
        return (
            self.diagonal_coefficients[0].real + 2.0 * (phase @ self.diagonal_coefficients[1:]).real
        )

    def absorptive_potential_eV(self, x_ang):
        """Imaginary optical potential ``W(x) = (hbar v / 2) C(x, x)`` [eV].

        Validation: cr-lindblad-jump-operators
        """
        return 0.5 * self.hbar_v_eV_ang * self.diagonal(x_ang)

    def kept_diagonal(self, x_ang):
        """``C(x, x)`` of the kept momentum band [1/angstrom]; at most :meth:`diagonal`."""
        x = np.atleast_1d(np.asarray(x_ang, dtype=float))
        phase = np.exp(1j * self.g_inv_ang * np.multiply.outer(x, self.orders))
        total = np.einsum("xi,rij,xj->x", phase, self.momentum_kernel, phase.conj()).real
        return total.reshape(np.shape(x_ang)) if np.ndim(x_ang) else float(total[0])

    def cell_kernel(self, n_grid: int) -> np.ndarray:
        """Return ``K[j, a, b] = C(x_a, x_b + j d)`` on ``x_a = a d / n_grid``.

        ``j`` is the cell offset modulo ``n_cells``. ``n_grid`` must resolve
        the kept band, ``n_grid > 2 H + 1``, so the band does not alias.

        Validation: cr-lindblad-jump-operators
        """
        if n_grid <= 2 * self.max_order + 1:
            raise ValueError("n_grid must exceed 2 * max_order + 1")
        x = np.arange(n_grid) * self.period_ang / n_grid
        waves = np.exp(1j * np.multiply.outer(self.wavevectors(), x))  # (r, i, a)
        blocks = np.einsum("ria,rij,rjb->rab", waves, self.momentum_kernel, waves.conj())
        return np.fft.fft(blocks, axis=0)


def _kinematics(kinetic_energy_eV: float) -> float:
    if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0.0:
        raise ValueError("kinetic_energy_eV must be finite and positive")
    _, beta = longitudinal_kinematics(kinetic_energy_eV)
    return HBARC_EV_ANG * beta


def _check_grid(n_cells: int, max_order: int) -> None:
    if n_cells < 1 or max_order < 1:
        raise ValueError("n_cells and max_order must be positive")


def _transfer_bins(n_cells: int, max_order: int) -> np.ndarray:
    """Bin indices ``l = r + nu n_cells`` for every residue ``r`` and ``|nu| <= max_order``.

    Bin ``l`` covers momenta ``[(l - 1/2), (l + 1/2)] g / n_cells``; each
    residue then carries the same plane-wave orders, which keeps every
    residue block a complete principal submatrix.
    """
    return np.arange(-max_order * n_cells, (max_order + 1) * n_cells)


# ---------------------------------------------------------------------------
# TDS kernel
# ---------------------------------------------------------------------------


def _tds_strength(element, cov, p, harmonic_g, q_y):
    """``int dq_y/(2 pi)^2 v(p+G) v(p) [bracket]`` over the given ``q_y`` nodes.

    ``p``: (P,), ``harmonic_g``: (N,), ``q_y``: (nodes, weights) on ``[0, inf)``.
    Returns (P, N) in eV^2 angstrom^5.

    Validation: cr-lindblad-jump-operators
    """
    sxx, sxy, syy = cov
    nodes, weights = q_y
    ket = p[:, None, None] + harmonic_g[None, :, None]
    bra = p[:, None, None]
    qy2 = nodes**2
    v_ket = _potential(element, np.sqrt(ket**2 + qy2))
    v_bra = _potential(element, np.sqrt(bra**2 + qy2))
    quad = sxx * (ket**2 + bra**2) + 2.0 * syy * qy2
    if sxy == 0.0:
        damp = np.exp(-0.5 * quad)
    else:
        # v is even in q_y but S_xy flips sign with it: average +-q_y.
        cross = 2.0 * sxy * nodes * (ket + bra)
        damp = 0.5 * (np.exp(-0.5 * (quad + cross)) + np.exp(-0.5 * (quad - cross)))
    bracket = np.exp(-0.5 * sxx * harmonic_g[None, :, None] ** 2) - damp
    # q_y is integrated over (-inf, inf): twice the half line.
    return 2.0 * (v_ket * v_bra * bracket) @ weights / (2.0 * np.pi) ** 2


def _tds_diagonal(profile, sites, covariances, hbar_v, max_harmonic):
    """Fourier coefficients ``D_m`` of the full TDS ``C(x, x)``, symmetric polar quadrature.

    With ``k = (p + G/2, q_y) = r (cos t, sin t)`` the bracket is
    ``e^{-2 S_xx (G/2)^2} - e^{-k.S.k - S_xx (G/2)^2}``; ``t`` and ``-t`` share
    the potentials, so the half circle carries the average over both signs of
    the ``S_xy`` term.

    Validation: cr-lindblad-jump-operators
    """
    g = profile.g_inv_ang
    radius, w_r = _panels(_geometric_edges(0.05, _Q_MAX_INV_ANG), 8)
    angle, w_a = _GL[24]
    angle, w_a = 0.5 * np.pi * (angle + 1.0), 0.5 * np.pi * w_a  # [0, pi], doubled below
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    r2 = radius[:, None] ** 2
    m = np.arange(max_harmonic + 1)
    coefficients = np.zeros(m.size, dtype=complex)
    for element, density in sites.items():
        sxx, sxy, syy = covariances[element]
        k_s_k = r2 * (sxx * cos_t**2 + syy * sin_t**2)[None, :]
        k_cross = r2 * (2.0 * sxy * cos_t * sin_t)[None, :]
        for index in m:
            half = 0.5 * index * g
            cross = 2.0 * radius[:, None] * half * cos_t[None, :]
            base = radius[:, None] ** 2 + half**2
            v_ket = _potential(element, np.sqrt(np.maximum(base + cross, 0.0)))
            v_bra = _potential(element, np.sqrt(np.maximum(base - cross, 0.0)))
            exponent = k_s_k + sxx * half**2
            if sxy == 0.0:
                damp = np.exp(-exponent)
            else:
                damp = 0.5 * (np.exp(-(exponent + k_cross)) + np.exp(-(exponent - k_cross)))
            bracket = np.exp(-2.0 * sxx * half**2) - damp
            integral = 2.0 * (radius * w_r) @ (v_ket * v_bra * bracket) @ w_a
            coefficients[index] += density[max_harmonic + index] * integral
    return coefficients / (2.0 * np.pi * hbar_v) ** 2


def tds_kernel(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    *,
    n_cells: int = DEFAULT_CELLS,
    max_order: int = DEFAULT_MAX_ORDER,
    beam_uvw=None,
) -> ScatteringKernel:
    """Return the Einstein-model thermal diffuse scattering kernel.

    Implements the phase-covariance kernel of the module docstring with
    static site densities from the crystal basis, per-element transverse
    covariance ``S`` from the profile (``a^2 I``, ``a^2 = B(T) / 8 pi^2``, for
    isotropic sites; rotated into the beam frame for anisotropic ones, so
    ``beam_uvw`` is required when ``S`` depends on the beam azimuth) and the
    Mott--Bethe potential of
    :func:`~pyrite.materials.channeling.potential.planar_profile`. The ``p``
    integral uses two Gauss--Legendre nodes per bin of width ``g / n_cells``,
    at the same offsets in every bin, which keeps each residue block an exact
    positive-weight sum of covariance matrices.

    Assumptions: first Born approximation per atom (``Z alpha / beta << 1``),
    uncorrelated (Einstein) Gaussian displacements, quasi-elastic phonon
    scattering (no energy loss) and a static planar lattice for ``n_G``.
    Limits: ``T -> 0`` keeps only zero-point TDS; ``a -> 0`` gives ``C = 0``;
    ``C ∝ 1/v^2``, so ``W ∝ 1/v`` with no ``gamma``.

    Validation: cr-lindblad-jump-operators
    """
    _check_grid(n_cells, max_order)
    hbar_v = _kinematics(kinetic_energy_eV)
    g = profile.g_inv_ang
    covariances = _transverse_covariances(profile, beam_uvw)
    a2_min = min(_min_eigenvalue(cov) for cov in covariances.values())
    # Terms carry exp(-a^2 G^2 / 4) at most (a^2 the smallest eigenvalue of S);
    # stop below 1e-12.
    max_harmonic = int(np.ceil(np.sqrt(4.0 * 27.7 / a2_min) / g))
    sites = _site_densities(profile, max_harmonic)
    width = g / n_cells
    bins = _transfer_bins(n_cells, max_order)
    sub, sub_w = _GL[2]
    p = (bins[:, None] * width + 0.5 * width * sub[None, :]).ravel()
    q_y = _panels(_geometric_edges(0.05, _Q_MAX_INV_ANG), 8)
    size = 2 * max_order + 1
    kernel = np.zeros((n_cells, size, size), dtype=complex)
    residue = bins % n_cells
    bra_order = (bins - residue) // n_cells
    for n in range(-min(max_harmonic, 2 * max_order), min(max_harmonic, 2 * max_order) + 1):
        ket_order = bra_order + n
        keep = np.abs(ket_order) <= max_order
        if not keep.any():
            continue
        p_keep = p.reshape(bins.size, sub.size)[keep].ravel()
        entry = np.zeros(int(keep.sum()), dtype=complex)
        for element, density in sites.items():
            strength = _tds_strength(element, covariances[element], p_keep, np.array([n * g]), q_y)[
                :, 0
            ]
            binned = strength.reshape(-1, sub.size) @ (0.5 * width * sub_w)
            entry += density[max_harmonic + n] * binned
        kernel[residue[keep], ket_order[keep] + max_order, bra_order[keep] + max_order] += entry
    kernel /= hbar_v**2
    kernel = 0.5 * (kernel + np.conj(np.swapaxes(kernel, 1, 2)))
    return ScatteringKernel(
        label="tds",
        period_ang=profile.period_ang,
        n_cells=n_cells,
        orders=np.arange(-max_order, max_order + 1),
        momentum_kernel=kernel,
        diagonal_coefficients=_tds_diagonal(profile, sites, covariances, hbar_v, max_harmonic),
        hbar_v_eV_ang=hbar_v,
    )


# ---------------------------------------------------------------------------
# Electronic kernel
# ---------------------------------------------------------------------------


def _profile_oscillators(profile, oscillators):
    if oscillators is not None:
        return oscillators
    return catalog_shell_oscillators(profile.crystal)


def _breakpoint_panels(geometric: np.ndarray, breaks, order: int = 8):
    """Composite nodes over ``geometric`` edges refined at the ``breaks`` inside them."""
    breaks = np.asarray(breaks, dtype=float)
    inside = breaks[(breaks > geometric[0]) & (breaks < geometric[-1])]
    return _panels(np.unique(np.concatenate([geometric, inside])), order)


def _electronic_edges(channels: ElectronicChannels) -> np.ndarray:
    """Radial ``q_perp`` panel edges, geometric and split at every channel edge."""
    q_floor = 0.25 * float(np.min(channels.distant_qmin_inv_ang))
    ridge = np.sqrt(
        np.maximum(channels.distant_qmax_inv_ang**2 - channels.distant_qmin_inv_ang**2, 0.0)
    )
    breaks = np.concatenate(
        [
            ridge,
            channels.close_transfer(channels.close_lower_eV),
            [channels.close_transfer(0.5 * channels.energy_eV)],
        ]
    )
    geometric = _geometric_edges(q_floor, _Q_MAX_INV_ANG, 1.5)
    inside = breaks[(breaks > geometric[0]) & (breaks < geometric[-1])]
    return np.unique(np.concatenate([geometric, inside]))


def _electronic_radial(channels: ElectronicChannels):
    """Radial ``q_perp`` nodes and ``2 pi q dq`` weights, split at every channel edge."""
    q, w = _panels(_electronic_edges(channels), 8)
    return q, 2.0 * np.pi * q * w


# J0(q Delta) oscillation periods resolved before the tail is tapered off.
_HANKEL_PERIODS = 400


def _oscillatory_panels(edges: np.ndarray, delta: float):
    """Nodes and tapered weights for ``int f(q) J0(q delta) dq`` over ``edges``.

    Panels are at most half a ``J0`` period wide up to ``Q1 = 2 pi N / delta``
    (``N`` periods). Beyond ``Q1`` the oscillating integral of a smooth
    integrand is ``O((Q1 delta)^-1/2 / delta)`` and is dropped, with a
    ``cos^2`` taper over the last quarter of ``[0, Q1]`` so no boundary term
    survives. ``delta = 0`` keeps every panel untapered.
    """
    if delta <= 0.0:
        return _panels(edges, 8)
    top = min(float(edges[-1]), 2.0 * np.pi * _HANKEL_PERIODS / delta)
    width = np.pi / delta
    uniform = np.arange(0.0, top, width)
    inner = edges[edges < top]
    q, w = _panels(np.unique(np.concatenate([inner, uniform, [top]])), 8)
    if top < edges[-1]:
        ramp = np.clip((q - 0.75 * top) / (0.25 * top), 0.0, 1.0)
        w = w * np.cos(0.5 * np.pi * ramp) ** 2
    return q, w


def _close_loss_nodes(channels: ElectronicChannels):
    """Loss nodes and weights over every close range, split at each shell's edges."""
    lower = float(np.min(channels.close_lower_eV))
    upper = float(np.max(channels.close_upper_eV))
    count = max(2, int(4 * np.log(upper / lower)) + 2)
    breaks = np.concatenate([channels.close_lower_eV, channels.close_upper_eV])
    return _breakpoint_panels(np.geomspace(lower, upper, count), breaks)


def _electronic_nodes(channels: ElectronicChannels) -> tuple[np.ndarray, np.ndarray]:
    """Transfer nodes ``q`` and weights ``c`` with ``int d^2q w(q) f(q) ~ sum c f(q)``.

    Distant nodes are radial ``2 pi q dq`` panels; close nodes are loss panels
    mapped to their free-electron transfer.
    """
    q, w = _electronic_radial(channels)
    loss, weight = _close_loss_nodes(channels)
    return (
        np.concatenate([q, channels.close_transfer(loss)]),
        np.concatenate([w * channels.distant(q), channels.close_loss(loss) * weight]),
    )


def _electronic_transform(channels: ElectronicChannels, g) -> np.ndarray:
    """``int d^2q w(q) exp(-g^2 / 2 q^2)`` [angstrom^2] at ``|g|``."""
    q, c = _electronic_nodes(channels)
    g = np.asarray(g, dtype=float)
    return (c @ np.exp(-0.5 * (g.ravel()[None, :] / q[:, None]) ** 2)).reshape(g.shape)


def _electronic_diagonal(profile, channels, max_harmonic):
    """``D_G = rho_G int d^2q w(q) exp(-G^2 / 2 q^2)``, the full ``C_el(x, x)`` harmonics."""
    rho = np.zeros(max_harmonic + 1, dtype=complex)
    count = min(rho.size, profile.electron_density.size)
    rho[:count] = profile.electron_density[:count]
    return rho * _electronic_transform(channels, np.arange(rho.size) * profile.g_inv_ang)


def electronic_kernel(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    *,
    n_cells: int = DEFAULT_CELLS,
    max_order: int = DEFAULT_MAX_ORDER,
    oscillators: MaterialShellOscillators | None = None,
) -> ScatteringKernel:
    """Return the local-density electronic-excitation kernel.

    Each jump ``L = sqrt(w(q) d^2q) s_q(x) e^{i q_x x}`` with
    ``s_q = sqrt(rho_q)``. Per transfer bin ``p`` the kernel is
    ``int dq_y w(q) s_q s_q^dagger``, a positive sum of rank-one terms, so the
    binned kernel stays positive semidefinite. Transfers ``|q_x| > max_order g``
    are dropped from the kept band but remain in the diagonal.

    ``oscillators`` defaults to the catalog shell oscillators of
    ``profile.crystal``; crystals without measured conduction-band data must
    pass their own.

    Assumptions: first Born, the PENELOPE-2024 δ-oscillator GOS of
    :func:`~pyrite.montecarlo.transport.shell_gos.shell_gos_moments`, every
    shell localized like the total atomic electron density at resolution
    ``1/|q_perp|`` (local-density approximation), and energy loss not fed back
    into the transverse state. Limits: ``rho_e`` uniform gives pure momentum
    diffusion ``int w(q) (1 - cos q_x Delta)``; ``q -> 0`` jumps do not
    decohere.

    Validation: cr-lindblad-jump-operators
    """
    _check_grid(n_cells, max_order)
    hbar_v = _kinematics(kinetic_energy_eV)
    channels = electronic_channels(_profile_oscillators(profile, oscillators), kinetic_energy_eV)
    g = profile.g_inv_ang
    root = _root_density(profile, 2 * max_order)
    width = g / n_cells
    bins = _transfer_bins(n_cells, max_order)
    sub, sub_w = _GL[3]
    p = bins[:, None] * width + 0.5 * width * sub[None, :]
    p_w = np.broadcast_to(0.5 * width * sub_w, p.shape).copy()
    # The central bin holds the 1/(q^2 + q_min^2) peak: refine it geometrically.
    q_floor = 0.25 * float(np.min(channels.distant_qmin_inv_ang))
    centre_x, centre_w = _panels(_geometric_edges(q_floor, 0.5 * width, 1.5), 8)
    centre_p = np.concatenate([-centre_x, centre_x])
    centre_pw = np.concatenate([centre_w, centre_w])
    q_y, q_yw = _panels(_geometric_edges(q_floor, _Q_MAX_INV_ANG, 1.5), 8)
    # Abel-type weights onto the log-q table: A[l, t] = int_bin dp int dq_y w(q) hat_t(q).
    abel = np.zeros((bins.size, root.log_q.size))
    for index, (nodes, node_w) in enumerate(zip(p, p_w, strict=True)):
        if bins[index] == 0:
            nodes, node_w = centre_p, centre_pw
        q = np.sqrt(nodes[:, None] ** 2 + q_y[None, :] ** 2).ravel()
        weight = (node_w[:, None] * 2.0 * q_yw[None, :]).ravel() * channels.weight(q)
        lo, w_lo, hi, w_hi = root.hat_weights(q)
        abel[index] = np.bincount(lo, weight * w_lo, root.log_q.size) + np.bincount(
            hi, weight * w_hi, root.log_q.size
        )
    coefficients = root.coefficients
    outer = np.einsum("ti,tj->tij", coefficients, coefficients.conj())
    s_orders = root.orders
    size = 2 * max_order + 1
    kernel = np.zeros((n_cells, size, size), dtype=complex)
    for index, bin_index in enumerate(bins):
        r = bin_index % n_cells
        nu = (bin_index - r) // n_cells
        ket = nu + s_orders
        keep = np.abs(ket) <= max_order
        block = np.tensordot(abel[index], outer, axes=1)[np.ix_(keep, keep)]
        slots = ket[keep] + max_order
        kernel[r][np.ix_(slots, slots)] += block
    kernel = 0.5 * (kernel + np.conj(np.swapaxes(kernel, 1, 2)))
    max_harmonic = int(np.ceil(_Q_MAX_INV_ANG / g))
    max_harmonic = min(max_harmonic, profile.electron_density.size - 1)
    return ScatteringKernel(
        label="electronic",
        period_ang=profile.period_ang,
        n_cells=n_cells,
        orders=np.arange(-max_order, max_order + 1),
        momentum_kernel=kernel,
        diagonal_coefficients=_electronic_diagonal(profile, channels, max_harmonic),
        hbar_v_eV_ang=hbar_v,
    )


# ---------------------------------------------------------------------------
# Decoherence diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoherenceRate:
    """Cell-averaged decoherence rate ``Gamma(Delta x)`` [1/angstrom] per mechanism.

    ``Gamma(Delta) = < C(x,x)/2 + C(x+Delta, x+Delta)/2 - Re C(x, x+Delta) >_x``:
    the depth rate at which a coherence across ``Delta`` decays. ``1/Gamma`` is
    the decoherence length.
    """

    delta_ang: np.ndarray
    tds: np.ndarray
    electronic: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Sum of the mechanisms [1/angstrom]."""
        return self.tds + self.electronic


def decoherence_rate(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    delta_ang,
    *,
    oscillators: MaterialShellOscillators | None = None,
    beam_uvw=None,
) -> DecoherenceRate:
    """Return the cell-averaged decoherence rate ``Gamma(Delta x)``.

    Only the ``G = 0`` momentum-diagonal part of ``C`` survives the cell
    average, so, with every momentum transfer kept,

    - TDS: ``Gamma = sum_s n_s/(hbar v)^2 (1/2 pi) int q dq v_s(q)^2
      (1 - e^{-a_s^2 q^2}) (1 - J0(q Delta))`` for isotropic sites. For an
      anisotropic transverse covariance ``S`` (eigenvalues ``lambda_min``,
      ``lambda_min + 2 rho``; ``S_xx - S_yy = 2 rho cos phi``,
      ``S_xy = rho sin phi``) the angular average of
      ``e^{-q.S.q} cos(q_x Delta)`` is
      ``sum_n eps_n e^{-q^2 lambda_min} ive(n, rho q^2) cos(n phi) J_2n(q Delta)``
      (``eps_0 = 1``, else 2), which replaces ``e^{-a^2 q^2} J0``. ``beam_uvw``
      is as in :func:`tds_kernel`;
    - electronic: ``Gamma = 2 pi int q dq w(q) [rho_bar - J0(q Delta)
      <s_q(x) s_q(x + Delta)>_x]``.

    Limits: ``Gamma(0) = 0``; ``Gamma(Delta -> inf)`` is the mean loss rate
    for TDS and approaches it for the electronic term; soft electronic
    transfers ``q Delta << 1`` contribute ``O((q Delta)^2)``.

    Validation: cr-lindblad-jump-operators
    """
    hbar_v = _kinematics(kinetic_energy_eV)
    delta = np.atleast_1d(np.asarray(delta_ang, dtype=float))
    if np.any(delta < 0.0) or not np.all(np.isfinite(delta)):
        raise ValueError("delta_ang must be finite and non-negative")
    sites = _site_densities(profile, 0)
    covariances = _transverse_covariances(profile, beam_uvw)
    tds_edges = _geometric_edges(0.05, _Q_MAX_INV_ANG)

    def tds_density(q, separation=None):
        """``q dq`` density of the mean TDS rate, or of its overlap at ``separation``."""
        total = np.zeros_like(q)
        for element, density in sites.items():
            cov = covariances[element]
            lam = _min_eigenvalue(cov)
            rho = 0.5 * (cov[0] + cov[2]) - lam
            if rho == 0.0:
                factor = -np.expm1(-lam * q**2)
                if separation is not None:
                    factor = factor * j0(q * separation)
            else:
                # e^{-q^2 lambda} I_n(rho q^2) < 1e-16 once n^2 > 70 rho q^2 with
                # lambda q^2 < 37 (Gaussian tail of I_n / I_0).
                orders = np.arange(int(np.ceil(np.sqrt(70.0 * 37.0 * rho / lam))) + 4)
                phi = np.arctan2(cov[1], 0.5 * (cov[0] - cov[2]))
                weight = np.where(orders == 0, 1.0, 2.0) * np.cos(orders * phi)
                series = (
                    ive(orders[None, :], rho * q[:, None] ** 2)
                    * np.exp(-lam * q**2)[:, None]
                    * weight
                )
                if separation is None:
                    factor = 1.0 - series[:, 0]
                else:
                    bessel = jv(2 * orders[None, :], q[:, None] * separation)
                    factor = j0(q * separation) - (series * bessel).sum(axis=1)
            total += density[0].real * q * _potential(element, q) ** 2 * factor
        return total / (2.0 * np.pi * hbar_v**2)

    q, w = _panels(tds_edges, 8)
    tds_mean = float(w @ tds_density(q))
    tds = np.empty(delta.shape)
    for index, value in enumerate(delta):
        q, w = _oscillatory_panels(tds_edges, value)
        tds[index] = tds_mean - w @ tds_density(q, value)

    channels = electronic_channels(_profile_oscillators(profile, oscillators), kinetic_energy_eV)
    root = _root_density(profile, 4 * max(1, profile.electron_density.size))
    q_el, w_el = _electronic_radial(channels)
    mean = float(profile.electron_density[0].real)
    el_mean = mean * float(w_el @ channels.weight(q_el))
    el_edges = _electronic_edges(channels)
    electronic = np.empty(delta.shape)
    for index, value in enumerate(delta):
        q, w = _oscillatory_panels(el_edges, value)
        overlap = root.autocorrelation(q, profile.g_inv_ang, value)
        weight = 2.0 * np.pi * q * w * channels.weight(q)
        electronic[index] = el_mean - weight @ (j0(q * value) * overlap)
    return DecoherenceRate(delta_ang=delta, tds=tds, electronic=electronic)
