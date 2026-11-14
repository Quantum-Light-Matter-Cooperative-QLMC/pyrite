"""Zone-row PXR and diffracted channeling radiation (DCR) source (#1166).

On a planar channeling zone row the electron is in the density-matrix
solver's tracked transverse Bloch states. For a reflection ``g`` off the
plane's own reciprocal row, photon ``(E, n)`` and polarization ``e_s``, the
reciprocity (photon-mode) amplitude of Feranchuk et al. Eq. (2) is, per
transition ``i -> f`` (fibre ``kappa`` to ``kappa - Q_x``, ``Q = k + g``)::

    a_fi,s(E) = chi_g [M_fi,s(E) K_fi,s(E) + M^pre_fi,s(E) K_pre,s(E)]
    M_fi,s = beta (v_hat.F_s) A_fi + (F_s.x_hat) B_fi,  F_s = Q (g.e_s) - k^2 e_s
    A = <f|exp(-i Q_x x)|i>,  B = <f|{v_x / c, exp(-i Q_x x)} / 2|i>

**Photon mode.** The time-reversed two-beam Laue mode enters through the exit
face with unit transmitted and zero diffracted amplitude. Along the inward
depth ``u = zeta_L - zeta`` measured from the exit face it obeys
``dP/du = i N P`` with ``N = [[kappa_0, a], [b, q + kappa_h]]``, where
``q = -det / (2 Q_s)``, ``det = g^2 + 2 k.g``, ``a = k^2 chi_-g C_s / (2 k_s)``,
``b = k^2 chi_g C_s / (2 Q_s)``, ``kappa_0 = k chi_0 / (2 n_s)`` and
``kappa_h = k^2 chi_0 / (2 Q_s)`` (subscript ``s`` the inward surface-normal
component, ``C_s`` the two-beam polarization factor). The eigenvalues
``lambda_1,2`` give the diffracted profile relative to the kinematic
particular amplitude ``chi_g F_s / det``::

    P(z) = -q [exp(i lambda_1 s (L - z)) - exp(i lambda_2 s (L - z))] / (lambda_1 - lambda_2)

with ``s = s_hat.v_hat`` and ``z`` the electron path. Its kinematic limit
(``ab -> 0``, ``chi_0 -> 0``) is ``1 - exp(i q s (L - z))``: the first term is
PXR and the second the free diffracted wave, i.e. forward channeling
radiation Bragg-diffracted into ``n`` (DCR). ``Im chi_0`` carries the photon
escape absorption, ``Re chi_0`` the refraction shift, and the coupling
``ab`` the dynamical (Pendelloesung) depletion; the kinematic pole ``1/det``
is removed. Then, with ``phi_L(x) = int_0^L exp(i x z) dz``::

    K_fi = sum_j (+-) exp(i lambda_j s L) phi_L(Delta_fi - lambda_j s) / (2 Q_s (lambda_1 - lambda_2))
    K_pre = [exp(i lambda_1 s L) - exp(i lambda_2 s L)] / (2 Q_s (lambda_1 - lambda_2)) / (i (Delta_pre - q s))
    Delta_fi = D (E - E_fi) / (hbar c beta),  E_fi = E_0 + (eps_i - eps_f + C Q_x^2) / D

``E_0`` is the exact-recoil free root, ``D = 1 - beta v_hat.n`` and
``C = (hbar c)^2 / (2 gamma m c^2)``; ``+ C Q_x^2`` removes the transverse
recoil that ``E_0`` already holds, so free states return to ``E_0``.

**Pre-crystal path.** The diffracted wave leaves the entrance face upstream,
where the electron is the entrance plane wave ``|p>``. The electron crosses
that wave before entering, which gives ``K_pre`` with the vacuum mismatch
``Delta_pre = D (E - E_0) / (hbar c beta) - 2 C p Q_x / (hbar c beta)``. This
mismatch never phase-matches. Because ``exp(-i Q_x x)`` and the velocity
vertex act on ``|p>``, ``sum_i <i|p> M_fi = <f|M|p>``: the term belongs on
the ``exp(-i Q_x x)`` vertex of *every* pair, not only on transverse-elastic
ones, and on the velocity vertex only through the entrance velocity,
``M^pre_fi,s = [beta v_hat.F_s + (F_s.x_hat) v_x(p - Q_x / 2) / c] A_fi``
(second fresh-context verifier, #1148). ``p = kappa + nu G`` is the true
entrance plane-wave momentum: the entry density is an incoherent mixture of
plane waves (beam divergence, source spot), so each entering order ``nu`` of a
fibre carries its own pre-crystal term. A state's intensity is averaged over
the entering waves with weight ``w_m |<i|kappa + m G>|^2`` (normalized per
state; :attr:`ZoneStates.entrance`), so it interferes only with the waves that
fed it. A fibre the entry leaves empty (fed only by scattering) keeps a
pre-crystal term at ``kappa`` folded into ``[-G/2, G/2)``: the same
stationary path-mean approximation that weights the pre-crystal term by
path-mean rather than entrance populations. For a
transverse-elastic pair the kinematic ``K_fi + K_pre`` is the single-line
first-order Born result ``phi_L(Delta) (-q s) / ((Delta - q s) det)``.

**Estimator.** ``d2N/dE dOmega = alpha (E / hbar c) |chi_g|^2 / (4 pi^2 hbar c beta^2)
sum_s sum_fi p_i |M_fi,s K_fi,s + M^pre_fi,s K_pre,s|^2``. This is the native
``alpha omega / (4 pi^2 hbar c) |A|^2 t_L^2`` (``t_L = L / beta``) with the
frozen-root ``t_L sinc`` replaced by the slab kernel. ``p_i`` are the
path-averaged diagonal populations of the retained density per incident
electron.

Assumptions: two-beam Laue geometry (``n.s_hat > 0``, ``Q_s > 0``); flat
``chi_g`` and ``chi_0`` across the line window (taken at ``E_0``) and
``chi_-g chi_g = |chi_g|^2`` (Friedel, non-anomalous); first-order ``q`` and
first-order (slowly varying amplitude) two-beam equations; stationary
path-averaged density with diagonal populations; finite state lifetimes and
CBS distorted-wave resolvents omitted; transverse momentum along the plane
carried by the beam only. The whole ``F_s`` takes the homogeneous (DCR and
pre-crystal) waves, although its longitudinal part
``det (e_s.Q_hat) Q_hat`` is a local medium response that should carry only
``phi_L``: zero for sigma and at the pole, it moved a bound-state band by 0.2%
24 eV from the pole (fresh-context verifier, #1166). PXR--CBS interference is
included (:func:`reflection_spectrum` ``u_g``).

Lengths in angstroms, energies in eV. CPU only.

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.crystal import HBARC_EV_ANG, M_E_EV

ALPHA_FS = 7.2973525693e-3
_SERIES = 1e-6


@dataclass(frozen=True)
class ZoneStates:
    """Tracked transverse Bloch states of one planar zone and their populations.

    Attributes
    ----------
    potentials
        ``(n_fibre, n_wave, n_wave)`` transverse Hamiltonian minus its kinetic
        diagonal in the plane-wave basis [eV].
    energies, vectors
        Bloch eigenvalues ``(n_fibre, n_state)`` [eV] and eigenvectors
        ``(n_fibre, n_wave, n_state)``.
    populations
        ``(n_fibre, n_state)`` path-averaged diagonal populations per incident
        electron (trace ``<= 1``).
    orders, kappa
        Plane-wave orders ``m`` (``k = kappa + m G``) and fibre quasi-momenta [1/angstrom].
    period_ang, gamma
        Plane spacing [angstrom] and electron Lorentz factor.
    entrance
        Optional ``(n_fibre, n_wave)`` entry plane-wave weights (the diagonal of
        the entry density over ``kappa + m G``). ``None`` takes every fibre's
        entrance wave as ``kappa`` folded into ``[-G/2, G/2)``.
    """

    potentials: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    populations: np.ndarray
    orders: np.ndarray
    kappa: np.ndarray
    period_ang: float
    gamma: float
    entrance: np.ndarray | None = None

    @property
    def beta(self) -> float:
        return float(np.sqrt(1.0 - 1.0 / self.gamma**2))

    def entrance_waves(
        self, fibre: int, states, tol: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """Entrance momenta ``kappa + m G`` [1/angstrom] of ``fibre`` and their weights per state.

        Returns ``(p, w)`` with ``w`` of shape ``(len(states), len(p))``: entry
        weight ``w_m`` times the overlap ``|<i|kappa + m G>|^2`` of each state in
        ``states``, normalized per state, so a state interferes only with the
        entering waves that fed it (exact in the free limit). A state with no
        overlap takes the bare entry weights. Orders below ``tol`` of the fibre's
        largest entry weight are dropped. A fibre with no entry weight returns
        ``kappa`` folded into ``[-G/2, G/2)``.
        """
        G = 2.0 * np.pi / self.period_ang
        kappa = float(self.kappa[fibre])
        states = np.asarray(states, dtype=int)
        weight = None if self.entrance is None else np.asarray(self.entrance[fibre], float)
        if weight is None or not np.any(weight > 0.0):
            return np.array([kappa - G * np.floor(kappa / G + 0.5)]), np.ones((states.size, 1))
        keep = np.flatnonzero(weight > tol * weight.max())
        bare = weight[keep] / weight[keep].sum()
        joint = bare[None, :] * np.abs(self.vectors[fibre][keep][:, states].T) ** 2
        total = joint.sum(axis=1, keepdims=True)
        per_state = np.where(total > 0.0, joint / np.where(total > 0.0, total, 1.0), bare[None, :])
        return kappa + self.orders[keep] * G, per_state

    @property
    def recoil_eV_ang2(self) -> float:
        """``C = (hbar c)^2 / (2 gamma m c^2)`` [eV angstrom^2]."""
        return HBARC_EV_ANG**2 / (2.0 * self.gamma * M_E_EV)


@dataclass(frozen=True)
class SlabGeometry:
    """Lab-frame zone geometry: beam ``v_hat``, plane normal ``x_hat``, inward face normal ``s_hat``."""

    beam: np.ndarray
    x_hat: np.ndarray
    s_hat: np.ndarray
    path_ang: float


def phi(x, length):
    """``int_0^L exp(i x z) dz`` for complex ``x``, series-safe near ``x = 0``."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x * length) < _SERIES
    safe = np.where(small, 1.0, x)
    exact = np.expm1(1j * safe * length) / (1j * safe)
    return np.where(small, length * (1.0 + 0.5j * x * length), exact)


def psi(x, mu, length):
    """``exp(i mu L) phi_L(x - mu) = int_0^L exp(i x z + i mu (L - z)) dz``.

    Formed as ``(exp(i x L) - exp(i mu L)) / (i (x - mu))`` for real ``x``
    and ``Im mu >= 0``, so both exponentials stay bounded: an absorbing mode
    never overflows where ``exp(i mu L)`` underflows.
    """
    x, mu = np.asarray(x, dtype=complex), np.asarray(mu, dtype=complex)
    d = x - mu
    small = np.abs(d * length) < _SERIES
    safe = np.where(small, 1.0, d)
    exact = (np.exp(1j * x * length) - np.exp(1j * mu * length)) / (1j * safe)
    return np.where(small, length * np.exp(1j * mu * length) * (1.0 + 0.5j * d * length), exact)


@dataclass(frozen=True)
class PhotonMode:
    """Two-beam mode quantities on an energy grid, one row per polarization ``(sigma, pi)``.

    ``alpha``/``bx`` are ``beta v_hat.F_s`` and ``F_s.x_hat`` [1/angstrom^2];
    ``lam`` holds ``(lambda_1, lambda_2)`` [1/angstrom]; ``q`` the kinematic
    mismatch rate and ``Qs`` the diffracted inward component; ``cbs`` the
    native exact CBS factor ``f_cbs`` (``A_CBS = (U_g / m c^2) f_cbs``) and
    ``kappa0`` the transmitted-wave diagonal ``k chi_0 / (2 n_s)``.

    Validation: cr-planar-zone-pxr-modulation
    """

    energy_eV: np.ndarray
    alpha: np.ndarray  # (2, n_E)
    bx: np.ndarray  # (2, n_E)
    lam: np.ndarray  # (2, 2, n_E): polarization, branch, energy
    q: np.ndarray  # (n_E,)
    Qs: np.ndarray  # (n_E,)
    Qx: float
    cbs: np.ndarray  # (2, n_E)
    kappa0: np.ndarray  # (n_E,)

    def profile(self, z, svel, length):
        """Diffracted profile ``P(z)`` relative to ``chi_g F_s / det``: ``(2, n_z, n_E)``."""
        lam1, lam2 = self.lam[:, 0][:, None], self.lam[:, 1][:, None]
        u = (svel * (length - np.asarray(z, float)))[None, :, None]
        return -self.q * (np.exp(1j * lam1 * u) - np.exp(1j * lam2 * u)) / (lam1 - lam2)

    def entrance_vertex(self, p_inv_ang, gamma):
        """Pre-crystal vertex ``beta v_hat.F_s + (F_s.x_hat) v_x(p - Q_x / 2) / c``: ``(2, n_E)``.

        ``<f|M_s|p> = entrance_vertex * <f|exp(-i Q_x x)|p>`` for the entrance plane wave ``|p>``.
        """
        cv = HBARC_EV_ANG / (gamma * M_E_EV)
        return self.alpha + self.bx * cv * (p_inv_ang - self.Qx / 2.0)

    def kernels(self, delta, delta_pre, svel, length):
        """``(K_fi, K_pre)`` per polarization: ``(2, *delta.shape)`` and ``(2, n_E)`` [angstrom^3].

        ``delta`` broadcasts against the energy axis (last); ``delta_pre`` is ``(n_E,)``.
        """
        d = np.asarray(delta)
        lead = (1,) * (d.ndim - 1)
        lam1, lam2 = self.lam[:, 0], self.lam[:, 1]
        split = 2.0 * self.Qs * (lam1 - lam2)  # (2, n_E)
        e1, e2 = np.exp(1j * lam1 * svel * length), np.exp(1j * lam2 * svel * length)

        def lift(a):
            return a.reshape((2,) + lead + a.shape[-1:])

        inside = (
            psi(d[None], lift(lam1 * svel), length) - psi(d[None], lift(lam2 * svel), length)
        ) / lift(split)
        pre = (e1 - e2) / split / (1j * (np.asarray(delta_pre) - self.q * svel))
        return inside, pre

    def cbs_kernel(self, delta, svel, length):
        """``int_0^L exp(i Delta z) P_0(z) dz`` per polarization: ``(2, *delta.shape)`` [angstrom].

        ``P_0`` is the transmitted component of the same mode (absorption,
        refraction and two-beam depletion of the CBS photon); kinematically
        ``exp(i kappa_0 s (L - z))``.
        """
        d = np.asarray(delta)
        lead = (1,) * (d.ndim - 1)
        lam1, lam2 = self.lam[:, 0], self.lam[:, 1]

        def lift(a):
            return a.reshape((2,) + lead + a.shape[-1:])

        k0 = self.kappa0[None, :]
        return (
            lift(k0 - lam2) * psi(d[None], lift(lam1 * svel), length)
            - lift(k0 - lam1) * psi(d[None], lift(lam2 * svel), length)
        ) / lift(lam1 - lam2)


def photon_mode(
    energy_eV,
    n_hat,
    g,
    geometry: SlabGeometry,
    beta,
    chi_g_abs,
    chi_0,
    dynamical=True,
    qx_at_eV=None,
):
    """Return the :class:`PhotonMode` of reflection ``g`` towards ``n_hat``.

    ``chi_0`` uses the ``exp(-i omega t)`` convention (``Im chi_0 > 0``
    absorbs). ``dynamical=False`` drops the two-beam coupling and ``chi_0``,
    recovering the kinematic profile ``1 - exp(i q s (L - z))``. The
    transverse transfer ``Q_x`` is frozen at ``qx_at_eV`` (default: the grid
    midpoint); the electron elements use that single fibre shift.

    Validation: cr-planar-zone-pxr-modulation
    """
    from ..._backend import _to_cpu, xp
    from ..spectrum.cxr._recoil import _quantum_line_factors, _recoil_compton_ang

    E = np.atleast_1d(np.asarray(energy_eV, dtype=float))
    n = np.asarray(n_hat, float)
    shat = geometry.s_hat
    k = E / HBARC_EV_ANG
    Q = k[:, None] * n[None, :] + g[None, :]
    det = g @ g + 2.0 * k * (n @ g)
    Qs = Q @ shat
    ns = float(n @ shat)
    if ns <= 0.0 or np.any(Qs <= 0.0):
        raise ValueError("zone PXR source supports Laue geometry only (n.s > 0, Q.s > 0)")
    q = -det / (2.0 * Qs)
    es = np.cross(n, g)
    es /= np.linalg.norm(es)
    ep = np.cross(es, n)
    Qhat = Q / np.linalg.norm(Q, axis=1)[:, None]
    gamma = 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))
    v = beta * geometry.beam
    kap0 = k * chi_0 / (2.0 * ns) if dynamical else np.zeros_like(k)
    alpha, bx, lam, cbs = [], [], [], []
    for e in (es, ep):
        _num, f_cbs = _quantum_line_factors(
            _recoil_compton_ang(),
            gamma,
            float(v @ g),
            xp.asarray(k * (v @ n)),
            xp.asarray(k * (n @ g)),
            float(g @ g),
            xp.asarray(k**2),
            float(v @ e),
            float(g @ e),
            0.0,
        )
        cbs.append(np.asarray(_to_cpu(f_cbs), dtype=float))
        F = Q * (g @ e) - (k**2)[:, None] * e[None, :]
        alpha.append(beta * F @ geometry.beam)
        bx.append(F @ geometry.x_hat)
        if dynamical:
            C = np.linalg.norm(e[None, :] - Qhat * (Qhat @ e)[:, None], axis=1)
            ab = k**4 * chi_g_abs**2 * C**2 / (4.0 * k * ns * Qs)
            kaph = k**2 * chi_0 / (2.0 * Qs)
        else:
            ab = kaph = np.zeros_like(k)
        lam.append(_branches(kap0, q + kaph, ab))
    return PhotonMode(
        energy_eV=E,
        alpha=np.array(alpha),
        bx=np.array(bx),
        lam=np.array(lam),
        q=q,
        Qs=Qs,
        Qx=float(
            (E[len(E) // 2] if qx_at_eV is None else qx_at_eV) / HBARC_EV_ANG * n @ geometry.x_hat
            + g @ geometry.x_hat
        ),
        cbs=np.array(cbs),
        kappa0=np.asarray(kap0, dtype=complex),
    )


def _branches(d0, dh, ab):
    """Eigenvalues of ``[[d0, a], [b, dh]]`` as ``(2, n)``; the second tends to ``d0``.

    The near-``d0`` root comes from the determinant, so it keeps full
    relative accuracy when ``ab << (dh - d0)^2``.
    """
    d0, dh, ab = (np.asarray(v, dtype=complex) for v in (d0, dh, ab))
    half = 0.5 * (dh - d0)
    root = np.sqrt(half * half + ab)
    root = np.where((root * np.conj(half)).real < 0, -root, root)
    lam1 = 0.5 * (d0 + dh) + root
    with np.errstate(divide="ignore", invalid="ignore"):
        lam2 = np.where(lam1 != 0, (d0 * dh - ab) / lam1, 0.5 * (d0 + dh) - root)
    return np.array([lam1, lam2])


def fibre_elements(states: ZoneStates, fibre: int, Qx: float):
    """``(A, B, eps_f)`` from fibre ``kappa`` to ``kappa - Q_x`` in the same plane-wave orders.

    Validation: cr-planar-zone-pxr-modulation
    """
    G = 2.0 * np.pi / states.period_ang
    pw = states.kappa[fibre] + states.orders * G
    C = states.recoil_eV_ang2
    ef, uf = np.linalg.eigh(states.potentials[fibre] + np.diag(C * (pw - Qx) ** 2))
    vec = states.vectors[fibre]
    cv = HBARC_EV_ANG / (states.gamma * M_E_EV)
    A = uf.conj().T @ vec
    B = uf.conj().T @ ((cv * (pw - Qx / 2.0))[:, None] * vec)
    return A, B, ef


@dataclass(frozen=True)
class _Pairs:
    """Retained transitions of one reflection and direction, per fibre."""

    entrance: list
    final: list
    initial: list
    A: list
    B: list
    population: list
    centre_eV: list
    lost_weight: float

    def energies(self) -> np.ndarray:
        return np.concatenate(self.centre_eV) if self.centre_eV else np.zeros(0)


def _pairs(states, geometry, g, n, E0_eV, weight_tol) -> _Pairs:
    """Transitions ordered by frozen weight ``p_i sum_s |M_fi,s(E_0)|^2``; drop the weakest ``weight_tol``."""
    beta = states.beta
    D = 1.0 - beta * float(geometry.beam @ n)
    centre = photon_mode([E0_eV], n, g, geometry, beta, 0.0, 0.0, False, qx_at_eV=E0_eV)
    a0, b0, Qx = centre.alpha[:, 0], centre.bx[:, 0], centre.Qx
    C = states.recoil_eV_ang2
    rows = []
    for r in range(states.kappa.size):
        A, B, ef = fibre_elements(states, r, Qx)
        pop = states.populations[r]
        weight = sum(np.abs(a * A + b * B) ** 2 for a, b in zip(a0, b0, strict=True)) * pop[None, :]
        rows.append((A, B, ef, pop, weight))
    flat = np.sort(np.concatenate([w.ravel() for *_, w in rows]))
    total = float(flat.sum())
    dropped = int(np.searchsorted(np.cumsum(flat), weight_tol * total, side="right"))
    cut = max(float(flat[dropped - 1]) if dropped else 0.0, 0.0)
    out = _Pairs([], [], [], [], [], [], [], 0.0)
    kept = 0.0
    for r, (A, B, ef, pop, weight) in enumerate(rows):
        f, i = np.nonzero(weight > cut)
        if not f.size:
            continue
        kept += float(weight[f, i].sum())
        out.entrance.append(states.entrance_waves(r, i))
        out.final.append(f)
        out.initial.append(i)
        out.A.append(A[f, i])
        out.B.append(B[f, i])
        out.population.append(pop[i])
        out.centre_eV.append(E0_eV + (states.energies[r][i] - ef[f] + C * Qx**2) / D)
    return _Pairs(
        out.entrance,
        out.final,
        out.initial,
        out.A,
        out.B,
        out.population,
        out.centre_eV,
        lost_weight=1.0 - kept / total if total else 0.0,
    )


def reflection_spectrum(
    states: ZoneStates,
    geometry: SlabGeometry,
    g,
    n_hat,
    E0_eV: float,
    energy_eV,
    chi_g: complex,
    chi_0: complex = 0.0,
    *,
    u_g: complex = 0.0,
    dynamical: bool = True,
    weight_tol: float = 1e-7,
    chunk: int = 2_000_000,
    pairs: _Pairs | None = None,
):
    """``d2N/dE dOmega`` of reflection ``g`` towards ``n_hat`` [photons/eV/sr/electron].

    Evaluated on ``energy_eV``, which must resolve the slab line width
    ``2 pi hbar c beta / (D L)`` and span every transition line. The weakest
    transitions holding a fraction ``weight_tol`` of the summed frozen weight
    ``p_i sum_s |M_fi,s(E_0)|^2`` are skipped. ``chi_g`` and ``u_g = U_g / m c^2``
    are complex: their relative phase sets the PXR--CBS interference, as in
    native scoring.

    Validation: cr-planar-zone-pxr-modulation
    """
    E = np.asarray(energy_eV, dtype=float)
    n = np.asarray(n_hat, float)
    g = np.asarray(g, float)
    beta = states.beta
    D = 1.0 - beta * float(geometry.beam @ n)
    svel = float(geometry.s_hat @ geometry.beam)
    L = geometry.path_ang
    mode = photon_mode(E, n, g, geometry, beta, abs(chi_g), chi_0, dynamical, qx_at_eV=E0_eV)
    if pairs is None:
        pairs = _pairs(states, geometry, g, n, E0_eV, weight_tol)
    C = states.recoil_eV_ang2
    scale = HBARC_EV_ANG * beta
    intensity = np.zeros(E.size)
    for (momenta, weights), A, B, pop, centre in zip(
        pairs.entrance, pairs.A, pairs.B, pairs.population, pairs.centre_eV, strict=True
    ):
        step = max(1, chunk // max(E.size, 1))
        for lo in range(0, A.size, step):
            sl = slice(lo, lo + step)
            delta = D * (E[None, :] - centre[sl, None]) / scale
            Af = A[sl][None, :, None]
            K, _ = mode.kernels(delta, delta[:1, :], svel, L)  # (2, pairs, n_E)
            body = (mode.alpha[:, None, :] * Af + mode.bx[:, None, :] * B[sl][None, :, None]) * K
            body *= chi_g
            if u_g:
                body += (u_g * mode.cbs[:, None, :] * Af) * mode.cbs_kernel(delta, svel, L)
            # Each entering plane wave is incoherent with the others (entry mixture).
            for p, w in zip(momenta, weights[sl].T, strict=True):
                delta_pre = D * (E - E0_eV) / scale - 2.0 * C * p * mode.Qx / scale
                _, K_pre = mode.kernels(delta_pre[None, :], delta_pre, svel, L)  # (2, n_E)
                pre = mode.entrance_vertex(p, states.gamma) * K_pre * chi_g
                amp = body + pre[:, None, :] * Af
                intensity += np.einsum("p,spe->e", w * pop[sl], np.abs(amp) ** 2)
    pref = ALPHA_FS * (E / HBARC_EV_ANG) / (4.0 * np.pi**2 * HBARC_EV_ANG * beta**2)
    return pref * intensity


def line_window(states, geometry, g, n_hat, E0_eV, pairs: _Pairs, *, widths=200.0, span_eV=1500.0):
    """``(lo, hi, step)`` [eV] spanning every PXR and DCR line of ``pairs``.

    PXR lines sit at ``E_fi``. DCR lines phase-match ``Delta_fi = q(E) s``,
    solved with ``q`` linear in ``E`` about ``E_0``. The window pads both by
    ``widths`` slab line widths ``w = 2 pi hbar c beta / (D L)`` (the
    ``sinc^2`` tail beyond ``X`` widths holds ``1 / (pi^2 X)``), is clipped
    to ``E_0 +- span_eV``, and is sampled at ``w / 4``: ``|phi_L|^2`` is
    band-limited, so the trapezoid band integral is exact well below that
    step.

    Validation: cr-planar-zone-pxr-modulation
    """
    n = np.asarray(n_hat, float)
    g = np.asarray(g, float)
    beta = states.beta
    D = 1.0 - beta * float(geometry.beam @ n)
    svel = float(geometry.s_hat @ geometry.beam)
    width = 2.0 * np.pi * HBARC_EV_ANG * beta / (D * geometry.path_ang)
    centres = pairs.energies()
    if not centres.size:
        return E0_eV, E0_eV, width / 4.0
    k0 = E0_eV / HBARC_EV_ANG
    Qs0 = float((k0 * n + g) @ geometry.s_hat)
    q0 = -(g @ g + 2.0 * k0 * (n @ g)) / (2.0 * Qs0)
    slope = svel * float(n @ g) / (HBARC_EV_ANG * Qs0)  # -d(q s)/dE
    a = D / (HBARC_EV_ANG * beta) + slope
    inelastic = centres[np.abs(centres - E0_eV) > width]  # elastic pairs never phase-match
    dcr = (D * inelastic / (HBARC_EV_ANG * beta) + svel * q0 + slope * E0_eV) / a
    lines = np.concatenate((centres, dcr[np.isfinite(dcr)]))
    lo = max(float(lines.min()) - widths * width, E0_eV - span_eV)
    hi = min(float(lines.max()) + widths * width, E0_eV + span_eV)
    return lo, hi, width / 4.0


def binned_reflection_spectrum(
    states: ZoneStates,
    geometry: SlabGeometry,
    g,
    n_hat,
    E0_eV: float,
    grid_eV,
    chi_g: complex,
    chi_0: complex = 0.0,
    *,
    u_g: complex = 0.0,
    quadrature: str = "bin-mean",
    weight_tol: float = 1e-7,
    dynamical: bool = True,
):
    """``d2N/dE dOmega`` of one reflection on an output photon grid [photons/eV/sr/electron].

    ``"node"`` evaluates the source at the grid nodes inside the
    :func:`line_window`; ``"bin-mean"`` averages it over the bins between node
    midpoints (outer bins mirrored) on the fine window grid, as native line
    scoring does.

    Validation: cr-planar-zone-pxr-modulation
    """
    grid = np.asarray(grid_eV, dtype=float)
    n = np.asarray(n_hat, float)
    g = np.asarray(g, float)
    pairs = _pairs(states, geometry, g, n, E0_eV, weight_tol)
    lo, hi, step = line_window(states, geometry, g, n, E0_eV, pairs)
    out = np.zeros(grid.size)
    if hi <= lo or grid.size == 0:
        return out
    edges = _bin_edges(grid)
    lo, hi = max(lo, edges[0]), min(hi, edges[-1])
    if hi <= lo:
        return out
    if quadrature == "node":
        inside = (grid >= lo) & (grid <= hi)
        out[inside] = reflection_spectrum(
            states, geometry, g, n, E0_eV, grid[inside], chi_g, chi_0,
            u_g=u_g, dynamical=dynamical, pairs=pairs,
        )  # fmt: skip
        return out
    fine = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    spec = reflection_spectrum(
        states, geometry, g, n, E0_eV, fine, chi_g, chi_0, u_g=u_g, dynamical=dynamical, pairs=pairs
    )
    if quadrature != "bin-mean":
        raise ValueError(f"unknown line quadrature {quadrature!r}")
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (spec[1:] + spec[:-1]) * np.diff(fine))))
    integral = np.interp(np.clip(edges, fine[0], fine[-1]), fine, cum)
    return np.diff(integral) / np.diff(edges)


def _bin_edges(grid):
    if grid.size == 1:
        return np.array([grid[0] - 0.5, grid[0] + 0.5])
    mid = 0.5 * (grid[1:] + grid[:-1])
    return np.concatenate(([2 * grid[0] - mid[0]], mid, [2 * grid[-1] - mid[-1]]))


def free_states(orders, period_ang: float, gamma: float, kappa: float = 0.0) -> ZoneStates:
    """A free plane-wave electron at transverse momentum ``kappa`` (the ``m = 0`` wave).

    Validation: cr-planar-zone-pxr-modulation
    """
    orders = np.asarray(orders, dtype=int)
    G = 2.0 * np.pi / period_ang
    C = HBARC_EV_ANG**2 / (2.0 * gamma * M_E_EV)
    pw = kappa + orders * G
    pop = (orders == 0).astype(float)
    return ZoneStates(
        potentials=np.zeros((1, orders.size, orders.size), complex),
        energies=(C * pw**2)[None],
        vectors=np.eye(orders.size, dtype=complex)[None],
        populations=pop[None],
        orders=orders,
        kappa=np.array([float(kappa)]),
        period_ang=float(period_ang),
        gamma=float(gamma),
    )


def plane_wave_band_yield(
    geometry: SlabGeometry, g, n_hat, E0_eV, gamma, chi_g, escape=1.0, u_g=0.0
) -> float:
    """Native frozen-root band yield ``dN/dOmega`` [photons/sr/electron].

    ``alpha E_0 sum_s |chi_g (beta v_hat.F_s) / det + u_g f_cbs,s|^2 L T / (2 pi hbar c beta D)``:
    the energy integral of the native ``alpha omega / (4 pi^2 hbar c) |A|^2
    t_L^2 sinc^2`` line with escape mean ``T``.

    Validation: cr-planar-zone-pxr-modulation
    """
    beta = float(np.sqrt(1.0 - 1.0 / gamma**2))
    n = np.asarray(n_hat, float)
    g = np.asarray(g, float)
    D = 1.0 - beta * float(geometry.beam @ n)
    mode = photon_mode([E0_eV], n, g, geometry, beta, 0.0, 0.0, False)
    det = float(g @ g + 2.0 * E0_eV / HBARC_EV_ANG * (n @ g))
    a2 = float(np.sum(np.abs(chi_g * mode.alpha[:, 0] / det + u_g * mode.cbs[:, 0]) ** 2))
    return (
        ALPHA_FS * E0_eV * a2 * geometry.path_ang * escape
        / (2.0 * np.pi * HBARC_EV_ANG * beta * D)
    )  # fmt: skip
