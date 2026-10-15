"""Channeling-radiation emission from the transverse density matrix (ADR-0018, P3).

**Operator.** A photon ``(k, epsilon)`` is emitted by the first-order
interaction ``(e/c) v . A``. In the transverse Bloch basis the transition
operator is ``A_fi = <f| (v . epsilon* / c) exp(-i k . r) |i>`` with
``v = (p_x / (gamma m), 0, beta c)``. The golden rule with the longitudinal
energy--momentum balance ``hbar omega (1 - beta cos theta) = Delta E_perp``
gives, per unit path,

``dN / dz dOmega = alpha (omega / c) |A_fi|^2 / (2 pi beta (1 - beta cos theta))``.

For ``q = k_x -> 0`` (state size ``<= 1 A`` against a rest-frame wavelength of
``>= 10^2 A``) ``A_fi -> -i x_fi (omega / c)[(1 - beta cos theta) eps_x* + beta n_x eps_z*]``,
which is the boosted electric dipole: the rest-frame rate
``A' = (4/3) alpha omega'^3 |x_fi|^2 / c^2`` at ``omega' = gamma Delta E_perp / hbar``,
the rest-frame pattern ``(3 / 8 pi) sin^2 Theta'`` about the plane normal
``x``, aberration ``dOmega'/dOmega = 1 / (gamma^2 (1 - beta cos theta)^2)`` and
the lab path rate ``A' / (gamma beta c)``.

**Estimator.** Lines are scored as the expectation
``<d_c^dagger d_c>_rho(z)`` per cluster ``c`` of transitions degenerate
within their widths, ``d_c = sum a_fi |f><i|`` with ``|a_fi|^2`` the dipole
path rate. ``rho`` is block-diagonal in Bloch ``k``, so lines of different
blocks are incoherent; within a block of a 1-D well the bound levels are
nondegenerate and every cluster reduces to ``P_i |a_fi|^2``. Only bound ->
bound transitions are scored (v1, maintainer decision 2026-10-14):
free -> free is coherent bremsstrahlung (PXR/CBS kernel) and free -> bound
capture is the diagnostic :func:`capture_bound`. Per-trajectory MCWF field
amplitudes are never squared.

**Line shape.** Each line is a Voigt profile: the coherence ``rho_if``
decays at ``kappa_if = (Gamma_i + Gamma_f)/2 - Re <i|J(|i><f|)|f>`` per unit
depth (secular Lindblad), a Lorentzian of transverse half width
``hbar v kappa_if`` scaled by the Doppler factor, convolved with the Gaussian
of the beam energy spread. The detector aperture and the in-plane beam
divergence add the Doppler--geometric width by quadrature over directions.
Photons are attenuated on the way out by
:func:`~pyrite.materials.attenuation._stack_tau`.

Lengths in angstroms, energies in eV, angles in radians. CPU only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as _C_M_PER_S
from scipy.special import voigt_profile

from ...materials.attenuation import _resolve_composition, _stack_tau, plasma_energy_eV
from ...materials.channeling.potential import PlanarProfile
from ...materials.channeling.transverse import (
    HBAR2_2M_EV_ANG2,
    TransverseStates,
    _hamiltonian,
    doppler_denominator,
    transverse_states,
)
from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from .density import DensityMatrixSolver

C_ANG_PER_S = _C_M_PER_S * 1e10
DEFAULT_CAPTURE_THRESHOLD = 0.03


def _beta(gamma):
    gamma = np.asarray(gamma, dtype=float)
    return np.sqrt(1.0 - 1.0 / gamma**2)


def _denominator(gamma, theta_rad):
    """``1 - beta cos theta`` for a Lorentz factor (cancellation-free)."""
    return doppler_denominator((np.asarray(gamma, dtype=float) - 1.0) * M_E_EV, theta_rad)


# ---------------------------------------------------------------------------
# Dipole rate, boosted pattern and polarization
# ---------------------------------------------------------------------------


def rest_frame_dipole_rate_inv_s(delta_eV, dipole_ang, gamma):
    """Return the rest-frame spontaneous dipole rate ``A'`` [1/s].

    ``A' = (4/3) alpha omega'^3 |x_fi|^2 / c^2`` (Gaussian units; equal to
    ``e^2 omega'^3 |x|^2 / (3 pi eps_0 hbar c^3)``), with
    ``omega' = gamma Delta E_perp / hbar``: transverse energies are lab-frame
    (``z``-as-time) eigenvalues, and the average rest frame of the longitudinal
    motion dilates their frequency by ``gamma``. The transverse dipole ``x_fi``
    is unchanged by the boost.

    Assumptions: electric dipole (rest-frame wavelength ``>= 10^2`` angstrom
    against a state extent ``<= 1`` angstrom), recoil and spin neglected.
    Limit: ``gamma = 1`` is the textbook atomic rate.

    Validation: cr-dipole-rate
    """
    k_rest = np.asarray(gamma, dtype=float) * np.asarray(delta_eV, dtype=float) / HBARC_EV_ANG
    return 4.0 / 3.0 * ALPHA_FS * k_rest**3 * np.abs(dipole_ang) ** 2 * C_ANG_PER_S


def dipole_rate_per_ang(delta_eV, dipole_ang, gamma):
    """Return the lab photon yield per unit path of a dipole line [1/angstrom].

    ``dN/dz = A' / (gamma beta c) = (4/3) alpha gamma^2 (Delta E_perp / hbar c)^3
    |x_fi|^2 / beta``: one ``gamma`` for time dilation, ``beta c`` to convert
    lab time to path. Photon number is invariant, so this is also the
    solid-angle integral of :func:`boosted_dipole_pattern` times ``A'/gamma``
    per ``beta c``.

    Validation: cr-dipole-rate
    """
    gamma = np.asarray(gamma, dtype=float)
    return rest_frame_dipole_rate_inv_s(delta_eV, dipole_ang, gamma) / (
        gamma * _beta(gamma) * C_ANG_PER_S
    )


def boosted_dipole_pattern(theta_rad, phi_rad, gamma):
    """Return the lab angular density of dipole photons, normalized to 1 [1/sr].

    The rest-frame pattern ``(3 / 8 pi) sin^2 Theta'`` about the dipole axis
    ``x`` (plane normal) is mapped by aberration,
    ``n'_x = sin theta cos phi / (gamma (1 - beta cos theta))`` and
    ``dOmega'/dOmega = 1 / (gamma^2 (1 - beta cos theta)^2)``, so

    ``dN/dOmega = (3 / 8 pi) [1 - n'_x^2] / (gamma^2 (1 - beta cos theta)^2)``.

    ``theta`` is the photon angle to the electron velocity, ``phi`` its azimuth
    from the plane normal. The lab rate per unit time and solid angle is
    ``(A' / gamma)`` times this pattern; its integral over the sphere is
    exactly 1 (change of variables), so the lab total is ``A' / gamma``.
    ``gamma -> 1`` gives the static dipole ``(3 / 8 pi) sin^2 Theta``.

    Validation: cr-boosted-dipole-angular
    """
    gamma = np.asarray(gamma, dtype=float)
    denom = _denominator(gamma, theta_rad)
    nx_rest = np.sin(theta_rad) * np.cos(phi_rad) / (gamma * denom)
    return 3.0 / (8.0 * np.pi) * (1.0 - nx_rest**2) / (gamma * denom) ** 2


def _photon_frame(theta_rad, phi_rad):
    """Photon direction ``n`` and the polarization basis ``(e1, e2)``.

    ``e1`` is the plane normal ``x`` projected perpendicular to ``n``;
    ``e2 = n x e1``. Shapes ``(..., 3)``.
    """
    theta = np.asarray(theta_rad, dtype=float)
    phi = np.asarray(phi_rad, dtype=float)
    theta, phi = np.broadcast_arrays(theta, phi)
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    x_hat = np.array([1.0, 0.0, 0.0])
    e1 = x_hat - n * n[..., :1]
    norm = np.linalg.norm(e1, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("photon direction along the plane normal has no dipole emission")
    e1 = e1 / norm
    e2 = np.cross(n, e1)
    return n, e1, e2


def dipole_stokes(theta_rad, phi_rad, gamma):
    """Return the normalized Stokes ``(Q, U)`` of planar dipole radiation.

    The lab field of a charge moving at ``beta z`` with acceleration along
    ``x`` is ``E ∝ n x ((n - beta) x x) = n_x (n - beta z) - x (1 - beta cos theta)``
    (Liénard--Wiechert), equivalent to boosting the rest-frame dipole
    polarization. ``Q`` and ``U`` refer to ``e1`` (the plane normal projected
    perpendicular to ``n``) and ``e2 = n x e1``. The field is real, so ``V = 0``
    and the radiation is fully linearly polarized; forward and at
    ``phi = 90°`` it is along ``e1`` (``Q = 1``). Every planar dipole line
    shares this polarization at a given direction.

    Validation: cr-boosted-dipole-angular
    """
    n, e1, e2 = _photon_frame(theta_rad, phi_rad)
    beta = _beta(gamma)
    denom = _denominator(gamma, np.asarray(theta_rad))
    field = n[..., :1] * (n - beta * np.array([0.0, 0.0, 1.0])) - np.asarray(denom)[
        ..., None
    ] * np.array([1.0, 0.0, 0.0])
    a1 = np.sum(field * e1, axis=-1)
    a2 = np.sum(field * e2, axis=-1)
    intensity = a1**2 + a2**2
    return np.stack([(a1**2 - a2**2) / intensity, 2.0 * a1 * a2 / intensity], axis=-1)


# ---------------------------------------------------------------------------
# Generic first-order transition operator
# ---------------------------------------------------------------------------


def transition_operator(
    initial: TransverseStates,
    final: TransverseStates,
    q_inv_ang: float,
    polarization: Sequence[complex],
) -> np.ndarray:
    """Return ``<f| (v . eps* / c) exp(-i q x) |i>`` for every band pair.

    ``initial`` holds Bloch states at ``k``; ``final`` must sit at
    ``k - q`` (modulo ``g``), where ``q`` is the photon's transverse
    wavevector along the plane normal. ``exp(-i q x)`` maps plane wave
    ``k + m g`` to ``k + m g - q``; the velocity acts on the initial plane
    wave, ``v_x / c = hbar c (k + m g) / (gamma m c^2)``, and the longitudinal
    velocity is the constant ``beta`` (the ``O(E_perp / p v)`` transverse
    correction is dropped). Ordering is immaterial because
    ``eps . k = 0``. ``polarization`` is ``(eps_x, eps_z)``; ``eps_y`` couples
    to no planar motion.

    Returns an array ``(N_final, N_initial)`` (dimensionless). The dipole
    limit ``q -> 0`` is
    ``-i x_fi (omega / c)[(1 - beta cos theta) eps_x* + beta n_x eps_z*]``.

    Validation: cr-dipole-rate
    """
    if not np.isclose(initial.gamma, final.gamma):
        raise ValueError("initial and final states need the same gamma")
    g = initial.profile.g_inv_ang
    shift_real = (initial.k_inv_ang - q_inv_ang - final.k_inv_ang) / g
    shift = int(np.round(shift_real))
    if abs(shift_real - shift) > 1e-9:
        raise ValueError("final states must be solved at k - q")
    eps_x, eps_z = (complex(value) for value in polarization)
    beta = float(_beta(initial.gamma))
    waves = initial.k_inv_ang + initial.orders * g
    weight = (
        np.conj(eps_x) * HBARC_EV_ANG * waves / (initial.gamma * M_E_EV) + np.conj(eps_z) * beta
    )
    target = initial.orders + shift
    lookup = {int(m): row for row, m in enumerate(final.orders)}
    rows = np.array([lookup.get(int(m), -1) for m in target])
    keep = rows >= 0
    projected = np.zeros((final.orders.size, initial.orders.size), dtype=complex)
    projected[rows[keep], :] = weight[keep, None] * initial.coefficients[keep, :]
    return final.coefficients.conj().T @ projected


@dataclass(frozen=True)
class FirstOrderTransition:
    """Golden-rule emission of one band pair into one direction.

    Attributes
    ----------
    photon_energy_eV
        Self-consistent lab photon energy ``(E_i(k) - E_f(k - q)) / (1 - beta cos theta)``.
    q_inv_ang
        Photon transverse wavevector along the plane normal [1/angstrom].
    amplitudes
        ``A_fi`` for polarizations ``(e1, e2)`` of :func:`dipole_stokes`.
    rate_per_ang_sr
        ``dN / dz dOmega`` summed over polarizations [1/(angstrom sr)].
    """

    photon_energy_eV: float
    q_inv_ang: float
    amplitudes: np.ndarray
    rate_per_ang_sr: float


def first_order_transition(
    profile: PlanarProfile,
    gamma: float,
    upper: int,
    lower: int,
    theta_rad: float,
    phi_rad: float = 0.0,
    *,
    k_inv_ang: float = 0.0,
    n_plane_waves: int = 32,
    iterations: int = 6,
) -> FirstOrderTransition:
    """Evaluate the generic first-order emission rate for ``upper -> lower``.

    ``dN / dz dOmega = alpha (omega / c) sum_eps |A_fi|^2 / (2 pi beta D)``
    with ``D = 1 - beta cos theta`` from the golden rule: mode density
    ``omega^2 dOmega d omega / (2 pi c)^3``, coupling ``(e/c) v . A`` and
    ``int d omega delta(Delta E_perp - hbar omega D) = 1 / (hbar D)``. The
    final state is solved at ``k - q`` and ``omega`` iterated to
    self-consistency (the band shift is tiny for bound states). Its
    ``q -> 0`` limit is :func:`dipole_rate_per_ang` times
    :func:`boosted_dipole_pattern`.

    Validation: cr-dipole-rate
    """
    initial = transverse_states(profile, gamma, k_inv_ang, n_plane_waves)
    n, e1, e2 = _photon_frame(theta_rad, phi_rad)
    beta = float(_beta(gamma))
    denom = float(_denominator(gamma, theta_rad))
    delta = initial.energies_eV[upper] - initial.energies_eV[lower]
    final = initial
    q = 0.0
    for _ in range(max(1, iterations)):
        q = float(delta / denom / HBARC_EV_ANG * n[0])
        final = transverse_states(profile, gamma, initial.k_inv_ang - q, n_plane_waves)
        delta = initial.energies_eV[upper] - final.energies_eV[lower]
    if delta <= 0.0:
        raise ValueError("upper state must lie above the lower state")
    omega_c = delta / denom / HBARC_EV_ANG
    amplitudes = np.array(
        [
            transition_operator(initial, final, q, (basis[0], basis[2]))[lower, upper]
            for basis in (e1, e2)
        ]
    )
    rate = ALPHA_FS * omega_c * np.sum(np.abs(amplitudes) ** 2) / (2.0 * np.pi * beta * denom)
    return FirstOrderTransition(
        photon_energy_eV=float(delta / denom),
        q_inv_ang=q,
        amplitudes=amplitudes,
        rate_per_ang_sr=float(rate),
    )


# ---------------------------------------------------------------------------
# Line catalog and expected-value estimator on rho(z)
# ---------------------------------------------------------------------------


def _block_dipoles(solver: DensityMatrixSolver, block: int, states: np.ndarray) -> np.ndarray:
    """``x_fi`` [angstrom] between eigenstates ``states`` of one block."""
    basis = solver.basis
    vectors = basis.vectors[block][:, states]
    waves = solver.kappa[block] + solver.orders * solver.profile.g_inv_ang
    momentum = vectors.conj().T @ (waves[:, None] * vectors)
    energies = basis.energies[block][states]
    gap = energies[:, None] - energies[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dipole = -1j * 2.0 * HBAR2_2M_EV_ANG2 / basis.gamma * momentum / gap
    dipole[~np.isfinite(dipole)] = 0.0
    np.fill_diagonal(dipole, 0.0)
    return dipole


def _coherence_decay(solver: DensityMatrixSolver, block: int, upper: int, lower: int) -> float:
    """Secular decay rate ``kappa_if`` [1/angstrom] of ``rho_if`` in one block."""
    size = solver.orders.size
    rho = np.zeros((solver.n_cells, size, size), dtype=complex)
    rho[block, upper, lower] = 1.0
    inside, _, _ = solver.jump(rho)
    gains = inside[block, upper, lower].real
    losses = []
    for state in (upper, lower):
        diag = np.zeros_like(rho)
        diag[block, state, state] = 1.0
        losses.append(solver.loss_rate(diag))
    return 0.5 * (losses[0] + losses[1]) - gains


@dataclass(frozen=True)
class EmissionLines:
    """Bound -> bound dipole lines of a density-matrix solver's basis.

    One entry per cluster of transitions in one Bloch block. Members are the
    ``(lower, upper, amplitude)`` triples of the cluster; ``|amplitude|^2`` is
    the dipole path rate [1/angstrom] and its phase follows the eigenvector
    gauge of the solver's basis (rates use ``rho`` in the same gauge).

    Attributes
    ----------
    gamma, beta
        Longitudinal kinematics of the basis.
    block
        Bloch block of each line.
    upper, lower
        Band indices of each line's strongest member.
    delta_eV
        Rate-weighted transverse transition energy [eV].
    hwhm_eV
        Transverse Lorentzian half width ``hbar v kappa_if`` [eV].
    gamma_exponent
        ``d ln Delta E_perp / d ln gamma`` at fixed band indices.
    strength_per_ang
        Sum of member path rates ``|a_fi|^2`` [1/angstrom] (rate per unit
        population of a nondegenerate line).
    members
        Per line, ``(lower, upper, amplitude)`` arrays.
    """

    gamma: float
    beta: float
    block: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    delta_eV: np.ndarray
    hwhm_eV: np.ndarray
    gamma_exponent: np.ndarray
    strength_per_ang: np.ndarray
    members: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]

    @property
    def size(self) -> int:
        """Number of lines."""
        return int(self.block.size)

    def rates(self, rho: np.ndarray) -> np.ndarray:
        """Expected photon yield per unit path of each line [1/angstrom].

        ``<d_c^dagger d_c> = sum_f sum_{i, i'} a_fi' * a_fi rho_ii'`` over the
        members of each cluster, on a density matrix in the solver's
        eigenbasis, shape ``(n_cells, N, N)``. It is linear in ``rho``, so it is
        an unbiased estimator on any ensemble of states.

        Validation: cr-yield-assembly
        """
        out = np.zeros(self.size)
        for line, (lower, upper, amplitude) in enumerate(self.members):
            sub = rho[self.block[line]][np.ix_(upper, upper)]
            same = lower[:, None] == lower[None, :]
            out[line] = np.sum(same * amplitude[:, None] * np.conj(amplitude[None, :]) * sub).real
        return out

    def forward_energy_eV(self) -> np.ndarray:
        """Forward (``theta = 0``) lab photon energy of each line [eV]."""
        return self.delta_eV / _denominator(self.gamma, 0.0)


def emission_lines(
    solver: DensityMatrixSolver,
    *,
    cluster_tol_eV: float | None = None,
    gamma_step: float = 1e-4,
    relative_cut: float = 1e-12,
) -> EmissionLines:
    """Build the bound -> bound line catalog of a density-matrix solver.

    In each Bloch block, every pair of bound states (``E < barrier``) with
    ``E_i > E_f`` contributes ``a_fi = sqrt(dN/dz) x_fi / |x_fi|`` with
    :func:`dipole_rate_per_ang`; ``x_fi`` comes from the commutator in the
    solver's eigenbasis, so parity selection emerges; transitions below
    ``relative_cut`` of the block's strongest (parity-forbidden to rounding)
    are dropped. Transitions of one block into the same final state are
    merged into one cluster when their energies differ by at most
    ``cluster_tol_eV`` (default: half the sum of their Lorentzian half widths,
    i.e. unresolved within the line width); only such clusters carry cross
    terms in ``<d^dagger d>``. In a 1-D well the bound levels of one block are
    nondegenerate, so in practice every cluster is one transition. Widths use the secular coherence
    decay ``kappa_if = (Gamma_i + Gamma_f)/2 - Re <i|J(|i><f|)|f>`` of the
    solver's kernel at its entry energy.

    Zero bound states give an empty catalog, hence exactly zero emission.

    Validation: cr-dipole-rate, cr-line-shape
    """
    basis = solver.basis
    gamma = float(basis.gamma)
    beta = float(_beta(gamma))
    hbar_v = HBARC_EV_ANG * beta
    bound = basis.energies < solver.barrier_eV
    shifted = []
    for factor in (1.0 + gamma_step, 1.0 - gamma_step):
        shifted.append(
            np.array(
                [
                    np.linalg.eigvalsh(
                        _hamiltonian(solver.profile, gamma * factor, kappa, solver.orders)
                    )
                    for kappa in solver.kappa
                ]
            )
        )
    rows: list[dict] = []
    for block in range(solver.n_cells):
        states = np.flatnonzero(bound[block])
        if states.size < 2:
            continue
        dipoles = _block_dipoles(solver, block, states)
        energies = basis.energies[block][states]
        pairs = []
        for a, upper in enumerate(states):
            for b, lower in enumerate(states[:a]):
                delta = energies[a] - energies[b]
                x = dipoles[b, a]
                if delta <= 1e-12 or abs(x) == 0.0:
                    continue
                rate = float(dipole_rate_per_ang(delta, abs(x), gamma))
                if rate == 0.0:
                    continue
                kappa = _coherence_decay(solver, block, int(upper), int(lower))
                up = shifted[0][block, upper] - shifted[0][block, lower]
                down = shifted[1][block, upper] - shifted[1][block, lower]
                exponent = (np.log(up) - np.log(down)) / (
                    np.log1p(gamma_step) - np.log1p(-gamma_step)
                )
                pairs.append(
                    {
                        "upper": int(upper),
                        "lower": int(lower),
                        "delta": float(delta),
                        "amp": np.sqrt(rate) * x / abs(x),
                        "rate": rate,
                        "hwhm": max(hbar_v * kappa, 0.0),
                        "exponent": float(exponent),
                    }
                )
        if not pairs:
            continue
        floor = relative_cut * max(item["rate"] for item in pairs)
        pairs = [item for item in pairs if item["rate"] >= floor]
        # Cross terms of <d^dagger d> need a shared final state and initial
        # states degenerate within the widths, so clusters never mix finals.
        pairs.sort(key=lambda item: (item["lower"], item["delta"]))
        clusters: list[list[dict]] = []
        for pair in pairs:
            if clusters and clusters[-1][-1]["lower"] == pair["lower"]:
                last = clusters[-1][-1]
                tol = (
                    0.5 * (last["hwhm"] + pair["hwhm"])
                    if cluster_tol_eV is None
                    else float(cluster_tol_eV)
                )
                if pair["delta"] - last["delta"] <= tol:
                    clusters[-1].append(pair)
                    continue
            clusters.append([pair])
        for cluster in clusters:
            weight = np.array([item["rate"] for item in cluster])
            strongest = cluster[int(np.argmax(weight))]

            def mean(key, cluster=cluster, weight=weight):
                return float(np.dot(weight, [item[key] for item in cluster]) / weight.sum())

            rows.append(
                {
                    "block": block,
                    "upper": strongest["upper"],
                    "lower": strongest["lower"],
                    "delta": mean("delta"),
                    "hwhm": mean("hwhm"),
                    "exponent": mean("exponent"),
                    "strength": float(weight.sum()),
                    "members": (
                        np.array([item["lower"] for item in cluster]),
                        np.array([item["upper"] for item in cluster]),
                        np.array([item["amp"] for item in cluster], dtype=complex),
                    ),
                }
            )

    def column(key, dtype=float):
        return np.array([row[key] for row in rows], dtype=dtype)

    return EmissionLines(
        gamma=gamma,
        beta=beta,
        block=column("block", int),
        upper=column("upper", int),
        lower=column("lower", int),
        delta_eV=column("delta"),
        hwhm_eV=column("hwhm"),
        gamma_exponent=column("exponent"),
        strength_per_ang=column("strength"),
        members=tuple(row["members"] for row in rows),
    )


# ---------------------------------------------------------------------------
# Line shape
# ---------------------------------------------------------------------------


def line_shape(photon_energy_eV, centre_eV, lorentz_hwhm_eV, gaussian_sigma_eV):
    """Return the unit-area Voigt line profile [1/eV].

    Lorentzian of half width ``lorentz_hwhm_eV`` (occupation/coherence
    lifetime) convolved with a Gaussian of RMS ``gaussian_sigma_eV`` (beam
    energy spread). The Doppler--geometric width is added by the caller's
    direction quadrature. A pure Lorentzian or pure Gaussian is the limit of
    a vanishing other width; both zero is a delta function and is refused.

    Validation: cr-line-shape
    """
    lorentz = np.asarray(lorentz_hwhm_eV, dtype=float)
    sigma = np.asarray(gaussian_sigma_eV, dtype=float)
    if np.any((lorentz <= 0.0) & (sigma <= 0.0)):
        raise ValueError("line has zero width: supply a scattering kernel or a beam energy spread")
    detuning = np.asarray(photon_energy_eV, dtype=float) - np.asarray(centre_eV, dtype=float)
    return voigt_profile(detuning, sigma, lorentz)


# ---------------------------------------------------------------------------
# Depth history and spectrum assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _History:
    z_ang: np.ndarray
    rates: np.ndarray  # (n_z, n_lines) [1/angstrom]
    bound: np.ndarray  # (n_z,)
    capture: np.ndarray  # (n_z,) diagonal free -> bound rate [1/angstrom]


def _capture_rates(solver: DensityMatrixSolver, gamma: float) -> np.ndarray:
    """Diagonal free -> bound capture rate of every basis state [1/angstrom]."""
    basis = solver.basis
    out = np.zeros(basis.energies.shape)
    for block in range(solver.n_cells):
        bound = np.flatnonzero(basis.energies[block] < solver.barrier_eV)
        free = np.flatnonzero(basis.energies[block] >= solver.barrier_eV)
        if bound.size == 0 or free.size == 0:
            continue
        states = np.concatenate([bound, free])
        dipoles = _block_dipoles(solver, block, states)
        energies = basis.energies[block][states]
        delta = energies[None, bound.size :] - energies[: bound.size, None]
        rate = dipole_rate_per_ang(delta, dipoles[: bound.size, bound.size :], gamma)
        out[block, free] = rate.sum(axis=0)
    return out


def _depth_history(
    solver: DensityMatrixSolver,
    lines: EmissionLines,
    thickness_ang: float,
    dz_ang: float,
    sample_ang: float,
    incidence_rad: float,
    divergence_rad: float,
) -> _History:
    if thickness_ang <= 0.0 or dz_ang <= 0.0 or sample_ang <= 0.0:
        raise ValueError("thickness_ang, dz_ang and sample_ang must be positive")
    inner = max(1, int(round(sample_ang / dz_ang)))
    segments = max(1, int(np.ceil(thickness_ang / (inner * dz_ang))))
    seg = thickness_ang / segments
    step = seg / inner
    capture = _capture_rates(solver, lines.gamma)
    rho, free, outside = solver.entry_state(incidence_rad, divergence_rad)
    entry = float(free.sum()) + outside
    z, rates, bound, caps = [0.0], [lines.rates(rho)], [solver.bound_population(rho)], []

    def capture_rate(matrix):
        return float(np.sum(np.einsum("rnn->rn", matrix).real * capture))

    caps.append(capture_rate(rho))
    for index in range(1, segments + 1):
        result = solver.propagate(seg, step, initial=(rho, entry))
        rho, entry = result.final_density, result.total_flux[-1]
        z.append(index * seg)
        rates.append(lines.rates(rho))
        bound.append(float(result.bound_population[-1]))
        caps.append(capture_rate(rho))
    return _History(
        z_ang=np.asarray(z),
        rates=np.asarray(rates).reshape(len(z), lines.size),
        bound=np.asarray(bound),
        capture=np.asarray(caps),
    )


@dataclass(frozen=True)
class ChannelingSpectrum:
    """Diagnostic channeling-radiation spectrum at the crystal exit.

    Attributes
    ----------
    photon_energy_eV
        Photon energy grid [eV].
    theta_rad, phi_rad
        Detector directions (polar angle to the beam axis, azimuth from the
        plane normal), broadcast shape ``S``.
    spectrum
        ``d^2N / dE dOmega`` [photons/eV/sr/electron], shape ``S + (n_E,)``.
    stokes
        Intensity-weighted normalized Stokes ``(Q, U)``, shape
        ``S + (n_E, 2)``; NaN where the spectrum is zero.
    lines
        Line catalog.
    z_ang, line_rates
        Sampled depths [angstrom] and the estimator per line [1/angstrom],
        shape ``(n_z, n_lines)``.
    bound_population
        Channeled (bound) population at each sampled depth.
    line_yield
        Depth-integrated photons per electron per line over ``4 pi``, before
        absorption.
    """

    photon_energy_eV: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    spectrum: np.ndarray
    stokes: np.ndarray
    lines: EmissionLines
    z_ang: np.ndarray
    line_rates: np.ndarray
    bound_population: np.ndarray
    line_yield: np.ndarray

    @property
    def total_line_yield(self) -> float:
        """Photons per electron summed over every line (``4 pi``, unabsorbed)."""
        return float(self.line_yield.sum())


def _direction_nodes(aperture_rad: float, n_aperture: int):
    """Offsets ``(r, chi, weight)`` over a cone of half angle ``aperture_rad``."""
    if aperture_rad <= 0.0:
        return np.zeros(1), np.zeros(1), np.ones(1)
    x, w = np.polynomial.legendre.leggauss(n_aperture)
    radius = 0.5 * aperture_rad * (x + 1.0)
    radial = 0.5 * aperture_rad * w * np.sin(radius)
    n_chi = 2 * n_aperture
    chi = 2.0 * np.pi * (np.arange(n_chi) + 0.5) / n_chi
    r_grid, chi_grid = np.meshgrid(radius, chi, indexing="ij")
    weight = np.repeat(radial[:, None], n_chi, axis=1) * (2.0 * np.pi / n_chi)
    return r_grid.ravel(), chi_grid.ravel(), (weight / weight.sum()).ravel()


def channeling_spectrum(
    solver: DensityMatrixSolver,
    thickness_ang: float,
    photon_energy_eV,
    theta_rad,
    phi_rad=0.0,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    divergence_y_rad: float = 0.0,
    energy_spread_eV: float = 0.0,
    aperture_rad: float = 0.0,
    dz_ang: float = 50.0,
    sample_ang: float = 500.0,
    absorption: bool = True,
    layers: Sequence | None = None,
    cluster_tol_eV: float | None = None,
    n_aperture: int = 6,
    n_divergence: int = 9,
) -> ChannelingSpectrum:
    """Return the bound -> bound channeling-radiation spectrum per electron.

    ``d^2N/dE dOmega = sum_lines int_0^t dz <d_c^dagger d_c>_rho(z)
    P(n_e) L_c(E - E_c(theta_e)) exp(-tau(z, E_c, n))``,

    averaged over the detector cone (``aperture_rad``) and the in-plane
    (``y``) beam divergence. ``rho(z)`` is propagated by ``solver`` from the
    divergence-averaged entry state at constant energy, sampled every
    ``sample_ang`` and integrated by the trapezoid rule; ``P`` is
    :func:`boosted_dipole_pattern` in the electron frame (photon angle
    ``theta_e`` to the electron velocity), ``E_c = Delta E_perp / (1 - beta cos
    theta_e)`` and ``L_c`` the :func:`line_shape` with Lorentzian half width
    ``hbar v kappa / (1 - beta cos theta_e)`` and Gaussian
    ``sigma = E_c s sigma_gamma / gamma``, where
    ``s = cos theta_e / (gamma^2 beta D) + d ln Delta E_perp / d ln gamma``
    (``-> 2 + d ln Delta E / d ln gamma`` forward; the walkthrough's
    ``2 delta gamma / gamma`` is its kinematic part). The face normal is the
    mean beam axis ``z``; photons leave through the exit face along the lab
    direction ``n``, attenuated by ``_stack_tau`` over ``layers`` (default:
    one slab of ``solver.profile.crystal`` over ``[0, thickness]``).

    Parameters
    ----------
    solver
        Density-matrix solver (profile, entry energy, kernel and margin).
    thickness_ang
        Crystal thickness [angstrom].
    photon_energy_eV
        Photon energy grid [eV].
    theta_rad, phi_rad
        Detector direction(s), broadcast together.
    incidence_rad, divergence_rad
        Mean incidence and RMS divergence normal to the plane (entry state).
    divergence_y_rad
        RMS beam divergence along the plane (electron direction spread).
    energy_spread_eV
        RMS beam kinetic-energy spread [eV].
    aperture_rad
        Half angle of a circular detector cone [rad]; 0 is a point detector.
    dz_ang, sample_ang
        Solver step and estimator sampling interval [angstrom].
    absorption
        Apply exit attenuation.
    layers
        ``_stack_tau`` layers ``[(z_top, z_bot, composition), ...]``.
    cluster_tol_eV
        Override the cluster tolerance of :func:`emission_lines`.
    n_aperture, n_divergence
        Quadrature orders of the aperture and in-plane divergence averages.

    Returns
    -------
    ChannelingSpectrum
        Spectrum in photons/eV/sr/electron with line diagnostics.

    Assumptions: bound -> bound dipole lines only; constant electron energy
    in the zone (no ``gamma(z)`` drift, ``Delta T / T << 1``); recoil
    ``O(hbar omega / E)`` and the radiative natural width neglected; the
    estimator is scored on the solver's subspace (dechanneled electrons
    emit no lines). Limits: zero bound states give exactly zero; with
    ``absorption=False`` the energy and solid-angle integral of a point
    detector equals ``line_yield``.

    Validation: cr-yield-assembly, cr-line-shape
    """
    energies = np.asarray(photon_energy_eV, dtype=float)
    theta, phi = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float), np.asarray(phi_rad, dtype=float)
    )
    shape = theta.shape
    lines = emission_lines(solver, cluster_tol_eV=cluster_tol_eV)
    history = _depth_history(
        solver, lines, thickness_ang, dz_ang, sample_ang, incidence_rad, divergence_rad
    )
    line_yield = np.trapezoid(history.rates, history.z_ang, axis=0)
    spectrum = np.zeros(shape + (energies.size,))
    q_sum = np.zeros_like(spectrum)
    u_sum = np.zeros_like(spectrum)
    if lines.size and np.any(line_yield > 0.0):
        if absorption and layers is None:
            layers = [(0.0, float(thickness_ang), _resolve_composition(solver.profile.crystal))]
        gamma, beta = lines.gamma, lines.beta
        sigma_gamma = float(energy_spread_eV) / M_E_EV
        radius, chi, ap_weight = _direction_nodes(float(aperture_rad), n_aperture)
        if divergence_y_rad > 0.0:
            nodes, psi_weight = np.polynomial.hermite_e.hermegauss(n_divergence)
            psi = divergence_y_rad * nodes
            psi_weight = psi_weight / psi_weight.sum()
        else:
            psi, psi_weight = np.zeros(1), np.ones(1)
        flat_theta, flat_phi = theta.ravel(), phi.ravel()
        flat_spec = spectrum.reshape(-1, energies.size)
        flat_q = q_sum.reshape(-1, energies.size)
        flat_u = u_sum.reshape(-1, energies.size)
        for d, (t0, p0) in enumerate(zip(flat_theta, flat_phi, strict=True)):
            n0 = np.array([np.sin(t0) * np.cos(p0), np.sin(t0) * np.sin(p0), np.cos(t0)])
            t1 = np.array([np.cos(t0) * np.cos(p0), np.cos(t0) * np.sin(p0), -np.sin(t0)])
            t2 = np.array([-np.sin(p0), np.cos(p0), 0.0])
            for r, c, wa in zip(radius, chi, ap_weight, strict=True):
                n = np.cos(r) * n0 + np.sin(r) * (np.cos(c) * t1 + np.sin(c) * t2)
                if n[2] <= 0.0:
                    continue
                for s, wp in zip(psi, psi_weight, strict=True):
                    ye = n[1] * np.cos(s) - n[2] * np.sin(s)
                    ze = n[1] * np.sin(s) + n[2] * np.cos(s)
                    theta_e = float(np.arctan2(np.hypot(n[0], ye), ze))
                    phi_e = float(np.arctan2(ye, n[0]))
                    denom = float(_denominator(gamma, theta_e))
                    centre = lines.delta_eV / denom
                    pattern = float(boosted_dipole_pattern(theta_e, phi_e, gamma))
                    if absorption:
                        e_grid = np.broadcast_to(centre[:, None], (lines.size, history.z_ang.size))
                        tau = _stack_tau(layers, history.z_ang[None, :], float(n[2]), e_grid)
                        attenuated = np.exp(-np.asarray(tau, dtype=float))
                        yields = np.trapezoid(history.rates.T * attenuated, history.z_ang, axis=1)
                    else:
                        yields = line_yield
                    slope = np.cos(theta_e) / (gamma**2 * beta * denom) + lines.gamma_exponent
                    sigma = np.abs(centre * slope) * sigma_gamma / gamma
                    profile = line_shape(
                        energies[None, :],
                        centre[:, None],
                        (lines.hwhm_eV / denom)[:, None],
                        sigma[:, None],
                    )
                    contribution = wa * wp * pattern * (yields[:, None] * profile).sum(axis=0)
                    stokes = dipole_stokes(theta_e, phi_e, gamma)
                    flat_spec[d] += contribution
                    flat_q[d] += contribution * stokes[0]
                    flat_u[d] += contribution * stokes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        stokes_out = np.stack([q_sum / spectrum, u_sum / spectrum], axis=-1)
    return ChannelingSpectrum(
        photon_energy_eV=energies,
        theta_rad=theta,
        phi_rad=phi,
        spectrum=spectrum,
        stokes=stokes_out,
        lines=lines,
        z_ang=history.z_ang,
        line_rates=history.rates,
        bound_population=history.bound,
        line_yield=line_yield,
    )


# ---------------------------------------------------------------------------
# Free -> bound capture bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureBound:
    """Free -> bound radiative-capture diagnostic (not scored in v1).

    Attributes
    ----------
    estimate
        ``int dz sum_i rho_ii(z) Gamma_i^cap`` over above-barrier states kept
        in the solver's subspace [photons/electron, ``4 pi``].
    upper_bound
        ``max_i Gamma_i^cap int dz (1 - P_bound(z))`` over every above-barrier
        basis state [photons/electron]: every unbound electron, dechanneled
        or not, captures at most at the fastest rate.
    line_yield
        Bound -> bound yield of the same solve [photons/electron].
    max_rate_per_ang
        Largest diagonal capture rate ``Gamma_i^cap`` [1/angstrom].
    edge_eV
        Forward capture edge ``(barrier - E_0) / (1 - beta)`` per block [eV];
        the continuum starts near it.
    threshold
        Fraction of the line yield above which capture should be scored.
    """

    estimate: float
    upper_bound: float
    line_yield: float
    max_rate_per_ang: float
    edge_eV: np.ndarray
    threshold: float

    @property
    def estimate_ratio(self) -> float:
        """``estimate / line_yield``."""
        return self.estimate / self.line_yield if self.line_yield > 0.0 else np.inf

    @property
    def bound_ratio(self) -> float:
        """``upper_bound / line_yield``."""
        return self.upper_bound / self.line_yield if self.line_yield > 0.0 else np.inf

    @property
    def promote(self) -> bool:
        """True when the estimate exceeds ``threshold`` of the line yield."""
        return self.estimate_ratio > self.threshold


def capture_bound(
    solver: DensityMatrixSolver,
    thickness_ang: float,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    dz_ang: float = 50.0,
    sample_ang: float = 500.0,
    threshold: float = DEFAULT_CAPTURE_THRESHOLD,
) -> CaptureBound:
    """Bound the free -> bound radiative-capture yield against the line yield.

    Capture of an above-barrier state ``i`` into bound ``f`` is the same
    dipole emission as a line, ``Gamma_i^cap = sum_f dN_fi/dz``
    (:func:`dipole_rate_per_ang`), at ``hbar omega >= (barrier - E_f)/(1 - beta
    cos theta)``. Above-barrier Bloch states at one ``k`` are nondegenerate,
    so the expected value is diagonal. The *estimate* scores the free states
    the solver keeps (``barrier <= E < barrier + margin``); build the solver
    with a large ``margin_eV`` for the most inclusive value. The *upper bound*
    charges every unbound electron, including dechanneled flux, at the
    fastest capture rate in the basis. Maintainer rule (issue #505): score
    capture only if it exceeds a few percent (``threshold``) of the line
    yield in an anchor configuration.

    Validation: cr-capture-bound
    """
    lines = emission_lines(solver)
    history = _depth_history(
        solver, lines, thickness_ang, dz_ang, sample_ang, incidence_rad, divergence_rad
    )
    rates = _capture_rates(solver, lines.gamma)
    estimate = float(np.trapezoid(history.capture, history.z_ang))
    max_rate = float(rates.max()) if rates.size else 0.0
    unbound = float(np.trapezoid(1.0 - history.bound, history.z_ang))
    ground = solver.basis.energies[:, 0]
    edge = (solver.barrier_eV - ground) / _denominator(lines.gamma, 0.0)
    return CaptureBound(
        estimate=estimate,
        upper_bound=max_rate * unbound,
        line_yield=float(np.trapezoid(history.rates, history.z_ang, axis=0).sum()),
        max_rate_per_ang=max_rate,
        edge_eV=np.asarray(edge),
        threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Transition-radiation bound
# ---------------------------------------------------------------------------


def transition_radiation(material, kinetic_energy_eV: float, photon_energy_eV, theta_rad):
    """Return single-boundary forward transition radiation [photons/eV/sr].

    Ginzburg--Frank radiation of a charge crossing a vacuum--medium boundary
    in the X-ray regime (``eps = 1 - (omega_p / omega)^2``), ultrarelativistic
    small-angle form (Jackson, *Classical Electrodynamics* 3rd ed., Eq. 13.91;
    Ter-Mikaelian 1972):

    ``d^2N / dE dOmega = (alpha / (pi^2 E)) theta^2
    [1 / (gamma^-2 + theta^2) - 1 / (gamma^-2 + theta^2 + (omega_p / omega)^2)]^2``.

    Assumptions: ``gamma >> 1``, ``theta << 1``, ``omega >> omega_p``
    (Drude medium from :func:`~pyrite.materials.attenuation.plasma_energy_eV`),
    no absorption. Limits: vanishes as ``omega_p -> 0`` (no boundary contrast)
    and as ``omega >> gamma omega_p`` (as ``(gamma omega_p / omega)^4``).

    Validation: cr-transition-radiation-bound
    """
    plasma = plasma_energy_eV(material)
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    energy = np.asarray(photon_energy_eV, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    vacuum = 1.0 / (gamma**-2 + theta**2)
    medium = 1.0 / (gamma**-2 + theta**2 + (plasma / energy) ** 2)
    return ALPHA_FS / (np.pi**2 * energy) * theta**2 * (vacuum - medium) ** 2


def transition_radiation_per_eV(material, kinetic_energy_eV: float, photon_energy_eV):
    """Return angle-integrated single-boundary transition radiation [photons/eV].

    ``dN/dE = (alpha / (pi E)) [(1 + 2 nu^2) ln(1 + 1/nu^2) - 2]`` with
    ``nu = omega / (gamma omega_p)`` (Jackson Eq. 13.92 per photon), the
    small-angle integral of :func:`transition_radiation`. Its energy integral
    is ``alpha gamma hbar omega_p / 3``.

    Validation: cr-transition-radiation-bound
    """
    plasma = plasma_energy_eV(material)
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    energy = np.asarray(photon_energy_eV, dtype=float)
    nu2 = (energy / (gamma * plasma)) ** 2
    return ALPHA_FS / (np.pi * energy) * ((1.0 + 2.0 * nu2) * np.log1p(1.0 / nu2) - 2.0)


@dataclass(frozen=True)
class TransitionRadiationBound:
    """Transition-radiation background against the channeling lines.

    Attributes
    ----------
    photons
        Two-boundary upper bound in the band [photons/electron]: four times
        one boundary, the maximum of the entrance--exit interference
        ``|1 - e^{i phi}|^2`` with no absorption of the entrance radiation.
    peak_density
        Two-boundary bound on ``d^2N/dE dOmega`` at each line's forward
        energy and ``theta = 0..aperture`` maximum [photons/eV/sr].
    line_density
        Peak of the channeling spectrum at each line [photons/eV/sr].
    ratio
        ``peak_density / line_density`` per line.
    energy_eV
        Single-boundary emitted energy ``alpha gamma hbar omega_p / 3`` [eV].
    """

    photons: float
    peak_density: np.ndarray
    line_density: np.ndarray
    ratio: np.ndarray
    energy_eV: float


def transition_radiation_bound(
    spectrum: ChannelingSpectrum,
    material,
    kinetic_energy_eV: float,
    band_eV: tuple[float, float],
    *,
    theta_max_rad: float | None = None,
    n_band: int = 512,
) -> TransitionRadiationBound:
    """Compare the transition-radiation bound with a channeling spectrum.

    The TR density is maximized over ``0 <= theta <= theta_max`` (default
    ``3 / gamma``) at each line's forward energy, doubled for two
    boundaries and doubled again for constructive interference; the CR
    density is the maximum of ``spectrum.spectrum`` within one half width of
    each forward line energy over the spectrum's directions.

    Validation: cr-transition-radiation-bound
    """
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    plasma = plasma_energy_eV(material)
    theta_max = 3.0 / gamma if theta_max_rad is None else float(theta_max_rad)
    grid = np.geomspace(band_eV[0], band_eV[1], n_band)
    photons = 4.0 * float(
        np.trapezoid(transition_radiation_per_eV(material, kinetic_energy_eV, grid), grid)
    )
    angles = np.linspace(0.0, theta_max, 400)
    forward = spectrum.lines.forward_energy_eV()
    peak = np.array(
        [4.0 * transition_radiation(material, kinetic_energy_eV, e, angles).max() for e in forward]
    )
    energies = spectrum.photon_energy_eV
    flat = spectrum.spectrum.reshape(-1, energies.size)
    step = float(np.min(np.diff(energies))) if energies.size > 1 else 0.0
    width = np.maximum(spectrum.lines.hwhm_eV / _denominator(gamma, 0.0), step)
    line = np.array(
        [
            flat[:, np.abs(energies - e) <= w].max(initial=0.0)
            for e, w in zip(forward, width, strict=True)
        ]
    )
    with np.errstate(divide="ignore"):
        ratio = peak / line
    return TransitionRadiationBound(
        photons=photons,
        peak_density=peak,
        line_density=line,
        ratio=ratio,
        energy_eV=ALPHA_FS * gamma * plasma / 3.0,
    )
