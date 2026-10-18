"""Photon transition primitives for channeling radiation (ADR-0018, P3).

A photon ``(k, epsilon)`` is emitted by the first-order interaction
``(e/c) v . A``; :func:`transition_operator` evaluates its matrix element in
the transverse Bloch basis and :func:`first_order_transition` the golden-rule
path rate. Its long-wavelength limit is the boosted electric dipole:
:func:`rest_frame_dipole_rate_inv_s`, :func:`dipole_rate_per_ang`,
:func:`boosted_dipole_pattern` and :func:`dipole_stokes`. :mod:`.emission`
builds the line catalog and spectrum on these.

Lengths in angstroms, energies in eV, angles in radians. CPU only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as _C_M_PER_S

from ...materials.channeling.potential import PlanarProfile
from ...materials.channeling.transverse import (
    TransverseStates,
    doppler_denominator,
    transverse_states,
)
from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV

C_ANG_PER_S = _C_M_PER_S * 1e10


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


def dipole_far_field(theta_x_rad, theta_y_rad, gamma, dipole) -> np.ndarray:
    """Return the lab far-field amplitude of a complex transverse dipole line [sr^-1/2].

    The Liénard--Wiechert field of :func:`dipole_stokes` is linear in the
    acceleration, so a dipole ``d = (d_x, d_y, 0)`` oscillating as
    ``d e^{-i omega t}`` radiates ``E ∝ (n . d)(n - beta z) - (1 - beta cos theta) d``
    toward ``n = (sin theta cos phi, sin theta sin phi, cos theta)`` with
    ``tan phi = theta_y / theta_x`` and ``theta = hypot(theta_x, theta_y)``.
    Returned are its components ``(E_+, E_-, E_z)`` on the fixed lab basis
    ``e_+- = (x +- i y)/sqrt 2`` (``E_+- = e_+-^* . E``) and ``z``, scaled by
    ``sqrt(3/8 pi) / (gamma (1 - beta cos theta)^2 |d|)`` so that
    ``|E_+|^2 + |E_-|^2 + |E_z|^2`` is the photon density per sr per emitted
    photon (unit sphere integral). ``E_+-`` are the circular components a
    fixed polarizer selects; their phase winds as ``exp(i l phi)`` with
    ``l = J_z - (+-1)`` for a ``J_z`` line (paraxially, orbital angular momentum).
    The global phase is that of ``d``.

    Assumptions: electric-dipole line, point source on the axis, far field.
    Limits: ``d = x`` reproduces :func:`boosted_dipole_pattern` and the
    linear polarization of :func:`dipole_stokes`; ``d = (x + i y)/sqrt 2``
    (``J_z = +1``) gives on-axis ``E_-= 0`` and ``E_-`` winding twice.

    Validation: cr-dipole-far-field-amplitude
    """
    d = np.asarray(dipole, dtype=complex)
    if d.shape != (2,) or not np.linalg.norm(d) > 0.0:
        raise ValueError("dipole needs two complex components, not both zero")
    tx, ty = np.broadcast_arrays(
        np.asarray(theta_x_rad, dtype=float), np.asarray(theta_y_rad, dtype=float)
    )
    theta = np.hypot(tx, ty)
    phi = np.arctan2(ty, tx)
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    beta = float(_beta(gamma))
    denom = _denominator(gamma, theta)
    axes = (3,) + (1,) * theta.ndim
    z_hat = np.array([0.0, 0.0, 1.0]).reshape(axes)
    d3 = np.array([d[0], d[1], 0.0]).reshape(axes)
    field = (n[0] * d[0] + n[1] * d[1]) * (n - beta * z_hat) - denom * d3
    field *= np.sqrt(3.0 / (8.0 * np.pi)) / (float(gamma) * denom**2 * np.linalg.norm(d))
    return np.stack(
        [
            (field[0] - 1j * field[1]) / np.sqrt(2.0),
            (field[0] + 1j * field[1]) / np.sqrt(2.0),
            field[2],
        ]
    )


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
