"""Discrete-energy longitudinal-coherence and recoil reference (research, #629).

One electron occupies a comb of total energies ``E_n = E_0 + n*hbar*Omega``
with a longitudinal density matrix ``rho[n, m]``. The crystal or slab is static,
so every first-order one-photon amplitude conserves energy exactly and a
component ``n`` that emits ``hbar*omega`` ends at ``E_n - hbar*omega``. The
functions here are deterministic CPU references for the research note
``docs/research/physics/longitudinal-coherence-recoil.md``. They are not
selected by production scoring and supply no absolute coupling constant: an
emission amplitude is a caller-supplied complex number per comb order.

Units: energies in eV, wavenumbers in inverse angstroms, lengths in angstroms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import jv

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

__all__: list[str] = []


def _orders(orders: ArrayLike) -> NDArray[np.int64]:
    result = np.asarray(orders)
    if result.ndim != 1 or result.size == 0 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("orders must be a nonempty one-dimensional integer array")
    if np.any(np.diff(result) != 1):
        raise ValueError("orders must be consecutive and increasing")
    return result.astype(np.int64)


def _momentum_ev(kinetic: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return p*c in eV without subtracting nearly equal total energies."""
    return np.sqrt(kinetic) * np.sqrt(kinetic + 2.0 * M_E_EV)


def comb_wavenumbers_inv_ang(
    kinetic_eV: float, spacing_eV: float, orders: ArrayLike
) -> NDArray[np.float64]:
    """Return exact on-shell vacuum wavenumbers ``k_n = p(E_n)/hbar``.

    Validation: longitudinal-quantum-bunching
    """
    order = _orders(orders)
    kinetic = np.asarray(float(kinetic_eV) + order * float(spacing_eV), dtype=np.float64)
    if np.any(kinetic <= 0) or not np.all(np.isfinite(kinetic)):
        raise ValueError("every comb kinetic energy must be finite and positive")
    return _momentum_ev(kinetic) / HBARC_EV_ANG


def talbot_length_ang(kinetic_eV: float, spacing_eV: float) -> float:
    """Return ``z_T = 4*pi*(pc)^3*hbar*c / ((m c^2)^2 (hbar*Omega)^2)``.

    Quadratic expansion of the on-shell phase ``k(E_0 + n*hbar*Omega)*z``:
    ``d^2k/dE^2 = -(m c^2)^2/((pc)^3 hbar c)``, so the order-``n`` phase
    ``(1/2)|k''|(n hbar Omega)^2 z`` equals ``2*pi*n^2*z/z_T``.

    Validation: longitudinal-quantum-bunching
    """
    if kinetic_eV <= 0 or spacing_eV <= 0:
        raise ValueError("kinetic_eV and spacing_eV must be positive")
    pc = float(_momentum_ev(np.asarray(kinetic_eV, dtype=np.float64)))
    return 4.0 * np.pi * pc**3 * HBARC_EV_ANG / (M_E_EV**2 * spacing_eV**2)


def pinem_amplitudes(coupling: complex, orders: ArrayLike) -> NDArray[np.complex128]:
    """Return ideal PINEM comb amplitudes ``c_n = J_n(2|g|) exp(i n arg g)``.

    The sign of ``arg g`` follows the optical-field phase convention of the
    caller; only relative phases enter the observables here. Truncation leaks
    ``1 - sum |c_n|^2``, which callers check.

    Validation: longitudinal-quantum-bunching
    """
    order = _orders(orders)
    g = complex(coupling)
    return jv(order, 2.0 * abs(g)) * np.exp(1j * order * np.angle(g))


def bunching_coefficient(
    rho: ArrayLike, wavenumbers: ArrayLike, z_ang: float, harmonic: int
) -> complex:
    """Return the harmonic-``l`` density coefficient after drift ``z``.

    At fixed ``z`` the electron density is
    ``sum_nm rho[n,m] exp(i(k_n-k_m)z - i(n-m)Omega t)``. This returns the
    coefficient of ``exp(-i l Omega t)``: ``sum_n rho[n+l, n] exp(i(k_{n+l}-k_n)z)``.

    Validation: longitudinal-quantum-bunching
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    k = np.asarray(wavenumbers, dtype=np.float64)
    ell = int(harmonic)
    if matrix.shape != (k.size, k.size):
        raise ValueError("rho must be square and match the wavenumbers")
    if abs(ell) >= k.size:
        return 0j
    diagonal = np.diagonal(matrix, offset=-ell)
    upper = k[max(ell, 0) : k.size + min(ell, 0)]
    lower = k[max(-ell, 0) : k.size - max(ell, 0)]
    return complex(np.sum(diagonal * np.exp(1j * (upper - lower) * z_ang)))


def pinem_bunching_closed_form(coupling: complex, harmonic: int, z_over_talbot: float) -> float:
    """Return ``|J_l(4|g| sin(2*pi*l*z/z_T))|`` for an ideal PINEM drift.

    Neumann--Graf addition of ``sum_n J_n(x) J_{n+l}(x) exp(i n phi)`` with the
    quadratic phase of :func:`talbot_length_ang`. Its classical limit
    ``z/z_T -> 0`` at fixed ``X = 8*pi*|g|*z/z_T`` is :func:`klystron_bunching`.

    Validation: longitudinal-quantum-bunching
    """
    ell = int(harmonic)
    argument = 4.0 * abs(complex(coupling)) * np.sin(2.0 * np.pi * ell * z_over_talbot)
    return float(abs(jv(ell, argument)))


def klystron_bunching(harmonic: int, bunching_parameter: float) -> float:
    """Return the classical ballistic bunching factor ``|J_l(l X)|``.

    Validation: longitudinal-quantum-bunching
    """
    ell = int(harmonic)
    return float(abs(jv(ell, ell * bunching_parameter)))


def pair_amplitude(rho: ArrayLike, amplitudes: ArrayLike, harmonic: int) -> complex:
    """Return ``Tr(rho M)`` for photons at exactly ``l*hbar*Omega``.

    ``M|n> = A_n |n - l>``: emission of ``l`` comb quanta takes order ``n`` to
    order ``n - l`` (energy conservation in a static medium). Hence
    ``Tr(rho M) = sum_n rho[n, n-l] A_n``, i.e. ``<psi|M|psi>`` for a pure state. This enters only the
    inter-electron pair term ``N(N-1)|Tr(rho M)|^2``; the self term is
    ``sum_n rho[n,n] |A_n|^2`` and bounds it (Cauchy--Schwarz).

    Validation: longitudinal-quantum-bunching
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    amplitude = np.asarray(amplitudes, dtype=np.complex128)
    ell = int(harmonic)
    size = amplitude.size
    if matrix.shape != (size, size):
        raise ValueError("rho must be square and match the amplitudes")
    if ell <= 0 or ell >= size:
        return 0j
    return complex(np.sum(np.diagonal(matrix, offset=-ell) * amplitude[ell:]))


def pair_amplitude_bound(populations: ArrayLike, harmonic: int) -> float:
    """Return ``sum_n sqrt(p_n p_{n-l})``, a state-independent pair bound.

    Positivity gives ``|rho[n, n-l]|^2 <= p_n p_{n-l}`` for every density
    matrix with diagonal ``p``, so
    ``|Tr(rho M)| <= max|A| * sum_n sqrt(p_n p_{n-l})``: the Bhattacharyya
    overlap of the measured energy distribution with itself shifted by the
    photon energy. It is attained by the pure state with nonnegative
    amplitudes ``sqrt(p_n)`` (constant ``A``); for a Gaussian of rms ``sigma``
    it is ``exp(-(hbar omega)^2/(8 sigma^2))``. A beam whose total energy
    spread is narrow compared with ``hbar*omega`` cannot radiate a pair term
    there, whatever its preparation.

    Validation: longitudinal-quantum-bunching
    """
    p = np.asarray(populations, dtype=np.float64)
    ell = abs(int(harmonic))
    if p.ndim != 1 or np.any(p < 0):
        raise ValueError("populations must be a nonnegative one-dimensional array")
    if ell >= p.size:
        return 0.0
    return float(np.sum(np.sqrt(p[ell:] * p[: p.size - ell])))


def photon_density_matrix(rho: ArrayLike, amplitudes: ArrayLike, subdivisions: int) -> NDArray:
    """Return the reduced one-photon frequency density matrix by explicit trace.

    ``amplitudes[n, a]`` is ``A_n(omega_a)`` on the photon grid
    ``omega_a = omega_0 + a*hbar*Omega/subdivisions``. The joint
    electron--photon state places component ``n`` with photon ``a`` at final
    electron energy index ``f = n*subdivisions - a``. Tracing ``f`` gives
    ``rho_ph[a, b] = sum rho[n, m] A_n(a) conj(A_m(b))`` over pairs with
    ``n*subdivisions - a == m*subdivisions - b``. Distinct ``n`` at one ``a``
    never share ``f``, so the diagonal (photon counts) contains no electron
    coherence; coherences appear only at ``a - b = (n - m)*subdivisions``.

    Validation: longitudinal-energy-diagonal-emission
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    amplitude = np.asarray(amplitudes, dtype=np.complex128)
    m_sub = int(subdivisions)
    if m_sub < 1:
        raise ValueError("subdivisions must be positive")
    orders, photons = amplitude.shape
    if matrix.shape != (orders, orders):
        raise ValueError("rho must be square and match amplitudes' first axis")
    final = np.arange(orders)[:, None] * m_sub - np.arange(photons)[None, :]
    offset = int(final.min())
    joint = np.zeros((int(final.max()) - offset + 1, orders, photons), dtype=np.complex128)
    rows, cols = np.indices((orders, photons))
    joint[final - offset, rows, cols] = amplitude
    # rho_ph[a,b] = sum_f sum_nm rho[n,m] Psi_n[f,a] conj(Psi_m[f,b])
    return np.einsum("nm,fna,fmb->ab", matrix, joint, joint.conj())


def slab_mismatch_inv_ang(
    photon_eV: ArrayLike,
    kinetic_eV: ArrayLike,
    g_ang_inv: ArrayLike,
    photon_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Return the fixed-energy longitudinal mismatch of one stationary slab.

    The electron moves along +z, the slab normal; the slab is laterally
    infinite and static, so energy and transverse momentum are conserved:
    ``p_f_perp = -hbar(g_perp + k_perp)`` and
    ``p_f_z = sqrt(p(E - photon)^2 - p_f_perp^2)``. With the
    ``line-recoil-resonance`` convention ``p_f = p_i - hbar g - hbar k``, the
    mismatch is ``(p_i - hbar g_z - hbar k_z - p_f_z)/hbar``. For a forward
    final electron (``p_i - hbar g_z - hbar k_z > 0``) it vanishes exactly at
    the vacuum root of ``materials.recoil.recoil_resonance_eV`` for ``u = z``;
    at that root a backward final electron gives ``2 p_f_z/hbar``. It is NaN
    where the final longitudinal momentum is evanescent.

    Validation: longitudinal-energy-diagonal-emission
    """
    photon = np.asarray(photon_eV, dtype=np.float64)
    kinetic = np.asarray(kinetic_eV, dtype=np.float64)
    g = np.asarray(g_ang_inv, dtype=np.float64)
    n = np.asarray(photon_direction, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    k_photon = photon[..., None] * n / HBARC_EV_ANG
    transverse = g[..., :2] + k_photon[..., :2]
    final_kinetic = kinetic - photon
    with np.errstate(invalid="ignore"):
        k_final = _momentum_ev(final_kinetic) / HBARC_EV_ANG
        k_final_z = np.sqrt(k_final**2 - np.sum(transverse**2, axis=-1))
    k_initial = _momentum_ev(kinetic) / HBARC_EV_ANG
    return k_initial - g[..., 2] - k_photon[..., 2] - k_final_z


def slab_formation_amplitude(mismatch_inv_ang: ArrayLike, thickness_ang: float) -> NDArray:
    """Return ``int_0^L exp(i*Delta*z) dz`` with its ``Delta -> 0`` limit ``L``.

    Validation: longitudinal-energy-diagonal-emission
    """
    delta = np.asarray(mismatch_inv_ang, dtype=np.float64)
    half = 0.5 * delta * thickness_ang
    return thickness_ang * np.exp(1j * half) * np.sinc(half / np.pi)


def channeling_recoil_line_energy_eV(
    kinetic_eV: ArrayLike,
    initial_level_eV: ArrayLike,
    final_level_eV: ArrayLike,
    final_transverse_kinetic_eV: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
) -> NDArray[np.float64]:
    """Return the recoil-exact planar channeling line for a lowering transition.

    Paraxial planar continuum model: plane normal ``x``, free in-plane ``y``,
    channel axis ``z``; ``(p_z c)^2 + (p_y c)^2 = E^2 - (mc^2)^2 - 2E*eps(E)``
    with ``eps`` an eigenvalue of ``H_perp = p_x^2 c^2/(2E) + U(x)``. The
    static plane absorbs photon momentum along ``x`` without energy; ``y`` and
    ``z`` momenta and the energy are conserved. The final level at
    ``E' = E - photon`` uses Hellmann--Feynman,
    ``d eps_f/d ln E = -<T_perp>_f``, linearized in ``ln(E'/E)``. With
    ``a = sin^2(theta) cos^2(phi)`` the root of
    ``photon*[2(E - p_z c cos theta) - 2 eps_f(E')] - a photon^2
    = 2E[eps_i - eps_f(E')]`` is returned (NaN if none). Any fixed potential zero
    may be used (offsets enter at ``O(eps/E)``); ``theta`` is measured from ``z`` and ``phi``
    from ``x``. Forward with ``<T_perp>_f = 0`` it equals the Baier--Katkov
    form ``w_D/(1 + w_D/E)`` of the Doppler line ``w_D`` to ``O(eps/E)``.

    Validation: cr-recoil-line-energy
    """
    kinetic, eps_i, eps_f, t_f, theta_a, phi_a = np.broadcast_arrays(
        *(
            np.asarray(v, dtype=np.float64)
            for v in (
                kinetic_eV,
                initial_level_eV,
                final_level_eV,
                final_transverse_kinetic_eV,
                theta,
                phi,
            )
        )
    )
    total = kinetic + M_E_EV
    pz2 = _momentum_ev(kinetic) ** 2 - 2.0 * total * eps_i
    with np.errstate(invalid="ignore", divide="ignore"):
        pz = np.sqrt(pz2)
        # E - p_z cos(theta) without cancellation: (E^2 - p_z^2)/(E + p_z) + p_z(1 - cos).
        doppler = (M_E_EV**2 + 2.0 * total * eps_i) / (total + pz)
        doppler += 2.0 * pz * np.sin(0.5 * theta_a) ** 2
        a = (np.sin(theta_a) * np.cos(phi_a)) ** 2
        photon = np.zeros_like(total)
        for _ in range(64):
            level_f = eps_f - t_f * np.log1p(-photon / total)
            c = 2.0 * total * (eps_i - level_f)
            b = 2.0 * (doppler - level_f)
            disc = b * b - 4.0 * a * c
            updated = 2.0 * c / (b + np.sqrt(disc))
            if np.all(np.abs(updated - photon) <= 4e-16 * np.abs(updated)):
                photon = updated
                break
            photon = updated
    valid = (pz2 > 0) & (photon > 0) & (photon <= kinetic) & np.isfinite(photon)
    return np.where(valid, photon, np.nan)
