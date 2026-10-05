"""Numba kernels of the opt-in shell soft/hard inelastic transport mode.

The host builder in :mod:`.shell_transport` tabulates the stopping-closed
PENELOPE-like shell partition of :mod:`.shell_partition`; the CPU cores in
:mod:`.cores` read those tables through the scalar kernels below. This module
imports no host physics module, so the cores stay cheap to import.

RNG layout. Hard events draw from a counter-addressed SplitMix64 stream keyed
by ``_hard_stream_key_scalar(stream_key)``: the electron's own stream key
re-hashed with a salt distinct from the Urban straggling salt, so hard draws
never share addresses with the free-path/scattering or straggling streams.
Per electron the counter advances sequentially through one uniform for each
physical flight's hard optical depth and four uniforms (channel, transfer,
recoil, azimuth) for each hard event. Soft-loss fluctuations use the existing
per-``(flight, substep)`` straggling key domain, which the Urban sampler does
not use in this mode.
"""

import numpy as np
from numba import njit, uint64
from scipy.constants import c, e, m_e

from .core_geometry import _rotate_direction_scalar
from .kinematics import (
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
    _splitmix64,
    _stream_uniform_scalar,
    stream_keys,
)

_MC2_EV = m_e * c * c / e
# ``simulate_trajectories(inelastic_model=...)`` values; "continuous" is the
# historical default.
INELASTIC_MODELS = ("continuous", "shell-soft-hard")
# Differs from the Urban salt and from every constant of the counter hash.
_HARD_STREAM_SALT = np.uint64(0x3C6EF372FE94F82B)
# PENELOPE-2024 Eq. 4.59 width factor that restores the variance of a
# Gaussian truncated at three standard deviations.
_TRUNCATED_GAUSSIAN_WIDTH = 1.015387
_BISECTION_STEPS = 200


def validate_inelastic_args(
    model, cutoff_eV, materials, *, energy_model, groove, stopping_tables
) -> bool:
    """Check :func:`simulate_trajectories` inelastic arguments; True in shell mode.

    The per-material ``W_c > W_cb`` check runs with the shell tables.
    """
    if model not in INELASTIC_MODELS:
        raise ValueError(f"inelastic_model must be one of {', '.join(INELASTIC_MODELS)}")
    if model == "continuous":
        if cutoff_eV is not None or materials is not None:
            raise ValueError(
                "inelastic_cutoff_eV and inelastic_materials require "
                "inelastic_model='shell-soft-hard'"
            )
        return False
    if energy_model != "midpoint":
        raise ValueError("inelastic_model='shell-soft-hard' requires energy_model='midpoint'")
    if groove is not None:
        raise NotImplementedError(
            "inelastic_model='shell-soft-hard' is not implemented for grooved transport"
        )
    if stopping_tables is None:
        raise ValueError("inelastic_model='shell-soft-hard' requires stopping_tables")
    if cutoff_eV is None or materials is None:
        raise ValueError(
            "inelastic_model='shell-soft-hard' requires inelastic_cutoff_eV and inelastic_materials"
        )
    return True


@njit(uint64(uint64), cache=True)
def _hard_stream_key_scalar(stream_key):
    """Per-electron hard-event key, disjoint from the other transport streams."""
    return _splitmix64(stream_key ^ _HARD_STREAM_SALT)


def hard_stream_keys(seed, Ne):
    """Host twin of :func:`_hard_stream_key_scalar` over electrons ``[0, Ne)``."""
    return hard_keys_from_stream_keys(stream_keys(seed, Ne))


def hard_keys_from_stream_keys(keys):
    """Host twin of :func:`_hard_stream_key_scalar` over explicit stream keys."""
    x = np.asarray(keys, dtype=np.uint64) ^ _HARD_STREAM_SALT
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


@njit(cache=True)
def _log_grid_frac(log_grid, n, log_e):
    """Lower node and fraction of ``log_e`` on a sorted, possibly uneven grid.

    Clamps to the first/last interval; the host validates the transport range
    against the grid, so the clamp only absorbs endpoint roundoff.
    """
    if log_e <= log_grid[0]:
        return 0, 0.0
    if log_e >= log_grid[n - 1]:
        return n - 2, 1.0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if log_grid[mid] <= log_e:
            lo = mid
        else:
            hi = mid
    return lo, (log_e - log_grid[lo]) / (log_grid[hi] - log_grid[lo])


@njit(cache=True)
def _triangle_cdf(peak_end, lower, w):
    """Unnormalized ``int_lower^w p_dis(W)/W dW`` for the triangle ``p_dis``."""
    return peak_end * np.log(w / lower) - (w - lower)


@njit(cache=True)
def _moller_j0(energy_eV, prime_eV, w):
    """``J_0^(-)`` antiderivative of ``F^(-)(E', W)/W^2`` (shell_gos twin)."""
    a = (energy_eV / (energy_eV + _MC2_EV)) ** 2
    rest = prime_eV - w
    return -1.0 / w + 1.0 / rest + (1.0 - a) / prime_eV * np.log(rest / w) + a * w / prime_eV**2


@njit(cache=True)
def _hard_loss_bounds(energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV):
    """``(lower, upper, peak_end)`` of one channel's hard loss interval [eV].

    Matches ``shell_sampling._loss_bounds``: close losses run from
    ``max(U or W_cb, W_c)`` to ``(E+U)/2``; bound distant losses from
    ``max(U, W_c)`` to ``min(W_dis, (E+U)/2)`` with the near-threshold
    ``W_dis = E`` of PENELOPE Eqs. 3.77-3.80.
    """
    if branch == 2:
        base = ionization_eV if ionization_eV > 0.0 else resonance_eV
        return max(base, cutoff_eV), 0.5 * (energy_eV + ionization_eV), 0.0
    if ionization_eV == 0.0:
        return resonance_eV, resonance_eV, 0.0
    peak_end = 3.0 * resonance_eV - 2.0 * ionization_eV
    if energy_eV <= peak_end:
        peak_end = energy_eV
    upper = min(peak_end, 0.5 * (energy_eV + ionization_eV))
    return max(ionization_eV, cutoff_eV), upper, peak_end


@njit(cache=True)
def _sample_hard_transfer_eV(energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV, u):
    """Invert one channel's restricted hard loss CDF at uniform ``u`` [eV].

    Source: PENELOPE-2024 Eqs. 3.76, 3.87, 3.94, 3.96 and 3.104, restricted
    to ``W > W_c`` as in Eq. 3.124: bound distant losses have density
    ``p_dis(W)/W`` (the adopted convention of ``shell_sampling``); close
    losses ``F^(-)(E+U, W)/W^2``; a conduction-band distant loss is a delta at
    ``W_cb``. Bisection to the float64 resolution of the loss interval
    replaces the host sampler's ``brentq``; both invert the same CDF.

    Limits: ``u = 0`` returns the lower bound; a degenerate interval (only
    reachable where an interpolated rate straddles a channel threshold)
    returns its lower bound. Validation: penelope-shell-hard-loss-sampling

    Validation: shell-soft-hard-transport
    """
    lower, upper, peak_end = _hard_loss_bounds(
        energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV
    )
    if branch != 2 and ionization_eV == 0.0:
        return resonance_eV
    if not upper > lower:
        return lower
    prime = energy_eV + ionization_eV
    if branch == 2:
        base = _moller_j0(energy_eV, prime, lower)
        target = u * (_moller_j0(energy_eV, prime, upper) - base)
    else:
        target = u * _triangle_cdf(peak_end, lower, upper)
    lo, hi = lower, upper
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if branch == 2:
            value = _moller_j0(energy_eV, prime, mid) - base
        else:
            value = _triangle_cdf(peak_end, lower, mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _qmin_ev_scalar(energy_eV, transfer_eV):
    """Scalar twin of ``inelastic._qmin_ev`` (Geant4 PRM Penelope Eq. 127)."""
    p0 = np.sqrt(energy_eV * (energy_eV + 2.0 * _MC2_EV))
    remaining = energy_eV - transfer_eV
    p1 = np.sqrt(remaining * (remaining + 2.0 * _MC2_EV))
    dp = transfer_eV * (2.0 * (energy_eV + _MC2_EV) - transfer_eV) / (p0 + p1)
    return dp * dp / (np.sqrt(dp * dp + _MC2_EV * _MC2_EV) + _MC2_EV)


@njit(cache=True)
def _hard_primary_cosine(energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u):
    """Primary polar cosine after a hard collision, relative to the flight.

    Source: PENELOPE-2024 Eqs. 3.126-3.129 (longitudinal ``Q`` with density
    ``1/[Q(1+Q/2mc^2)]`` on ``[Q_-, Q'_k]`` and the modified resonance) and
    Eq. 3.134 (close, ``Q = W``); transverse events do not deflect. Scalar
    twin of ``shell_sampling.sample_shell_hard_collision``.

    Limit: ``u = 0`` gives the minimum longitudinal ``Q`` and no deflection.
    An empty recoil interval (interpolation-edge only) returns 1.
    Validation: penelope-shell-hard-recoil

    Validation: shell-soft-hard-transport
    """
    two_mc2 = 2.0 * _MC2_EV
    if branch == 1:
        return 1.0
    if branch == 2:
        remaining = energy_eV - transfer_eV
        return np.sqrt(remaining / energy_eV * (energy_eV + two_mc2) / (remaining + two_mc2))
    if ionization_eV > 0.0:
        full_width = 3.0 * resonance_eV - 2.0 * ionization_eV
        if energy_eV > full_width:
            resonance, q_upper = resonance_eV, ionization_eV
        else:
            resonance = (energy_eV + 2.0 * ionization_eV) / 3.0
            q_upper = ionization_eV * energy_eV / full_width
    else:
        resonance, q_upper = resonance_eV, resonance_eV
    if not resonance < energy_eV:
        return 1.0
    q_lower = _qmin_ev_scalar(energy_eV, resonance)
    if not 0.0 < q_lower < q_upper:
        return 1.0
    log_lower = np.log(q_lower / (q_lower + two_mc2))
    log_upper = np.log(q_upper / (q_upper + two_mc2))
    log_ratio = (1.0 - u) * log_lower + u * log_upper
    recoil = two_mc2 / np.expm1(-log_ratio)
    p0_sq = energy_eV * (energy_eV + two_mc2)
    remaining = energy_eV - resonance
    p1_sq = remaining * (remaining + two_mc2)
    cosine = (p0_sq + p1_sq - recoil * (recoil + two_mc2)) / (2.0 * np.sqrt(p0_sq * p1_sq))
    return min(1.0, max(-1.0, cosine))


@njit(cache=True)
def _hard_secondary_cosine(energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u):
    """Secondary polar cosine of a hard collision, relative to the flight.

    Source: PENELOPE-2024 §3.2.5.4, Eqs. 3.137-3.138: the emitted electron
    follows the momentum transfer. Longitudinal events use the recoil ``Q``
    that :func:`_hard_primary_cosine` samples from the same uniform ``u`` and
    ``cos = [p(E)^2 + p(Q)^2 - p(E - W'_k)^2] / [2 p(E) p(Q)]``; close events
    ``Q = W``; transverse events the fixed cosine 0.5 of the Geant4 Penelope
    implementation. Scalar twin of ``shell_sampling.sample_shell_hard_collision``.

    Limit: an empty recoil interval (interpolation-edge only) returns 1, as the
    primary twin does. Validation: penelope-shell-secondary-direction,
    shell-secondary-transport
    """
    two_mc2 = 2.0 * _MC2_EV
    if branch == 1:
        return 0.5
    if branch == 2:
        return np.sqrt(transfer_eV / energy_eV * (energy_eV + two_mc2) / (transfer_eV + two_mc2))
    if ionization_eV > 0.0:
        full_width = 3.0 * resonance_eV - 2.0 * ionization_eV
        if energy_eV > full_width:
            resonance, q_upper = resonance_eV, ionization_eV
        else:
            resonance = (energy_eV + 2.0 * ionization_eV) / 3.0
            q_upper = ionization_eV * energy_eV / full_width
    else:
        resonance, q_upper = resonance_eV, resonance_eV
    if not resonance < energy_eV:
        return 1.0
    q_lower = _qmin_ev_scalar(energy_eV, resonance)
    if not 0.0 < q_lower < q_upper:
        return 1.0
    log_lower = np.log(q_lower / (q_lower + two_mc2))
    log_upper = np.log(q_upper / (q_upper + two_mc2))
    log_ratio = (1.0 - u) * log_lower + u * log_upper
    recoil = two_mc2 / np.expm1(-log_ratio)
    p0_sq = energy_eV * (energy_eV + two_mc2)
    remaining = energy_eV - resonance
    p1_sq = remaining * (remaining + two_mc2)
    q_sq = recoil * (recoil + two_mc2)
    cosine = (p0_sq + q_sq - p1_sq) / (2.0 * np.sqrt(p0_sq * q_sq))
    return min(1.0, max(-1.0, cosine))


@njit(cache=True)
def _hard_secondary_direction(
    dx, dy, dz, energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u_q, u_phi
):
    """Laboratory direction of a hard collision's secondary; no extra draw.

    Rotates :func:`_hard_secondary_cosine` about the pre-collision flight
    ``(dx, dy, dz)`` at azimuth ``(2 pi u_phi + pi) mod 2 pi``, opposite the
    primary's, in the frame the primary recoil uses
    (``shell_sampling.shell_collision_world_directions``).
    Validation: penelope-shell-secondary-direction, shell-secondary-transport
    """
    cosine = _hard_secondary_cosine(
        energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u_q
    )
    phi = (2.0 * np.pi * u_phi + np.pi) % (2.0 * np.pi)
    return _rotate_direction_scalar(dx, dy, dz, cosine, phi)


@njit(cache=True)
def _soft_loss_sample_keV(mean, variance, key, counter):
    """Soft energy loss with the first two soft moments, ``(loss, counter)``.

    Source: PENELOPE-2024 Eqs. 4.54-4.63. With ``<w> = S_s s`` and
    ``var(w) = Omega_s^2 s``: case I (``<w>^2 > 9 var``) a Gaussian of width
    ``1.015387 sigma`` truncated at ``|w - <w>| < 3 sigma`` (Box-Muller with
    rejection, two uniforms per trial); case II (``3 var < <w>^2 <= 9 var``)
    uniform on ``<w> -/+ sqrt(3) sigma``; case III a delta at zero with
    weight ``a`` plus a uniform on ``[0, w_0)``, one uniform.

    Assumptions: the soft DCS is frozen at the row's start energy (the
    Eq. 4.65 energy-dependence correction is not applied; ``max_dE_frac``
    bounds it). Limits: ``var = 0`` returns the mean; every case has mean
    ``<w>`` and variance ``var`` exactly (case I up to the published
    truncation factor), and no case returns a negative loss.
    Validation: shell-soft-hard-transport
    """
    if mean <= 0.0:
        return 0.0, counter
    if variance <= 0.0:
        return mean, counter
    sigma = np.sqrt(variance)
    mean_sq = mean * mean
    if mean_sq > 9.0 * variance:
        width = _TRUNCATED_GAUSSIAN_WIDTH * sigma
        while True:
            u1 = _stream_uniform_scalar(key, counter)
            counter = counter + _SM64_ONE
            u2 = _stream_uniform_scalar(key, counter)
            counter = counter + _SM64_ONE
            z = np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
            offset = width * z
            if abs(offset) < 3.0 * sigma:
                return mean + offset, counter
    u = _stream_uniform_scalar(key, counter)
    counter = counter + _SM64_ONE
    if mean_sq > 3.0 * variance:
        half = np.sqrt(3.0) * sigma
        return mean - half + 2.0 * half * u, counter
    weight = (3.0 * variance - mean_sq) / (3.0 * variance + 3.0 * mean_sq)
    top = (3.0 * variance + 3.0 * mean_sq) / (2.0 * mean)
    if u < weight:
        return 0.0, counter
    return top * (u - weight) / (1.0 - weight), counter
