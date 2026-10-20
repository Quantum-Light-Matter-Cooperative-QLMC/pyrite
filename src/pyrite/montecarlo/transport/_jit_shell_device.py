"""Device helpers of the opt-in shell soft/hard inelastic mode on CUDA.

Scalar twins of the Numba kernels in :mod:`.hard_inelastic`, called from the
exact CUDA transport kernel in :mod:`._jit_kernel`. Each is a transcription of
its host twin in the same operation order, so the port is checkable against a
per-electron CPU run from identical draws. Host and device share the
constants below by import rather than transcription, so they cannot drift.

Device functions return one value, so the host ``_log_grid_frac`` pair is
split into :func:`_log_grid_lower` and :func:`_log_grid_fraction`, and the
host soft sampler's advanced counter (discarded by every caller) is dropped.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__``.

Validation: shell-secondary-transport
Validation: shell-soft-hard-transport
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import (
    F64_HALF,
    F64_ONE,
    F64_PI,
    F64_TWO,
    F64_ZERO,
    I32_ONE,
    I32_TWO,
    I32_ZERO,
    U64_ONE,
    _stream_uniform,
)
from .hard_inelastic import (
    _BISECTION_STEPS,
    _MC2_EV,
    _TRUNCATED_GAUSSIAN_WIDTH,
    POSITRON_BRANCH_OFFSET,
)

F64_MC2_EV = np.float64(_MC2_EV)
F64_TWO_MC2_EV = np.float64(2.0 * _MC2_EV)
F64_THREE = np.float64(3.0)
F64_NINE = np.float64(9.0)
F64_SQRT3 = np.float64(np.sqrt(3.0))
F64_GAUSS_WIDTH = np.float64(_TRUNCATED_GAUSSIAN_WIDTH)
I32_BISECTION_STEPS = np.int32(_BISECTION_STEPS)
I32_POSITRON_BRANCH = np.int32(POSITRON_BRANCH_OFFSET)
# ``hard_channel`` of a row without a hard collision.
I16_NO_CHANNEL = np.int16(-1)


@jit.rawkernel(device=True)
def _log_grid_lower(log_grid, row, n, log_e):
    """Lower node of ``log_e`` on one flattened grid row; host ``_log_grid_frac``."""
    if log_e <= log_grid[row]:
        return I32_ZERO
    if log_e >= log_grid[row + n - I32_ONE]:
        return n - I32_TWO
    lower_idx = I32_ZERO
    upper_idx = n - I32_ONE
    while upper_idx - lower_idx > I32_ONE:
        mid_idx = (lower_idx + upper_idx) // I32_TWO
        if log_grid[row + mid_idx] <= log_e:
            lower_idx = mid_idx
        else:
            upper_idx = mid_idx
    return lower_idx


@jit.rawkernel(device=True)
def _log_grid_fraction(log_grid, row, n, log_e, lo):
    """Fraction of ``log_e`` above node ``lo``, clamped like ``_log_grid_frac``."""
    if log_e <= log_grid[row]:
        return F64_ZERO
    if log_e >= log_grid[row + n - I32_ONE]:
        return F64_ONE
    lower = log_grid[row + lo]
    return (log_e - lower) / (log_grid[row + lo + I32_ONE] - lower)


@jit.rawkernel(device=True)
def _triangle_cdf(peak_end, lower, w):
    """Unnormalized ``int_lower^w p_dis(W)/W dW``; host ``_triangle_cdf``."""
    return peak_end * xp.log(w / lower) - (w - lower)


@jit.rawkernel(device=True)
def _moller_j0(energy_eV, prime_eV, w):
    """``J_0^(-)`` antiderivative of ``F^(-)(E', W)/W^2``; host ``_moller_j0``."""
    ratio = energy_eV / (energy_eV + F64_MC2_EV)
    a = ratio * ratio
    rest = prime_eV - w
    return (
        -F64_ONE / w
        + F64_ONE / rest
        + (F64_ONE - a) / prime_eV * xp.log(rest / w)
        + a * w / (prime_eV * prime_eV)
    )


@jit.rawkernel(device=True)
def _bhabha_j0(energy_eV, w):
    """``J_0^(+)`` antiderivative of ``F^(+)(E, W)/W^2``; host ``_bhabha_j0``.

    Validation: bhabha-close
    """
    gamma = F64_ONE + energy_eV / F64_MC2_EV
    ratio = (gamma - F64_ONE) / gamma
    g = ratio * ratio
    gp1 = gamma + F64_ONE
    b1 = (gamma - F64_ONE) / (gamma * gamma * gp1) * (F64_TWO * gp1 * gp1 - F64_ONE)
    b2 = g * (F64_THREE * gp1 * gp1 + F64_ONE) / (gp1 * gp1)
    b3 = g * F64_TWO * gamma * (gamma - F64_ONE) / (gp1 * gp1)
    b4 = g * (gamma - F64_ONE) * (gamma - F64_ONE) / (gp1 * gp1)
    x = w / energy_eV
    return (
        -F64_ONE / w
        + (-b1 * xp.log(w) + b2 * x - b3 * x * x / F64_TWO + b4 * x * x * x / F64_THREE) / energy_eV
    )


@jit.rawkernel(device=True)
def _sample_hard_transfer_eV(energy_eV, ionization_eV, resonance_eV, branch, cutoff_eV, u):
    """Invert one channel's restricted hard loss CDF; host ``_sample_hard_transfer_eV``.

    ``_hard_loss_bounds`` and ``_split_branch`` are inlined: a positron-tagged
    branch takes ``W_max = E`` and the Bhabha close CDF (#276).
    Validation: penelope-shell-hard-loss-sampling, bhabha-close
    """
    positron = branch >= I32_POSITRON_BRANCH
    if positron:
        branch = branch - I32_POSITRON_BRANCH
    w_max = F64_HALF * (energy_eV + ionization_eV)
    if positron:
        w_max = energy_eV
    peak_end = F64_ZERO
    if branch == I32_TWO:
        base = ionization_eV
        if not ionization_eV > F64_ZERO:
            base = resonance_eV
        lower = xp.maximum(base, cutoff_eV)
        upper = w_max
    elif ionization_eV == F64_ZERO:
        return resonance_eV
    else:
        peak_end = F64_THREE * resonance_eV - F64_TWO * ionization_eV
        if energy_eV <= peak_end:
            peak_end = energy_eV
        upper = xp.minimum(peak_end, w_max)
        lower = xp.maximum(ionization_eV, cutoff_eV)
    if not upper > lower:
        return lower
    prime = energy_eV + ionization_eV
    base_j = F64_ZERO
    if branch == I32_TWO and positron:
        base_j = _bhabha_j0(energy_eV, lower)
        target = u * (_bhabha_j0(energy_eV, upper) - base_j)
    elif branch == I32_TWO:
        base_j = _moller_j0(energy_eV, prime, lower)
        target = u * (_moller_j0(energy_eV, prime, upper) - base_j)
    else:
        target = u * _triangle_cdf(peak_end, lower, upper)
    lower_transfer_eV = lower
    upper_transfer_eV = upper
    step = I32_ZERO
    while step < I32_BISECTION_STEPS:
        step += I32_ONE
        mid_transfer_eV = F64_HALF * (lower_transfer_eV + upper_transfer_eV)
        if mid_transfer_eV <= lower_transfer_eV or mid_transfer_eV >= upper_transfer_eV:
            break
        if branch == I32_TWO and positron:
            value = _bhabha_j0(energy_eV, mid_transfer_eV) - base_j
        elif branch == I32_TWO:
            value = _moller_j0(energy_eV, prime, mid_transfer_eV) - base_j
        else:
            value = _triangle_cdf(peak_end, lower, mid_transfer_eV)
        if value < target:
            lower_transfer_eV = mid_transfer_eV
        else:
            upper_transfer_eV = mid_transfer_eV
    return F64_HALF * (lower_transfer_eV + upper_transfer_eV)


@jit.rawkernel(device=True)
def _qmin_ev(energy_eV, transfer_eV):
    """Minimum recoil energy [eV]; host ``_qmin_ev_scalar``."""
    p0 = xp.sqrt(energy_eV * (energy_eV + F64_TWO_MC2_EV))
    remaining = energy_eV - transfer_eV
    p1 = xp.sqrt(remaining * (remaining + F64_TWO_MC2_EV))
    dp = transfer_eV * (F64_TWO * (energy_eV + F64_MC2_EV) - transfer_eV) / (p0 + p1)
    return dp * dp / (xp.sqrt(dp * dp + F64_MC2_EV * F64_MC2_EV) + F64_MC2_EV)


@jit.rawkernel(device=True)
def _hard_primary_cosine(energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u):
    """Primary polar cosine after a hard collision; host ``_hard_primary_cosine``.

    Validation: penelope-shell-hard-recoil
    """
    two_mc2 = F64_TWO_MC2_EV
    if branch >= I32_POSITRON_BRANCH:
        branch = branch - I32_POSITRON_BRANCH
    if branch == I32_ONE:
        return F64_ONE
    if branch == I32_TWO:
        remaining = energy_eV - transfer_eV
        return xp.sqrt(remaining / energy_eV * (energy_eV + two_mc2) / (remaining + two_mc2))
    if ionization_eV > F64_ZERO:
        full_width = F64_THREE * resonance_eV - F64_TWO * ionization_eV
        if energy_eV > full_width:
            resonance = resonance_eV
            q_upper = ionization_eV
        else:
            resonance = (energy_eV + F64_TWO * ionization_eV) / F64_THREE
            q_upper = ionization_eV * energy_eV / full_width
    else:
        resonance = resonance_eV
        q_upper = resonance_eV
    if not resonance < energy_eV:
        return F64_ONE
    q_lower = _qmin_ev(energy_eV, resonance)
    if not (q_lower > F64_ZERO and q_lower < q_upper):
        return F64_ONE
    log_lower = xp.log(q_lower / (q_lower + two_mc2))
    log_upper = xp.log(q_upper / (q_upper + two_mc2))
    log_ratio = (F64_ONE - u) * log_lower + u * log_upper
    recoil = two_mc2 / xp.expm1(-log_ratio)
    p0_sq = energy_eV * (energy_eV + two_mc2)
    remaining = energy_eV - resonance
    p1_sq = remaining * (remaining + two_mc2)
    cosine = (p0_sq + p1_sq - recoil * (recoil + two_mc2)) / (F64_TWO * xp.sqrt(p0_sq * p1_sq))
    return xp.minimum(F64_ONE, xp.maximum(-F64_ONE, cosine))


@jit.rawkernel(device=True)
def _hard_secondary_cosine(energy_eV, ionization_eV, resonance_eV, branch, transfer_eV, u):
    """Secondary polar cosine of a hard collision; host ``_hard_secondary_cosine``.

    Validation: penelope-shell-secondary-direction, shell-secondary-transport
    """
    two_mc2 = F64_TWO_MC2_EV
    if branch >= I32_POSITRON_BRANCH:
        branch = branch - I32_POSITRON_BRANCH
    if branch == I32_ONE:
        return F64_HALF
    if branch == I32_TWO:
        return xp.sqrt(transfer_eV / energy_eV * (energy_eV + two_mc2) / (transfer_eV + two_mc2))
    if ionization_eV > F64_ZERO:
        full_width = F64_THREE * resonance_eV - F64_TWO * ionization_eV
        if energy_eV > full_width:
            resonance = resonance_eV
            q_upper = ionization_eV
        else:
            resonance = (energy_eV + F64_TWO * ionization_eV) / F64_THREE
            q_upper = ionization_eV * energy_eV / full_width
    else:
        resonance = resonance_eV
        q_upper = resonance_eV
    if not resonance < energy_eV:
        return F64_ONE
    q_lower = _qmin_ev(energy_eV, resonance)
    if not (q_lower > F64_ZERO and q_lower < q_upper):
        return F64_ONE
    log_lower = xp.log(q_lower / (q_lower + two_mc2))
    log_upper = xp.log(q_upper / (q_upper + two_mc2))
    log_ratio = (F64_ONE - u) * log_lower + u * log_upper
    recoil = two_mc2 / xp.expm1(-log_ratio)
    p0_sq = energy_eV * (energy_eV + two_mc2)
    remaining = energy_eV - resonance
    p1_sq = remaining * (remaining + two_mc2)
    q_sq = recoil * (recoil + two_mc2)
    cosine = (p0_sq + q_sq - p1_sq) / (F64_TWO * xp.sqrt(p0_sq * q_sq))
    return xp.minimum(F64_ONE, xp.maximum(-F64_ONE, cosine))


@jit.rawkernel(device=True)
def _store_rotated(out, at, dx, dy, dz, cos_t, phi):
    """Write ``(dx, dy, dz)`` rotated by ``(cos_t, phi)`` to ``out[at:at+3]``.

    The host ``core_geometry._rotate_direction_scalar`` arithmetic, as the
    kernel inlines it for the primary. Returns 0 (device functions return one
    value). Validation: shell-secondary-transport
    """
    sin_theta_squared = F64_ONE - cos_t * cos_t
    if sin_theta_squared < F64_ZERO:
        sin_theta_squared = F64_ZERO
    sin_theta = xp.sqrt(sin_theta_squared)
    cos_phi = xp.cos(phi)
    sin_phi = xp.sin(phi)
    if xp.abs(dx) < np.float64(0.9):
        reference_x = F64_ONE
        reference_y = F64_ZERO
    else:
        reference_x = F64_ZERO
        reference_y = F64_ONE
    basis_u_x = -dz * reference_y
    basis_u_y = dz * reference_x
    basis_u_z = dx * reference_y - dy * reference_x
    basis_u_norm = xp.sqrt(basis_u_x * basis_u_x + basis_u_y * basis_u_y + basis_u_z * basis_u_z)
    basis_u_x /= basis_u_norm
    basis_u_y /= basis_u_norm
    basis_u_z /= basis_u_norm
    basis_w_x = dy * basis_u_z - dz * basis_u_y
    basis_w_y = dz * basis_u_x - dx * basis_u_z
    basis_w_z = dx * basis_u_y - dy * basis_u_x
    rotation_u = sin_theta * cos_phi
    rotation_w = sin_theta * sin_phi
    rotated_x = cos_t * dx + rotation_u * basis_u_x + rotation_w * basis_w_x
    rotated_y = cos_t * dy + rotation_u * basis_u_y + rotation_w * basis_w_y
    rotated_z = cos_t * dz + rotation_u * basis_u_z + rotation_w * basis_w_z
    direction_norm = xp.sqrt(rotated_x * rotated_x + rotated_y * rotated_y + rotated_z * rotated_z)
    out[at] = rotated_x / direction_norm
    out[at + I32_ONE] = rotated_y / direction_norm
    out[at + I32_TWO] = rotated_z / direction_norm
    return F64_ZERO


@jit.rawkernel(device=True)
def _soft_loss_sample_keV(mean, variance, key):
    """Two-moment soft energy loss [keV]; host ``_soft_loss_sample_keV``.

    Draws from ``key`` starting at counter 0, as every host call site does.
    Validation: shell-soft-hard-transport
    """
    if mean <= F64_ZERO:
        return F64_ZERO
    if variance <= F64_ZERO:
        return mean
    sigma = xp.sqrt(variance)
    mean_sq = mean * mean
    counter = np.uint64(0)
    if mean_sq > F64_NINE * variance:
        width = F64_GAUSS_WIDTH * sigma
        loss = mean
        accepted = False
        while not accepted:
            u1 = _stream_uniform(key, counter)
            counter = counter + U64_ONE
            u2 = _stream_uniform(key, counter)
            counter = counter + U64_ONE
            z = xp.sqrt(-F64_TWO * xp.log(F64_ONE - u1)) * xp.cos(F64_TWO * F64_PI * u2)
            offset = width * z
            if xp.abs(offset) < F64_THREE * sigma:
                loss = mean + offset
                accepted = True
        return loss
    u = _stream_uniform(key, counter)
    if mean_sq > F64_THREE * variance:
        half = F64_SQRT3 * sigma
        return mean - half + F64_TWO * half * u
    weight = (F64_THREE * variance - mean_sq) / (F64_THREE * variance + F64_THREE * mean_sq)
    top = (F64_THREE * variance + F64_THREE * mean_sq) / (F64_TWO * mean)
    if u < weight:
        return F64_ZERO
    return top * (u - weight) / (F64_ONE - weight)
