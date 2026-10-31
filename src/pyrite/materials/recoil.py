"""Exact vacuum line-recoil reference, not yet selected by production scoring.

These deterministic plane-wave kinematics conserve the final electron's mass
shell. They do not supply quantum PXR/CBS amplitudes, medium/interface physics,
or a rule for coherence between different electron final states.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystal import HBARC_EV_ANG, M_E_EV

__all__ = [
    "recoil_resonance_eV",
    "recoil_shift_fraction",
    "recoil_mismatch_inv_ang",
    "recoil_mismatch_slope",
]


def _real(value: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real and finite")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be real and finite") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be real and finite")
    return result


def _vector(value: ArrayLike, name: str, *, unit: bool = False) -> NDArray[np.float64]:
    result = _real(value, name)
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have a final axis of length three")
    if unit:
        norm = np.linalg.norm(result, axis=-1)
        if not np.all(np.isclose(norm, 1.0, rtol=0.0, atol=1e-12)):
            raise ValueError(f"{name} must contain unit vectors")
        result = result / norm[..., None]
    return result


def _state(kinetic_eV, electron_direction, g_ang_inv, photon_direction):
    kinetic = _real(kinetic_eV, "kinetic_eV")
    if np.any(kinetic <= 0):
        raise ValueError("kinetic_eV must be strictly positive")
    direction = _vector(electron_direction, "electron_direction", unit=True)
    reciprocal = _vector(g_ang_inv, "g_ang_inv")
    normal = _vector(photon_direction, "photon_direction", unit=True)
    return _kinematics(np, kinetic, direction, reciprocal, normal)


def _kinematics(xp, kinetic, direction, reciprocal, normal):
    """Broadcast validated float64 inputs with array module ``xp``."""
    energy_rows, direction, reciprocal, normal = xp.broadcast_arrays(
        kinetic[..., None], direction, reciprocal, normal
    )
    kinetic = energy_rows[..., 0]
    total = kinetic + M_E_EV
    momentum = xp.sqrt(kinetic) * xp.sqrt(kinetic + 2 * M_E_EV)
    return kinetic, total, momentum / total, direction, HBARC_EV_ANG * reciprocal, normal


def _coefficients(total, beta, direction, transfer, normal):
    """Conservation polynomial divided by twice the incident total energy.

    Source: Shi et al. (2024), Eq. (6), with their reciprocal vector negated.
    Vacuum unit directions, static infinitely massive crystal; the low-transfer
    limit gives the classical Doppler numerator and denominator. Forward
    subtraction uses 1-beta=(m/E)^2/(1+beta) and
    1-cos(theta)=|u-n|^2/2 to retain the mass term when beta rounds to one.

    Validation: line-recoil-resonance
    """
    numerator = beta * (direction * transfer).sum(axis=-1)
    numerator -= (transfer * transfer).sum(axis=-1) / (2 * total)
    denominator = (M_E_EV / total) ** 2 / (1 + beta)
    denominator += 0.5 * beta * ((direction - normal) ** 2).sum(axis=-1)
    denominator += (transfer * normal).sum(axis=-1) / total
    return numerator, denominator


def recoil_resonance_eV(
    kinetic_eV: ArrayLike,
    electron_direction: ArrayLike,
    g_ang_inv: ArrayLike,
    photon_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Return the exact physical vacuum photon root, or NaN if forbidden.

    Source: Shi et al., Nature Communications 15, 7803 (2024), Eq. (6),
    DOI 10.1038/s41467-024-52050-w; use its exact conservation statement,
    not the second-order small-recoil expansion in Eq. (7).

    The convention is p_f*c = p_i*c - hbar*c*g - photon_eV*n. The crystal
    takes +hbar*g and no energy (infinite mass). With E=T+m*c^2 and
    Q=hbar*c*g, the root is
    (beta*u.Q-Q.Q/(2E))/(1-beta*u.n+Q.n/E).
    Only positive roots no larger than T belong to the positive final-electron
    mass shell. No clipping of a large classical root supplies this condition.

    Parameters
    ----------
    kinetic_eV
        Strictly positive incident electron kinetic energy, in eV.
    electron_direction, photon_direction
        Unit vectors with a final axis of length three. Roundoff-sized norm
        errors (absolute tolerance 1e-12) are normalized.
    g_ang_inv
        Signed reciprocal vector in inverse angstroms, final axis length three.
        Its sign is opposite to the cited paper's reciprocal label.

    Returns
    -------
    numpy.ndarray
        Photon energies in eV, with NumPy-broadcast leading axes (a zero-D
        array for scalar inputs). Forbidden/degenerate branches return NaN.
        Nonfinite inputs, nonpositive T and nonunit directions raise ValueError.

    Notes
    -----
    The classical limit requires both lattice-recoil and Doppler-denominator
    corrections to be small; photon/E alone is insufficient. Near a vanishing
    numerator the allowed/forbidden boundary is ill-conditioned. This vacuum
    reference does not certify high-recoil radiation intensity or coherence.

    Validation: line-recoil-resonance
    """
    state = _state(kinetic_eV, electron_direction, g_ang_inv, photon_direction)
    return np.asarray(_exact_root(np, *state))


def _exact_root(xp, kinetic, total, beta, direction, transfer, normal):
    numerator, denominator = _coefficients(total, beta, direction, transfer, normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = numerator / denominator
    # Allow only roundoff at the final electron-at-rest endpoint. A root above
    # this tolerance is an extraneous negative-energy branch of the squared
    # mass-shell equation, never a photon to clip down to the beam energy.
    endpoint_tolerance = 8 * np.finfo(np.float64).eps * kinetic
    valid = (numerator > 0) & (denominator > 0) & xp.isfinite(root) & (root > 0)
    valid &= root <= kinetic + endpoint_tolerance
    return xp.where(valid, xp.minimum(root, kinetic), xp.nan)


def recoil_shift_fraction(
    kinetic_eV: ArrayLike,
    electron_direction: ArrayLike,
    g_ang_inv: ArrayLike,
    photon_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Return the exact vacuum root over its classical limit, minus one.

    The classical limit drops the transfer terms of the exact root,
    (beta*u.Q)/(1-beta*u.n), evaluated with the same stable forward
    denominator. Arguments follow recoil_resonance_eV. Returns NaN where the
    classical root is not positive (no line), and +inf where it is positive
    but the exact root is forbidden. To first order in Q/E the fraction is
    -Q.Q/(2E*beta*u.Q) - (Q.n/E)/(1-beta*u.n). Forward lines with Q along u
    give -photon/(beta*E)*(1+(1-beta)/(2*beta)), tending to -photon/E; the
    first (longitudinal lattice-recoil) term has magnitude Q/(2*beta*E) for Q
    along u, about 1 % at 30 keV for HOPG (002). Transverse transfer can exceed
    photon/E.

    Validation: line-recoil-validity-screen
    """
    state = _state(kinetic_eV, electron_direction, g_ang_inv, photon_direction)
    return np.asarray(_shift_fraction(np, *state))


def _recoil_shift_fraction_unchecked(xp, kinetic_eV, electron_direction, g_ang_inv, normal):
    """:func:`recoil_shift_fraction` with array module ``xp`` and no input checks.

    The caller guarantees float64, finite, strictly positive ``kinetic_eV`` and
    finite, nonzero direction rows; the directions are normalized here exactly
    as the checked reference normalizes them, so NumPy inputs give bitwise
    the reference result. Used by the device line-recoil screen.

    Validation: line-recoil-validity-screen
    """
    direction = electron_direction / xp.linalg.norm(electron_direction, axis=-1)[..., None]
    normal = normal / xp.linalg.norm(normal, axis=-1)[..., None]
    return _shift_fraction(xp, *_kinematics(xp, kinetic_eV, direction, g_ang_inv, normal))


def _shift_fraction(xp, kinetic, total, beta, direction, transfer, normal):
    exact = _exact_root(xp, kinetic, total, beta, direction, transfer, normal)
    numerator = beta * (direction * transfer).sum(axis=-1)
    denominator = (M_E_EV / total) ** 2 / (1 + beta)
    denominator += 0.5 * beta * ((direction - normal) ** 2).sum(axis=-1)
    classical = numerator / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = xp.where(xp.isfinite(exact), exact / classical - 1, xp.inf)
    return xp.where(classical > 0, fraction, xp.nan)


def _photon(photon_eV, kinetic):
    photon = _real(photon_eV, "photon_eV")
    photon, kinetic = np.broadcast_arrays(photon, kinetic)
    if np.any((photon < 0) | (photon > kinetic)):
        raise ValueError("photon_eV must lie between zero and kinetic_eV")
    return photon


def recoil_mismatch_inv_ang(
    photon_eV: ArrayLike,
    kinetic_eV: ArrayLike,
    electron_direction: ArrayLike,
    g_ang_inv: ArrayLike,
    photon_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Return (E_f+photon_eV-E_i)/(hbar*c), in inverse angstroms.

    Source: the interaction-picture finite-time integral in Shi et al. (2024),
    scattering elements preceding Eq. (6), and its exact conservation Eq. (6).
    Assumes fixed incident momentum and one vacuum plane-wave final-electron
    mode. Arguments and broadcasting follow recoil_resonance_eV; photon_eV
    may range from zero through kinetic_eV.

    Over a prescribed duration c*dt=T_ang, the centred time integral is
    T_ang*sinc(mismatch*T_ang/(2*pi)). A midpoint age adds
    exp(i*mismatch*t_mid_ang); constant-momentum pieces then sum exactly.
    This is not a rule for joining different momenta or electron final states.
    The low-transfer limit is (photon*(1-beta*u.n)-hbar*c*beta*u.g)/(hbar*c).
    Rationalization avoids subtracting nearly equal total electron energies.

    Validation: line-recoil-formation
    """
    kinetic, total, beta, direction, transfer, normal = _state(
        kinetic_eV, electron_direction, g_ang_inv, photon_direction
    )
    photon = _photon(photon_eV, kinetic)
    numerator, denominator = _coefficients(total, beta, direction, transfer, normal)
    final_momentum = total[..., None] * beta[..., None] * direction
    final_momentum = final_momentum - transfer - photon[..., None] * normal
    final_energy = np.hypot(M_E_EV, np.linalg.norm(final_momentum, axis=-1))
    return np.asarray(
        (2 * total / (final_energy + total - photon))
        * (photon * denominator - numerator)
        / HBARC_EV_ANG
    )


def recoil_mismatch_slope(
    photon_eV: ArrayLike,
    kinetic_eV: ArrayLike,
    electron_direction: ArrayLike,
    g_ang_inv: ArrayLike,
    photon_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Return the vacuum mismatch derivative (1-beta_f.n)/(hbar*c).

    Source: derivative of the exact final-electron mass shell in Shi et al.
    (2024), Eq. (6). Same arguments/domain as recoil_mismatch_inv_ang. Units
    are inverse angstroms/eV; the energy Jacobian is its positive reciprocal.
    The derivative is strictly positive for a massive final electron. Its
    low-transfer limit uses beta_i. A forward mass-shell rearrangement retains
    the slope when direct subtraction of beta_f.n from one would vanish.

    Validation: line-recoil-formation
    """
    kinetic, total, beta, direction, transfer, normal = _state(
        kinetic_eV, electron_direction, g_ang_inv, photon_direction
    )
    photon = _photon(photon_eV, kinetic)
    final_momentum = total[..., None] * beta[..., None] * direction
    final_momentum = final_momentum - transfer - photon[..., None] * normal
    parallel = np.sum(final_momentum * normal, axis=-1)
    transverse = final_momentum - parallel[..., None] * normal
    energy = np.hypot(M_E_EV, np.linalg.norm(final_momentum, axis=-1))
    stable_forward = (M_E_EV / energy) ** 2 + np.sum((transverse / energy[..., None]) ** 2, axis=-1)
    stable_forward /= 1 + np.maximum(parallel, 0) / energy
    return np.asarray(np.where(parallel >= 0, stable_forward, 1 - parallel / energy) / HBARC_EV_ANG)
