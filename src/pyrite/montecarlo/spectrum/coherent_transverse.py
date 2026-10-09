"""Analytic transverse bunch form factor for a Gaussian spot on a tilted face.

An electron with lab spot offset ``w = (u, v)`` enters the face at
``p0 = o + s* b`` and, with the face-arrival delay transport adds to ``t0``,
``s*/beta`` after its bunch time. On a translation-invariant target its field
is the offset-free field times ``exp(i phi)``,

    phi = omega t0 - Q . p0 + omega s* / beta,      Q = omega n_hat + g,

and ``p0_z = 0``, ``s* = p0 . b`` make ``phi = omega t0 - K . p0_xy`` with

    K = (Q - (omega / beta) b)_xy = omega (n_hat - b / beta)_xy + g_xy.

``p0_xy = J w`` is linear in the lab offset, so a Gaussian spot of lab
covariance ``Sigma`` gives a Gaussian face point of covariance
``Sigma_f = J Sigma J^T`` and

    F_perp = |<exp(-i K . p0_xy)>|^2 = exp(-K^T Sigma_f K).

Equivalently ``kappa = -J^T K`` and ``F_perp = exp(-kappa^T Sigma kappa)``.
Limits: zero tilt (``b_xy = 0``, ``J = I``) recovers ``exp(-Q_perp^T Sigma
Q_perp)``; ``Sigma -> 0`` gives one; ``K = 0`` is fully coherent for any spot.

Validation: transverse-bunch-form-factor
"""

from dataclasses import dataclass

import numpy as np

from ..._backend import _to_cpu
from ...materials.crystal import HBARC_EV_ANG
from ..geometry import beam_entry_basis, beam_entry_jacobian
from ..transport.kinematics import beta_from_keV_scalar

#: Gaussian standard deviations of the face spot that must clear the footprint
#: edge after the offset-free emission and escape reach. Two-sided tail mass
#: beyond it is 2e-9 per axis.
FOOTPRINT_SUPPORT_SIGMAS = 6.0


@dataclass(frozen=True)
class TransverseSpot:
    """Face covariance ``Sigma_f`` [Ang^2] and ``b_xy / beta`` of one beam."""

    face_covariance: np.ndarray
    beam_xy_over_beta: np.ndarray


def transverse_spot(record):
    """Resolve the transport ``beam_entry`` record, refusing unsupported beams.

    Returns ``None`` when no spot covariance is recorded. Refuses Twiss beams
    whose position and slope are correlated (``alpha != 0``) and, on a tilted
    face, an energy spread: either couples the per-electron field to its
    offset, so the average no longer factorizes.

    Validation: transverse-bunch-form-factor
    """
    if not record or record.get("spot_covariance_ang2") is None:
        return None
    if not record.get("face_arrival_delay"):
        raise ValueError(
            "the analytic transverse form factor needs transport's face-arrival delay; "
            "re-run transport with this version"
        )
    if record.get("position_slope_correlated"):
        raise ValueError(
            "the analytic transverse form factor does not support position-slope "
            "correlated (alpha != 0) Courant-Snyder spots"
        )
    tilt, azim = float(record["tilt_polar_rad"]), float(record["tilt_azim_rad"])
    if tilt and record.get("energy_spread"):
        raise ValueError(
            "the analytic transverse form factor on a tilted face does not support an "
            "energy spread: the face-arrival delay couples beta to the offset"
        )
    covariance = np.asarray(record["spot_covariance_ang2"], dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("spot_covariance_ang2 must be a finite 2 x 2 matrix")
    J = beam_entry_jacobian(tilt, azim)
    _, _, b = beam_entry_basis(tilt, azim)
    beta = beta_from_keV_scalar(float(record["E0_keV"]))
    return TransverseSpot(J @ covariance @ J.T, b[:2] / beta)


def transverse_form_factor(spot, omega, n_hat, g_vec, *, xp=np, amplitude=False):
    """``exp(-K^T Sigma_f K)`` on ``omega`` [1/Ang]; its square root if ``amplitude``.

    ``K = omega (n_hat - b/beta)_xy + g_xy`` in the sample frame, matching the
    reducer's phase ``omega d - g . r``. Pass float64 ``omega``; the caller
    casts. The amplitude form is the (real)
    characteristic function ``<exp(i phi)>`` of the zero-mean Gaussian phase.

    Validation: transverse-bunch-form-factor
    """
    S = spot.face_covariance.tolist()
    a = np.asarray(_to_cpu(n_hat), dtype=float)[:2] - spot.beam_xy_over_beta
    g = np.asarray(_to_cpu(g_vec), dtype=float)[:2]
    # Form K itself rather than expanding K^T S K in omega: near K = 0 the
    # expanded terms cancel catastrophically.
    kx = omega * float(a[0]) + float(g[0])
    ky = omega * float(a[1]) + float(g[1])
    exponent = kx * kx * S[0][0] + 2.0 * kx * ky * S[0][1] + ky * ky * S[1][1]
    exponent = xp.maximum(exponent, 0.0)  # positive semidefinite up to rounding
    return xp.exp(-(0.5 if amplitude else 1.0) * exponent)


def require_footprint_support(
    spot, seg_r_geom, n_hat, thickness_ang, width_ang, height_ang, *, xp=np
):
    """Refuse unless every spot offset keeps fields translation invariant.

    Offset-free emission points ``r`` and their slab exits along ``n_hat``
    (``z = 0`` or ``z = thickness``) span a lateral reach ``X, Y``. Translating
    them by any face offset within ``FOOTPRINT_SUPPORT_SIGMAS`` standard
    deviations must stay inside the footprint, so entry, transport and escape
    attenuation never meet a side face.

    Validation: transverse-bunch-form-factor
    """
    n = np.asarray(_to_cpu(n_hat), dtype=float)
    if abs(n[2]) < 1e-12:
        raise ValueError("finite-footprint transverse average needs n_hat to leave a slab face")
    reach = np.zeros(2)
    if seg_r_geom.shape[0]:
        # Reduce on the active backend; only two scalars per axis come back.
        face = 0.0 if n[2] < 0.0 else float(thickness_ang)
        xy = seg_r_geom[:, :2]
        exit_xy = xy + ((face - seg_r_geom[:, 2]) / n[2])[:, None] * xp.asarray(n[:2])[None, :]
        reach = np.maximum(
            np.asarray(_to_cpu(xp.abs(xy).max(axis=0)), dtype=float),
            np.asarray(_to_cpu(xp.abs(exit_xy).max(axis=0)), dtype=float),
        )
    sigma = np.sqrt(np.diag(spot.face_covariance))
    need = reach + FOOTPRINT_SUPPORT_SIGMAS * sigma
    half = 0.5 * np.array([float(width_ang), float(height_ang)])
    if np.any(need > half):
        raise ValueError(
            "finite-footprint coherent transverse average needs the spot well inside the "
            f"footprint: emission/escape reach {reach.tolist()} Ang plus "
            f"{FOOTPRINT_SUPPORT_SIGMAS:g} face sigma {sigma.tolist()} Ang exceeds the "
            f"half-widths {half.tolist()} Ang"
        )


def transverse_envelope(spot, n_hat, g_vec):
    """Nonincreasing upper bound ``E -> sup_{E' >= E} F_perp(E')`` for windows.

    ``q(omega) = K^T Sigma_f K`` is a convex quadratic with minimizer
    ``omega* = -c1 / (2 c2)``: above it ``F_perp`` itself is nonincreasing;
    below it the supremum is ``exp(-q_min)``. The product with the
    nonincreasing ``F_z`` is then a nonincreasing bound on the reducer's
    ``F_z F_perp``, which window rules evaluate at interval starts. Rounding
    of ``q_min`` is relaxed downward (toward a larger bound).

    Validation: transverse-bunch-form-factor
    """
    S = spot.face_covariance
    a = np.asarray(_to_cpu(n_hat), dtype=float)[:2] - spot.beam_xy_over_beta
    g = np.asarray(_to_cpu(g_vec), dtype=float)[:2]
    c2, c1, c0 = float(a @ S @ a), float(2.0 * a @ S @ g), float(g @ S @ g)
    if c2 > 0.0:
        omega_star = -c1 / (2.0 * c2)
        drop = c1 * c1 / (4.0 * c2)
        q_min = max(0.0, c0 - drop - 1e-12 * (abs(c0) + drop))
    else:
        omega_star, q_min = -np.inf, 0.0

    def bound(energy_eV):
        omega = np.asarray(energy_eV, dtype=float) / HBARC_EV_ANG
        exact = transverse_form_factor(spot, omega, n_hat, g_vec)
        return np.where(omega >= omega_star, exact, np.exp(-q_min))

    return bound
