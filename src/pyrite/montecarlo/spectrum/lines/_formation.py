"""Complex formation factor of a coherent line piece under absorption.

A segment's line field is the formation integral of the emitter's phase along
its constant-velocity flight. With ``t'`` the time from the piece midpoint,
``T = t_L`` its duration, and the propagation phase the coherent route already
applies between segments,

    Phi(t') = omega d(t') - g.r(t') - delta(E) omega L_esc(t'),

the field amplitude also decays along the flight by ``exp(-tau(t')/2)``,
``tau = mu(E_res) L_esc``. On one linear escape piece (``segment_escape``)
``L_esc`` -- and so ``tau`` and the refractive phase -- is affine in ``t'``,
so the integrand is a single exponential and

    int_{-T/2}^{T/2} exp(i Phi(t') - tau(t')/2) dt'
        = exp(i Phi_c - tau_c/2) T sinh(w)/w,
    w = i v - q,   q = (tau_end - tau_start)/4,
    v = (T/2) [(1 - v.n_hat) omega - v.g] - delta(E) omega (L_end - L_start)/2
      = a_vac (E - E_vac) - delta(E) omega dL/2,

with ``a_vac = (1 - v.n_hat) T / (2 hbar c)`` and ``E_vac = hbar c v.g /
(1 - v.n_hat)`` the vacuum sinc width and centre, ``start``/``end`` ordered
along travel and ``_c`` the piece-midpoint values. ``mu -> 0`` with ``dL = 0``
recovers ``T sinc(v)``. The midpoint factors stay where the route has always
put them (the phase ``Phi_c``); the attenuation ``exp(-tau_c/2)`` is folded in
here so opaque pieces cannot overflow:

    F = exp(-tau_c/2) sinh(w)/w
      = [exp(-tau_end/2) e^{iv} - exp(-tau_start/2) e^{-iv}] / (2 w).

The intra-piece slope is the time derivative of the SAME phase that separates
pieces, so splitting a straight segment into collinear pieces sums back to it
exactly. The bulk in-medium denominator ``1 - Re n (v.n_hat)`` would instead
charge a refractive slope ``delta omega (v.n_hat)`` on top of the escape-path
one, counting the medium twice inside a piece; it keeps its role for the
amplitudes (couplings at ``E_res``) and the incoherent route.

Parseval: ``int |F|^2 dv = pi <exp(-tau)>``, against ``int sinc^2 = pi`` for
the undamped line, so a piece's integrated coherent self-term carries exactly
the segment-mean transmission of ``segment-escape-average`` -- the incoherent
route's integrated yield. Absorption only reshapes the line (a damped,
``q``-broadened sinc); refraction only shifts it.

Validation: coherent-formation-absorption
"""

import numpy as np

from ...._backend import REAL
from ...transport import beta_from_keV
from ..segment_escape import segment_escape_paths
from ._kernels import _SEG_ARRAYS

# |w|^2 below which F takes its second-order series. The closed form loses
# at most absolute rounding (epsilon times the 1/|v| envelope), so the switch
# only guards the 0/0 at w = 0. The series keeps cosh(q) exactly in a+b but
# cuts the bracket, so its truncation is ~q^4/8, about 1e-13 at the switch.
FORMATION_SERIES_W2 = 1.0e-6
# |q| below which a+b and b-a are formed from exp(-tau_c/2) and cosh/sinh(q):
# b - a cancels for small q. Above it both exponentials are <= 1 and far apart.
FORMATION_SMALL_Q = 0.5


def formation_coefficients(tau_start, tau_end, xp=np):
    """Per-piece constants of :func:`formation_factor`.

    Returns ``(apb, bma, q)``: ``apb = a + b``, ``bma = b - a`` with ``a =
    exp(-tau_start/2)``, ``b = exp(-tau_end/2)``, and ``q = (tau_end -
    tau_start)/4``. Every factor is bounded by 2 for ``tau >= 0``, so a piece
    running from the surface into an opaque depth cannot overflow even where
    its midpoint transmission underflows. Validation: coherent-formation-absorption
    """
    q = 0.25 * (tau_end - tau_start)
    small = xp.abs(q) < FORMATION_SMALL_Q
    q_small = xp.where(small, q, 0.0)
    c0 = xp.exp(-0.25 * (tau_start + tau_end))
    a = xp.exp(-0.5 * tau_start)
    b = xp.exp(-0.5 * tau_end)
    apb = xp.where(small, 2.0 * c0 * xp.cosh(q_small), a + b)
    bma = xp.where(small, -2.0 * c0 * xp.sinh(q_small), b - a)
    return apb, bma, q


def formation_factor(v, apb, bma, q, xp=np):
    """Complex ``exp(-tau_c/2) sinh(w)/w`` with ``w = i v - q``.

    ``v`` is the refraction-shifted vacuum sinc argument; ``apb``, ``bma``,
    ``q`` come from :func:`formation_coefficients` and broadcast against it.
    With ``q = 0`` and ``bma = 0`` this is ``exp(-tau/2) sin(v)/v`` to
    rounding. ``|F| <= 1/|v|``: the damped profile never exceeds the undamped
    sinc envelope, so a sinc cutoff on ``|v|`` stays a conservative tail cut.
    Validation: coherent-formation-absorption
    """
    w2 = q * q + v * v
    series = w2 < FORMATION_SERIES_W2
    safe = xp.where(series, 1.0, 2.0 * w2)
    sv = xp.sin(v)
    cv = xp.cos(v)
    half = 0.5 * apb
    re = xp.where(
        series,
        half * (1.0 - q * q / 3.0 - v * v / 6.0),
        (v * apb * sv - q * bma * cv) / safe,
    )
    im = xp.where(series, -half * q * v / 3.0, -(v * bma * cv + q * apb * sv) / safe)
    return re + 1j * im


def formation_argument(E_grid, delta_omega, E_vac, a_vac, half_dL, xp=np):
    """``v[j, k] = a_vac_j (E_k - E_vac_j) - (dL_j/2) delta(E_k) omega(E_k)``.

    Row arrays are ``(n,)``; the result is ``(n, n_E)``.
    Validation: coherent-formation-absorption
    """
    return (
        a_vac[:, None] * (E_grid[None, :] - E_vac[:, None])
        - half_dL[:, None] * delta_omega[None, :]
    )


def formation_profile(E_grid, delta_omega, E_vac, a_vac, half_dL, apb, bma, q, *, sinc_cutoff, xp):
    """``F[j, k]`` for rows ``j`` over the energies ``E_grid``, windowed.

    ``sinc_cutoff`` (or None) zeroes ``|v| > sinc_cutoff``, the same unscaled
    argument bound the undamped sinc used. Validation: coherent-formation-absorption
    """
    v = formation_argument(E_grid, delta_omega, E_vac, a_vac, half_dL, xp=xp)
    F = formation_factor(v, apb[:, None], bma[:, None], q[:, None], xp=xp)
    if sinc_cutoff is not None:
        F = xp.where(xp.abs(v) <= sinc_cutoff, F, 0.0)
    return F


def formation_window_half_width(a_vac, half_dL, delta_omega_max, sinc_cutoff):
    """Energy half-width around ``E_vac`` outside which ``|v| > sinc_cutoff``.

    ``|dL/2 delta omega| <= |dL/2| max|delta omega|`` bounds the refractive
    shift, so the window is conservative. Validation: coherent-formation-absorption
    """
    return (sinc_cutoff + abs(half_dL) * delta_omega_max) / a_vac


def expand_escape_pieces(segments, n_hat, *, groove=None, xp=np):
    """Split every segment at its linear escape pieces into collinear rows.

    Returns ``(pieces, owner, L_start, L_end)``. ``pieces`` is a copy of
    ``segments`` whose per-row arrays (``_SEG_ARRAYS``) are gathered by
    ``owner``, with each piece's own length, midpoint, and start age -- the
    parent's ``t_ang`` advanced by the piece offset over the parent's speed at
    the spectrum's evaluation energy, so the flight clock is unchanged.
    ``L_start``/``L_end`` are the escape distances at the piece ends, ordered
    along travel. Single-slab only (the coherent and flight-grouped routes
    refuse layers). The pieces of a straight flight sum exactly to its field.
    Validation: coherent-formation-absorption
    """
    n_rows = int(xp.asarray(segments["L_ang"]).shape[0])
    owner, fraction, path_start, path_end = segment_escape_paths(
        segments, xp.arange(n_rows), n_hat, layers=None, groove=groove, xp=xp
    )
    # ``owner`` ascends and each owner's pieces are ordered along travel, so a
    # piece's start offset is the running fraction of its owner's pieces.
    counts = xp.bincount(owner, minlength=n_rows)
    first = (xp.cumsum(counts) - counts)[owner]
    cumulative = xp.cumsum(fraction) - fraction
    offset = cumulative - cumulative[first]

    pieces = dict(segments)
    for key in _SEG_ARRAYS:
        if key in pieces and pieces[key] is not None:
            pieces[key] = xp.asarray(pieces[key])[owner]
    L_parent = xp.asarray(segments["L_ang"], dtype=np.float64)[owner]
    v_hat = xp.asarray(pieces["v_hat"], dtype=np.float64)
    r_mid = xp.asarray(pieces["r_mid"], dtype=np.float64)
    shift = (offset + 0.5 * fraction - 0.5) * L_parent
    pieces["r_mid"] = (r_mid + shift[:, None] * v_hat).astype(REAL)
    pieces["L_ang"] = (fraction * L_parent).astype(REAL)
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    beta = beta_from_keV(xp.asarray(pieces[E_field], dtype=np.float64))
    # Absent ``t_ang`` means every segment starts at age zero (the setup's
    # default); its pieces still need their offsets along the flight.
    t_start = (
        xp.zeros(owner.size, dtype=np.float64)
        if pieces.get("t_ang") is None
        else xp.asarray(pieces["t_ang"], dtype=np.float64)
    )
    pieces["t_ang"] = (t_start + offset * L_parent / beta).astype(REAL)
    return pieces, owner, path_start[:, 0].astype(REAL), path_end[:, 0].astype(REAL)


__all__ = [
    "FORMATION_SERIES_W2",
    "FORMATION_SMALL_Q",
    "expand_escape_pieces",
    "formation_argument",
    "formation_coefficients",
    "formation_factor",
    "formation_profile",
    "formation_window_half_width",
]
