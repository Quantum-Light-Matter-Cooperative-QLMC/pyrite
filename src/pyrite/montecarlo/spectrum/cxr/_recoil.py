"""Exact electron-recoil line kinematics shared by every CXR line route.

Array forms of the ``materials.recoil`` vacuum root, its mismatch slope and
the formation-piece recoil phase in the routes' ``1/Angstrom`` units
(issue #1099). ``_RECOIL_HBARC_EV_ANG = 0`` is the ``hbar g -> 0`` test hook.

Validation: line-recoil-in-medium-root
"""

from ...._backend import xp
from ....materials.crystal import HBARC_EV_ANG, M_E_EV

# hbar*c in the line-recoil terms only. Production never changes it; zero is
# the hbar -> 0 hook under which every route reproduces the classical root,
# width and phase bit for bit (``x + 0``, ``x - 0`` and ``x * 1`` are exact).
_RECOIL_HBARC_EV_ANG = HBARC_EV_ANG


def _recoil_inverse_energy(gamma):
    """``hbar c / E`` [Angstrom] with ``E = gamma m c^2``; zero under the hook.

    Validation: line-recoil-in-medium-root
    """
    return _RECOIL_HBARC_EV_ANG / (gamma * M_E_EV)


def _recoil_compton_ang():
    """Reduced Compton wavelength ``hbar c / m c^2`` [Angstrom]; zero under the hook.

    Validation: cbs-amplitude
    """
    return _RECOIL_HBARC_EV_ANG / M_E_EV


def _quantum_line_factors(h, gamma, vdg, k_dot_v, k_dot_g, g2, k2, v_dot_e, g_dot_e, k_dot_e):
    """Exact second-order PXR numerator and CBS factor of one polarization.

    Returns ``(pxr_num, f_cbs)`` with ``A_PXR = chi pxr_num / detuning`` and
    ``A_CBS = (U_g/m) f_cbs``. The electron enters with ``p_i = gamma m v``,
    absorbs the lattice transfer ``-hbar g`` and emits ``hbar k``. ``f_cbs`` is
    the sum of both time orderings of the second-order CB element, with the
    exact intermediate denominators ``E_1 - E_i`` and ``E_f - E_2``. Both
    terms carry convective transition currents ``(p + p')/(E + E')`` and are
    combined algebraically, so no ``1/hbar`` cancellation remains. The PXR
    numerator takes the same current between the initial and final states
    in place of ``v``. Energies are in units of ``m c^2`` and ``h = hbar c /
    m c^2`` [Angstrom]. At ``h = 0`` these reduce identically to the classical
    ``v.(k+g)(g.e + k.e) - k^2 v.e`` and
    ``-({g;e} + (v.e){k;g}/(v.g)) / (gamma v.g)``. Valid away from the
    real intermediate-state poles ``n_1, n_2 -> 0``, which is the quantum
    form of the classical ``v.g -> 0`` limit.
    Derivation: docs/validation/radiation-physics/cbs-amplitude.md.

    Validation: cbs-amplitude
    """
    gg = gamma * gamma
    n_1 = h * g2 - 2.0 * gamma * vdg
    n_2 = n_1 + 2.0 * h * k_dot_g
    n_f = h * (g2 + 2.0 * k_dot_g + k2) - 2.0 * gamma * (vdg + k_dot_v)
    e_1 = xp.sqrt(gg + h * n_1)
    e_2 = xp.sqrt(gg + h * (h * k2 - 2.0 * gamma * k_dot_v))
    e_f = xp.sqrt(gg + h * n_f)
    s_if = gamma + e_f
    ge_ke = g_dot_e + k_dot_e
    pxr_num = (-n_f * ge_ke - k2 * (2.0 * gamma * v_dot_e - h * ge_ke)) / s_if
    s_1 = e_1 + gamma
    s_3 = e_1 + e_f
    s_4 = gamma + e_2
    x = 2.0 * gamma * v_dot_e - h * k_dot_e
    sigma = s_if + e_1 + e_2
    f_cbs = 2.0 * g_dot_e * s_1 / (s_3 * n_1) + x * (
        n_1 * (n_f / s_if) * sigma - 2.0 * s_1 * s_4 * k_dot_g
    ) / (s_3 * s_4 * n_1 * n_2)
    return pxr_num, f_cbs


def _line_recoil_terms(gamma, g2, n_dot_g):
    """``(r_num, r_den, hbarc/E)`` of the exact recoil root, per row.

    With ``Q = hbar c g`` and total energy ``E = gamma m c^2``, the exact
    vacuum root (``materials.recoil``) is
    ``hbar omega = (beta u.Q - Q^2/2E) / (1 - beta u.n + Q.n/E)``.
    In the route's ``1/Angstrom`` units, ``omega = (v.g - r_num) / (D + r_den)``
    with ``r_num = hbar c g^2 / (2E)`` [1/Angstrom] and
    ``r_den = hbar c n.g / E``. ``D`` is the classical (vacuum or
    first-order Snell) denominator. ``hbarc/E`` [Angstrom] also forms
    the slope factor of :func:`_recoil_slope_factor`.

    Validation: line-recoil-in-medium-root
    """
    inv_E = _recoil_inverse_energy(gamma)
    return 0.5 * g2 * inv_E, n_dot_g * inv_E, inv_E


def _recoil_slope_factor(omega, inv_E):
    """``kappa = E / (E - hbar omega)``, the exact-slope factor at the root.

    The mismatch slope at the exact root is ``(1 - beta_f.n)/hbar c =
    kappa D_exact / hbar c`` (``line-recoil-formation``), so the sinc
    half-width, and with it the energy Jacobian, carries ``kappa``.

    Validation: line-recoil-in-medium-root
    """
    return 1.0 / (1.0 - omega * inv_E)


def _recoil_line_allowed(denom, omega, inv_E, gamma):
    """Positive-mass-shell guard: ``D_exact > 0`` and ``hbar omega <= T``.

    ``hbar omega <= T`` is ``omega hbarc/E <= 1 - 1/gamma``; it is vacuous
    at ``inv_E = 0``. Validation: line-recoil-in-medium-root
    """
    return (denom > 0) & (omega * inv_E <= 1.0 - 1.0 / gamma)


def _recoil_formation_line(vdg, denom_vac, t_L, gamma, g2, n_dot_g, age):
    """Exact vacuum sinc centre/width and the recoil phase of a formation piece.

    Returns ``(E_c, a, d_shift, g_shift)``. The piece's phase is
    ``E d/hbar c - g.r``; its exact mismatch, linear about the root,
    ``kappa (D E/hbar c - N)`` with ``N = v.g - r_num``, ``D = D_vac + r_den``,
    exceeds the classical ``D_vac E/hbar c - v.g`` by
    ``(kappa D - D_vac) E/hbar c - (kappa N - v.g)``. Its accumulated phase
    over the electron's own age ``t`` [Angstrom] is carried by
    ``d -> d + d_shift`` and ``g.r -> g.r + g_shift``. ``kappa - 1 =
    kappa x`` with ``x = hbar omega/E`` keeps both shifts cancellation free.
    The photon escape phase ``delta omega L_esc`` stays geometric (no
    ``kappa``), inside the piece (``half_dL``) and between pieces alike, which
    keeps the coherent sum invariant under segment splitting. A coherent
    self-term therefore peaks where ``kappa D (E - E_c) = delta E v.grad L``,
    an ``O(x delta)`` relative offset from the incoherent in-medium root,
    below the first-order Snell order of that root.

    Validation: line-recoil-in-medium-root
    """
    r_num, r_den, inv_E = _line_recoil_terms(gamma, g2, n_dot_g)
    numerator = vdg - r_num
    denom = denom_vac + r_den
    E_c = HBARC_EV_ANG * numerator / denom
    x = numerator / denom * inv_E
    kappa = 1.0 / (1.0 - x)
    a = kappa * denom * t_L / (2.0 * HBARC_EV_ANG)
    d_shift = kappa * (x * denom_vac + r_den) * age
    g_shift = kappa * (x * vdg - r_num) * age
    return E_c, a, d_shift, g_shift
