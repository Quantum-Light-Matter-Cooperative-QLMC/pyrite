"""Host-side PENELOPE-2024 shell GOS energy-loss moments for electrons.

Evaluates the integrated cross sections of PENELOPE-2024 §3.2.3 from the
oscillators of :func:`~.shell_oscillators.build_shell_oscillators`. The raw
moments are neither calibrated nor sampled, and no transport mode uses them.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e, physical_constants
from scipy.optimize import brentq

from .inelastic import _qmin_ev
from .shell_oscillators import MaterialShellOscillators, ShellOscillator

_MC2_EV = m_e * c * c / e
_RE_CM = 100.0 * physical_constants["classical electron radius"][0]
# 2 pi e^4 / (m_e c^2) in eV cm^2; divide by beta^2 for 2 pi e^4 / (m_e v^2).
_PREF_EV_CM2 = 2.0 * np.pi * _RE_CM**2 * _MC2_EV


@dataclass(frozen=True, slots=True)
class ShellGOSMoments:
    """Raw per-formula-unit moments ``sigma^(n)``, ``n = 0, 1, 2``.

    Each channel array has shape ``(n_oscillators, 3)`` and units
    ``(cm^2, eV cm^2, eV^2 cm^2)``, in the order of ``oscillators``.
    """

    energy_eV: float
    oscillators: tuple[ShellOscillator, ...]
    density_effect: float
    distant_longitudinal: np.ndarray
    distant_transverse: np.ndarray
    close: np.ndarray

    @property
    def per_shell(self) -> np.ndarray:
        """Distant plus close moments of each oscillator (Eq. 3.103)."""
        return self.distant_longitudinal + self.distant_transverse + self.close

    @property
    def total(self) -> np.ndarray:
        """Material moments ``(sigma_in, sigma_in^(1), sigma_in^(2))``."""
        return self.per_shell.sum(axis=0)


def _kinematics(energy_eV: float) -> tuple[float, float]:
    gamma = 1.0 + energy_eV / _MC2_EV
    return gamma, 1.0 - 1.0 / (gamma * gamma)


def formula_units_per_angstrom3(material: MaterialShellOscillators) -> float:
    """Formula-unit number density implied by the material plasma energy.

    Source: PENELOPE-2024 Eq. 3.51, ``Omega_p^2 = 4 pi N Z hbar^2 e^2/m_e``
    (Gaussian), i.e. ``N Z = eps_0 m_e omega_p^2 / e^2`` in SI. Using the same
    ``Omega_p`` as the oscillators keeps ``N`` consistent with Eq. 3.62–3.63.
    Limit: ``Omega_p -> 0`` gives ``N -> 0``.

    Validation: penelope-shell-gos-moments
    """
    omega = material.plasma_energy_eV * e / hbar
    electrons_m3 = epsilon_0 * m_e * omega * omega / (e * e)
    return float(electrons_m3 * 1e-30 / material.electrons_per_formula)


def density_effect_correction(material: MaterialShellOscillators, energy_eV: float) -> float:
    """Fermi density-effect correction ``delta_F`` of the oscillator OOS.

    Source: PENELOPE-2024 Eqs. 3.70–3.72. ``L^2`` is the positive root of
    ``(Omega_p^2/Z) sum_k f_k/(W_k^2 + L^2) = 1 - beta^2``; then
    ``delta_F = (1/Z) sum_k f_k ln(1 + L^2/W_k^2) - (L^2/Omega_p^2)(1 - beta^2)``.
    The unmodified resonances ``W_k`` (Eq. 3.59) define the OOS.
    Limits: ``delta_F = 0`` when ``1 - beta^2 >= F(0)``; for ``beta -> 1``,
    ``delta_F -> ln(Omega_p^2/((1 - beta^2) I^2)) - 1`` (Eq. 3.73).

    Validation: penelope-shell-gos-moments
    """
    _, beta2 = _kinematics(energy_eV)
    target = 1.0 - beta2
    f = np.array([o.strength for o in material.oscillators])
    w2 = np.array([o.resonance_energy_eV for o in material.oscillators]) ** 2
    scale = material.plasma_energy_eV**2 / material.electrons_per_formula

    def excess(l2: float) -> float:
        return float(scale * np.sum(f / (w2 + l2)) - target)

    if excess(0.0) <= 0.0:
        return 0.0
    upper = material.plasma_energy_eV**2 / target
    l2 = brentq(excess, 0.0, upper, xtol=1e-300, rtol=1e-15, maxiter=500)
    delta = np.sum(f * np.log1p(l2 / w2)) / material.electrons_per_formula
    return float(delta - l2 * target / material.plasma_energy_eV**2)


def _triangle_moments(lower: float, peak_end: float, upper: float) -> np.ndarray:
    """``int W^(n-1) p_dis dW`` for n = 0, 1, 2 over ``[U, min(W_dis, W_max)]``.

    ``p_dis = 2 (W_dis - W)/(W_dis - U)^2`` on ``[U, W_dis]`` (Eq. 3.76).
    """
    b = min(peak_end, upper)
    norm = 2.0 / (peak_end - lower) ** 2
    m0 = norm * (peak_end * np.log(b / lower) - (b - lower))
    m1 = norm * (peak_end * (b - lower) - 0.5 * (b * b - lower * lower))
    m2 = norm * (0.5 * peak_end * (b * b - lower * lower) - (b**3 - lower**3) / 3.0)
    return np.array([m0, m1, m2])


def _moller_integrals(energy_eV: float, prime_eV: float, w: float) -> np.ndarray:
    """Antiderivatives ``J_n^(-)`` (Eqs. 3.108–3.110) with ``E -> E'`` in ``F``.

    ``F^(-)`` uses ``E' = E + U_k`` in its ``W/(E'-W)`` and ``W^2/E'^2`` terms
    and ``a = (E/(E + m_e c^2))^2`` (Eqs. 3.85, 3.87). Constants of
    integration differ from the manual's ``J_2``; only differences are used.
    """
    a = (energy_eV / (energy_eV + _MC2_EV)) ** 2
    ep = prime_eV
    rest = ep - w
    j0 = -1.0 / w + 1.0 / rest + (1.0 - a) / ep * np.log(rest / w) + a * w / ep**2
    j1 = np.log(w) + ep / rest + (2.0 - a) * np.log(rest) + a * w * w / (2.0 * ep**2)
    j2 = (3.0 - a) * w + ep * ep / rest + (3.0 - a) * ep * np.log(rest) + a * w**3 / (3.0 * ep**2)
    return np.array([j0, j1, j2])


def shell_gos_moments(material: MaterialShellOscillators, energy_eV: float) -> ShellGOSMoments:
    """Raw zeroth, first and second energy-loss moments for an electron.

    Source: PENELOPE-2024 §§3.2.2–3.2.3 (NEA/MBDAV/R(2024)1). Per oscillator,
    with ``pref = 2 pi e^4/(m_e v^2)``:

    - distant longitudinal, Eq. 3.104:
      ``pref f_k ln[Q'_k (Q_- + 2mc^2)/(Q_- (Q'_k + 2mc^2))] int W^(n-1) p_dis``;
    - distant transverse, Eq. 3.105:
      ``pref f_k [ln(1/(1-beta^2)) - beta^2 - delta_F] int W^(n-1) p_dis``;
    - close Møller, Eqs. 3.86–3.87, 3.96, 3.106–3.110:
      ``pref f_k int_{Q_k}^{W_max} W^(n-2) F^(-)(E, W) dW``, with
      ``Q_k = U_k`` for bound shells (Eq. 3.96) rather than Eq. 3.106's ``Q'_k``.

    Both distant terms require ``Q_- < Q'_k``; ``Q_-`` is the zero-angle
    recoil of the (modified) resonance, Eq. 3.83. Bound shells (``U_k > 0``)
    use the triangle ``p_dis`` on ``[U_k, W_dis]`` (Eq. 3.76), with
    ``W_dis = 3W'_k - 2U_k`` and the near-threshold ``W'_k``, ``Q'_k`` of
    Eqs. 3.77–3.80; ``Q_k = U_k`` (Eq. 3.56). The conduction band uses
    ``delta(W - W_cb)`` and ``Q_cb = W_cb``. ``W_max = (E + U_k)/2``
    (Eq. 3.88) bounds every integral, and ``F^(-)`` uses ``E' = E + U_k``.
    ``delta_F`` follows Eqs. 3.70–3.72.

    Assumptions: first Born, δ-oscillator GOS with zero-width Bethe ridge;
    every bound shell (not only K/L/M) is broadened; Eqs. 3.94/3.104's
    ``W^(n-1) p_dis`` form is used rather than Eq. 3.81's ``p_dis/W_k``
    (printed for the double-differential DCS); the transverse bracket is
    clipped at zero where a conductor-like OOS makes ``delta_F`` exceed
    ``ln(1/(1-beta^2)) - beta^2``; a shell with ``E <= U_k`` contributes
    nothing, and its close moments vanish continuously as ``E -> U_k``.
    Limits: for ``E >> U_k``, ``sigma^(1)`` tends to the Bethe formula
    (Eqs. 3.115–3.121); an untruncated ``p_dis`` has ``<W> = W_k``
    (Eq. 3.77), so ``sigma_dis^(2) = W_k sigma_dis^(1)``.

    Units: E in eV; per formula unit in cm^2, eV cm^2, eV^2 cm^2.
    Validation: penelope-shell-gos-moments
    """
    if not np.isfinite(energy_eV) or energy_eV <= 0.0:
        raise ValueError("electron kinetic energy must be finite and positive")
    gamma, beta2 = _kinematics(energy_eV)
    pref = _PREF_EV_CM2 / beta2
    delta = density_effect_correction(material, energy_eV)
    # A conductor-like OOS gives delta_F > 0 at low E; keep the DCS non-negative.
    transverse = max(np.log(gamma * gamma) - beta2 - delta, 0.0)
    n = len(material.oscillators)
    dis_l, dis_t, close = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for i, osc in enumerate(material.oscillators):
        u, w, f = osc.ionization_energy_eV, osc.resonance_energy_eV, osc.strength
        w_max = 0.5 * (energy_eV + u)
        if u > 0.0:
            if energy_eV <= u:
                continue
            w_dis_full = 3.0 * w - 2.0 * u
            if energy_eV > w_dis_full:
                w_mod, q_mod = w, u
            else:
                w_mod, q_mod = (energy_eV + 2.0 * u) / 3.0, u * energy_eV / w_dis_full
            loss = _triangle_moments(u, 3.0 * w_mod - 2.0 * u, w_max)
        else:
            w_mod, q_mod = w, w
            loss = np.array([1.0 / w, 1.0, w]) if w < w_max else np.zeros(3)
        if w_mod < energy_eV and loss[1] > 0.0:
            q_minus = float(_qmin_ev(energy_eV, w_mod))
            if q_minus < q_mod:
                longitudinal = np.log(
                    (q_mod / q_minus) * ((q_minus + 2.0 * _MC2_EV) / (q_mod + 2.0 * _MC2_EV))
                )
                dis_l[i] = pref * f * longitudinal * loss
                dis_t[i] = pref * f * transverse * loss
        # Eq. 3.96: close losses start at Q_k = U_k (Eq. 3.56), not Q'_k, so a
        # knock-on energy W - U_k is never negative; the band keeps Q_cb = W_cb.
        q_close = u if u > 0.0 else q_mod
        if q_close < w_max:
            prime = energy_eV + u
            j = _moller_integrals(energy_eV, prime, w_max) - _moller_integrals(
                energy_eV, prime, q_close
            )
            close[i] = pref * f * j
    return ShellGOSMoments(float(energy_eV), material.oscillators, delta, dis_l, dis_t, close)


def bethe_stopping_cs(material: MaterialShellOscillators, energy_eV: float) -> float:
    """High-energy electron Bethe stopping cross section, eV cm^2 per formula unit.

    Source: PENELOPE-2024 Eqs. 3.120–3.121,
    ``pref Z [ln(E^2 (gamma+1)/(2 I^2)) + f^(-)(gamma) - delta_F]`` with
    ``f^(-) = 1 - beta^2 - (2 gamma - 1) ln 2/gamma^2 + ((gamma-1)/gamma)^2/8``
    and the same oscillator ``delta_F``. Assumption: ``E >> U_k``; this is
    the limit :func:`shell_gos_moments` must reach, not a low-energy model.

    Validation: penelope-shell-gos-moments
    """
    gamma, beta2 = _kinematics(energy_eV)
    f_minus = (
        1.0
        - beta2
        - (2.0 * gamma - 1.0) / gamma**2 * np.log(2.0)
        + ((gamma - 1.0) / gamma) ** 2 / 8.0
    )
    log_term = np.log(energy_eV**2 * (gamma + 1.0) / (2.0 * material.mean_excitation_eV**2))
    delta = density_effect_correction(material, energy_eV)
    return float(
        _PREF_EV_CM2 / beta2 * material.electrons_per_formula * (log_term + f_minus - delta)
    )


def path_moments(moments: ShellGOSMoments, formula_units_per_A3: float) -> np.ndarray:
    """Inverse IMFP (1/Å), stopping power (eV/Å), straggling (eV²/Å).

    Source: PENELOPE-2024 Eqs. 3.100–3.102, ``N sigma^(n)``.
    Limit: zero number density gives zero for every moment.

    Validation: penelope-shell-gos-moments
    """
    return moments.total * formula_units_per_A3 * 1e24 * 1e-8
