"""Angular deflection by atomic electrons as an elastic-rate correction.

The elastic tables describe scattering by the screened nucleus, whose rate
scales as ``Z^2``. Collisions with the ``Z`` atomic electrons also deflect the
primary; transport that resolves only part of them (none in continuous mode,
``W > W_c`` in shell mode) folds the rest into the elastic rate by replacing
``Z^2`` with ``Z(Z + xi)``, keeping the elastic angular shape:

    sigma_i -> sigma_i * (1 + xi / Z_i),
    xi(T, T_c) = xi_0 - g_M(tau, tau_c) / g_R(eta),   clipped to [0, xi_0],

with ``xi_0 = 1``. ``g_M`` is the ``sin^2 theta`` moment of Moller collisions
with ``W > T_c`` per atomic electron and ``g_R`` the same moment of the
screened-Rutherford nucleus per ``Z^2``, both in the normalization of
Kawrakow (1997) and EGSnrc PIRS-701 Eqs. 4.7.21-4.7.22; ``eta`` is the
Moliere screening parameter of EGSnrc Eqs. 4.7.6-4.7.8. ``T_c = inf`` gives
``xi = xi_0``. See
docs/physics/beam-transport/atomic-electron-deflection.md.

Validation: inelastic-angular-deflection
"""

import math

import numpy as np

from ...materials._transport_data import TRANSPORT_ELEMENTS

ATOMIC_ELECTRON_DEFLECTION_MODELS = ("kawrakow",)
_MC2_EV = 510998.95
_ALPHA2 = 5.325135453e-5  # fine-structure constant squared
# EGSnrc Eq. 4.7.6: chi_cc^2 / b_c = 0.1569 MeV^2 / 7821.6 * exp((Z_X - Z_E)/Z_S).
_CHI2_OVER_B_MEV2 = 0.1569 / 7821.6
_XI0 = 1.0


def validate_atomic_electron_deflection(model) -> bool:
    """Return whether the correction is on; ``None`` is the bit-for-bit opt-out."""
    if model is None:
        return False
    if model not in ATOMIC_ELECTRON_DEFLECTION_MODELS:
        raise ValueError(
            "atomic_electron_deflection must be None or one of "
            + ", ".join(repr(m) for m in ATOMIC_ELECTRON_DEFLECTION_MODELS)
        )
    return True


def moliere_screening_eta(composition, tau):
    """Moliere screening parameter ``eta = chi_a^2 / 4`` (EGSnrc Eqs. 4.7.6-4.7.8).

    ``composition`` is ``[(element, number_density), ...]``; only the atom
    fractions enter. ``tau = T / mc^2``.

    Validation: inelastic-angular-deflection
    """
    z = np.asarray([float(TRANSPORT_ELEMENTS[el]["Z"]) for el, _ in composition])
    p = np.asarray([float(n) for _, n in composition])
    p = p / p.sum()
    w = p * z * (z + _XI0)
    z_s = w.sum()
    z_e = (w * (-2.0 / 3.0) * np.log(z)).sum()
    z_x = (w * np.log(1.0 + 3.34 * _ALPHA2 * z * z)).sum()
    m2 = (_MC2_EV * 1e-6) ** 2
    tau = np.asarray(tau, dtype=np.float64)
    return _CHI2_OVER_B_MEV2 * math.exp((z_x - z_e) / z_s) / (4.0 * m2 * tau * (tau + 2.0))


def screened_rutherford_g(eta):
    """``g_R(eta) = (1 + 2 eta) ln(1 + 1/eta) - 2``: half the ``sin^2`` moment.

    Validation: inelastic-angular-deflection
    """
    eta = np.asarray(eta, dtype=np.float64)
    return (1.0 + 2.0 * eta) * np.log1p(1.0 / eta) - 2.0


def moller_g(tau, tau_c):
    """Moller ``sin^2`` moment for ``W in [T_c, T/2]`` (PIRS-701 Eq. 4.7.21).

    Zero where no hard Moller collision exists (``tau <= 2 tau_c``).

    Validation: inelastic-angular-deflection
    """
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    m = tau > 2.0 * tau_c
    t = tau[m]
    d0 = (t + 2.0) / (t + 1.0)
    d1 = (t + 1.0) ** 2
    out[m] = (
        np.log(0.5 * t / tau_c)
        + (1.0 + d0 * d0) * np.log(2.0 * (t - tau_c + 2.0) / (t + 4.0))
        - (0.25 * (t + 2.0) ** 2 + (t + 2.0) * (t + 0.5) / d1)
        * np.log((t + 4.0) * (t - tau_c) / (t * (t - tau_c + 2.0)))
        + 0.5 * (t - 2.0 * tau_c) * (t + 2.0) * (1.0 / (t - tau_c) - 1.0 / d1)
    )
    return out


def atomic_electron_xi(energy_eV, composition, cutoff_eV=None):
    """``xi(T, T_c)`` on ``energy_eV`` for one layer.

    ``cutoff_eV=None`` means no atomic-electron collision is simulated
    explicitly (continuous stopping), so ``xi = xi_0`` everywhere.

    Validation: inelastic-angular-deflection
    """
    energy_eV = np.asarray(energy_eV, dtype=np.float64)
    if cutoff_eV is None:
        return np.full(energy_eV.shape, _XI0)
    tau = energy_eV / _MC2_EV
    ratio = moller_g(tau, float(cutoff_eV) / _MC2_EV) / screened_rutherford_g(
        moliere_screening_eta(composition, tau)
    )
    return _XI0 * (1.0 - np.minimum(ratio / _XI0, 1.0))


def elastic_rate_scale(Z, xi):
    """Per-element factor ``1 + xi / Z`` on the elastic rate.

    Validation: inelastic-angular-deflection
    """
    return 1.0 + np.asarray(xi, dtype=np.float64) / np.asarray(Z, dtype=np.float64)
