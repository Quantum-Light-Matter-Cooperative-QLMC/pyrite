"""Refit the small-k/T shape of the packaged Seltzer--Berger (1985) eta table.

Below ``k/T = 0.1`` PyRITE writes ``eta`` as a stretched exponential with a
scale ``s`` and a power ``p`` per Table 4 row (see
``pyrite._bremslib_electron``). Two Seltzer--Berger constraints fix them:

* Table 4's ``etabar`` (SB85 Eq. 27), the chi_n-weighted mean of ``eta``;
* the Seltzer--Berger (1986) total at their only node inside ``(0, 0.1)``,
  ``kappa = 0.05``, which with a smooth chi_n implies ``eta(0.05)``.

chi_n is chi_total / (1 + eta/Z) at the Seltzer--Berger nodes where Table 4
gives ``eta`` (``kappa`` = 0, 0.1, 0.2), joined by the quadratic through them
below 0.2 and linearly above. For each row from 2 to 200 MeV and each
tabulated element the script solves for ``(s, p)`` meeting both constraints,
stores the Z-mean ``p`` and refits ``s`` at it. Below 2 MeV the implied
``eta(0.05)`` depends on how chi_n is joined by more than the shape changes
it, so ``p = 1`` and ``s`` meets ``etabar`` alone. It prints the per-element
fits, the stored values and the residuals of both constraints.

    pixi run python checks/sb85_eta_small_kappa_fit.py

Validation: bremslib-electron-electron
"""

import tomllib

import numpy as np
from scipy.optimize import brentq

from pyrite._bremslib_electron import ELECTRON_ELECTRON_DATA
from pyrite.validation.brem_sources import load_seltzer_berger

X = np.unique(np.r_[np.geomspace(1e-7, 0.1, 2000), np.linspace(0.1, 1.0, 901), 0.0])
#: Lowest Table 4 row whose implied ``eta(0.05)`` constrains the power.
JOINT_MIN_MEV = 2.0


def _g(x, inverse_scale, power):
    edge = np.exp(-((0.1 * inverse_scale) ** power))
    return (np.exp(-((x * inverse_scale) ** power)) - edge) / (1.0 - edge)


def _eta(nodes, row, inverse_scale, power):
    eta = np.interp(X, nodes, row)
    small = X < 0.1
    eta[small] = row[1] + (row[0] - row[1]) * _g(X[small], inverse_scale, power)
    return eta


def _constraints(sb, nodes, row, z, t1):
    """``(chi_n on X, implied eta(0.05))`` for one element and row."""
    total = sb.chi(z, t1)
    chi_n_nodes = total / (1.0 + np.interp(sb.kappa, nodes, row) / z)
    near = [np.flatnonzero(np.isclose(sb.kappa, k))[0] for k in (0.0, 0.1, 0.2)]
    quadratic = np.polyfit(sb.kappa[near], chi_n_nodes[near], 2)
    chi_n = np.interp(X, sb.kappa, chi_n_nodes)
    chi_n[X < 0.2] = np.polyval(quadratic, X[X < 0.2])
    at_node = np.flatnonzero(np.isclose(sb.kappa, 0.05))[0]
    return chi_n, z * (total[at_node] / np.polyval(quadratic, 0.05) - 1.0)


def _mean_eta(nodes, row, chi_n, inverse_scale, power):
    eta = _eta(nodes, row, inverse_scale, power)
    return np.trapezoid(chi_n * eta, X) / np.trapezoid(chi_n, X)


def _fit_scale(nodes, row, chi_n, target, power):
    def residual(log_inverse):
        return _mean_eta(nodes, row, chi_n, 10.0**log_inverse, power) - target

    return 10.0 ** brentq(residual, -2.0, 6.0, xtol=1e-9)


def _eta_05(row, inverse_scale, power):
    return row[1] + (row[0] - row[1]) * _g(0.05, inverse_scale, power)


def _fit_power(nodes, row, chi_n, target, implied):
    def residual(power):
        scale = _fit_scale(nodes, row, chi_n, target, power)
        return _eta_05(row, scale, power) - implied

    return brentq(residual, 0.6, 3.0, xtol=1e-5)


def main():
    with ELECTRON_ELECTRON_DATA.open("rb") as handle:
        raw = tomllib.load(handle)
    sb = load_seltzer_berger()
    nodes = np.asarray(raw["reduced_energy"])
    t1 = np.asarray(raw["incident_energy_MeV"])
    stored_scale = np.asarray(raw["inverse_small_kappa_scale"])
    stored_power = np.asarray(raw["small_kappa_power"])
    for i in np.flatnonzero((t1 >= 0.5) & (t1 <= 200.0)):
        cases = []
        for z in raw["atomic_number"]:
            row = np.asarray(raw[f"Z{z}"]["eta"][i])
            chi_n, implied = _constraints(sb, nodes, row, z, t1[i])
            cases.append((z, row, chi_n, raw[f"Z{z}"]["etabar"][i], implied))
        if t1[i] >= JOINT_MIN_MEV:
            powers = np.array([_fit_power(nodes, r, c, tg, im) for _, r, c, tg, im in cases])
            power = powers.mean()
        else:
            powers, power = np.ones(len(cases)), 1.0
        scales = np.array([_fit_scale(nodes, r, c, tg, power) for _, r, c, tg, _ in cases])
        scale = scales.mean()
        etabar_error = max(
            abs(_mean_eta(nodes, r, c, scale, power) - tg) for _, r, c, tg, _ in cases
        )
        node_error = max(
            abs((1.0 + _eta_05(r, scale, power) / z) / (1.0 + im / z) - 1.0)
            for z, r, _, _, im in cases
        )
        print(
            f"{t1[i]:7g} MeV  p {np.round(powers, 2)} -> {power:.2f} (stored {stored_power[i]})"
            f"  1/kappa_s {np.round(scales, 1)} -> {scale:.1f} (stored {stored_scale[i]})"
            f"  |d etabar| <= {etabar_error:.4f}  |d(1+eta/Z)| at 0.05 <= {node_error:.4f}"
        )


if __name__ == "__main__":
    main()
