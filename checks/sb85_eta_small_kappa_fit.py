"""Refit the small-k/T scale of the packaged Seltzer--Berger (1985) eta table.

For every Table 4 row from 0.5 to 200 MeV and each tabulated element, solve
for the ``kappa_s`` whose small-``k/T`` shape (see
``pyrite._bremslib_electron``) makes the chi_n-weighted mean of eta equal the
row's ``etabar``, with chi_n = chi_total / (1 + eta/Z) from the pinned
Seltzer--Berger digital table. Prints the per-element fits, their spread and
the stored Z-averaged ``inverse_small_kappa_scale``.

    uv run python checks/sb85_eta_small_kappa_fit.py

Validation: bremslib-electron-electron
"""

import tomllib

import numpy as np
from scipy.optimize import brentq

from pyrite._bremslib_electron import ELECTRON_ELECTRON_DATA
from pyrite.validation.brem_sources import load_seltzer_berger

X = np.unique(np.r_[np.geomspace(1e-7, 0.1, 400), np.linspace(0.1, 1.0, 901), 0.0])


def _eta(nodes, row, inverse_scale):
    eta = np.interp(X, nodes, row)
    small = X < 0.1
    if inverse_scale > 0.0:
        edge = 1.0 / (1.0 + 0.1 * inverse_scale)
        g = (1.0 / (1.0 + X[small] * inverse_scale) - edge) / (1.0 - edge)
        eta[small] = row[1] + (row[0] - row[1]) * g
    return eta


def main():
    with ELECTRON_ELECTRON_DATA.open("rb") as handle:
        raw = tomllib.load(handle)
    sb = load_seltzer_berger()
    nodes = np.asarray(raw["reduced_energy"])
    t1 = np.asarray(raw["incident_energy_MeV"])
    stored = np.asarray(raw["inverse_small_kappa_scale"])
    for i in np.flatnonzero((t1 >= 0.5) & (t1 <= 200.0)):
        fits = []
        for z in raw["atomic_number"]:
            row = np.asarray(raw[f"Z{z}"]["eta"][i])
            target = raw[f"Z{z}"]["etabar"][i]
            total = np.interp(X, sb.kappa, sb.chi(z, t1[i]))

            def residual(log_inverse, z=z, row=row, target=target, total=total):
                eta = _eta(nodes, row, 10.0**log_inverse)
                chi_n = total / (1.0 + eta / z)
                return np.trapezoid(chi_n * eta, X) / np.trapezoid(chi_n, X) - target

            fits.append(10.0 ** brentq(residual, -4.0, 6.0, xtol=1e-6))
        fits = np.asarray(fits)
        print(
            f"{t1[i]:7g} MeV  1/kappa_s {np.round(fits, 1)}  "
            f"spread {np.ptp(fits) / fits.mean():.3f}  mean {fits.mean():.1f}  stored {stored[i]}"
        )


if __name__ == "__main__":
    main()
