"""Electron-electron bremsstrahlung for the BremsLib electron-nucleus tables.

BremsLib is an electron-atom partial-wave library for the field of the
screened nucleus only. Seltzer and Berger (Nucl. Instrum. Methods B 12, 95
(1985), Eqs. 18-19) write the electron-atom scaled SDCS as

    Z**2 chi_n + Z chi_e = Z (Z + eta) chi_n,     eta = chi_e / chi_n,

so the total is the electron-nucleus value times ``1 + eta(Z, T, k/T) / Z``.
``eta`` is their Table 4 (Haug's electron-electron cross section with the
Fedyushin Coulomb factor and Hartree-Fock incoherent-scattering screening),
packaged in ``data/bremsstrahlung_electron_electron.toml`` for Z = 6, 13, 29,
47, 74 and 92.

Interpolation: linear in ``ln Z`` and in ``ln T`` between tabulated rows,
linear in ``k/T`` for ``k/T >= 0.1``. Below 0.1, where Table 4 has no node but
``eta`` falls steeply from its soft-photon value,

    eta(x) = eta(0.1) + (eta(0) - eta(0.1)) g(x),
    g(x) = (1/(1 + x/kappa_s) - 1/(1 + 0.1/kappa_s)) / (1 - 1/(1 + 0.1/kappa_s)),

with ``1/kappa_s(T)`` fitted so that the ``chi_n``-weighted mean of ``eta``
reproduces Table 4's ``etabar`` (Eq. 27); see the data file.

Assumptions: the electron-electron photon has the electron-nucleus angular
distribution at the same ``(T, k/T)`` (both are forward-peaked within
``~1/gamma`` at high energy; SB85 gives no angular data); ``eta`` is held at
the Z = 6 (Z = 92) values below (above) the tabulated range. Limiting cases:
``eta -> 0`` at ``k/T -> 1`` (``k_max < T`` for an electron target); the
radiative stopping integral reproduces ``(1 + etabar/Z)`` times the
electron-nucleus one.

Validation: bremslib-electron-electron
"""

import tomllib
from functools import cache

import numpy as np

from .paths import data_dir

#: Packaged Seltzer--Berger (1985) Table 4.
ELECTRON_ELECTRON_DATA = data_dir() / "bremsstrahlung_electron_electron.toml"

#: Upper end of the steep small-``k/T`` region; Table 4's first interior node.
_SMALL_X = 0.1


@cache
def _table():
    with ELECTRON_ELECTRON_DATA.open("rb") as handle:
        raw = tomllib.load(handle)
    atomic_number = np.asarray(raw["atomic_number"], dtype=float)
    eta = np.stack([np.asarray(raw[f"Z{z}"]["eta"], dtype=float) for z in raw["atomic_number"]])
    return (
        atomic_number,
        np.asarray(raw["incident_energy_MeV"], dtype=float),
        np.asarray(raw["reduced_energy"], dtype=float),
        np.asarray(raw["inverse_small_kappa_scale"], dtype=float),
        eta,
    )


def _along(log_grid, log_value):
    """Lower index and weight for linear interpolation, clamped to the grid."""
    log_value = np.clip(log_value, log_grid[0], log_grid[-1])
    lower = np.clip(np.searchsorted(log_grid, log_value, side="right") - 1, 0, log_grid.size - 2)
    weight = (log_value - log_grid[lower]) / (log_grid[lower + 1] - log_grid[lower])
    return lower, weight


def electron_electron_eta(atomic_number: int, t1_MeV, reduced_energy) -> np.ndarray:
    """``eta(Z, T, k/T)`` on the ``(T, k/T)`` outer grid; shape ``(n_T, n_x)``.

    ``reduced_energy`` is ``(n_x,)`` or ``(n_T, n_x)``.

    Validation: bremslib-electron-electron
    """
    z_grid, t_grid, x_grid, inverse_scale, eta = _table()
    t1 = np.asarray(t1_MeV, dtype=float)
    x = np.broadcast_to(
        np.asarray(reduced_energy, dtype=float), (t1.size, np.shape(reduced_energy)[-1])
    )
    z_lower, z_weight = _along(np.log(z_grid), np.log(float(atomic_number)))
    rows = (1.0 - z_weight) * eta[z_lower] + z_weight * eta[z_lower + 1]  # (n_T_table, n_x_table)
    t_lower, t_weight = _along(np.log(t_grid), np.log(t1))
    w = t_weight[:, None]
    at_t = (1.0 - w) * rows[t_lower] + w * rows[t_lower + 1]  # (n_T, n_x_table)
    inverse = (1.0 - t_weight) * inverse_scale[t_lower] + t_weight * inverse_scale[t_lower + 1]
    out = np.stack([np.interp(x[i], x_grid, at_t[i]) for i in range(t1.size)])
    small = x < _SMALL_X
    scale = np.broadcast_to(inverse[:, None], x.shape)
    curved = small & (scale > 0.0)
    if np.any(curved):
        s, xs = scale[curved], x[curved]
        edge = 1.0 / (1.0 + _SMALL_X * s)
        g = (1.0 / (1.0 + xs * s) - edge) / (1.0 - edge)
        row = np.nonzero(curved)[0]
        out[curved] = at_t[row, 1] + (at_t[row, 0] - at_t[row, 1]) * g
    return out


def electron_electron_factor(atomic_number: int, t1_MeV, reduced_energy) -> np.ndarray:
    """Electron-atom over electron-nucleus SDCS, ``1 + eta/Z``; shape ``(n_T, n_x)``.

    Validation: bremslib-electron-electron
    """
    return 1.0 + electron_electron_eta(atomic_number, t1_MeV, reduced_energy) / float(atomic_number)
