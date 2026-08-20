"""Precomputed energy/depth lookup table for the LUT transport cores."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .stopping import _dEds_spliced_compound


@dataclass(frozen=True)
class TransportLUTConfig:
    """Energy-grid policy for the ungrooved transport hot loops.

    The LUT is deliberately uniform in kinetic energy so an event needs only one
    multiply, one integer conversion, and linear interpolation -- no search or
    logarithm just to locate a table entry. ``step_keV`` is a target spacing;
    ``max_points`` bounds memory for unusually wide energy ranges.
    """

    enabled: bool = True
    step_keV: float = 0.025
    min_points: int = 256
    max_points: int = 16384


DEFAULT_TRANSPORT_LUT_CONFIG = TransportLUTConfig()


@dataclass(frozen=True)
class TransportEnergyLUT:
    """Per-run transport tables shared by lockstep CPU, per-electron CPU and CUDA."""

    E_min_keV: float
    inv_dE_keV: float
    n_energy: int
    n_el: np.ndarray
    total_rate: np.ndarray
    dEds: np.ndarray
    inv_beta: np.ndarray
    cdf: np.ndarray
    alpha: np.ndarray


@njit(cache=True, inline="always")
def _lut_index_frac_scalar(E_keV, E_min_keV, inv_dE_keV, n_energy):
    """Return lower LUT index and interpolation fraction, clamped to the grid."""
    x = (E_keV - E_min_keV) * inv_dE_keV
    if x <= 0.0:
        return 0, 0.0
    last = n_energy - 1
    if x >= last:
        return last - 1, 1.0
    i = int(x)
    return i, x - i


@njit(cache=True, inline="always")
def _lut_lerp_1d(table, i, f):
    return table[i] + f * (table[i + 1] - table[i])


@njit(cache=True, inline="always")
def _lut_lerp_2d(table, row, i, f):
    return table[row, i] + f * (table[row, i + 1] - table[row, i])


@njit(cache=True, inline="always")
def _lut_lerp_3d(table, row, col, i, f):
    return table[row, col, i] + f * (table[row, col, i + 1] - table[row, col, i])


def build_transport_energy_lut(
    E_min_keV,
    E_max_keV,
    elastic_model_code,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    mott_tables,
    config=DEFAULT_TRANSPORT_LUT_CONFIG,
):
    """Precompute all energy-dependent scalar transport physics for one run.

    Tables are material/layer specific.  Runtime transport then needs only uniform
    grid indexing plus linear interpolation for the total elastic rate, stopping
    power, 1/beta, collision-element CDF and scattering screening parameter.
    """
    E_min_keV = float(E_min_keV)
    E_max_keV = float(E_max_keV)
    if not np.isfinite(E_min_keV) or not np.isfinite(E_max_keV):
        raise ValueError("transport LUT energy bounds must be finite")
    if E_min_keV <= 0.0:
        raise ValueError("transport LUT requires positive kinetic energies")
    if E_max_keV <= E_min_keV:
        E_max_keV = np.nextafter(E_min_keV, np.inf)

    step = float(config.step_keV)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("transport LUT step_keV must be finite and positive")
    min_points = max(2, int(config.min_points))
    max_points = max(min_points, int(config.max_points))
    requested = int(np.ceil((E_max_keV - E_min_keV) / step)) + 1
    n_energy = min(max_points, max(min_points, requested))

    E_grid = np.linspace(E_min_keV, E_max_keV, n_energy, dtype=np.float64)
    inv_dE_keV = np.float64((n_energy - 1) / (E_max_keV - E_min_keV))
    n_layers = len(L_Zs)
    max_el = max(arr.size for arr in L_Zs)
    n_el = np.asarray([arr.size for arr in L_Zs], dtype=np.int32)

    total_rate = np.empty((n_layers, n_energy), dtype=np.float64)
    dEds = np.empty((n_layers, n_energy), dtype=np.float64)
    cdf = np.ones((n_layers, max_el, n_energy), dtype=np.float64)
    alpha = np.zeros((n_layers, max_el, n_energy), dtype=np.float64)

    g = 1.0 + E_grid / 510.99895
    beta = np.sqrt(1.0 - 1.0 / (g * g))
    inv_beta = 1.0 / beta
    sqrt_E = np.sqrt(E_grid)
    E_511_over_1024 = (E_grid + 511.0) / (E_grid + 1024.0)
    rel2 = E_511_over_1024 * E_511_over_1024

    for L in range(n_layers):
        n = int(n_el[L])
        rates = np.empty((n, n_energy), dtype=np.float64)

        # The LUT bakes dE/ds, so it must carry the same spliced model the direct
        # cores evaluate or the LUT paths would silently keep the old physics.
        dEds[L] = _dEds_spliced_compound(
            np.asarray(L_Js[L], dtype=np.float64),
            np.asarray(L_ks[L], dtype=np.float64),
            np.asarray(L_coeffs[L], dtype=np.float64),
            np.asarray(L_E_cross[L], dtype=np.float64),
            0.0,
            E_grid.copy(),
        )

        for i_el in range(n):
            sr_joy = float(L_sr_joy_numer[L][i_el])
            if elastic_model_code == 1:
                rates[i_el] = float(L_mott_numer[L][i_el]) / (
                    E_grid
                    + float(L_mott_denom1[L][i_el]) * sqrt_E
                    + float(L_mott_denom2[L][i_el]) / sqrt_E
                )
            else:
                a = sr_joy / E_grid
                rates[i_el] = (
                    float(L_sr_rate_numer[L][i_el]) / (E_grid * E_grid) / (a * (1.0 + a)) * rel2
                )

            table = mott_tables[L][i_el]
            if elastic_model_code == 1 and table is not None:
                logE, logA = table
                alpha[L, i_el] = 10.0 ** np.interp(
                    np.log10(E_grid * 1e3),
                    np.asarray(logE, dtype=np.float64),
                    np.asarray(logA, dtype=np.float64),
                )
            else:
                alpha[L, i_el] = sr_joy / E_grid

        layer_total = rates.sum(axis=0)
        if np.any(~np.isfinite(layer_total)) or np.any(layer_total <= 0.0):
            raise ValueError("transport LUT produced a non-positive elastic rate")
        total_rate[L] = layer_total
        layer_cdf = np.cumsum(rates, axis=0) / layer_total[None, :]
        layer_cdf[-1, :] = 1.0
        cdf[L, :n, :] = layer_cdf

    return TransportEnergyLUT(
        E_min_keV=np.float64(E_min_keV),
        inv_dE_keV=inv_dE_keV,
        n_energy=n_energy,
        n_el=n_el,
        total_rate=total_rate,
        dEds=dEds,
        inv_beta=inv_beta,
        cdf=cdf,
        alpha=alpha,
    )
