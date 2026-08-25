"""Elastic scattering: Browning free paths, NIST-Mott-calibrated
screened-Rutherford angles, and the analytic SR fallback."""

import logging
import os
from functools import cache

import numpy as np
from numba import njit

from ... import DATA_DIR

logger = logging.getLogger(__name__)

MOTT_DIR = str(DATA_DIR / "mott_transport_cross_sections")
A0_SQ_CM2 = 2.8002852e-17  # Bohr radius squared [cm^2] (NIST SRD 64 unit)


# ---- elastic scattering models ------------------------------------------------
@njit(cache=True)
def _sigma_browning_cm2(Z, E_keV):
    """
    Browning et al., J. Appl. Phys. 76, 2016 (1994): empirical fit to the
    tabulated Mott TOTAL elastic cross sections [cm^2], valid 0.1-30 keV,
    Z <= 92.
    """
    out = np.empty_like(E_keV)

    z17 = Z**1.7
    numerator = 3.0e-18 * z17
    z_exp = 0.005 * z17
    z_squared = 0.0007 * Z * Z

    for i in range(E_keV.size):
        E = E_keV[i]
        sqrt_E = np.sqrt(E)

        out[i] = numerator / (E + z_exp * sqrt_E + z_squared / sqrt_E)
    return out


@njit(cache=True)
def _sigma_browning_cm2_scalar(mott_numer, mott_denom1, mott_denom2, E_i):
    """
    Browning et al., J. Appl. Phys. 76, 2016 (1994): empirical fit to the
    tabulated Mott TOTAL elastic cross sections [cm^2], valid 0.1-30 keV,
    Z <= 92.
    """
    sqrt_E_i = np.sqrt(E_i)

    return mott_numer / (E_i + mott_denom1 * sqrt_E_i + mott_denom2 / sqrt_E_i)


def _alpha_sr_joy(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return 3.4e-3 * Z**0.67 / E_keV


@njit(cache=True)
def _alpha_sr_joy_scalar(sr_joy_numer, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return sr_joy_numer / E_keV


@njit(cache=True)
def _alpha_sr_joy_numba(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    out = np.empty_like(E_keV)
    for j in range(E_keV.size):
        out[j] = 3.4e-3 * Z**0.67 / E_keV[j]
    return out


def _alpha_from_first_moment(target):
    """
    Invert <1-cos(theta)> = 2 a [(1+a) ln(1+1/a) - 1] for the screened-
    Rutherford screening parameter a (monotonic; vectorized bisection in
    log10 a). target must lie in (0, 1); values outside are clipped.
    """
    t = np.clip(np.asarray(target, dtype=float), 1e-12, 0.999)
    lo = np.full(t.shape, -12.0)
    hi = np.full(t.shape, 4.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = 10.0**mid
        val = 2.0 * a * ((1.0 + a) * np.log1p(1.0 / a) - 1.0)
        small = val < t
        lo = np.where(small, mid, lo)
        hi = np.where(small, hi, mid)
    return 10.0 ** (0.5 * (lo + hi))


@cache
def _load_mott_transport(element):
    """
    NIST SRD 64 relativistic Mott TRANSPORT cross sections
    sigma_tr = integral (1-cos theta) dsigma, from
    mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv
    (50 eV - 300 keV, 401 points). Returns (E_eV, sigma_tr_cm2).
    """
    path = os.path.join(MOTT_DIR, f"DisplayCalcTCSTableFor{element}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No NIST Mott transport table for '{element}' ({path}). "
            f"Download from https://srdata.nist.gov/srd64/ or use "
            f"elastic_model='sr'."
        )
    E, sig = [], []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                E.append(float(parts[1]))
                sig.append(float(parts[2]))
    return np.array(E), np.array(sig) * A0_SQ_CM2


@cache
def _mott_alpha_table(element, Z):
    """
    Screening parameter alpha(E) calibrated so the screened-Rutherford
    angular distribution reproduces the NIST Mott transport cross section:
        sigma_tr / sigma_el = <1-cos theta>(alpha)
    with sigma_el from the Browning fit to the Mott totals. Returns
    (log10_E_eV_grid, log10_alpha_grid) for interpolation.
    """
    E_eV, sig_tr = _load_mott_transport(element)
    sig_el = _sigma_browning_cm2(Z, E_eV / 1e3)
    alpha = _alpha_from_first_moment(sig_tr / sig_el)
    return np.log10(E_eV), np.log10(alpha)


_NO_MOTT = set()  # elements with no NIST Mott table -> screened-Rutherford


@njit(cache=True)
def _scatter_rates_mott_scalar(E_i, mott_numer, mott_denom1, mott_denom2):
    sig_i = _sigma_browning_cm2_scalar(mott_numer, mott_denom1, mott_denom2, E_i)
    return sig_i


@njit(cache=True)
def _scatter_rates_sr_scalar(E_i, sr_rate_numer, sr_joy_numer):
    """Per-element elastic scattering rates [1/cm] at energies Ea (one layer)."""
    a = _alpha_sr_joy_scalar(sr_joy_numer, E_i)
    E_i_plus_511 = E_i + 511.0
    E_i_plus_1024 = E_i + 1024.0
    E_i_511_over_1024 = E_i_plus_511 / E_i_plus_1024

    sig_i = sr_rate_numer / (E_i * E_i) / (a * (1.0 + a)) * (E_i_511_over_1024 * E_i_511_over_1024)
    return sig_i


@njit(cache=True)
def _sample_cos_theta_sr_numba(Z, E_keV, R):
    out = np.empty_like(E_keV)

    Zfac = 3.4e-3 * Z**0.67

    for i in range(E_keV.size):
        alpha = Zfac / E_keV[i]

        out[i] = 1.0 - 2.0 * alpha * R[i] / (1.0 + alpha - R[i])

    return out


@njit(cache=True)
def _sample_cos_theta_from_alpha(alpha, R):
    return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)


@njit(cache=True)
def _interp_mott_log_alpha_scalar(logE_eV, logE_flat, logA_flat, start, length):
    """Linear interpolation with the same endpoint clamping as ``np.interp``."""
    if length <= 0:
        raise ValueError("Mott interpolation table must contain at least one point")

    first = start
    last = start + length - 1
    if logE_eV <= logE_flat[first]:
        return logA_flat[first]
    if logE_eV >= logE_flat[last]:
        return logA_flat[last]

    lo = first
    hi = last
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if logE_flat[mid] <= logE_eV:
            lo = mid
        else:
            hi = mid

    x0 = logE_flat[lo]
    x1 = logE_flat[hi]
    y0 = logA_flat[lo]
    y1 = logA_flat[hi]
    return y0 + (logE_eV - x0) * (y1 - y0) / (x1 - x0)


@njit(cache=True)
def _sample_cos_theta_joy_scalar(Z, E_keV, R):
    alpha = 3.4e-3 * Z**0.67 / E_keV
    return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)


@njit(cache=True)
def _sample_cos_theta_mott_scalar(E_keV, rng, logE, logA):
    """Polar scattering angle from the screened-Rutherford inversion, with the
    screening parameter from the chosen model."""
    R = rng.random(E_keV.shape)
    alpha = 10.0 ** np.interp(np.log10(E_keV * 1e3), logE, logA)
    return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)


def _sample_cos_theta(Z, E_keV, rng, elastic_model, element):
    """Polar scattering angle from the screened-Rutherford inversion, with the
    screening parameter from the chosen model. If elastic_model="mott" but no
    NIST Mott transport table exists for `element` (e.g. W), fall back to the
    analytic screened-Rutherford screening for that element. The miss is cached
    in _NO_MOTT so we don't re-stat the filesystem every transport step
    (lru_cache doesn't cache the FileNotFoundError); logged once per element
    per process at DEBUG (silent by default -- set PYRITE_MC_DEBUG=1 to see it;
    a ProcessPoolExecutor worker pool re-logs once per worker, since each
    worker gets its own _NO_MOTT cache)."""
    R = rng.random(E_keV.shape)
    if elastic_model == "mott" and element not in _NO_MOTT:
        try:
            logE, logA = _mott_alpha_table(element, Z)
            alpha = 10.0 ** np.interp(np.log10(E_keV * 1e3), logE, logA)

            return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)

        except FileNotFoundError:
            logger.debug(
                "no Mott transport table for %s; transport will use the analytic fallback",
                element,
            )
            _NO_MOTT.add(element)

    return _sample_cos_theta_sr_numba(Z, E_keV, R)
