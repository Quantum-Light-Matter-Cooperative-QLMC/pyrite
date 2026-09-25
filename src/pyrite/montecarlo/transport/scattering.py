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


def _flatten_mott_tables(mott_tables):
    """Flatten per-layer, per-element Mott tables into the cores' mott_group arrays."""
    n_layers = len(mott_tables)
    max_elements = max(len(layer_tables) for layer_tables in mott_tables)
    # Numba only sees numeric Mott data. Tables are flattened because different
    # elements may have different grid lengths; start/length locate each table.
    mott_has_table = np.zeros((n_layers, max_elements), dtype=np.bool_)
    mott_start = np.zeros((n_layers, max_elements), dtype=np.int64)
    mott_len = np.zeros((n_layers, max_elements), dtype=np.int64)
    mott_logE_chunks = []
    mott_logA_chunks = []
    offset = 0
    for L, layer_tables in enumerate(mott_tables):
        for i_el, table in enumerate(layer_tables):
            if table is None:
                continue
            logE, logA = table
            logE = np.asarray(logE, dtype=float)
            logA = np.asarray(logA, dtype=float)
            if logE.size != logA.size or logE.size == 0:
                raise ValueError("invalid Mott interpolation table")
            mott_has_table[L, i_el] = True
            mott_start[L, i_el] = offset
            mott_len[L, i_el] = logE.size
            mott_logE_chunks.append(logE)
            mott_logA_chunks.append(logA)
            offset += logE.size

    if mott_logE_chunks:
        mott_logE_flat = np.concatenate(mott_logE_chunks)
        mott_logA_flat = np.concatenate(mott_logA_chunks)
    else:
        mott_logE_flat = np.empty(0, dtype=float)
        mott_logA_flat = np.empty(0, dtype=float)
    return mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat


# ---- ELSEPA tabulated elastic scattering -------------------------------------
def elsepa_angular_pdf(mu, dcs):
    """Normalize tabulated DCS rows into piecewise-linear angular densities.

    For an azimuthally symmetric DCS, ``dOmega = 4*pi*dmu`` with
    ``mu = (1 - cos(theta))/2``, so ``p(mu) = DCS(mu) / integral_0^1 DCS du``.
    The integral is the trapezoid rule on ELSEPA's native grid, matching the
    ``angular_cdf`` stored by :mod:`pyrite.xsgen.elsepa`; returns ``(pdf, cdf)``
    with ``cdf`` the exact integral of the piecewise-linear ``pdf``.

    Validation: elsepa-elastic-sampling
    """
    mu = np.asarray(mu, dtype=np.float64)
    dcs = np.atleast_2d(np.asarray(dcs, dtype=np.float64))
    increments = 0.5 * (dcs[:, :-1] + dcs[:, 1:]) * np.diff(mu)
    norm = increments.sum(axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("ELSEPA DCS rows need a positive finite angular integral")
    pdf = dcs / norm[:, None]
    cdf = np.zeros_like(pdf)
    cdf[:, 1:] = np.cumsum(increments, axis=1) / norm[:, None]
    cdf[:, -1] = 1.0
    return pdf, cdf


def pack_elsepa_tables(elastic_tables, L_ncm3):
    """Flatten per-layer, per-element ELSEPA tables into the cores' group.

    ``elastic_tables[L][i]`` is a mapping with ``energy_eV``,
    ``total_elastic_cm2``, ``mu`` and ``dcs_cm2_sr``, one per element of
    layer ``L`` in composition order. ``None`` packs an empty group, which the
    cores never read unless ``elastic_model="elsepa"``.

    Returns ``(has, start, length, logE, log_rate, cdf, pdf, mu)``: rates are
    stored as ``ln(n_cm3 * sigma_el)`` [1/cm] so energy interpolation is
    log-log in the macroscopic cross section.
    """
    if elastic_tables is None:
        empty2 = np.zeros((1, 1), dtype=np.float64)
        return (
            np.zeros((1, 1), dtype=np.bool_),
            np.zeros((1, 1), dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            empty2,
            empty2,
            np.zeros(1, dtype=np.float64),
        )
    n_layers = len(elastic_tables)
    if n_layers != len(L_ncm3):
        raise ValueError("elastic_tables needs one entry per transport layer")
    max_el = max(len(row) for row in L_ncm3)
    has = np.zeros((n_layers, max_el), dtype=np.bool_)
    start = np.zeros((n_layers, max_el), dtype=np.int64)
    length = np.zeros((n_layers, max_el), dtype=np.int64)
    logE, log_rate, cdf, pdf = [], [], [], []
    mu_ref = None
    offset = 0
    for L, (layer, ncm3) in enumerate(zip(elastic_tables, L_ncm3, strict=True)):
        if len(layer) != len(ncm3):
            raise ValueError(f"elastic_tables layer {L} needs one table per element")
        for i_el, (table, n_cm3) in enumerate(zip(layer, ncm3, strict=True)):
            energy = np.asarray(table["energy_eV"], dtype=np.float64)
            sigma = np.asarray(table["total_elastic_cm2"], dtype=np.float64)
            mu = np.asarray(table["mu"], dtype=np.float64)
            if mu_ref is None:
                mu_ref = mu
            elif not np.array_equal(mu, mu_ref):
                raise ValueError("every ELSEPA table must share one angular grid")
            if energy.size < 2 or np.any(np.diff(energy) <= 0.0) or np.any(sigma <= 0.0):
                raise ValueError("ELSEPA tables need >= 2 increasing energies, positive sigma")
            row_pdf, row_cdf = elsepa_angular_pdf(mu, table["dcs_cm2_sr"])
            has[L, i_el] = True
            start[L, i_el] = offset
            length[L, i_el] = energy.size
            logE.append(np.log(energy))
            log_rate.append(np.log(float(n_cm3) * sigma))
            cdf.append(row_cdf)
            pdf.append(row_pdf)
            offset += energy.size
    return (
        has,
        start,
        length,
        np.concatenate(logE),
        np.concatenate(log_rate),
        np.vstack(cdf),
        np.vstack(pdf),
        np.ascontiguousarray(mu_ref),
    )


def check_elsepa_coverage(elastic_tables, E_min_keV, E_max_keV):
    """Reject a transport energy range outside any ELSEPA table; never extrapolate."""
    for layer_tables in elastic_tables or ():
        for table in layer_tables:
            energy_eV = np.asarray(table["energy_eV"], dtype=float)
            lower, upper = float(energy_eV[0]) / 1e3, float(energy_eV[-1]) / 1e3
            if E_min_keV < lower or E_max_keV > upper:
                raise ValueError(
                    f"transport energy range must be within ELSEPA table [{lower:g}, {upper:g}] keV"
                )


@njit(cache=True)
def _elsepa_bracket(logE, logE_flat, start, length):
    """Grid interval and fraction for ``logE``, clamped to the table's ends."""
    first = start
    last = start + length - 1
    if logE <= logE_flat[first]:
        return first, 0.0
    if logE >= logE_flat[last]:
        return last - 1, 1.0
    lo = first
    hi = last
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if logE_flat[mid] <= logE:
            lo = mid
        else:
            hi = mid
    return lo, (logE - logE_flat[lo]) / (logE_flat[hi] - logE_flat[lo])


@njit(cache=True)
def _elsepa_rate_scalar(E_keV, logE_flat, log_rate_flat, start, length):
    """Macroscopic elastic cross section [1/cm], log-log in energy."""
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    return np.exp(log_rate_flat[row] + f * (log_rate_flat[row + 1] - log_rate_flat[row]))


@njit(cache=True)
def _elsepa_invert_row(cdf, pdf, mu, row, xi):
    """Invert one piecewise-linear angular density exactly at ``xi``.

    Within ``[mu_k, mu_k+1]`` the density is linear, so the CDF is quadratic
    in ``t = (mu - mu_k)/dmu``:
    ``F = F_k + dmu*(p_k*t + (p_k+1 - p_k)*t^2/2)``. The root is taken in the
    cancellation-free form ``t = 2r / (b + sqrt(b^2 + 2*dmu*(p_k+1 - p_k)*r))``
    with ``b = dmu*p_k`` and ``r = xi - F_k``.
    """
    n = mu.size
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cdf[row, mid] <= xi:
            lo = mid
        else:
            hi = mid
    dmu = mu[lo + 1] - mu[lo]
    p0 = pdf[row, lo]
    p1 = pdf[row, lo + 1]
    r = xi - cdf[row, lo]
    b = dmu * p0
    disc = b * b + 2.0 * dmu * (p1 - p0) * r
    if disc < 0.0:
        disc = 0.0
    denom = b + np.sqrt(disc)
    t = 2.0 * r / denom if denom > 0.0 else 0.0
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return mu[lo] + t * dmu


@njit(cache=True)
def _sample_cos_theta_elsepa(E_keV, xi, logE_flat, cdf, pdf, mu, start, length):
    """Polar deflection cosine from tabulated ELSEPA angular distributions.

    One uniform ``xi`` is inverted on the two bracketing energy nodes and the
    quantiles are interpolated linearly in ``ln E``. Quantile interpolation
    keeps the sample monotone in ``xi`` and moves the forward peak's width
    smoothly between nodes; at a node it reproduces that node's distribution
    exactly.

    Validation: elsepa-elastic-sampling
    """
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    mu0 = _elsepa_invert_row(cdf, pdf, mu, row, xi)
    mu1 = _elsepa_invert_row(cdf, pdf, mu, row + 1, xi)
    return 1.0 - 2.0 * ((1.0 - f) * mu0 + f * mu1)
