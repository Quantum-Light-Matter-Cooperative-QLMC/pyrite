"""
montecarlo.transport

Single-scattering electron transport (Zhai SI S2): element data, elastic
scattering models (Browning free paths + NIST-Mott-calibrated screened-
Rutherford angles, analytic SR fallback), Joy-Luo stopping power, and a
Numba-compiled scalar event-driven transport core. CPU-only; the spectrum phase
consumes the segment arrays returned here.
"""

import logging
import os
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
from numba import njit

from .. import DATA_DIR
from ..materials._transport_data import TRANSPORT_ELEMENTS
from ..materials.attenuation import _normalize_composition
from ._backend import _GPU
from .geometry import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    project_beam_entry,
    validate_transverse_dimensions,
)
from .groove import _first_surface_event_scalar_numba, entry_points

if _GPU:
    import cupy as xp
else:
    import numpy as np

logger = logging.getLogger(__name__)

MOTT_DIR = str(DATA_DIR / "mott_transport_cross_sections")
A0_SQ_CM2 = 2.8002852e-17  # Bohr radius squared [cm^2] (NIST SRD 64 unit)

# Speed of light in transport-clock units: the electron clock sum(L/beta) is in
# Angstrom (c=1), so a longitudinal bunch length in fs converts via
# c = 2997.924580 Ang/fs. Mirrors sweep.C_ANG_PER_FS / plots.trajectories
# .C_ANG_PER_FS (same physical constant; defined locally to keep transport free
# of a plots/sweep import cycle).
C_ANG_PER_FS = 2997.924580


def _sample_bunch_offsets(
    Ne,
    bunch_length_fs,
    long_shape,
    long_offsets_fs,
    seed,
    longitudinal_distribution: Mapping[str, Any] | None = None,
):
    """Per-electron longitudinal arrival offset ``Delta t`` [Angstrom, c=1],
    centered on the bunch centroid (decision 3).

    Source/convention: a longitudinal bunch is a distribution of electron
    arrival TIMES; ``Delta t_e`` [fs] converts to the transport clock's Angstrom
    units via ``c = 2997.924580 Ang/fs`` (:data:`C_ANG_PER_FS`). ``long_shape``
    selects the sampling law about a zero centroid:

    * ``"gaussian"`` -- ``Delta t ~ Normal(0, sigma)`` with RMS ``sigma =
      bunch_length_fs`` (the RMS convention, NOT FWHM).
    * ``"uniform"`` -- a flat-top of the SAME RMS: half-width ``sqrt(3)*sigma``.

    ``long_offsets_fs`` supplies explicit per-particle offsets (measured /
    arbitrary / microbunched profiles) and OVERRIDES ``long_shape`` /
    ``bunch_length_fs``. The draw uses an independent RNG child stream
    (``SeedSequence(seed).spawn(4)[3]`` -- the next index after ``beam_rng``'s
    ``spawn(2)[1]`` and ``phase_rng``'s ``spawn(3)[2]``), so enabling the bunch
    NEVER perturbs the main free-path / scattering draws.

    Limiting case: ``bunch_length_fs=None`` and ``long_offsets_fs=None`` ->
    all-zero (the legacy point bunch, bit-for-bit); ``bunch_length_fs -> 0``
    recovers it continuously.

    Validation: longitudinal-bunch-sampling
    """
    if longitudinal_distribution is not None:
        if bunch_length_fs is not None or long_offsets_fs is not None or long_shape != "gaussian":
            raise ValueError("longitudinal_distribution is incompatible with legacy bunch fields")
        kind = longitudinal_distribution.get("kind")
        bunch_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
        if kind in ("gaussian", "compressed"):
            sigma_fs = longitudinal_distribution.get("rms_duration_fs")
            if sigma_fs is None:
                raise ValueError(f"{kind} resolution requires rms_duration_fs")
            dt = bunch_rng.normal(0.0, float(sigma_fs), size=Ne)
        elif kind == "microtrain":
            envelope_fs = float(longitudinal_distribution["envelope_rms_fs"])
            microbunch_fs = float(longitudinal_distribution["microbunch_rms_fs"])
            spacing_fs = float(longitudinal_distribution["spacing_fs"])
            jitter_fs = float(longitudinal_distribution["timing_jitter_fs"])
            depth = float(longitudinal_distribution["modulation_depth"])
            if not (
                np.isfinite(envelope_fs)
                and np.isfinite(microbunch_fs)
                and np.isfinite(spacing_fs)
                and envelope_fs > 0.0
                and microbunch_fs >= 0.0
                and spacing_fs > 0.0
                and jitter_fs >= 0.0
                and 0.0 <= depth <= 1.0
            ):
                raise ValueError("invalid resolved microtrain parameters")
            center_variance = envelope_fs**2 - microbunch_fs**2 - jitter_fs**2
            if center_variance <= 0.0:
                raise ValueError(
                    "microtrain envelope RMS must exceed combined microbunch width and jitter"
                )
            center_sigma_fs = np.sqrt(center_variance)
            centers = np.rint(bunch_rng.normal(0.0, center_sigma_fs / spacing_fs, size=Ne))
            train = (
                centers * spacing_fs
                + bunch_rng.normal(0.0, microbunch_fs, size=Ne)
                + bunch_rng.normal(0.0, jitter_fs, size=Ne)
            )
            if depth == 1.0:
                dt = train
            elif depth == 0.0:
                dt = bunch_rng.normal(0.0, envelope_fs, size=Ne)
            else:
                unmodulated = bunch_rng.normal(0.0, envelope_fs, size=Ne)
                mask = bunch_rng.random(Ne) < depth
                dt = np.where(mask, train, unmodulated)
        else:
            raise ValueError(f"unknown resolved longitudinal kind {kind!r}")
        dt = dt * C_ANG_PER_FS
    elif long_offsets_fs is not None:
        dt = np.asarray(long_offsets_fs, dtype=float)
        if dt.ndim != 1:
            raise ValueError("long_offsets_fs must be one-dimensional")
        if dt.size != Ne:
            raise ValueError(
                f"long_offsets_fs has {dt.size} entries but Ne={Ne}; supply one "
                "explicit longitudinal offset per electron"
            )
        if not np.all(np.isfinite(dt)):
            raise ValueError("long_offsets_fs must contain only finite values")
        dt = dt * C_ANG_PER_FS
    elif bunch_length_fs is not None:
        sigma_fs = float(bunch_length_fs)
        if not np.isfinite(sigma_fs) or sigma_fs < 0.0:
            raise ValueError("bunch_length_fs must be finite and non-negative")
        sigma_ang = sigma_fs * C_ANG_PER_FS
        bunch_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
        if long_shape == "gaussian":
            dt = bunch_rng.normal(0.0, sigma_ang, size=Ne)
        elif long_shape == "uniform":
            half_width = np.sqrt(3.0) * sigma_ang  # flat-top of the same RMS
            dt = bunch_rng.uniform(-half_width, half_width, size=Ne)
        else:
            raise ValueError(
                f"long_shape must be 'gaussian' or 'uniform' (or supply "
                f"long_offsets_fs), got {long_shape!r}"
            )
    else:
        return np.zeros(Ne)
    return dt - dt.mean()  # center on the bunch centroid (t=0 == centroid)


def _beta_array(E_keV):
    g = 1.0 + E_keV / 510.99895
    return (1.0 - 1.0 / (g * g)) ** 0.5


if _GPU:
    beta_from_keV = xp.fuse(kernel_name="beta_from_keV")(_beta_array)
else:
    beta_from_keV = _beta_array


@njit(cache=True)
def beta_from_keV_scalar(E_i):
    g = 1.0 + E_i / 510.99895
    g_inv_square = 1.0 / (g * g)
    return (1.0 - g_inv_square) ** 0.5


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
def _sigma_browning_cm2_scalar(Z_i, E_i):
    """
    Browning et al., J. Appl. Phys. 76, 2016 (1994): empirical fit to the
    tabulated Mott TOTAL elastic cross sections [cm^2], valid 0.1-30 keV,
    Z <= 92.
    """

    z17 = Z_i**1.7
    numerator = 3.0e-18 * z17
    z_exp = 0.005 * z17
    z_squared = 0.0007 * Z_i * Z_i
    sqrt_E_i = np.sqrt(E_i)

    return numerator / (E_i + z_exp * sqrt_E_i + z_squared / sqrt_E_i)


def _alpha_sr_joy(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return 3.4e-3 * Z**0.67 / E_keV


@njit(cache=True)
def _alpha_sr_joy_scalar(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return 3.4e-3 * Z**0.67 / E_keV


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
def _scatter_rates_mott_scalar(E_i, Z_i, n_cm3_i):
    sig_i = _sigma_browning_cm2_scalar(Z_i, E_i)
    return sig_i * n_cm3_i


@njit(cache=True)
def _scatter_rates_sr_scalar(E_i, Z_i, n_cm3_i):
    """Per-element elastic scattering rates [1/cm] at energies Ea (one layer)."""
    a = _alpha_sr_joy_scalar(Z_i, E_i)
    E_i_plus_511 = E_i + 511.0
    E_i_plus_1024 = E_i + 1024.0
    E_i_511_over_1024 = E_i_plus_511 / E_i_plus_1024

    sig_i = (
        5.21e-21
        * (Z_i * Z_i)
        / (E_i * E_i)
        * 4.0
        * np.pi
        / (a * (1.0 + a))
        * (E_i_511_over_1024 * E_i_511_over_1024)
    )
    return n_cm3_i * sig_i


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
    per process at DEBUG (silent by default -- set CXR_MC_DEBUG=1 to see it;
    a ProcessPoolExecutor worker pool re-logs once per worker, since each
    worker gets its own _NO_MOTT cache)."""
    R = rng.random(E_keV.shape)
    if elastic_model == "mott" and element not in _NO_MOTT:
        try:
            logE, logA = _mott_alpha_table(element, Z)
            alpha = 10.0 ** np.interp(np.log10(E_keV * 1e3), logE, logA)

            return 1.0 - 2.0 * alpha * R / (1.0 + alpha - R)

        except FileNotFoundError:
            logger.debug(...)
            _NO_MOTT.add(element)

    return _sample_cos_theta_sr_numba(Z, E_keV, R)


def _dEds_keV_per_ang(Z, A, J_keV, rho_g_cm3, E_keV):
    """Joy-Luo modified Bethe stopping power [keV/Angstrom] (negative)."""
    k = 0.731 + 0.0688 * np.log10(Z)
    # 78500 keV/cm -> 7.85e-4 keV/Angstrom prefactor
    return -7.85e-4 * rho_g_cm3 * Z / (A * E_keV) * np.log(1.166 * (E_keV + k * J_keV) / J_keV)


@njit(cache=True)
def _dEds_compound_scalar(J_arr, k_arr, coeff_arr, E_i):

    total = 0.0
    for i in range(J_arr.size):
        J = J_arr[i]
        k = k_arr[i]
        coeff = coeff_arr[i]

        total += coeff * np.log(1.166 * (E_i + k * J) / J)

    return -7.85e-4 / E_i * total


@njit(cache=True)
def _dEds_compound(J_arr, k_arr, coeff_arr, E_keV):
    out = np.empty_like(E_keV)

    for j in range(E_keV.size):
        E = E_keV[j]
        total = 0.0
        for i in range(J_arr.size):
            J = J_arr[i]
            k = k_arr[i]
            coeff = coeff_arr[i]

            total += coeff * np.log(1.166 * (E + k * J) / J)

        out[j] = -7.85e-4 / E * total
    return out


@njit(cache=True)
def _rotate_direction_scalar(dx, dy, dz, cos_t, phi):
    sin2 = 1.0 - cos_t * cos_t
    if sin2 < 0.0:
        sin2 = 0.0
    sin_t = np.sqrt(sin2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if abs(dx) < 0.9:
        refx = 1.0
        refy = 0.0
    else:
        refx = 0.0
        refy = 1.0

    ux = -dz * refy
    uy = dz * refx
    uz = dx * refy - dy * refx

    u_mag = np.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= u_mag
    uy /= u_mag
    uz /= u_mag

    wx = dy * uz - dz * uy
    wy = dz * ux - dx * uz
    wz = dx * uy - dy * ux

    a = sin_t * cos_phi
    b = sin_t * sin_phi

    outx = cos_t * dx + a * ux + b * wx
    outy = cos_t * dy + a * uy + b * wy
    outz = cos_t * dz + a * uz + b * wz

    mag = np.sqrt(outx * outx + outy * outy + outz * outz)

    return outx / mag, outy / mag, outz / mag


@njit(cache=True)
def _rotate_directions(d, cos_t, phi):
    """Rotate unit vectors d (N,3) by polar angle theta, azimuth phi.
    First test @njit case, since it is the heaviest single item
    in the transport path."""
    out = np.empty_like(d)

    for i in range(d.shape[0]):
        dx = d[i, 0]
        dy = d[i, 1]
        dz = d[i, 2]

        cos_theta_i = cos_t[i]
        sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i
        if sin2_theta_i < 0.0:
            sin2_theta_i = 0.0

        sin_theta_i = np.sqrt(sin2_theta_i)

        phi_i = phi[i]
        cos_phi_i = np.cos(phi_i)
        sin_phi_i = np.sin(phi_i)

        if abs(dx) < 0.9:
            refx = 1.0
            refy = 0.0
        else:
            refx = 0.0
            refy = 1.0
        # refz left out because always zero

        ux = -dz * refy
        uy = dz * refx
        uz = dx * refy - dy * refx

        u_mag = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= u_mag
        uy /= u_mag
        uz /= u_mag

        wx = dy * uz - dz * uy
        wy = dz * ux - dx * uz
        wz = dx * uy - dy * ux

        sin_theta_cos_phi = sin_theta_i * cos_phi_i
        sin_theta_sin_phi = sin_theta_i * sin_phi_i

        outx = cos_theta_i * dx + sin_theta_cos_phi * ux + sin_theta_sin_phi * wx
        outy = cos_theta_i * dy + sin_theta_cos_phi * uy + sin_theta_sin_phi * wy
        outz = cos_theta_i * dz + sin_theta_cos_phi * uz + sin_theta_sin_phi * wz

        out_mag = np.sqrt(outx * outx + outy * outy + outz * outz)
        out[i, 0] = outx / out_mag
        out[i, 1] = outy / out_mag
        out[i, 2] = outz / out_mag

    return out


@njit(cache=True)
def _first_prism_exit_scalar(px, py, pz, dx, dy, dz, z_min, z_max, width, height):
    """Nearest positive ray/prism intersection, matching first_prism_exit face order."""
    best_t = np.inf
    best_face = -1
    half_w = 0.5 * width
    half_h = 0.5 * height

    if dx < 0.0:
        t = (-half_w - px) / dx
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = X_MIN
    if dx > 0.0:
        t = (half_w - px) / dx
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = X_MAX
    if dy < 0.0:
        t = (-half_h - py) / dy
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Y_MIN
    if dy > 0.0:
        t = (half_h - py) / dy
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Y_MAX
    if dz < 0.0:
        t = (z_min - pz) / dz
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Z_MIN
    if dz > 0.0:
        t = (z_max - pz) / dz
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Z_MAX

    return best_t, best_face


@njit(cache=True)
def _transport_core_ungrooved(
    Ne,
    alive,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    rng,
    pos,
    dirs,
    E_cut_keV,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_ncm3,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
):
    """Compiled ungrooved transport core, with optional finite x/y footprint.

    ``elastic_model_code`` is 0 for analytic screened Rutherford and 1 for
    Browning/Mott transport. Mott angular tables are preloaded by the Python
    wrapper; elements without a table use the analytic SR angular distribution
    while retaining the Browning total elastic collision rate, matching the
    legacy fallback behavior.
    """
    EPS = 1e-6
    nseg = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_alive = int(alive.sum())

    # Reuse one small rate buffer for all events; only the first Z_arr.size
    # entries are live for the current material layer.
    rate_arr = np.empty(mott_has_table.shape[1])

    lockstep_step = 0
    while lockstep_step < max_steps and n_alive > 0:
        lockstep_step += 1

        for g in range(Ne):
            if not alive[g]:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[g, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            n_cm3s = L_ncm3[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[g]

            # 1. Sample the next elastic-collision distance.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(E_j, Z_arr[i_el], n_cm3s[i_el])
                else:
                    rate = _scatter_rates_sr_scalar(E_j, Z_arr[i_el], n_cm3s[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            step_j = -lam_ang * np.log(rng.random())

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[g, 0]
            dy = dirs[g, 1]
            dz = dirs[g, 2]
            px = pos[g, 0]
            py = pos[g, 1]
            pz = pos[g, 2]

            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 3. Record the radiating material segment.
            dEds = _dEds_compound_scalar(J_arr, k_arr, coeff_arr, E_j)
            beta_j = beta_from_keV_scalar(E_j)

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[g]
            seg_id[nseg] = g
            seg_lay[nseg] = L
            nseg += 1

            # 4. Advance position, energy, and transport clock.
            pos[g, 0] = px + step_j * dx
            pos[g, 1] = py + step_j * dy
            pos[g, 2] = pz + step_j * dz
            E_keV[g] = E_j + dEds * step_j
            clock[g] += step_j / beta_j

            # 5. Exit, internal-boundary, or collision handling.
            died_j = exit_top_j or exit_bot_j or exit_side_j or E_keV[g] < E_cut_keV
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if died_j:
                alive[g] = False
                n_alive -= 1
                continue

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[g, 2] += (1.0 if dirs[g, 2] > 0.0 else -1.0) * EPS
                continue

            # A full flight ended in an elastic collision. Pick the element with
            # probability proportional to n_i * sigma_i(E).
            if Z_arr.size == 1:
                i_el = 0
            else:
                u = rng.random() * total_rate
                cumulative = 0.0
                i_el = Z_arr.size - 1
                for k in range(Z_arr.size):
                    cumulative += rate_arr[k]
                    if cumulative > u:
                        i_el = k
                        break

            Z_i = Z_arr[i_el]
            if elastic_model_code == 1 and mott_has_table[L, i_el]:
                log_alpha = _interp_mott_log_alpha_scalar(
                    np.log10(E_keV[g] * 1e3),
                    mott_logE_flat,
                    mott_logA_flat,
                    mott_start[L, i_el],
                    mott_len[L, i_el],
                )
                alpha = 10.0**log_alpha
            else:
                alpha = _alpha_sr_joy_scalar(Z_i, E_keV[g])

            cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
            phi = 2.0 * np.pi * rng.random()
            dx, dy, dz = _rotate_direction_scalar(dirs[g, 0], dirs[g, 1], dirs[g, 2], cos_t, phi)
            dirs[g, 0] = dx
            dirs[g, 1] = dy
            dirs[g, 2] = dz

    return nseg, n_back, n_trans, n_side


@njit(cache=True)
def _transport_core_grooved(
    Ne,
    alive,
    max_steps,
    max_segments,
    max_vac,
    n_layers,
    internal_bounds,
    elastic_model_code,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    groove_spacing,
    groove_depth,
    groove_st,
    groove_ct,
    clock,
    rng,
    pos,
    dirs,
    E_cut_keV,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_ncm3,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    vac_start,
    vac_end,
    vac_E,
    vac_t0,
    vac_id,
):
    """Compiled groove-aware scalar transport.

    Groove facet exits and vacuum re-entries use the exact sawtooth event search
    from :mod:`montecarlo.groove`, expressed entirely in scalar numeric form for
    Numba.  Material exit/re-entry pairs do not consume ``max_steps``; the same
    per-electron surface-event guard as the legacy implementation prevents
    pathological geometry from looping forever.
    """
    EPS = 1e-6
    machine_eps = 2.220446049250313e-16
    surface_eps = max(
        64.0 * machine_eps * groove_spacing,
        2.0e-12 * groove_spacing,
    )

    nseg = 0
    nvac = 0
    n_back = 0
    n_trans = 0
    n_side = 0

    zero_surface_events = np.zeros(Ne, dtype=np.int16)
    material_steps = np.zeros(Ne, dtype=np.int32)
    surface_events = np.zeros(Ne, dtype=np.int32)

    # One small reusable rate buffer; material layers can have different
    # element counts, bounded by the second Mott-table dimension.
    rate_arr = np.empty(mott_has_table.shape[1])

    # Initially every alive electron is eligible for a material event.  An
    # exit/re-entry pair leaves it eligible without consuming a material step.
    n_eligible = int(alive.sum())

    while n_eligible > 0:
        for g in range(Ne):
            if not alive[g] or material_steps[g] >= max_steps:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[g, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            n_cm3s = L_ncm3[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[g]

            # 1. Sample the next elastic-collision distance in the current layer.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(E_j, Z_arr[i_el], n_cm3s[i_el])
                else:
                    rate = _scatter_rates_sr_scalar(E_j, Z_arr[i_el], n_cm3s[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            step_j = -lam_ang * np.log(rng.random())

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            dx = dirs[g, 0]
            dy = dirs[g, 1]
            dz = dirs[g, 2]
            px = pos[g, 0]
            py = pos[g, 1]
            pz = pos[g, 2]

            # 2a. Candidate layer/prism boundary event.
            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px,
                    py,
                    pz,
                    dx,
                    dy,
                    dz,
                    z_top_L,
                    z_bot_L,
                    width_ang,
                    height_ang,
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 2b. Exact material->vacuum groove crossing competes with the
            # collision/layer/prism event.  Strict '<' matches legacy tie behavior.
            s_surface = _first_surface_event_scalar_numba(
                px,
                pz,
                dx,
                dz,
                groove_spacing,
                groove_depth,
                groove_st,
                groove_ct,
                1,  # exit
            )
            surface_first = s_surface < step_j
            if surface_first:
                step_j = s_surface
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

            # 3. Record the radiating material segment.
            dEds = _dEds_compound_scalar(J_arr, k_arr, coeff_arr, E_j)
            beta_j = beta_from_keV_scalar(E_j)

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[g]
            seg_id[nseg] = g
            seg_lay[nseg] = L
            nseg += 1

            # 4. Advance through material and apply continuous stopping.
            pos[g, 0] = px + step_j * dx
            pos[g, 1] = py + step_j * dy
            pos[g, 2] = pz + step_j * dz
            E_keV[g] = E_j + dEds * step_j
            clock[g] += step_j / beta_j

            below_cut = E_keV[g] < E_cut_keV
            active_surface = surface_first and not below_cut
            reentered = False

            if not active_surface:
                zero_surface_events[g] = 0
            else:
                # 5. From the groove surface, follow the unchanged ray through
                # vacuum to its first exact vacuum->material re-entry.
                sx = pos[g, 0]
                sy = pos[g, 1]
                sz = pos[g, 2]
                entry_distance = _first_surface_event_scalar_numba(
                    sx,
                    sz,
                    dx,
                    dz,
                    groove_spacing,
                    groove_depth,
                    groove_st,
                    groove_ct,
                    2,  # entry
                )

                if finite_footprint:
                    side_distance, side_face = _first_prism_exit_scalar(
                        sx,
                        sy,
                        sz,
                        dx,
                        dy,
                        dz,
                        0.0,
                        z_total,
                        width_ang,
                        height_ang,
                    )
                    if side_face < Z_MIN and side_distance < entry_distance:
                        exit_side_j = True
                        entry_distance = np.inf

                reentered = np.isfinite(entry_distance)
                if reentered:
                    if nvac >= max_vac:
                        raise RuntimeError("vacuum segment buffer exhausted")

                    ex = sx + entry_distance * dx
                    ey = sy + entry_distance * dy
                    ez = sz + entry_distance * dz

                    vac_start[nvac, 0] = sx
                    vac_start[nvac, 1] = sy
                    vac_start[nvac, 2] = sz
                    vac_end[nvac, 0] = ex
                    vac_end[nvac, 1] = ey
                    vac_end[nvac, 2] = ez
                    vac_E[nvac] = E_keV[g]
                    vac_t0[nvac] = clock[g]
                    vac_id[nvac] = g
                    nvac += 1

                    clock[g] += entry_distance / beta_from_keV_scalar(E_keV[g])
                    pos[g, 0] = ex + surface_eps * dx
                    pos[g, 1] = ey + surface_eps * dy
                    pos[g, 2] = ez + surface_eps * dz

                    surface_events[g] += 1
                    if surface_events[g] > max_steps:
                        raise RuntimeError("grooved surface event limit exhausted")
                elif not exit_side_j:
                    # No later material intersection: permanent escape through
                    # the grooved entrance surface.
                    exit_top_j = True

                if step_j <= EPS:
                    zero_surface_events[g] += 1
                else:
                    zero_surface_events[g] = 0
                if zero_surface_events[g] >= 2:
                    raise RuntimeError("repeated zero-length grooved surface events")

            # 6. Kill true exits / cutoff electrons and handle internal seams.
            died_j = exit_top_j or exit_bot_j or exit_side_j or below_cut
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1

            if died_j:
                alive[g] = False
                n_eligible -= 1
                continue

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[g, 2] += (1.0 if dirs[g, 2] > 0.0 else -1.0) * EPS

            # 7. Only a full material flight ends in an elastic collision.
            full_j = not cross_up_j and not cross_dn_j and not exit_side_j and not surface_first
            if full_j:
                if Z_arr.size == 1:
                    i_el = 0
                else:
                    u = rng.random() * total_rate
                    cumulative = 0.0
                    i_el = Z_arr.size - 1
                    for k in range(Z_arr.size):
                        cumulative += rate_arr[k]
                        if cumulative > u:
                            i_el = k
                            break

                Z_i = Z_arr[i_el]
                if elastic_model_code == 1 and mott_has_table[L, i_el]:
                    log_alpha = _interp_mott_log_alpha_scalar(
                        np.log10(E_keV[g] * 1e3),
                        mott_logE_flat,
                        mott_logA_flat,
                        mott_start[L, i_el],
                        mott_len[L, i_el],
                    )
                    alpha = 10.0**log_alpha
                else:
                    alpha = _alpha_sr_joy_scalar(Z_i, E_keV[g])

                cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
                phi = 2.0 * np.pi * rng.random()
                ndx, ndy, ndz = _rotate_direction_scalar(
                    dirs[g, 0], dirs[g, 1], dirs[g, 2], cos_t, phi
                )
                dirs[g, 0] = ndx
                dirs[g, 1] = ndy
                dirs[g, 2] = ndz

            # Legacy groove semantics: valid exit/re-entry pairs repeat without
            # consuming a material step; all other surviving material events do.
            if not reentered:
                material_steps[g] += 1
                if material_steps[g] >= max_steps:
                    n_eligible -= 1

    return nseg, nvac, n_back, n_trans, n_side


def simulate_trajectories(
    E0_keV,
    Ne,
    thickness_ang,
    element=None,
    n_atoms_per_ang3=None,
    E_cut_keV=5.0,
    seed=0,
    max_steps=20000,
    elastic_model="mott",
    beam_dir=None,
    composition=None,
    layers=None,
    beam_fwhm_mm=None,
    beam_fwhm_y_mm=None,
    bunch_length_fs=None,
    long_shape="gaussian",
    long_offsets_fs=None,
    longitudinal_distribution=None,
    crystal_width_mm=None,
    crystal_height_mm=None,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    groove=None,
):
    """
    Transport Ne electrons of energy E0_keV [keV] into a slab 0<=z<=thickness.
    Beam enters at the origin along +z. Electrons terminate when they exit
    either surface or drop below E_cut_keV (segments below the cutoff don't
    radiate in the spectral window of interest anyway).

    elastic_model:
      "mott" (default) -- Browning fit to the Mott TOTAL cross sections for
          the free path + screening parameter alpha(E) calibrated per element
          to reproduce the NIST SRD 64 relativistic Mott TRANSPORT cross
          section (so both the collision rate and the momentum-transfer rate
          match Mott data). Requires the NIST table in
          mott_transport_cross_sections/.
      "sr" -- classic analytic screened-Rutherford model (Joy), no data files.

    beam_dir: initial electron direction in the SLAB frame (default +z,
    i.e. normal incidence). For a tilted sample use tilted_geometry().

    composition: for COMPOUNDS, [(element, number_density_1_per_Ang3), ...]
    overriding element/n_atoms_per_ang3. Free paths and stopping are
    additive over elements; the scattering element at each collision is
    chosen with probability n_i sigma_i / sum.

    layers: optional film-on-substrate stack
    [(z_top, z_bot, composition), ...] (top/entrance first, contiguous, deepest
    z_bot = total thickness). Each electron's free path / stopping / scattering
    element switch by the layer it is currently in; a flight is truncated at an
    internal boundary (no collision -- the electron continues into the neighbor),
    so the substrate's higher-Z backscatter feeds electron path back into the
    film. None -> a single layer over [0, thickness_ang] (the old single-material
    transport, BIT-FOR-BIT). When given, thickness_ang is superseded by the
    stack's total thickness.

    beam_fwhm_mm: transverse size of the incident electron beam -- an
    azimuthally-symmetric Gaussian spot of the given FULL WIDTH AT HALF MAXIMUM
    [mm] (same FWHM convention as mosaic_fwhm_rad / eds_fwhm_eV / aperture_fwhm_eV
    elsewhere in this package). Each electron's entry point is drawn independently
    as x0, y0 ~ Normal(0, sigma), sigma = beam_fwhm_mm / (2 sqrt(2 ln 2)) in the
    LAB plane perpendicular to the fixed beam axis, then PROJECTED onto the
    tilted sample entrance face by geometry.project_beam_entry (the spot
    stretches by 1/cos(tilt_polar) along the tilt azimuth). At tilt_polar_rad=0
    the projection is the identity, so the entry is x0, y0 exactly. The result is
    used as the electron's initial transverse position. With a laterally infinite crystal,
    this rigidly translates the whole trajectory; beam_dir, common to the whole
    beam, is unaffected. None (default) is a strict no-op -- the old point-source
    (delta-function) beam entering at the origin, BIT-FOR-BIT. Limiting case:
    beam_fwhm_mm -> 0 recovers the point source exactly
    (sigma -> 0 -> x0 = y0 = 0).

    The offset is drawn from an RNG stream independent of `seed`'s main stream
    (a numpy SeedSequence child). When both crystal_width_mm and
    crystal_height_mm are None, enabling it NEVER perturbs the free-path /
    scattering-angle draws: every other returned array (E_keV, v_hat, L_ang,
    t_ang, elec_id, layer, n_backscattered, n_transmitted, n_stopped) is
    identical to the beam_fwhm_mm=None run; r_mid changes only by the constant
    per-electron transverse offset. In this laterally infinite limit, no
    downstream physics -- elastic scattering, stopping power, layer-boundary
    crossing, or the self-absorption path in mc_spectrum -- reads pos[:, :2],
    and a finite beam spot is a pure geometry/visualization refinement with zero
    effect on the emitted spectrum.

    With a finite crystal footprint, the sampled transverse positions classify
    missed entries and can cause side-face exits. Segment positions also affect
    the downstream six-face escape attenuation, so beam size can change the
    emitted radiation spectrum.

    beam_fwhm_y_mm: optional y-plane spot FWHM [mm] for an ELLIPTICAL beam
    (decision 8). None -> equals beam_fwhm_mm (isotropic), which draws
    sigma_x == sigma_y and is bit-for-bit with the historical scalar-spot path.

    bunch_length_fs, long_shape, long_offsets_fs, longitudinal_distribution:
    longitudinal bunch sampling. Each electron gets an arrival offset
    ``Delta t`` [Angstrom, c=1] via :func:`_sample_bunch_offsets`. The legacy
    fields provide Gaussian/uniform RMS sampling or explicit per-particle
    offsets. The mutually exclusive resolved distribution provides Gaussian,
    compressed-Gaussian, or directly indexed finite-train sampling without
    materializing its many centers. Offsets are centered on the bunch centroid
    and returned as per-segment ``t0_ang`` (and ``vacuum_t0_ang``), kept
    SEPARATE from relative-age ``t_ang``/``clock``. The independent RNG child
    (``spawn(4)[3]``) never perturbs transport draws. With no legacy or resolved
    distribution, offsets are all zero (the legacy point bunch, bit-for-bit).

    crystal_width_mm, crystal_height_mm: optional full transverse dimensions
    [mm] of a rectangular prism centered at the beam origin. Both must be
    supplied and strictly positive, or both omitted. Finite dimensions are
    converted once to Angstrom and define the transport volume
    ``[-width/2, width/2] x [-height/2, height/2] x [0, thickness]``. An
    incident Gaussian entry point outside that footprint is counted in
    ``n_missed`` and produces no segment, but remains in ``Ne`` so all yields
    retain their per-incident-electron normalization. Side-face exits are
    counted separately in ``n_side_exited``. The all-``None`` limiting case is
    the original laterally infinite slab and follows its legacy free-flight
    path without invoking the prism-exit helper. Each finite free flight is
    capped at the smallest positive ray boundary solution ``p + s d`` on a
    prism face; this assumes an axis-aligned rectangular footprint.

    tilt_polar_rad, tilt_azim_rad: sample tilt (Zhai convention, same angles
    passed to :func:`geometry.tilted_geometry`) used ONLY to project the
    ``beam_fwhm_mm`` Gaussian spot onto the tilted entrance face via
    :func:`geometry.project_beam_entry`. Both default to 0 (normal incidence),
    making the projection the identity and leaving every ``beam_fwhm_mm`` result
    bit-for-bit. They have no effect when ``beam_fwhm_mm`` is None.

    groove: optional :class:`~cxr_mc.montecarlo.groove.GrooveSpec` describing a
    blazed sawtooth material/vacuum boundary on the beam-entrance face. Initial
    rays enter through relief facets via ``entry_points``.

    Every later free flight is intersected with both periodic facet families

        n . r = k*spacing*cos(tp),  b . r = k*spacing*sin(tp),

    accepting only crossings in the physical depth band ``0 <= z <= h`` where
    ``h = spacing*sin(tp)*cos(tp)``. A material-to-vacuum event truncates the
    radiating segment at the facet. If the unchanged ray intersects a later
    facet from the vacuum side, it advances to that re-entry point with no
    scattering, stopping, or radiation, then resumes material transport.
    Vacuum flight advances the electron clock by ``L_vacuum / beta``. Resampling
    the elastic free path after re-entry is exact because the exponential
    collision-distance distribution is memoryless. Valid exit/re-entry pairs
    use a separate per-electron event counter and do not consume ``max_steps``
    material iterations; exhausting that event bound raises ``RuntimeError``.

    Groove gaps are returned as separate ``vacuum_*`` diagnostic arrays; they
    never enter the material segment sum. A ray with no later re-entry is a
    permanent entrance-face exit. When beam_fwhm_mm is None, lateral phase is
    sampled uniformly over one groove period using an RNG stream independent of
    the main transport draws. None is a strict no-op -- BIT-FOR-BIT identical to
    the ungrooved slab. Source: exact periodic ray-plane intersections; see
    ``docs/superpowers/specs/2026-07-24-groove-aware-transport-design.md``.
    Validation: blazed-groove-geometry

    Returns dict of per-segment arrays:
      "r_mid" (M,3) [Ang], "v_hat" (M,3), "L_ang" (M,), "E_keV" (M,),
      "t_ang" (M,), "t0_ang" (M,) [per-electron bunch offset], "elec_id" (M,),
      "layer" (M,) [emitting layer index]
    incident phase-space diagnostics (one row per sampled electron, including
    missed entries): "initial_r_ang" (Ne,3), "initial_v_hat" (Ne,3),
      "initial_E_keV" (Ne,), "initial_t0_ang" (Ne,)
    non-radiating groove-gap flights:
      "vacuum_start_ang" (V,3), "vacuum_end_ang" (V,3),
      "vacuum_E_keV" (V,), "vacuum_t_ang" (V,), "vacuum_t0_ang" (V,),
      "vacuum_elec_id" (V,)
    and diagnostics: "n_backscattered", "n_transmitted", "n_side_exited",
    "n_missed", "n_stopped", "n_layers".

    Validation: electron-transport, finite-beam-size, finite-transverse-crystal,
    grazing-beam-projection
    """
    width_mm, height_mm = validate_transverse_dimensions(
        crystal_width_mm, crystal_height_mm, unit="mm"
    )
    width_ang = None if width_mm is None else width_mm * 1.0e7
    height_ang = None if height_mm is None else height_mm * 1.0e7
    finite_footprint = width_ang is not None
    max_segments = Ne * max_steps
    max_vac = Ne * max_steps

    # Build the layer stack: explicit `layers` (film-on-substrate) overrides;
    # else a single layer spanning the slab (bit-for-bit the old transport).
    if layers is None:
        layers = [
            (
                0.0,
                float(thickness_ang),
                _normalize_composition(element, n_atoms_per_ang3, composition),
            )
        ]

    if elastic_model not in ("mott", "sr"):
        raise ValueError("elastic_model must be 'mott' or 'sr'")

    z_total = float(layers[-1][1])
    n_layers = len(layers)
    L_Zs = []
    L_Js = []
    L_ncm3 = []
    L_ks = []
    L_coeffs = []
    mott_tables = []

    for _, _, lc in layers:
        elements = []
        ncm3_arr = []
        Z_arr = []
        J_arr = []
        k_arr = []
        coeff_arr = []
        layer_mott_tables = []

        for el, n_i in lc:
            elements.append(el)
            params = TRANSPORT_ELEMENTS[el]
            Z_i = float(params["Z"])
            J_i = float(params["J_keV"])
            k_i = 0.731 + 0.0688 * np.log10(Z_i)
            coeff_i = (n_i / 0.602214076) * Z_i

            ncm3_arr.append(n_i * 1e24)
            Z_arr.append(Z_i)
            J_arr.append(J_i)
            k_arr.append(k_i)
            coeff_arr.append(coeff_i)

            table = None
            if elastic_model == "mott" and el not in _NO_MOTT:
                try:
                    table = _mott_alpha_table(el, Z_i)
                except FileNotFoundError:
                    logger.debug("No NIST Mott table for %s; using analytic SR angles", el)
                    _NO_MOTT.add(el)
            layer_mott_tables.append(table)

        L_Zs.append(np.asarray(Z_arr, dtype=float))
        L_Js.append(np.asarray(J_arr, dtype=float))
        L_ncm3.append(np.asarray(ncm3_arr, dtype=float))
        L_ks.append(np.asarray(k_arr, dtype=float))
        L_coeffs.append(np.asarray(coeff_arr, dtype=float))
        mott_tables.append(layer_mott_tables)

    L_top = np.asarray([float(a) for (a, _, _) in layers], dtype=float)
    L_bot = np.asarray([float(b) for (_, b, _) in layers], dtype=float)
    internal_bounds = L_bot[:-1].copy()

    # Numba only sees numeric Mott data. Tables are flattened because different
    # elements may have different grid lengths; start/length locate each table.
    max_elements = max(arr.size for arr in L_Zs)
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

    rng = np.random.default_rng(seed)
    pos = np.zeros((Ne, 3))
    if beam_fwhm_mm or beam_fwhm_y_mm:
        # independent child stream: does not consume from `rng`, so the main
        # transport draws (free path, scattering angle) are untouched -- see
        # the beam_fwhm_mm docstring paragraph above for the invariance this
        # buys.
        MM_TO_ANG = 1.0e7
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # Per-plane elliptical spot (decision 8). y defaults to x, so an
        # isotropic beam draws sigma_x == sigma_y and stays bit-for-bit with the
        # historical scalar-spot path: normal(0,1,(Ne,2)) scaled by a scalar sigma
        # IS normal(0,sigma,(Ne,2)) element-for-element and consumes the stream
        # identically.
        fwhm_y = beam_fwhm_mm if beam_fwhm_y_mm is None else beam_fwhm_y_mm
        sigma_x = float(beam_fwhm_mm or 0.0) * MM_TO_ANG * fwhm_to_sigma
        sigma_y = float(fwhm_y or 0.0) * MM_TO_ANG * fwhm_to_sigma
        beam_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        offsets = beam_rng.normal(0.0, 1.0, size=(Ne, 2))
        offsets[:, 0] *= sigma_x
        offsets[:, 1] *= sigma_y
        # Project the lab-frame Gaussian spot onto the tilted sample entrance
        # face (grazing-incidence footprint elongation). tilt=0 -> (u, v)
        # bit-for-bit, so the untilted beam draw is unchanged.
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif groove is not None:
        # Groove effects depend on x mod spacing; a point source samples one
        # phase only. Draw the lateral phase uniformly over one period from an
        # independent child stream (same spawn pattern as beam_rng, so the
        # main free-path/scattering draws are untouched).
        phase_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        pos[:, 0] = phase_rng.uniform(0.0, groove.spacing_ang, size=Ne)
    if groove is not None:
        # Slide each ray along the beam to its relief-facet entry point.
        # Later flights use exact groove exit/re-entry events below.
        # Validation: blazed-groove-geometry
        x_e, z_e = entry_points(pos[:, 0], groove)
        pos[:, 0] = x_e
        pos[:, 2] = z_e
    if beam_dir is None:
        beam_dir = np.array([0.0, 0.0, 1.0])
    beam_dir = np.asarray(beam_dir, dtype=float)
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    if beam_dir[2] <= 1e-6:
        raise ValueError("beam_dir must point into the slab (z component > 0)")
    dirs = np.tile(beam_dir, (Ne, 1))
    E_keV = np.full(Ne, float(E0_keV))
    if finite_footprint:
        assert height_ang is not None
        alive = (np.abs(pos[:, 0]) <= width_ang / 2.0) & (np.abs(pos[:, 1]) <= height_ang / 2.0)
        n_missed = int((~alive).sum())
    else:
        alive = np.ones(Ne, dtype=bool)
        n_missed = 0
    n_back = n_trans = n_side = 0
    # per-electron clock: cumulative flight "time" sum(L/beta) [Ang, c=1], the
    # same unit as the radiation interaction time t_L. Recorded at each segment's
    # START so a segment carries (depth, energy, age) -- consumed by the
    # penetration / electron-lifetime plots (-> fs via c = 2997.92 Ang/fs).

    clock = np.zeros(Ne)
    # per-electron longitudinal bunch offset [Ang, c=1], kept SEPARATE from the
    # relative-age clock: the coherent sum later reads absolute time
    # t_abs = t_ang + t0_ang. None/None -> all-zero (point bunch, bit-for-bit),
    # and nothing reads it in the incoherent spectrum today.
    t0_electron = _sample_bunch_offsets(
        Ne,
        bunch_length_fs,
        long_shape,
        long_offsets_fs,
        seed,
        longitudinal_distribution,
    )
    # Snapshot before transport mutates ``pos``, ``dirs``, and ``E``. These
    # arrays describe incident phase space, including particles that miss a
    # finite footprint.
    initial_r_ang = pos.copy()
    initial_v_hat = dirs.copy()
    initial_E_keV = E_keV.copy()

    # Preallocate fixed-capacity output buffers.  ``max_steps`` is already a
    # conservative safety bound in ordinary runs; groove re-entry events can add
    # material segments without consuming it, so the compiled core raises a
    # clear buffer error if a pathological case exceeds this capacity.
    seg_mid = np.empty((max_segments, 3), dtype=float)
    seg_dir = np.empty((max_segments, 3), dtype=float)
    seg_len = np.empty(max_segments, dtype=float)
    seg_E = np.empty(max_segments, dtype=float)
    seg_t0 = np.empty(max_segments, dtype=float)
    seg_id = np.empty(max_segments, dtype=np.int64)
    seg_lay = np.empty(max_segments, dtype=np.int16)

    elastic_model_code = 1 if elastic_model == "mott" else 0

    if groove is None:
        nseg, n_back, n_trans, n_side = _transport_core_ungrooved(
            Ne,
            alive,
            max_steps,
            max_segments,
            n_layers,
            internal_bounds,
            elastic_model_code,
            z_total,
            finite_footprint,
            0.0 if width_ang is None else float(width_ang),
            0.0 if height_ang is None else float(height_ang),
            clock,
            rng,
            pos,
            dirs,
            E_cut_keV,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_ncm3,
            L_top,
            L_bot,
            mott_has_table,
            mott_start,
            mott_len,
            mott_logE_flat,
            mott_logA_flat,
            E_keV,
            seg_dir,
            seg_mid,
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    else:
        vac_start_buf = np.empty((max_vac, 3), dtype=float)
        vac_end_buf = np.empty((max_vac, 3), dtype=float)
        vac_E_buf = np.empty(max_vac, dtype=float)
        vac_t0_buf = np.empty(max_vac, dtype=float)
        vac_id_buf = np.empty(max_vac, dtype=np.int64)

        tp = float(groove.tilt_polar_rad)
        groove_st = float(np.sin(tp))
        groove_ct = float(np.cos(tp))
        nseg, nvac, n_back, n_trans, n_side = _transport_core_grooved(
            Ne,
            alive,
            max_steps,
            max_segments,
            max_vac,
            n_layers,
            internal_bounds,
            elastic_model_code,
            z_total,
            finite_footprint,
            0.0 if width_ang is None else float(width_ang),
            0.0 if height_ang is None else float(height_ang),
            float(groove.spacing_ang),
            float(groove.depth_ang),
            groove_st,
            groove_ct,
            clock,
            rng,
            pos,
            dirs,
            E_cut_keV,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_ncm3,
            L_top,
            L_bot,
            mott_has_table,
            mott_start,
            mott_len,
            mott_logE_flat,
            mott_logA_flat,
            E_keV,
            seg_dir,
            seg_mid,
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            vac_start_buf,
            vac_end_buf,
            vac_E_buf,
            vac_t0_buf,
            vac_id_buf,
        )
        vac_start = vac_start_buf[:nvac]
        vac_end = vac_end_buf[:nvac]
        vac_E = vac_E_buf[:nvac]
        vac_t0 = vac_t0_buf[:nvac]
        vac_id = vac_id_buf[:nvac]

    r_mid = seg_mid[:nseg]
    v_hat = seg_dir[:nseg]
    L_ang = seg_len[:nseg]
    E_seg = seg_E[:nseg]
    t_ang = seg_t0[:nseg]
    elec_id = seg_id[:nseg]
    layer = seg_lay[:nseg]

    vacuum_start_ang = vac_start
    vacuum_end_ang = vac_end
    vacuum_E_keV = vac_E
    vacuum_t_ang = vac_t0
    vacuum_elec_id = vac_id

    t0_ang = t0_electron[elec_id] if elec_id.size else np.empty(0, dtype=float)
    vacuum_t0_ang = t0_electron[vacuum_elec_id] if vacuum_elec_id.size else np.empty(0, dtype=float)

    return {
        # Initial sampled phase space is diagnostic-only.  Keep per-electron
        # arrays (including missed entries), separate from per-segment arrays,
        # so beam metrics describe the incident bunch rather than its transport.
        "initial_r_ang": initial_r_ang,
        "initial_v_hat": initial_v_hat,
        "initial_E_keV": initial_E_keV,
        "initial_t0_ang": t0_electron.copy(),
        "r_mid": r_mid,
        "v_hat": v_hat,
        "L_ang": L_ang,
        "E_keV": E_seg,
        "t_ang": t_ang,  # segment-start age sum(L/beta) [Ang, c=1]
        "t0_ang": t0_ang,  # per-electron longitudinal bunch offset [Ang, c=1]
        "elec_id": elec_id,  # emitting electron index in [0, Ne)
        "layer": layer,  # emitting layer index in [0, n_layers)
        "vacuum_start_ang": vacuum_start_ang,
        "vacuum_end_ang": vacuum_end_ang,
        "vacuum_E_keV": vacuum_E_keV,
        "vacuum_t_ang": vacuum_t_ang,
        "vacuum_t0_ang": vacuum_t0_ang,
        "vacuum_elec_id": vacuum_elec_id,
        "n_backscattered": n_back,
        "n_transmitted": n_trans,
        "n_side_exited": n_side,
        "n_missed": n_missed,
        "n_stopped": int(Ne - n_back - n_trans - n_side - n_missed),
        "Ne": Ne,
        "thickness_ang": z_total,
        "crystal_width_ang": width_ang,
        "crystal_height_ang": height_ang,
        "n_layers": n_layers,
    }
