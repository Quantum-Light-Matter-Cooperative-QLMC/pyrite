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
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numba import float64, int64, njit, uint64

from .. import DATA_DIR
from .._compat import env_value
from ..materials._transport_data import STERNHEIMER_DENSITY_EFFECT, TRANSPORT_ELEMENTS
from ..materials.attenuation import _normalize_composition
from ..transverse import resolved_from_mapping, sample_transverse
from .geometry import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    beam_frame_basis,
    project_beam_entry,
    validate_transverse_dimensions,
)
from .groove import _first_surface_event_scalar_numba, entry_points

# No array-backend import is needed here.
# Electron transport itself is NumPy/Numba CPU code.

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


beta_from_keV = _beta_array


@njit(cache=True)
def beta_from_keV_scalar(E_i):
    g = 1.0 + E_i / 510.99895
    g_inv_square = 1.0 / (g * g)
    return (1.0 - g_inv_square) ** 0.5


# ---- counter-based per-electron RNG -------------------------------------------
# The lockstep core draws every random number from one shared `Generator`, so its
# stream order is "step-major, electron-minor" and cannot be reproduced by
# threads that run electrons to completion independently. The per-electron core
# below and its CUDA port instead address randomness by (seed, electron, draw
# index) through a SplitMix64 counter hash. Nothing is carried between draws, so
# a stream is replayable: re-running a batch after a capacity overflow, changing
# the batch size, or changing the CUDA launch geometry all give identical
# numbers. That is what lets the GPU kernel be bit-for-bit against a CPU
# reference rather than merely statistically similar.
#
# SplitMix64's finalizer is a bijection on 64 bits and passes BigCrush as a
# counter-mode generator (Steele, Lea & Flood, OOPSLA 2014). Streams are keyed
# through the same finalizer so adjacent electron ids do not produce correlated
# sequences.

_SM64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = np.uint64(0x94D049BB133111EB)
_SM64_S27 = np.uint64(27)
_SM64_S30 = np.uint64(30)
_SM64_S31 = np.uint64(31)
_SM64_S11 = np.uint64(11)
_SM64_ZERO = np.uint64(0)
_SM64_ONE = np.uint64(1)
# 2**-53: the same 53-bit mantissa scaling numpy's `random()` uses, so draws land
# in [0, 1) with uniform spacing.
_U53_SCALE = 1.0 / 9007199254740992.0


# Signatures are explicit because numba otherwise unifies these expressions to
# int64, which turns every `>>` into an arithmetic shift and silently biases the
# generator (draws lose their top bit).
@njit(uint64(uint64), cache=True)
def _splitmix64(x):
    """SplitMix64 finalizer, used here as a counter-based hash."""
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


@njit(uint64(int64, int64), cache=True)
def _stream_key_scalar(seed, elec_id):
    """Per-electron stream key. Independent of batch size and launch geometry."""
    return _splitmix64(np.uint64(seed) + _SM64_GOLDEN * (np.uint64(elec_id) + _SM64_ONE))


@njit(float64(uint64, uint64), cache=True)
def _stream_uniform_scalar(key, counter):
    """Draw ``counter`` of the stream identified by ``key``, in [0, 1)."""
    z = _splitmix64(key + _SM64_GOLDEN * (counter + _SM64_ONE))
    return np.float64(z >> _SM64_S11) * _U53_SCALE


def stream_keys(seed, Ne):
    """Per-electron stream keys for ``[0, Ne)``, as consumed by both cores.

    Keys are built on the host so the CUDA kernel needs no 64-bit integer casts
    in device code, and so both cores provably address the same streams.
    """
    e = np.arange(Ne, dtype=np.uint64)
    x = np.uint64(seed) + _SM64_GOLDEN * (e + _SM64_ONE)
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


# ---- counter-based straggling RNG stream (slice D) ----------------------------
# The Urban sampler (slice C) draws a variable number of uniforms per flight --
# a Poisson channel with a nonzero count consumes exactly one, an empty one
# consumes zero, and each continuum quantum consumes one more (see
# `_urban_poisson_scalar`, `_urban_sample_element_keV`). Sharing the electron's
# own counter (the free-path / scattering-angle stream above) would therefore
# make those draws' addressing depend on whether straggling is on, which is
# exactly the coupling the "off path is bit-for-bit" and "on path never
# perturbs the existing draws" requirements forbid. So straggling gets a
# disjoint key domain instead of a shared counter:
#
#   urban_key  = _urban_stream_key_scalar(stream_key)                  -- once
#                                                                    per electron
#   flight_key = _urban_flight_key_scalar(urban_key, flight, substep) -- once
#                                                                    per flight
#
# and the sampler draws from `(flight_key, counter=0)`. Every `(seed, electron,
# flight, substep)` tuple addresses its own SplitMix64 stream with no shared
# state and no stride to overflow, so the sampler's own draw count can vary
# freely per flight without perturbing anything else -- including itself on a
# later flight. `_URBAN_STREAM_SALT` only has to differ from the other
# constants in this module; XORing it into the electron's stream key before
# re-hashing is what keeps this domain disjoint from `_stream_uniform_scalar`'s.
_URBAN_STREAM_SALT = np.uint64(0xD6E8FEB86659FD93)


@njit(uint64(uint64), cache=True)
def _urban_stream_key_scalar(stream_key):
    """Per-electron straggling key, disjoint from the electron's own stream."""
    return _splitmix64(stream_key ^ _URBAN_STREAM_SALT)


@njit(uint64(uint64, int64, int64), cache=True)
def _urban_flight_key_scalar(urban_key, flight, substep):
    """Per-``(flight, substep)`` straggling key.

    ``(flight, substep)`` pack into one 64-bit index by a 32/32 bit split --
    exact and collision-free for any flight count or substep count below
    2**32, far beyond any reachable ``max_steps`` or ``max_dE_frac`` substep
    count, and simpler than bounding a stride the way a shared counter would
    need. The result is re-hashed through the same SplitMix64 finalizer as
    every other stream in this module.
    """
    combined = (np.uint64(flight) << np.uint64(32)) + np.uint64(substep)
    return _splitmix64(urban_key + _SM64_GOLDEN * combined)


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
            logger.debug(
                "no Mott transport table for %s; transport will use the analytic fallback",
                element,
            )
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


# ---- Berger-Seltzer / ICRU-37 relativistic collision stopping -----------------
# The relativistic replacement for the Joy-Luo law above:
#
#   -(1/rho) dE/dx = (2 pi r_e^2 m c^2 N_A / beta^2) (Z/A)
#                    [ ln(tau^2 (tau + 2) / (2 (I/mc^2)^2)) + F^-(tau) - delta ]
#   F^-(tau)       = 1 - beta^2 + [tau^2/8 - (2 tau + 1) ln 2] / (tau + 1)^2
#
# with tau = E/mc^2 and 2 pi r_e^2 m c^2 N_A = 0.1535 MeV cm^2/mol. The
# per-element coefficient c_i = n_i Z_i / 0.602214076 is an exact rewrite of
# rho Z/A (eq-stopping-compound-coefficient) and carries over unchanged, so
# Bragg additivity has the same form as Joy-Luo.
#
# 0.1535 MeV cm^2/mol against c_i in mol/cm^3 gives MeV/cm; 1e3 keV/MeV times
# 1e-8 cm/Angstrom is the 1e-5 that makes the prefactor 1.535e-6 keV/Angstrom.
# That pins it against the Joy-Luo path: non-relativistically beta^2 -> 2E/mc^2
# and (tau+2)/2 -> 1, so 1.535e-6/beta^2 * ln(E^2/I^2) -> (7.844e-4/E) ln(E/I)
# against the 7.85e-4 of eq-stopping-joy-luo -- 0.08%, the rounding in the
# conventional constant.
#
# delta is the density-effect correction. It is a bulk property of the medium
# rather than of an element, so it factors out of the Bragg sum and enters as
# one scalar per layer. Callers pass 0.0 until Sternheimer parameters land; over
# 1-300 keV beta*gamma stays below 1.24 and the term is bounded well under the
# Joy-Luo error it replaces.
_BS_PREFACTOR = 1.535e-6  # keV Angstrom^-1 per (mol cm^-3), from 0.1535 MeV cm^2/mol
_MC2_KEV = 510.99895
_LN2 = 0.6931471805599453


def _bs_tau_terms(E_keV):
    """Return ``(tau, beta^2, F^-(tau))`` -- the element-independent factors."""
    tau = E_keV / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    return tau, beta_sq, f_minus


def _dEds_bs_keV_per_ang(Z, A, J_keV, rho_g_cm3, E_keV, delta=0.0):
    """Berger-Seltzer collision stopping power [keV/Angstrom] (negative).

    Elemental form, mirroring :func:`_dEds_keV_per_ang`. Shell corrections are
    omitted, which is why NIST restricts ESTAR collision stopping to >=10 keV;
    the low-energy branch stays with Joy-Luo.
    """
    tau, beta_sq, f_minus = _bs_tau_terms(E_keV)
    I_rel = J_keV / _MC2_KEV
    log_term = np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel))
    return -_BS_PREFACTOR * rho_g_cm3 * Z / (A * beta_sq) * (log_term + f_minus - delta)


@njit(cache=True)
def _dEds_bs_compound_scalar(J_arr, coeff_arr, delta, E_i):
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    total = 0.0
    for i in range(J_arr.size):
        I_rel = J_arr[i] / _MC2_KEV
        total += coeff_arr[i] * (
            np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
        )

    return -_BS_PREFACTOR / beta_sq * total


@njit(cache=True)
def _dEds_bs_packed_scalar(L_Js, L_coeffs, L_deltas, L, n_el, E_i):
    """Berger-Seltzer stopping power read from the padded per-layer tables.

    Same arithmetic as :func:`_dEds_bs_compound_scalar`, but indexing the
    ``(n_layers, max_elements)`` rows the per-electron and CUDA cores use, with
    the per-layer density-effect scalar from ``L_deltas``.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    delta = L_deltas[L]

    total = 0.0
    for i in range(n_el):
        I_rel = L_Js[L, i] / _MC2_KEV
        total += L_coeffs[L, i] * (
            np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
        )
    return -_BS_PREFACTOR / beta_sq * total


@njit(cache=True)
def _dEds_bs_compound(J_arr, coeff_arr, delta, E_keV):
    out = np.empty_like(E_keV)

    for j in range(E_keV.size):
        E = E_keV[j]
        tau = E / _MC2_KEV
        gamma = 1.0 + tau
        beta_sq = 1.0 - 1.0 / (gamma * gamma)
        f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

        total = 0.0
        for i in range(J_arr.size):
            I_rel = J_arr[i] / _MC2_KEV
            total += coeff_arr[i] * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

        out[j] = -_BS_PREFACTOR / beta_sq * total
    return out


# ---- Joy-Luo / Berger-Seltzer splice ------------------------------------------
# Joy-Luo is kept as the low-energy branch: Berger-Seltzer omits shell
# corrections (why NIST restricts ESTAR collision stopping to >=10 keV) and the
# default E_cut_keV is 5 keV, so the low-energy branch is load-bearing.
#
# The crossover is strongly Z-dependent, so there is no good *global* splice
# energy. Measured over the 24 catalog elements the two laws cross at 2.66 keV
# (B) through 10.46 keV (Bi), monotone in I. Forcing one global energy leaves a
# step: the best available choice is 8.0 keV, which still mismatches by 1.84% in
# the worst element. The 2% agreement at 10 keV quoted in stopping-power.md is a
# carbon number and does not generalize.
#
# Because stopping is additive over elements, the splice does not have to be
# global. Each element's contribution is switched at *its own* crossover, so
# every term is continuous by construction and therefore so is the compound, for
# any material, with no per-material tuning and no fitted blend. Each element
# crosses exactly once above the energy where the Berger-Seltzer bracket is
# still positive (that zero sits below 0.71 keV for every catalog element, far
# under any crossover), so the crossover is well defined.
#
# The splice is C0, not C1: the log-slope d ln|dE/ds| / d ln E steps across it by
# 0.0145 for B (2.0% of the local slope) up to 0.0587 for Bi (8.9%), worst at
# high Z where the crossover sits highest. The midpoint solve
# (eq-stopping-cutoff-distance) only needs the value, but a kink inside the
# bracketing interval is the failure mode to check when the cores adopt this.


def _bs_joy_luo_crossover_keV(Z, A, J_keV):
    """Energy [keV] where Joy-Luo and Berger-Seltzer stopping powers are equal.

    Bisection rather than a closed form: the two laws differ in the shape of the
    logarithm, not just a constant. The result depends only on ``Z`` and
    ``J_keV`` (density cancels in the ratio), so it is a per-element constant
    computed once at table-build time.
    """
    Z = float(Z)
    A = float(A)
    J_keV = float(J_keV)

    def difference(E):
        return _dEds_keV_per_ang(Z, A, J_keV, 1.0, E) - _dEds_bs_keV_per_ang(Z, A, J_keV, 1.0, E)

    # Start above the zero of the Berger-Seltzer bracket, below which that law is
    # not physically signed and the difference is meaningless. That zero sits at
    # ~0.86 J for every catalog element, while the crossover is at ~35 J, so 2 J
    # brackets from below with room on both sides.
    lo = 2.0 * J_keV
    hi = 300.0
    if difference(lo) * difference(hi) > 0.0:
        raise ValueError(f"no Joy-Luo/Berger-Seltzer crossover in [{lo}, {hi}] keV for Z={Z}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if difference(lo) * difference(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_i):
    """Stopping power with each element on its own side of its own crossover."""
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = 0.0
    bs_total = 0.0
    for i in range(J_arr.size):
        J = J_arr[i]
        coeff = coeff_arr[i]
        if E_i < E_cross_arr[i]:
            joy_luo_total += coeff * np.log(1.166 * (E_i + k_arr[i] * J) / J)
        else:
            I_rel = J / _MC2_KEV
            bs_total += coeff * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

    return -7.85e-4 / E_i * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


@njit(cache=True)
def _dEds_spliced_compound(J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_keV):
    out = np.empty_like(E_keV)
    for j in range(E_keV.size):
        out[j] = _dEds_spliced_compound_scalar(
            J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_keV[j]
        )
    return out


@njit(cache=True)
def _dEds_spliced_packed_scalar(L_Js, L_ks, L_coeffs, L_E_cross, delta, L, n_el, E_i):
    """Spliced stopping power read from the padded per-layer tables.

    Same arithmetic as :func:`_dEds_spliced_compound_scalar`, but indexing the
    ``(n_layers, max_elements)`` rows the per-electron and CUDA cores use. The
    padding is zero-filled and not a valid element, so ``n_el`` bounds the loop.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = 0.0
    bs_total = 0.0
    for i in range(n_el):
        J = L_Js[L, i]
        coeff = L_coeffs[L, i]
        if E_i < L_E_cross[L, i]:
            joy_luo_total += coeff * np.log(1.166 * (E_i + L_ks[L, i] * J) / J)
        else:
            I_rel = J / _MC2_KEV
            bs_total += coeff * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

    return -7.85e-4 / E_i * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


# The crossover is a bisection, far too expensive for a hot loop, and it depends
# only on (Z, J) -- A and rho cancel in the ratio because both laws carry the
# same rho Z/A. Solve it once per element and reuse it for every layer that
# contains that element.
_CROSSOVER_CACHE: dict[str, float] = {}


def _element_crossover_keV(element, Z, A, J_keV):
    """Memoized Joy--Luo/Berger--Seltzer crossover energy [keV] for one element."""
    cached = _CROSSOVER_CACHE.get(element)
    if cached is None:
        cached = _bs_joy_luo_crossover_keV(Z, A, J_keV)
        _CROSSOVER_CACHE[element] = cached
    return cached


def spliced_stopping_keV_per_ang(composition, E_keV):
    """Spliced stopping power [keV/Angstrom] (negative) for a normalized composition.

    The host-side entry point for code outside the transport cores that needs
    the *same* model the cores evaluate: the cost proxy in ``campaign.sweep``
    and the frozen-rule replay in ``spectrum.diagnostics``. Both previously
    carried their own copy of the Joy--Luo constants and drifted the moment the
    model changed; delegating here is what keeps them honest.

    ``composition`` is the ``(element, n_i)`` sequence used everywhere else,
    with ``n_i`` in Angstrom^-3. Plain NumPy, no Numba: these are cold paths and
    should not pay a compile.
    """
    E = np.asarray(E_keV, dtype=float)
    tau = E / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = np.zeros_like(E)
    bs_total = np.zeros_like(E)
    for element, n_i in composition:
        params = TRANSPORT_ELEMENTS[element]
        Z = float(params["Z"])
        A = float(params["A"])
        J = float(params["J_keV"])
        k = 0.731 + 0.0688 * np.log10(Z)
        coeff = (n_i / 0.602214076) * Z
        I_rel = J / _MC2_KEV
        use_bs = E >= _element_crossover_keV(element, Z, A, J)
        joy_luo_total += np.where(use_bs, 0.0, coeff * np.log(1.166 * (E + k * J) / J))
        bs_total += np.where(
            use_bs,
            coeff * (np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus),
            0.0,
        )

    return -7.85e-4 / E * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


# ---- Urban energy-loss fluctuation model --------------------------------------
# Source: Geant4 Physics Reference Manual, "Energy loss fluctuations", Urban
# model (``G4UniversalFluctuation``), after Bichsel, Rev. Mod. Phys. 60, 663
# (1988). Selected in preference to Landau, Vavilov and Bohr because those three
# each *derive* the mean from their own xi and would therefore contradict the
# Joy--Luo/Berger--Seltzer splice above -- Urban instead takes C = |dE/dx| as its
# normalisation, so whatever mean the transport already computes is reproduced
# exactly. The measured regime forced that choice: per flight the process runs
# 0.003--0.16 close collisions and kappa = 2e-5--2.9e-2, and even integrated over
# the whole CSDA range kappa stays at 0.08--0.16, so neither the Landau nor the
# Gaussian limit is admissible anywhere in the catalog.
#
# The atom is modelled as two excitation levels plus an ionisation continuum:
#
#   E_0 = 10 eV                         ionisation threshold of the continuum
#   E_2 = 10 Z^2 eV,  f_2 = 2/Z         K shell (level ~ binding energy, Z f_2 = 2
#                                       recovers its occupancy)
#   f_1 = 1 - f_2,    f_1 ln E_1 + f_2 ln E_2 = ln I      loosely bound level
#   T_up = T_max = E/2                  Moller ceiling; unrestricted, no delta cut
#   r    = 0.55                         share of the mean carried by the continuum
#
#   Sigma_i = C (f_i/E_i) [ln(2 mc^2 (beta gamma)^2 / E_i) - beta^2]
#                        / [ln(2 mc^2 (beta gamma)^2 / I) - beta^2] (1 - r),  i = 1,2
#   Sigma_3 = C r (T_up - E_0) / (E_0 T_up ln(T_up/E_0))
#
# The loss over a step ``s`` is the compound Poisson sum
#
#   dE = n_1 E_1 + n_2 E_2 + sum_{k=1}^{n_3} E_k,   n_i ~ Poisson(s Sigma_i)
#
# with the continuum quanta drawn from the 1/E^2 spectrum on [E_0, T_up] by its
# exact inverse CDF, E_k = E_0 / (1 - u (T_up - E_0)/T_up), u ~ U[0,1).
#
# Units: C and Sigma_i E_i are keV/Angstrom, E_i and I are keV, s is Angstrom,
# so <n_i> = s Sigma_i is dimensionless and dE is keV. C is |dE/dx| -- the
# stopping powers in this module are negative, and the sampler returns a
# POSITIVE loss, matching the ``E -= |dEds| * s`` convention of the cores.
#
# Mean (the closure identity, and the reason this model was selected):
#   sum_{i=1,2} Sigma_i E_i = C (1-r)/L_I [f_1 L(E_1) + f_2 L(E_2)]
#                           = C (1-r)/L_I [ln(2 mc^2 (beta gamma)^2) - beta^2 - ln I]
#                           = C (1-r)
# using f_1 + f_2 = 1 and the log sum rule, and
#   Sigma_3 <E>_3 = C r,   <E>_3 = E_0 T_up ln(T_up/E_0) / (T_up - E_0),
# so <dE> = C s exactly, for any r, any Z and any I.
#
# Variance: a compound Poisson sum has no cross terms, so
#   Var(dE) = s (Sigma_1 E_1^2 + Sigma_2 E_2^2 + Sigma_3 E_0 T_up),
# using <E^2>_3 = E_0 T_up exactly for the 1/E^2 spectrum on [E_0, T_up].
#
# Limiting case: as s -> 0 every <n_i> -> 0, so P(dE = 0) -> 1 and both moments
# vanish linearly in s. The loss degenerates to the deterministic C s of the
# present cores in mean at every s, and in distribution as s -> 0.
#
# Assumptions and known limits, measured in slice B and accepted rather than
# tuned out:
#   - The PRM's own shape floor ("mean loss at least a few multiples of I") is
#     violated in nearly every catalog cell (dE/I per flight 0.017--2.58). Only
#     the first two moments are trusted here; the shape is not.
#   - Variance against the analytic Moller second moment xi T_max runs 0.73--1.42
#     over the catalog (sigma within -14%/+19%). The PRM's width correction is
#     deliberately NOT applied.
#   - A sampled loss may exceed E_i (E_1 can sit above T_up after the re-solve at
#     low E, and n_3 can exceed 1). Clamping would break the mean, so the sampler
#     does not clamp; the cutoff-crossing bookkeeping owns that.
_URBAN_E0_KEV = 1.0e-2  # ionisation level E_0 = 10 eV
_URBAN_E2_KEV_PER_Z2 = 1.0e-2  # E_2 = 10 Z^2 eV
_URBAN_RATE = 0.55  # r, the model's single tuned parameter
# Above this Poisson mean the inverse-CDF search is replaced by its Gaussian
# limit. Both branches consume a fixed number of uniforms so the stream advances
# deterministically; the regime this model was selected for keeps <n_i> under
# ~2.5 per flight, so the threshold is a guard for oversized steps, not a path
# the transport takes.
_URBAN_POISSON_GAUSS_MIN = 100.0


@njit(cache=True)
def _urban_levels_scalar(Z, J_keV, T_up, two_mc2_bg2, beta_sq):
    """Urban's ``(f_1, E_1, f_2, E_2)`` for one element [keV], with the re-solve.

    The K-shell level ``E_2 = 10 Z^2`` eV is a *parameterisation*, not a measured
    binding energy, and PyRITE runs it far below the energies Geant4 does. It is
    inadmissible when it sits above the Moller ceiling ``T_up = E/2`` (i.e. below
    ``E = 20 Z^2`` eV: C below 0.72 keV, Si below 3.92 keV, S below 5.12 keV, W
    below 109.5 keV) or when its logarithmic factor has gone non-positive. Geant4
    never reaches either boundary and simply clamps a negative count to zero;
    clamping here would DELETE a negative contribution and make the mean
    overshoot -- measured closure 1.0187/1.0098/1.0038/1.0010 for W at 1/2/5/10
    keV. Instead the sum rules are re-solved on the single remaining level,
    ``f_1 = 1``, ``E_1 = I``, which satisfies ``f_1 ln E_1 = ln I`` identically
    and so restores <dE> = C s exactly.
    """
    E_2 = _URBAN_E2_KEV_PER_Z2 * Z * Z
    f_2 = 2.0 / Z if Z > 2.0 else 1.0
    if Z > 2.0 and E_2 < T_up and np.log(two_mc2_bg2 / E_2) - beta_sq > 0.0:
        f_1 = 1.0 - f_2
        E_1 = np.exp((np.log(J_keV) - f_2 * np.log(E_2)) / f_1)
        if np.log(two_mc2_bg2 / E_1) - beta_sq > 0.0:
            return f_1, E_1, f_2, E_2
    return 1.0, J_keV, 0.0, E_2


@njit(cache=True)
def _urban_channels_scalar(Z, J_keV, C_keV_per_ang, E_i):
    """Urban's per-unit-length channel rates for one element.

    Returns ``(valid, Sigma_1, E_1, Sigma_2, E_2, Sigma_3)`` with ``Sigma_i`` in
    Angstrom^-1 and the levels in keV. ``C_keV_per_ang`` is ``|dE/dx|`` for this
    element alone -- the model is applied per element, matching the Bragg-additive
    form of the mean stopping power, which is what keeps the total mean equal to
    the transport's own ``dE/dx``.

    ``valid`` is 0.0 where the parameterisation has no admissible form at all:
    the continuum needs ``T_up > E_0`` (E above 20 eV) and every level needs a
    positive logarithmic factor, which fails once ``2 mc^2 (beta gamma)^2 <= I``
    (below ~0.18 keV for tungsten). Callers fall back to the deterministic loss
    there rather than sample a degenerate distribution.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    two_mc2_bg2 = 2.0 * _MC2_KEV * tau * (tau + 2.0)
    T_up = 0.5 * E_i

    if T_up <= _URBAN_E0_KEV or C_keV_per_ang <= 0.0:
        return 0.0, 0.0, J_keV, 0.0, 0.0, 0.0

    L_I = np.log(two_mc2_bg2 / J_keV) - beta_sq
    if L_I <= 0.0:
        return 0.0, 0.0, J_keV, 0.0, 0.0, 0.0

    f_1, E_1, f_2, E_2 = _urban_levels_scalar(Z, J_keV, T_up, two_mc2_bg2, beta_sq)
    soft = C_keV_per_ang * (1.0 - _URBAN_RATE) / L_I
    sigma_1 = soft * (f_1 / E_1) * (np.log(two_mc2_bg2 / E_1) - beta_sq)
    sigma_2 = 0.0
    if f_2 > 0.0:
        sigma_2 = soft * (f_2 / E_2) * (np.log(two_mc2_bg2 / E_2) - beta_sq)
    sigma_3 = (
        C_keV_per_ang
        * _URBAN_RATE
        * (T_up - _URBAN_E0_KEV)
        / (_URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV))
    )
    return 1.0, sigma_1, E_1, sigma_2, E_2, sigma_3


@njit(cache=True)
def _urban_moments_element_scalar(Z, J_keV, C_keV_per_ang, E_i, s_ang):
    """Analytic ``(mean, variance)`` of the Urban loss for one element [keV, keV^2].

    The mean is ``C s`` by the closure identity above; it is evaluated from the
    sampled channels rather than shortcut to ``C s`` so that the identity is what
    the tests measure.
    """
    valid, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(
        Z, J_keV, C_keV_per_ang, E_i
    )
    if valid == 0.0:
        return C_keV_per_ang * s_ang, 0.0
    T_up = 0.5 * E_i
    mean_3 = _URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV) / (T_up - _URBAN_E0_KEV)
    mean = s_ang * (sigma_1 * E_1 + sigma_2 * E_2 + sigma_3 * mean_3)
    var = s_ang * (sigma_1 * E_1 * E_1 + sigma_2 * E_2 * E_2 + sigma_3 * _URBAN_E0_KEV * T_up)
    return mean, var


@njit(cache=True)
def _urban_poisson_scalar(lam, key, counter):
    """Poisson variate with mean ``lam``, returning ``(n, counter)``.

    Inverse CDF by the recurrence ``p_{k+1} = p_k lam/(k+1)``, which consumes
    exactly one uniform however large ``n`` comes out -- a counter-addressed
    stream must advance by an amount known to the caller, and Knuth's product
    method would advance it by ``n + 1``. Above ``lam = 100`` the recurrence
    starts from ``exp(-lam) ~ 1e-44`` and costs O(lam) iterations, so it hands
    over to the Gaussian limit (skewness ``lam^-1/2`` <= 0.1 there), which
    consumes two uniforms through Box-Muller.
    """
    if lam <= 0.0:
        return 0, counter
    if lam < _URBAN_POISSON_GAUSS_MIN:
        u = _stream_uniform_scalar(key, counter)
        counter = counter + _SM64_ONE
        p = np.exp(-lam)
        cdf = p
        k = 0
        while u >= cdf and k < 10000:
            k += 1
            p = p * lam / k
            cdf += p
        return k, counter
    u1 = _stream_uniform_scalar(key, counter)
    u2 = _stream_uniform_scalar(key, counter + _SM64_ONE)
    counter = counter + _SM64_ONE + _SM64_ONE
    # u1 is in [0, 1); shift it off the log's singularity.
    z = np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
    n = int(np.floor(lam + np.sqrt(lam) * z + 0.5))
    if n < 0:
        n = 0
    return n, counter


@njit(cache=True)
def _urban_ionisation_keV(u, T_up):
    """Inverse CDF of the ``1/E^2`` continuum on ``[E_0, T_up]`` [keV].

    ``F(E) = (E_0 T_up/(T_up - E_0)) (1/E_0 - 1/E)``, so ``u = 0`` returns ``E_0``
    and ``u -> 1`` returns ``T_up``. The denominator stays in ``(E_0/T_up, 1]``.
    """
    return _URBAN_E0_KEV / (1.0 - u * (T_up - _URBAN_E0_KEV) / T_up)


@njit(cache=True)
def _urban_sample_element_keV(Z, J_keV, C_keV_per_ang, E_i, s_ang, key, counter):
    """Sample the Urban energy loss of one element over ``s_ang``, ``(dE, counter)``.

    ``dE`` is a positive loss in keV. Where the parameterisation has no admissible
    form the deterministic ``C s`` is returned instead, which keeps the mean exact
    and the fallback silent in every moment test.
    """
    valid, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(
        Z, J_keV, C_keV_per_ang, E_i
    )
    if valid == 0.0 or s_ang <= 0.0:
        return C_keV_per_ang * s_ang, counter

    n_1, counter = _urban_poisson_scalar(sigma_1 * s_ang, key, counter)
    n_2, counter = _urban_poisson_scalar(sigma_2 * s_ang, key, counter)
    n_3, counter = _urban_poisson_scalar(sigma_3 * s_ang, key, counter)

    dE = n_1 * E_1 + n_2 * E_2
    T_up = 0.5 * E_i
    for _ in range(n_3):
        u = _stream_uniform_scalar(key, counter)
        counter = counter + _SM64_ONE
        dE += _urban_ionisation_keV(u, T_up)
    return dE, counter


@njit(cache=True)
def _dEds_spliced_element_scalar(J, k, coeff, E_cross, delta, E_i):
    """One element's contribution to the spliced stopping power [keV/Angstrom].

    A deliberate duplicate of the body of :func:`_dEds_spliced_compound_scalar`
    rather than a factoring of it: that function accumulates the Joy--Luo and
    Berger--Seltzer sums separately and combines them once, and re-associating
    the sum would move the last bits of every existing transport result. The two
    agree to float64 rounding, which is what the closure test asserts.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    if E_i < E_cross:
        return -7.85e-4 / E_i * coeff * np.log(1.166 * (E_i + k * J) / J)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    I_rel = J / _MC2_KEV
    return (
        -_BS_PREFACTOR
        / beta_sq
        * coeff
        * (np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta)
    )


@njit(cache=True)
def _urban_sample_compound_keV(
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_i, s_ang, key, counter
):
    """Sample the Urban loss of a compound over ``s_ang`` [keV], ``(dE, counter)``.

    Bragg-additive, one independent Urban draw per element with that element's own
    ``C_i = |dE/dx|_i``, ``Z_i`` and ``I_i``. Summing the per-element means
    reproduces the compound stopping power, so the compound closure is inherited
    from the elemental one rather than asserted separately.
    """
    dE = 0.0
    for i in range(Z_arr.size):
        C_i = -_dEds_spliced_element_scalar(
            J_arr[i], k_arr[i], coeff_arr[i], E_cross_arr[i], delta, E_i
        )
        loss, counter = _urban_sample_element_keV(Z_arr[i], J_arr[i], C_i, E_i, s_ang, key, counter)
        dE += loss
    return dE, counter


def urban_element_table(composition):
    """``(Z, J_keV, k, coeff, E_cross)`` arrays for a normalized composition.

    The host-side table the Urban samplers index, built from the same
    :data:`TRANSPORT_ELEMENTS` entries and the same memoized crossover as
    :func:`spliced_stopping_keV_per_ang`, so the two cannot drift.
    """
    Z_arr = np.empty(len(composition))
    J_arr = np.empty(len(composition))
    k_arr = np.empty(len(composition))
    coeff_arr = np.empty(len(composition))
    E_cross_arr = np.empty(len(composition))
    for i, (element, n_i) in enumerate(composition):
        params = TRANSPORT_ELEMENTS[element]
        Z = float(params["Z"])
        A = float(params["A"])
        J = float(params["J_keV"])
        Z_arr[i] = Z
        J_arr[i] = J
        k_arr[i] = 0.731 + 0.0688 * np.log10(Z)
        coeff_arr[i] = (n_i / 0.602214076) * Z
        E_cross_arr[i] = _element_crossover_keV(element, Z, A, J)
    return Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr


def urban_loss_moments_keV(composition, E_keV, s_ang, delta=0.0):
    """Analytic ``(mean, variance)`` of the Urban loss over ``s_ang`` [keV, keV^2].

    Cold path, for tests, validation instruments and the observable measurement
    that follows; the cores use the scalar helpers. The mean is the closure
    identity ``|dE/dx| s`` evaluated through the model's own channels, not
    shortcut to it.
    """
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
    mean = 0.0
    var = 0.0
    for i in range(Z_arr.size):
        C_i = -_dEds_spliced_element_scalar(
            J_arr[i], k_arr[i], coeff_arr[i], E_cross_arr[i], delta, float(E_keV)
        )
        m_i, v_i = _urban_moments_element_scalar(
            Z_arr[i], J_arr[i], C_i, float(E_keV), float(s_ang)
        )
        mean += m_i
        var += v_i
    return mean, var


def sternheimer_delta(element, E_keV):
    """Sternheimer density-effect correction delta for one element at ``E_keV``.

    Source: Sternheimer, Berger & Seltzer, *Atomic Data and Nuclear Data Tables*
    **30**, 261 (1984), in the PDG's presentation, with coefficients read from
    :data:`~pyrite.materials._transport_data.STERNHEIMER_DENSITY_EFFECT`. With
    ``x = log10(beta gamma)``,

        x < x_0:        delta = delta_0 * 10^(2 (x - x_0))
        x_0 <= x < x_1: delta = 2 ln(10) x - C_bar + a (x_1 - x)^k
        x >= x_1:       delta = 2 ln(10) x - C_bar

    delta_0 is zero for non-conductors, so the first branch vanishes for them.

    NO TRANSPORT PATH CALLS THIS. eq-stopping-bs carries delta as a parameter
    and every call site passes 0.0; this function exists to measure how much
    that omission costs, and is the instrument behind the bound quoted in
    stopping-power.md and pinned by ``test_stopping_density_effect.py``.

    Being per element it is not a compound's delta -- delta is a bulk property
    of the medium and does not Bragg-add. Applying it per element would be
    wrong; bounding a compound by its constituents is what it supports.
    """
    p = STERNHEIMER_DENSITY_EFFECT[element]
    E = np.asarray(E_keV, dtype=float)
    tau = E / _MC2_KEV
    gamma = 1.0 + tau
    x = np.log10(np.sqrt(1.0 - 1.0 / (gamma * gamma)) * gamma)
    asymptote = 2.0 * np.log(10.0) * x - p["C_bar"]
    return np.where(
        x < p["x0"],
        p["delta0"] * 10.0 ** (2.0 * (x - p["x0"])),
        asymptote + np.where(x < p["x1"], p["a"] * np.abs(p["x1"] - x) ** p["k"], 0.0),
    )


# Generation marker for the collision-stopping model every core evaluates. The
# splice is unconditional physics, not a selector, so this is a CONSTANT and
# follows the ``line_kinematics`` precedent: identity payloads and the per-case
# content key hash it, which moves every digest exactly once and orphans the
# records minted under the retired pure Joy--Luo model (rev-and-re-run) rather
# than letting them resume into -- or be served from the CAS for -- a run that
# computes different numbers. Bump it whenever the evaluated model changes:
# adding the density-effect term delta, or moving a crossover, is such a change.
STOPPING_MODEL = "joy-luo/berger-seltzer-splice"


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


# ---- stochastic energy loss in transport (slice E) ----------------------------
# Slice C built the Urban sampler and slice D addressed it per
# `(electron, flight, substep)` without applying it. This block is the design
# record for actually applying it inside `_transport_core_ungrooved`, i.e. for
# the two questions the deterministic core answers by construction and a random
# loss reopens: where the cutoff crossing is, and what `max_dE_frac` substepping
# still guarantees. Both are gated behind `straggle_on`; with straggling off
# every line below is unreachable and the deterministic code path is textually
# unchanged.
#
# Source: Geant4 PRM "Energy loss fluctuations" (Urban model) for the loss
# itself -- see the derivation block above `_urban_levels_scalar`. Nothing here
# adds physics to that model; it is the transport-side integration of it.
#
# --- 1. The loss over a flight is a subordinator, not just a random number ----
#
# Urban's loss over a step of length s at frozen energy is the compound Poisson
# sum dE(s) = sum_i sum_{k=1}^{n_i(s)} E_{i,k} with n_i(s) ~ Poisson(s Sigma_i).
# Read as a function of s it is a Levy process with non-negative jumps: a
# subordinator. Two of its properties do all the work below.
#
#   (P1) MONOTONE. dE(s) is non-decreasing in s, so the electron's energy
#        E(s) = E_start - dE(s) is non-increasing, exactly as in the
#        deterministic model. Therefore
#            inf{ s' <= s : E(s') <= E_cut }  exists  <=>  dE(s) >= E_start-E_cut.
#        The *indicator* of "this row crosses the cutoff" is a function of the
#        total loss over the row alone -- which is precisely what the sampler
#        returns. So the crossing decision, and hence `n_cutoff_stopped`, is
#        EXACT under this model: no approximation enters it.
#   (P2) INFINITELY DIVISIBLE. For any partition s = sum_m s_m,
#            sum_m CP(s_m Sigma) =_d CP(s Sigma),
#        because sum_m Poisson(s_m Sigma_i) = Poisson(s Sigma_i) and the marks
#        are i.i.d. from the same law. At frozen Sigma this is exact, not
#        asymptotic. It is the substep invariance, derived in 3 below.
#
# --- 2. Cutoff crossing: exact indicator, fluid-interpolated location ---------
#
# Let Delta = E_start - E_cut > 0 (every alive electron satisfies this; a row
# that reaches E_cut is killed) and let dE be the sampled loss over the row's
# length s. By (P1) the row crosses iff dE >= Delta, and the crossing distance
# is the position of the jump that carries the running sum past Delta. The
# sampler returns the total, not the jump ladder, so the *location* needs a
# rule. The one used here places the crossing where the loss, accrued at the
# row's own REALIZED average rate dE/s, reaches Delta:
#
#       s_cut = s * Delta / dE,        E_end = E_cut,        cutoff_j = True.
#
# Why this rule:
#   - It degenerates ALGEBRAICALLY, not merely in gate, to the deterministic
#     solve. Put dE -> |dE/ds| s (the zero-fluctuation limit): the crossing
#     condition becomes s > Delta/|dE/ds| = cutoff_distance and
#     s_cut = s Delta / (|dE/ds| s) = Delta/|dE/ds| = cutoff_distance, which is
#     the frozen-model line `cutoff_distance = (E_cut - E_j) / dEds` verbatim.
#   - It is the same approximation the surrounding transport already makes.
#     The deterministic core spreads a flight's loss uniformly along the flight
#     even though the loss is physically a handful of discrete collisions; the
#     clock (`s / beta`) and `seg_mid` are built on that fluid picture. Using
#     the realized rate instead of the mean rate changes which number is spread,
#     not the spreading.
#   - It handles the overshoot case -- slice C decision 3: the sampler does not
#     clamp dE to E, and with n_3 up to 1.15 per flight a single row can sample
#     a loss far above Delta -- with no special case and no unphysical result:
#     dE >> Delta gives s_cut -> 0, i.e. "the electron ran out of energy right
#     at the start of this row". E_end is E_cut exactly, never negative, never
#     below the cutoff.
#   - It consumes no additional random numbers, so slice D's stream layout,
#     its off-path bit-for-bit claim, and its offline reproducibility of
#     `straggle_dE_keV` from `(electron, flight, substep)` all survive unchanged.
#
# What it costs: the crossing LOCATION is biased inside the crossing row. The
# true first-passage distance is the position of the crossing jump, which given
# one jump is uniform on [0, s]; the rule returns the deterministic fraction
# Delta/dE of the row instead, so a large overshoot places the stop earlier than
# the truth. The bias is bounded by one row length and applies only to the row
# that terminates the track, so it perturbs the end of the range straggling
# distribution by at most the final flight length -- which at E ~ E_cut is the
# elastic mean free path at a few keV, Angstroms to tens of Angstroms.
#
# Alternatives considered and rejected:
#   (a) Travel the full row, then stop if E_end <= E_cut. Rejected: it does not
#       degenerate to the deterministic solve at all (in the zero-fluctuation
#       limit it still overshoots by s - cutoff_distance), and it lengthens
#       every terminated track by half a flight on average, which is a
#       systematic range bias present even with the fluctuation switched off.
#   (b) Draw the crossing position uniformly on [0, s]. Exact for a single-jump
#       crossing, but wrong for a multi-jump one, wrong in the deterministic
#       limit (it would randomize a stopping point that is not random), and it
#       consumes a stream draw whose count depends on the outcome.
#   (c) Clamp the sampled loss to Delta and keep the analytic cutoff distance.
#       Rejected: clamping breaks <dE> = C s, the single property Urban was
#       selected for (slice B point 4), and makes `n_cutoff_stopped` blind to
#       the fluctuation it is supposed to reflect.
#   (d) Sample the jump ladder (counts and uniform positions) to get the exact
#       first passage. Correct, but it requires the sampler to return per-
#       element counts and to draw n_i extra position variates -- i.e. changing
#       slice C's sampler, which slice E does not own.
#
# `n_cutoff_stopped` bookkeeping: `cutoff_j` keeps its exact meaning ("this row
# ended because the electron reached E_cut"), so the increment, the `died_j`
# kill, and the geometry-flag clearing are unchanged; only the test that sets it
# is redefined. By (P1) the flag fires on exactly the rows on which the true
# first passage lies inside the row, so the count is exact, not approximate.
#
# The energy model: under straggling the loss over a row is the sampled dE and
# E_end = E_start - dE for BOTH `energy_model` codes. The midpoint
# predictor-corrector is a second-order quadrature of the deterministic ODE
# dE/ds = f(E); with a random loss there is no ODE to quadrature and the
# sampler's own mean is the left-endpoint one, C(E_start) s. `energy_model`
# therefore still selects the clock's representative energy -- beta at the row's
# realized midpoint (E_start + E_end)/2 versus at E_start -- and the
# `seg_E_end`/`seg_t_end` schema, but no longer the energy update itself. The
# residual left-endpoint bias this leaves in the mean is exactly the O(s^2) term
# derived in 3 below, and `max_dE_frac` is the lever that controls it.
#
# --- 3. Substep invariance under a stochastic loss ----------------------------
#
# `substep-radiation-invariance` currently states an ALGEBRAIC invariance:
# subdividing a flight leaves the deterministic result unchanged. That claim
# does not survive a random loss and is re-derived here as a DISTRIBUTIONAL one.
# (Docs are slices J/K; this block is the derivation for them to transcribe.)
#
# Setup: one physical flight of length s at start energy E, either taken whole
# (N = 1 row) or split by `max_dE_frac` into N substeps of lengths s_1..s_N with
# sum_m s_m = s, substep m starting at energy E^(m), E^(1) = E,
# E^(m+1) = E^(m) - X_m, and X_m the loss sampled over s_m at E^(m).
#
# (i) At frozen energy the invariance is EXACT. If every substep used the same
#     rates Sigma_i(E), then by (P2) sum_m X_m =_d X, the unsplit draw, for any
#     partition and any N. Not a limit, not a tolerance: the same distribution.
#     It is *distributional*, not pathwise -- each substep addresses its own
#     `(flight, substep)` key, so the realized numbers differ; only the law is
#     preserved. This is the strongest form the invariance can take and it is
#     what slice B's infinite-divisibility argument buys.
#
# (ii) The ONLY substep dependence is the drift of Sigma_i with E inside the
#     flight. Write C(E) = |dE/dx|(E) for the mean loss per unit length and
#     V(E) = Sigma_1 E_1^2 + Sigma_2 E_2^2 + Sigma_3 E_0 T_up for the variance
#     per unit length (both from the block above `_urban_levels_scalar`). Then
#         <sum_m X_m> = sum_m s_m C(E^(m)),   Var(sum_m X_m) = sum_m s_m V(E^(m))
#     (no cross terms: the substeps are independent). Expanding
#     C(E^(m)) = C(E) - C'(E) Y_{m-1} + O(Y^2) with Y_{m-1} = sum_{l<m} X_l and
#     <Y_{m-1}> = C(E) sigma_{m-1}, sigma_{m-1} = sum_{l<m} s_l, gives
#
#         <sum_m X_m> - <X> = -C C' sum_m s_m sigma_{m-1} + O(s^3)
#                           = -C C' s^2 (N-1)/(2N) + O(s^3)   [equal substeps]
#
#     and identically Var(sum_m X_m) - Var(X) = -V' C s^2 (N-1)/(2N) + O(s^3).
#     Both are monotone in N, vanish at N = 1, and saturate at N -> infinity.
#
# (iii) The N -> infinity limit is the CORRECT moment, so substepping converges
#     rather than drifting. The exactly integrated mean loss over the flight is
#         int_0^s C(E(s')) ds' = C s - (1/2) C C' s^2 + O(s^3),
#     since dC/ds = C'(E) dE/ds = -C C'. The N -> infinity substep mean above is
#     C s - (1/2) C C' s^2: the same second-order term. So the whole substep
#     dependence of the straggled loss is the pre-existing left-endpoint
#     quadrature error of the frozen energy model, and refining `max_dE_frac`
#     removes it at first order in the step, exactly as it does deterministically.
#     STRAGGLING INTRODUCES NO SUBSTEP DEPENDENCE OF ITS OWN.
#
# (iv) Bound, in the form a caller can check. Dividing (ii) by the unsplit
#     moments and using DeltaE = C s for the flight's own mean loss,
#
#         |<sum X_m> - <X>| / <X>          ~  (1/2) |dlnC/dlnE| (DeltaE / E)
#         |Var(sum X_m) - Var(X)| / Var(X)  ~  (1/2) |dlnV/dlnE| (DeltaE / E)
#
#     both with the (N-1)/N <= 1 factor dropped, and both to LEADING order --
#     the dropped O(s^3) remainder is itself of relative size DeltaE/E, so the
#     bound is an estimate that tightens as the flight's fractional loss falls,
#     not a hard inequality at large DeltaE/E. For the spliced stopping power
#     over 1--300 keV |dlnC/dlnE| is of order 1 (Joy--Luo is C ~ ln(...)/E,
#     Berger--Seltzer likewise), so the substep-induced shift in the mean is
#     about half the flight's fractional energy loss -- which under
#     `max_dE_frac = f` is at most f/2 per substep. `max_dE_frac` therefore
#     bounds the invariance violation directly, which is the property the doc
#     re-derivation needs.
#
# (v) Measured (`tests/montecarlo/test_straggling_transport_integration.py`),
#     graphite, E = 25 keV, s = 1e4 Ang (DeltaE/E = 0.09), 20000 repetitions:
#       - frozen, N = 1 vs N = 32: mean shift -0.0014 +- 0.0150 keV on a mean
#         of 2.243 keV, i.e. consistent with the exact invariance of (i);
#       - drifting, N = 32: shift +0.086 +- 0.015 keV against the leading-order
#         prediction +0.075, a ratio of 1.15 -- the 15% excess being the
#         O(s^3) term, which grows to a ratio of 1.30 at s = 1.5e4
#         (DeltaE/E = 0.135), as the expansion predicts;
#       - in transport, 600 electrons at 25 keV with `max_dE_frac` 0 vs 0.02:
#         mean per-electron straggled loss 19.61 vs 19.70 keV, 0.5%.
#
# The step-length control itself stays DETERMINISTIC under straggling:
# `max_dE_frac * E_j / (-dEds)` uses the mean rate, not the sampled loss. That
# is deliberate. A substep grid chosen from the realized loss would be a random
# partition, the partition and the increments would be dependent, and (P2) --
# which holds for any FIXED partition -- would no longer apply. `max_dE_frac` is
# a numerical control parameter and stays one.
#
# Ordering consequence: with straggling on the row's length must be settled
# before the loss can be sampled over it, so the `max_dE_frac` cap is applied
# BEFORE the sample and the cutoff test after it, the reverse of the
# deterministic order (which can afford to solve the cutoff first because the
# loss is a known function of distance). A substep cap that binds short of the
# crossing simply emits its row and lets the next substep cross, which is the
# same semantics, at finer resolution.


@njit(cache=True)
def _transport_core_ungrooved(
    Ne,
    alive,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    rng,
    pos,
    dirs,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
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
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    straggle_on,
    stream_keys_arr,
    stragg_dE,
):
    """Compiled ungrooved transport core, with optional finite x/y footprint.

    ``elastic_model_code`` is 0 for analytic screened Rutherford and 1 for
    Browning/Mott transport. Mott angular tables are preloaded by the Python
    wrapper; elements without a table use the analytic SR angular distribution
    while retaining the Browning total elastic collision rate, matching the
    legacy fallback behavior.

    ``energy_model_code`` is 0 for the frozen left-endpoint rule and 1 for the
    midpoint predictor-corrector: stopping and the ``L/beta`` clock are then
    evaluated at the flight's midpoint energy instead of its start energy, and
    ``seg_E_end``/``seg_t_end`` (sized 0 under the frozen rule) record the
    flight's end state.

    ``max_dE_frac`` above zero caps one row's fractional energy loss, splitting a
    physical flight into numerical substeps. The collision draw is then an
    optical-depth budget carried across those substeps and consumed at each
    substep's own hazard, so refining the cap neither redraws the collision nor
    shifts its statistics; ``seg_flight``/``seg_substep`` carry the resulting
    ``(flight_id, substep_id)`` identity.

    ``straggle_on`` replaces the deterministic per-row loss with a draw from the
    Urban compound-Poisson sampler, keyed on this row's own
    ``(electron, flight, substep)``. It also redefines the cutoff crossing --
    exact indicator, fluid-interpolated location -- and reverses the order of the
    cutoff test and the ``max_dE_frac`` cap, since the row's length must be
    settled before its loss can be sampled. See the module block comment
    "stochastic energy loss in transport (slice E)" immediately above this
    function for the derivation, the alternatives rejected, and the substep
    invariance that survives. With ``straggle_on`` false none of it is reachable
    and the deterministic path is bit-for-bit what it was before straggling
    existed.
    """
    EPS = 1e-6
    # ``tau_left`` is the current physical flight's unconsumed optical depth;
    # -1.0 marks "no flight open", which is the only place a collision is drawn.
    tau_left = np.full(Ne, -1.0)
    flight_of = np.zeros(Ne, dtype=np.int64)
    substep_of = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0
    nseg = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_cutoff = 0
    n_alive = int(alive.sum())

    # Reuse one small rate buffer for all events; only the first Z_arr.size
    # entries are live for the current material layer.
    rate_arr = np.empty(mott_has_table.shape[1])

    lockstep_step = 0
    while lockstep_step < max_steps and n_alive > 0:
        lockstep_step += 1
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]
            if not alive[e]:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            E_cross_arr = L_E_cross[L]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]

            # 1. Sample the next elastic-collision distance.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

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

            # 3. Close the flight's energy and clock, then record its row.
            dEds = _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j

            if straggle_on:
                # Slice E. The loss over this row is a draw from the Urban
                # compound-Poisson subordinator (slice C) rather than a known
                # function of distance, so the row's length has to be settled
                # first and the cutoff decided afterwards from the realized
                # loss. See the "stochastic energy loss in transport (slice E)"
                # block above this function for the derivation of both the
                # crossing rule and the substep invariance; every branch here is
                # unreachable with ``straggle_on`` false.
                if energy_controlled:
                    # Deterministic step control on purpose: a substep grid
                    # chosen from the sampled loss would be a random partition
                    # and would forfeit the infinite-divisibility invariance.
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_of[e], substep_of[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    Z_arr,
                    J_arr,
                    k_arr,
                    coeff_arr,
                    E_cross_arr,
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                # Diagnostic, unchanged from slice D: the SAMPLED loss, which on
                # a cutoff row exceeds the applied loss by exactly the overshoot
                # the truncation discards.
                stragg_dE[e] += stragg_loss

                # The loss process is non-decreasing, so "crosses E_cut somewhere
                # inside this row" is equivalent to "total loss over the row
                # reaches E_j - E_cut" -- an exact test, hence an exact
                # ``n_cutoff_stopped``. The tie-break matches the deterministic
                # branch below: a crossing exactly at the row's end yields to a
                # geometry event.
                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    # Fluid interpolation at the row's own realized rate. Reduces
                    # to `cutoff_distance = (E_cut - E_j)/dEds` term by term when
                    # the loss is deterministic, and sends an overshooting draw
                    # to a vanishing step rather than a negative energy.
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if energy_model_code == 1:
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    beta_j = beta_from_keV_scalar(E_j)
            else:
                if energy_model_code == 1:
                    # The midpoint rule makes E_end = E_cut at the cutoff by
                    # definition, so E_mid there is (E_start + E_cut)/2 exactly
                    # and the truncation distance solves the scheme rather than
                    # its left-endpoint linearization.
                    cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                    )
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

                if energy_controlled and not cutoff_j:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        # Predictor-corrector for the implicit midpoint rule
                        # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                        # step_j <= cutoff_distance and |dE/ds| grows as E falls,
                        # so the predictor never undershoots E_cut and the
                        # Joy-Luo log argument stays in range.
                        E_pred = E_j + dEds * step_j
                        E_end_j = E_j + step_j * _dEds_spliced_compound_scalar(
                            J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                        )
                    # One representative energy per flight also drives the clock:
                    # s / beta(E_mid) is the midpoint rule for int ds / beta(E(s)).
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_of[e]
                seg_substep[nseg] = substep_of[e]
            nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_of[e] += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1
            if died_j:
                alive[e] = False
                n_alive -= 1
                continue

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_of[e] += 1
            substep_of[e] = 0
            tau_left[e] = -1.0

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
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

            if elastic_model_code == 1 and mott_has_table[L, i_el]:
                log_alpha = _interp_mott_log_alpha_scalar(
                    np.log10(E_keV[e] * 1e3),
                    mott_logE_flat,
                    mott_logA_flat,
                    mott_start[L, i_el],
                    mott_len[L, i_el],
                )
                alpha = 10.0**log_alpha
            else:
                alpha = _alpha_sr_joy_scalar(sr_joy_numer[i_el], E_keV[e])

            cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
            phi = 2.0 * np.pi * rng.random()
            dx, dy, dz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = dx
            dirs[e, 1] = dy
            dirs[e, 2] = dz

    return nseg, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


@njit(cache=True)
def _transport_core_ungrooved_lut(
    Ne,
    alive,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    rng,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut_E_min_keV,
    lut_inv_dE_keV,
    lut_n_energy,
    lut_total_rate,
    lut_dEds,
    lut_inv_beta,
    lut_cdf,
    lut_alpha,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    straggle_on,
    stream_keys_arr,
    stragg_dE,
):
    """Ungrooved lockstep CPU core using pretabulated energy-dependent physics.

    ``energy_model_code`` matches the exact core: 0 for the frozen
    left-endpoint rule, 1 for the midpoint predictor-corrector rule, which also
    records ``seg_E_end``/``seg_t_end`` per flight. ``max_dE_frac`` matches the
    exact core's energy-controlled substepping and optical-depth budget.

    ``L_Js``/``L_Zs``/``L_ks``/``L_coeffs``/``L_E_cross`` are the exact-core
    per-element tables (slice D): the LUT bakes ``dE/ds`` as a single
    per-layer interpolant and carries no per-element split, but the Urban
    sampler needs one, so straggling re-derives its own per-element ``C_i``
    from these tables via :func:`_urban_sample_compound_keV` rather than the
    LUT's interpolated total. Unused when ``straggle_on`` is false.
    """
    EPS = 1e-6
    tau_left = np.full(Ne, -1.0)
    flight_of = np.zeros(Ne, dtype=np.int64)
    substep_of = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0
    nseg = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_cutoff = 0
    n_alive = int(alive.sum())

    lockstep_step = 0
    while lockstep_step < max_steps and n_alive > 0:
        lockstep_step += 1
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]
            if not alive[e]:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            lut_i, lut_f = _lut_index_frac_scalar(E_j, lut_E_min_keV, lut_inv_dE_keV, lut_n_energy)

            # 1. Sample the next elastic-collision distance.
            total_rate = _lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)
            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

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

            # 3. Stopping and clock factors come from the same energy interpolation.
            dEds = _lut_lerp_2d(lut_dEds, L, lut_i, lut_f)
            inv_beta_j = _lut_lerp_1d(lut_inv_beta, lut_i, lut_f)
            cutoff_j = False
            if energy_model_code == 1:
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2 rather than its left-endpoint
                # linearization -- same construction as the exact core.
                cut_i, cut_f = _lut_index_frac_scalar(
                    0.5 * (E_j + E_cut_e),
                    lut_E_min_keV,
                    lut_inv_dE_keV,
                    lut_n_energy,
                )
                cutoff_distance = (E_cut_e - E_j) / _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
            else:
                cutoff_distance = (E_cut_e - E_j) / dEds
            geometry_event = cross_up_j or cross_dn_j or exit_side_j
            if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                step_j = cutoff_distance
                cutoff_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            if energy_controlled and not cutoff_j:
                step_energy = max_dE_frac * E_j / (-dEds)
                if step_energy < step_j:
                    step_j = step_energy
                    limited_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                    E_pred = E_j + dEds * step_j
                    mid_i, mid_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_pred),
                        lut_E_min_keV,
                        lut_inv_dE_keV,
                        lut_n_energy,
                    )
                    E_end_j = E_j + step_j * _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                # The clock uses the same representative energy: the midpoint
                # rule for int ds / beta(E(s)) is s / beta(E_mid).
                clk_i, clk_f = _lut_index_frac_scalar(
                    0.5 * (E_j + E_end_j),
                    lut_E_min_keV,
                    lut_inv_dE_keV,
                    lut_n_energy,
                )
                t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                t_end_j = clock[e] + step_j * inv_beta_j

            # Straggling (slice D): see the comment in _transport_core_ungrooved.
            # Uses the exact per-element tables, not the LUT's interpolated
            # total dE/ds -- see the docstring above. Not applied to
            # E_end_j/E_keV; for test purposes only.
            if straggle_on:
                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_of[e], substep_of[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L],
                    L_Js[L],
                    L_ks[L],
                    L_coeffs[L],
                    L_E_cross[L],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_of[e]
                seg_substep[nseg] = substep_of[e]
            nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_of[e] += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1
            if died_j:
                alive[e] = False
                n_alive -= 1
                continue

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_of[e] += 1
            substep_of[e] = 0
            tau_left[e] = -1.0

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            # The interpolated CDF replaces the second constituent-rate calculation.
            if n_el == 1:
                i_el = 0
            else:
                u = rng.random()
                i_el = n_el - 1
                for k_el in range(n_el):
                    cumulative = _lut_lerp_3d(lut_cdf, L, k_el, lut_i, lut_f)
                    if cumulative > u:
                        i_el = k_el
                        break

            # Scattering angle uses the post-flight energy, matching the exact core.
            alpha_i, alpha_f = _lut_index_frac_scalar(
                E_keV[e], lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
            )
            alpha = _lut_lerp_3d(lut_alpha, L, i_el, alpha_i, alpha_f)
            cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
            phi = 2.0 * np.pi * rng.random()
            dx, dy, dz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = dx
            dirs[e, 1] = dy
            dirs[e, 2] = dz

    return nseg, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


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
    energy_model_code,
    max_dE_frac,
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
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
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
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    vac_start,
    vac_end,
    vac_E,
    vac_t0,
    vac_id,
    straggle_on,
    stream_keys_arr,
    stragg_dE,
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
    n_cutoff = 0

    zero_surface_events = np.zeros(Ne, dtype=np.int16)
    material_steps = np.zeros(Ne, dtype=np.int32)
    surface_events = np.zeros(Ne, dtype=np.int32)

    # Electrons are visited round-robin, so unlike the ungrooved cores the
    # optical-depth budget and flight identity have to survive across passes.
    # -1.0 marks "no flight open", the only state in which a collision is drawn.
    tau_left = np.full(Ne, -1.0)
    flight_id = np.zeros(Ne, dtype=np.int64)
    substep_id = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0

    # One small reusable rate buffer; material layers can have different
    # element counts, bounded by the second Mott-table dimension.
    rate_arr = np.empty(mott_has_table.shape[1])

    # Initially every alive electron is eligible for a material event.  An
    # exit/re-entry pair leaves it eligible without consuming a material step.
    n_eligible = int(alive.sum())

    while n_eligible > 0:
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]

            if not alive[e] or material_steps[e] >= max_steps:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            E_cross_arr = L_E_cross[L]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]

            # 1. Sample the next elastic-collision distance in the current layer.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

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
            dEds = _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j)
            cutoff_j = False
            if energy_model_code == 1:
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2, not its left-endpoint form.
                cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_compound_scalar(
                    J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                )
            else:
                cutoff_distance = (E_cut_e - E_j) / dEds
            geometry_event = cross_up_j or cross_dn_j or exit_side_j or surface_first
            if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                step_j = cutoff_distance
                cutoff_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False
                surface_first = False

            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and `flight_id`.
            limited_j = False
            if energy_controlled and not cutoff_j:
                step_energy = max_dE_frac * E_j / (-dEds)
                if step_energy < step_j:
                    step_j = step_energy
                    limited_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    surface_first = False

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                    E_pred = E_j + dEds * step_j
                    E_end_j = E_j + step_j * _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                    )
                beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            # Straggling (slice D): see the identical comment in
            # _transport_core_ungrooved. Same disjoint key domain, same
            # for-test-purposes-only scope; not applied to E_end_j/E_keV.
            if straggle_on:
                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_id[e], substep_id[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    Z_arr,
                    J_arr,
                    k_arr,
                    coeff_arr,
                    E_cross_arr,
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_id[e]
                seg_substep[nseg] = substep_id[e]
            nseg += 1

            # 4. Advance through material and apply continuous stopping.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_id[e] += 1
            else:
                flight_id[e] += 1
                substep_id[e] = 0
                tau_left[e] = -1.0

            active_surface = surface_first and not cutoff_j
            reentered = False

            if not active_surface:
                zero_surface_events[e] = 0
            else:
                # 5. From the groove surface, follow the unchanged ray through
                # vacuum to its first exact vacuum->material re-entry.
                sx = pos[e, 0]
                sy = pos[e, 1]
                sz = pos[e, 2]
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
                    vac_E[nvac] = E_keV[e]
                    vac_t0[nvac] = clock[e]
                    vac_id[nvac] = e
                    nvac += 1

                    clock[e] += entry_distance / beta_from_keV_scalar(E_keV[e])
                    pos[e, 0] = ex + surface_eps * dx
                    pos[e, 1] = ey + surface_eps * dy
                    pos[e, 2] = ez + surface_eps * dz

                    surface_events[e] += 1
                    if surface_events[e] > max_steps:
                        raise RuntimeError("grooved surface event limit exhausted")
                elif not exit_side_j:
                    # No later material intersection: permanent escape through
                    # the grooved entrance surface.
                    exit_top_j = True

                if step_j <= EPS:
                    zero_surface_events[e] += 1
                else:
                    zero_surface_events[e] = 0
                if zero_surface_events[e] >= 2:
                    raise RuntimeError("repeated zero-length grooved surface events")

            # 6. Kill true exits / cutoff electrons and handle internal seams.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1

            if died_j:
                alive[e] = False
                n_eligible -= 1
                continue

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS

            # 7. Only a full material flight ends in an elastic collision; a
            #    numerical substep resumes the open one instead.
            full_j = (
                not cross_up_j
                and not cross_dn_j
                and not exit_side_j
                and not surface_first
                and not limited_j
            )
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

                if elastic_model_code == 1 and mott_has_table[L, i_el]:
                    log_alpha = _interp_mott_log_alpha_scalar(
                        np.log10(E_keV[e] * 1e3),
                        mott_logE_flat,
                        mott_logA_flat,
                        mott_start[L, i_el],
                        mott_len[L, i_el],
                    )
                    alpha = 10.0**log_alpha
                else:
                    alpha = _alpha_sr_joy_scalar(sr_joy_numer[i_el], E_keV[e])

                cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
                phi = 2.0 * np.pi * rng.random()
                ndx, ndy, ndz = _rotate_direction_scalar(
                    dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi
                )
                dirs[e, 0] = ndx
                dirs[e, 1] = ndy
                dirs[e, 2] = ndz

            # Legacy groove semantics: valid exit/re-entry pairs repeat without
            # consuming a material step; all other surviving material events do.
            if not reentered:
                material_steps[e] += 1
                if material_steps[e] >= max_steps:
                    n_eligible -= 1

    return nseg, nvac, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


# ---- per-electron transport core (GPU-portable reference) ---------------------

# Exit classification, returned per electron instead of accumulated into shared
# counters, so the host can total them in a fixed order.
EXIT_CUTOFF_STOPPED = np.int8(0)
EXIT_BACKSCATTERED = np.int8(1)
EXIT_TRANSMITTED = np.int8(2)
EXIT_SIDE = np.int8(3)
EXIT_STEP_LIMITED = np.int8(4)
EXIT_NOT_ENTERED = np.int8(5)


@njit(cache=True)
def _searchsorted_right_scalar(bounds, x, n):
    """Scalar equivalent of ``np.searchsorted(bounds[:n], x, side="right")``."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if bounds[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _transport_core_ungrooved_perelectron(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    L_nel,
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
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
    straggle_on,
    stragg_dE,
):
    """Run electrons ``[e_start, e_start + e_count)`` to completion, independently.

    This is the executable specification of the CUDA transport kernel in
    :mod:`transport_jit_kernel`: same arithmetic, same draw order, same output
    addressing, one CPU iteration per CUDA thread. Keeping the two in one
    algorithm lets the GPU port be checked bit-for-bit instead of statistically.

    It is *not* bit-for-bit against :func:`_transport_core_ungrooved`. Both
    consume the same physics, but the lockstep core interleaves one shared RNG
    stream across electrons while this one gives each electron its own; the two
    therefore realize different samples of the same distribution.

    Segment ``s`` of local electron ``i`` is written to slot ``i * cap + s``, a
    pure function of the electron index, so output does not depend on execution
    order. ``seg_count[i]`` records the *true* segment count even when it exceeds
    ``cap``; the caller must treat any such electron's slots as invalid and
    replay the batch with a larger ``cap``. Replay is exact because the streams
    are counter-addressed.

    Per-layer element data is passed as ``(n_layers, max_elements)`` padded rows
    with live lengths in ``L_nel``, the layout the CUDA kernel needs.

    ``energy_model_code`` (0 frozen, 1 midpoint) and ``max_dE_frac`` carry the
    same meaning as in :func:`_transport_core_ungrooved`: the midpoint rule
    records ``seg_E_end``/``seg_t_end``, and a positive cap splits a physical
    flight into numerical substeps that share one optical-depth budget. Both are
    per-thread scalars here rather than the lockstep core's per-electron arrays,
    which is what the CUDA port needs.
    """
    EPS = 1e-6

    for i in range(e_count):
        e = e_start + i
        seg_count[i] = 0
        exit_code[i] = EXIT_NOT_ENTERED
        # A capacity replay re-runs this whole per-electron loop from the
        # snapshotted start state, so the accumulator is reset here (like
        # seg_count above) rather than trusting a stale value from a discarded
        # attempt.
        if straggle_on:
            stragg_dE[e] = 0.0
        if not alive[e]:
            continue
        exit_code[i] = EXIT_STEP_LIMITED

        key = stream_key[e]
        draw = _SM64_ZERO
        local_nseg = 0
        E_cut_e = E_cut_by_electrons[e]
        # ``tau_left`` is the open physical flight's unconsumed optical depth;
        # -1.0 marks "no flight open", the only state in which a collision is
        # drawn. Substeps of one flight share that draw and the flight identity.
        tau_left = -1.0
        flight_id = 0
        substep_id = 0
        energy_controlled = max_dE_frac > 0.0

        for _step in range(max_steps):
            if n_layers == 1:
                L = 0
            else:
                L = _searchsorted_right_scalar(internal_bounds, pos[e, 2], n_layers - 1)

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]

            # 1. Sample the next elastic-collision distance.
            total_rate = 0.0
            for i_el in range(n_el):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left < 0.0:
                tau_left = -np.log(_stream_uniform_scalar(key, draw))
                draw += _SM64_ONE
            step_j = tau_left * lam_ang

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

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

            # 3. Record the radiating material segment. Overflowing electrons
            #    keep transporting so `seg_count` reports the capacity actually
            #    needed for the replay.
            dEds = _dEds_spliced_packed_scalar(L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_j)
            cutoff_j = False
            if energy_model_code == 1:
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2 rather than its left-endpoint
                # linearization -- same construction as the lockstep core.
                cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_packed_scalar(
                    L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_cut_e)
                )
            else:
                cutoff_distance = (E_cut_e - E_j) / dEds
            geometry_event = cross_up_j or cross_dn_j or exit_side_j
            if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                step_j = cutoff_distance
                cutoff_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            if energy_controlled and not cutoff_j:
                step_energy = max_dE_frac * E_j / (-dEds)
                if step_energy < step_j:
                    step_j = step_energy
                    limited_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                    E_pred = E_j + dEds * step_j
                    E_end_j = E_j + step_j * _dEds_spliced_packed_scalar(
                        L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_pred)
                    )
                beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            # Straggling (slice D): see the comment in _transport_core_ungrooved.
            # Reuses this electron's own stream key `key` through the disjoint
            # salted-rehash domain, so it draws no uniforms from `key`'s own
            # counter `draw` and cannot perturb the free-path / scattering-angle
            # draws above. Not applied to E_end_j/E_keV; for test purposes only.
            if straggle_on:
                urban_key = _urban_stream_key_scalar(key)
                flight_key = _urban_flight_key_scalar(urban_key, flight_id, substep_id)
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L, :n_el],
                    L_Js[L, :n_el],
                    L_ks[L, :n_el],
                    L_coeffs[L, :n_el],
                    L_E_cross[L, :n_el],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

            if local_nseg < cap:
                slot = i * cap + local_nseg
                seg_dir[slot, 0] = dx
                seg_dir[slot, 1] = dy
                seg_dir[slot, 2] = dz
                seg_mid[slot, 0] = px + 0.5 * step_j * dx
                seg_mid[slot, 1] = py + 0.5 * step_j * dy
                seg_mid[slot, 2] = pz + 0.5 * step_j * dz
                seg_len[slot] = step_j
                seg_E[slot] = E_j
                seg_t0[slot] = clock[e]
                seg_id[slot] = e
                seg_lay[slot] = L
                if energy_model_code == 1:
                    seg_E_end[slot] = E_end_j
                    seg_t_end[slot] = t_end_j
                    seg_flight[slot] = flight_id
                    seg_substep[slot] = substep_id
            local_nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left -= step_j / lam_ang
            if tau_left < 0.0:
                tau_left = 0.0

            if limited_j:
                substep_id += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            if exit_top_j:
                exit_code[i] = EXIT_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = EXIT_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = EXIT_SIDE
            elif cutoff_j:
                exit_code[i] = EXIT_CUTOFF_STOPPED
            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                break

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_id += 1
            substep_id = 0
            tau_left = -1.0

            if cross_up_j or cross_dn_j:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            # A full flight ended in an elastic collision. Pick the element with
            # probability proportional to n_i * sigma_i(E). Rates are recomputed
            # rather than buffered so the kernel needs no per-thread local array.
            if n_el == 1:
                i_el = 0
            else:
                u = _stream_uniform_scalar(key, draw) * total_rate
                draw += _SM64_ONE
                cumulative = 0.0
                i_el = n_el - 1
                for k_el in range(n_el):
                    if elastic_model_code == 1:
                        rate = _scatter_rates_mott_scalar(
                            E_j,
                            mott_numer[k_el],
                            mott_denom1[k_el],
                            mott_denom2[k_el],
                        )
                    else:
                        rate = _scatter_rates_sr_scalar(
                            E_j, sr_rate_numer[k_el], sr_joy_numer[k_el]
                        )
                    cumulative += rate
                    if cumulative > u:
                        i_el = k_el
                        break

            if elastic_model_code == 1 and mott_has_table[L, i_el]:
                log_alpha = _interp_mott_log_alpha_scalar(
                    np.log10(E_keV[e] * 1e3),
                    mott_logE_flat,
                    mott_logA_flat,
                    mott_start[L, i_el],
                    mott_len[L, i_el],
                )
                alpha = 10.0**log_alpha
            else:
                alpha = _alpha_sr_joy_scalar(sr_joy_numer[i_el], E_keV[e])

            cos_t = _sample_cos_theta_from_alpha(alpha, _stream_uniform_scalar(key, draw))
            draw += _SM64_ONE
            phi = 2.0 * np.pi * _stream_uniform_scalar(key, draw)
            draw += _SM64_ONE
            ndx, ndy, ndz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = ndx
            dirs[e, 1] = ndy
            dirs[e, 2] = ndz

        seg_count[i] = local_nseg


@njit(cache=True)
def _transport_core_ungrooved_perelectron_lut(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut_E_min_keV,
    lut_inv_dE_keV,
    lut_n_energy,
    lut_total_rate,
    lut_dEds,
    lut_inv_beta,
    lut_cdf,
    lut_alpha,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    straggle_on,
    stragg_dE,
):
    """Per-electron CPU reference for the CUDA LUT transport kernel.

    ``energy_model_code`` and ``max_dE_frac`` match
    :func:`_transport_core_ungrooved_perelectron`; the LUT variant reads the
    midpoint stopping power and inverse speed from the same interpolation the
    lockstep LUT core uses.

    ``L_Js``/``L_Zs``/``L_ks``/``L_coeffs``/``L_E_cross`` are the same padded
    ``(n_layers, max_elements)`` per-element tables
    :func:`_transport_core_ungrooved_perelectron` receives, threaded in here
    only for straggling (slice D): the LUT carries no per-element split (see
    :func:`_transport_core_ungrooved_lut`). Unused when ``straggle_on`` is
    false.
    """
    EPS = 1e-6

    for i in range(e_count):
        e = e_start + i
        seg_count[i] = 0
        exit_code[i] = EXIT_NOT_ENTERED
        if straggle_on:
            stragg_dE[e] = 0.0
        if not alive[e]:
            continue
        exit_code[i] = EXIT_STEP_LIMITED

        key = stream_key[e]
        draw = _SM64_ZERO
        local_nseg = 0
        E_cut_e = E_cut_by_electrons[e]
        # Per-thread optical-depth budget and flight identity; -1.0 marks "no
        # flight open", the only state in which a collision is drawn.
        tau_left = -1.0
        flight_id = 0
        substep_id = 0
        energy_controlled = max_dE_frac > 0.0

        for _step in range(max_steps):
            if n_layers == 1:
                L = 0
            else:
                L = _searchsorted_right_scalar(internal_bounds, pos[e, 2], n_layers - 1)

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            lut_i, lut_f = _lut_index_frac_scalar(E_j, lut_E_min_keV, lut_inv_dE_keV, lut_n_energy)

            total_rate = _lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)
            lam_ang = 1e8 / total_rate
            if tau_left < 0.0:
                tau_left = -np.log(_stream_uniform_scalar(key, draw))
                draw += _SM64_ONE
            step_j = tau_left * lam_ang

            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

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

            dEds = _lut_lerp_2d(lut_dEds, L, lut_i, lut_f)
            inv_beta_j = _lut_lerp_1d(lut_inv_beta, lut_i, lut_f)
            cutoff_j = False
            if energy_model_code == 1:
                cut_i, cut_f = _lut_index_frac_scalar(
                    0.5 * (E_j + E_cut_e), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                )
                cutoff_distance = (E_cut_e - E_j) / _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
            else:
                cutoff_distance = (E_cut_e - E_j) / dEds
            geometry_event = cross_up_j or cross_dn_j or exit_side_j
            if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                step_j = cutoff_distance
                cutoff_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight.
            limited_j = False
            if energy_controlled and not cutoff_j:
                step_energy = max_dE_frac * E_j / (-dEds)
                if step_energy < step_j:
                    step_j = step_energy
                    limited_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    E_pred = E_j + dEds * step_j
                    mid_i, mid_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_pred), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                    )
                    E_end_j = E_j + step_j * _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                clk_i, clk_f = _lut_index_frac_scalar(
                    0.5 * (E_j + E_end_j), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                )
                t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                t_end_j = clock[e] + step_j * inv_beta_j

            # Straggling (slice D): see the comment in
            # _transport_core_ungrooved_perelectron. Uses the exact per-element
            # tables threaded in above, not the LUT's interpolated total dE/ds.
            # Not applied to E_end_j/E_keV; for test purposes only.
            if straggle_on:
                urban_key = _urban_stream_key_scalar(key)
                flight_key = _urban_flight_key_scalar(urban_key, flight_id, substep_id)
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L, :n_el],
                    L_Js[L, :n_el],
                    L_ks[L, :n_el],
                    L_coeffs[L, :n_el],
                    L_E_cross[L, :n_el],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

            if local_nseg < cap:
                slot = i * cap + local_nseg
                seg_dir[slot, 0] = dx
                seg_dir[slot, 1] = dy
                seg_dir[slot, 2] = dz
                seg_mid[slot, 0] = px + 0.5 * step_j * dx
                seg_mid[slot, 1] = py + 0.5 * step_j * dy
                seg_mid[slot, 2] = pz + 0.5 * step_j * dz
                seg_len[slot] = step_j
                seg_E[slot] = E_j
                seg_t0[slot] = clock[e]
                seg_id[slot] = e
                seg_lay[slot] = L
                if energy_model_code == 1:
                    seg_E_end[slot] = E_end_j
                    seg_t_end[slot] = t_end_j
                    seg_flight[slot] = flight_id
                    seg_substep[slot] = substep_id
            local_nseg += 1

            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left -= step_j / lam_ang
            if tau_left < 0.0:
                tau_left = 0.0

            if limited_j:
                substep_id += 1
                continue

            if exit_top_j:
                exit_code[i] = EXIT_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = EXIT_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = EXIT_SIDE
            elif cutoff_j:
                exit_code[i] = EXIT_CUTOFF_STOPPED
            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                break

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_id += 1
            substep_id = 0
            tau_left = -1.0

            if cross_up_j or cross_dn_j:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            if n_el == 1:
                i_el = 0
            else:
                u = _stream_uniform_scalar(key, draw)
                draw += _SM64_ONE
                i_el = n_el - 1
                for k_el in range(n_el):
                    cumulative = _lut_lerp_3d(lut_cdf, L, k_el, lut_i, lut_f)
                    if cumulative > u:
                        i_el = k_el
                        break

            alpha_i, alpha_f = _lut_index_frac_scalar(
                E_keV[e], lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
            )
            alpha = _lut_lerp_3d(lut_alpha, L, i_el, alpha_i, alpha_f)
            cos_t = _sample_cos_theta_from_alpha(alpha, _stream_uniform_scalar(key, draw))
            draw += _SM64_ONE
            phi = 2.0 * np.pi * _stream_uniform_scalar(key, draw)
            draw += _SM64_ONE
            ndx, ndy, ndz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = ndx
            dirs[e, 1] = ndy
            dirs[e, 2] = ndz

        seg_count[i] = local_nseg


def pack_layer_tables(
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
):
    """Pad the per-layer element lists into ``(n_layers, max_elements)`` rows.

    The lockstep core indexes a Python list of ragged arrays, which neither the
    per-electron core nor CUDA can do. Padding is zero-filled and never read:
    ``L_nel`` bounds every element loop.
    """
    n_layers = len(L_Zs)
    max_el = max(arr.size for arr in L_Zs)
    Js = np.zeros((n_layers, max_el), dtype=np.float64)
    Zs = np.zeros((n_layers, max_el), dtype=np.float64)
    ks = np.zeros((n_layers, max_el), dtype=np.float64)
    coeffs = np.zeros((n_layers, max_el), dtype=np.float64)
    E_cross = np.zeros((n_layers, max_el), dtype=np.float64)
    ncm3 = np.zeros((n_layers, max_el), dtype=np.float64)
    sr_rate_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_denom1s = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_denom2s = np.zeros((n_layers, max_el), dtype=np.float64)
    sr_joy_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    nel = np.zeros(n_layers, dtype=np.int32)
    for L in range(n_layers):
        n = L_Zs[L].size
        nel[L] = n
        Js[L, :n] = L_Js[L]
        Zs[L, :n] = L_Zs[L]
        ks[L, :n] = L_ks[L]
        sr_rate_numers[L, :n] = L_sr_rate_numer[L]
        mott_numers[L, :n] = L_mott_numer[L]
        mott_denom1s[L, :n] = L_mott_denom1[L]
        mott_denom2s[L, :n] = L_mott_denom2[L]
        sr_joy_numers[L, :n] = L_sr_joy_numer[L]
        coeffs[L, :n] = L_coeffs[L]
        E_cross[L, :n] = L_E_cross[L]
        ncm3[L, :n] = L_ncm3[L]

    packed_tables = (
        Js,
        Zs,
        ks,
        coeffs,
        E_cross,
        ncm3,
        sr_rate_numers,
        mott_numers,
        mott_denom1s,
        mott_denom2s,
        sr_joy_numers,
        nel,
    )

    return packed_tables


def _percentile_summary(values):
    """Compact, JSON-safe summary of one non-negative per-flight diagnostic."""
    if values.size == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    p50, p90, p99 = np.percentile(values, (50.0, 90.0, 99.0))
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(np.max(values)),
    }


def _flight_diagnostic_summary(
    E_start_keV,
    L_ang,
    elec_id,
    layer,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    elastic_model,
):
    """Estimate frozen-state transport error per physical flight, then reduce it.

    The current segment is one physical flight. Diagnostics recompute its
    left-endpoint stopping and elastic hazard at the stored start energy and at
    the predicted end energy. They do not feed values back into propagation and
    retain only fixed-size percentile summaries.
    """

    def _host(array):
        get = getattr(array, "get", None)
        return np.asarray(get() if get is not None else array)

    E_start = _host(E_start_keV).astype(float, copy=False)
    length = _host(L_ang).astype(float, copy=False)
    electron = _host(elec_id).astype(np.int64, copy=False)
    layer_index = _host(layer).astype(np.int64, copy=False)
    n_flights = E_start.size

    stopping = np.zeros(n_flights, dtype=float)
    for L, (J_arr, k_arr, coeff_arr, E_cross_arr) in enumerate(
        zip(L_Js, L_ks, L_coeffs, L_E_cross, strict=True)
    ):
        mask = layer_index == L
        E = E_start[mask]
        if E.size == 0:
            continue
        stopping[mask] = _dEds_spliced_compound(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E.copy())

    E_end = E_start + stopping * length
    hazard_start = np.zeros(n_flights, dtype=float)
    hazard_end = np.zeros(n_flights, dtype=float)
    for L, (Z_arr, ncm3_arr) in enumerate(zip(L_Zs, L_ncm3, strict=True)):
        mask = layer_index == L
        start = E_start[mask]
        end = E_end[mask]
        if start.size == 0:
            continue
        start_total = np.zeros(start.size, dtype=float)
        end_total = np.zeros(end.size, dtype=float)
        for Z, ncm3 in zip(Z_arr, ncm3_arr, strict=True):
            if elastic_model == "mott":
                start_total += _sigma_browning_cm2(Z, start) * ncm3
                end_total += _sigma_browning_cm2(Z, end) * ncm3
            else:
                alpha_start = 3.4e-3 * Z**0.67 / start
                alpha_end = 3.4e-3 * Z**0.67 / end
                start_rel = (start + 511.0) / (start + 1024.0)
                end_rel = (end + 511.0) / (end + 1024.0)
                start_total += (
                    ncm3
                    * 5.21e-21
                    * Z**2
                    / start**2
                    * 4.0
                    * np.pi
                    / (alpha_start * (1.0 + alpha_start))
                    * start_rel**2
                )
                end_total += (
                    ncm3
                    * 5.21e-21
                    * Z**2
                    / end**2
                    * 4.0
                    * np.pi
                    / (alpha_end * (1.0 + alpha_end))
                    * end_rel**2
                )
        hazard_start[mask] = start_total
        hazard_end[mask] = end_total

    midpoint = 0.5 * (E_start + E_end)
    left_clock = length / _beta_array(E_start)
    midpoint_clock = length / _beta_array(midpoint)
    fractional_loss = np.maximum(0.0, (E_start - E_end) / E_start)
    relative_hazard_change = np.abs(hazard_end - hazard_start) / hazard_start
    relative_clock_error = np.abs(midpoint_clock - left_clock) / midpoint_clock
    cutoff = np.asarray(E_cut_by_electrons, dtype=float)[electron]
    cutoff_overshoot = np.maximum(0.0, cutoff - E_end)

    return {
        "n_flights": int(n_flights),
        "fractional_energy_loss": _percentile_summary(fractional_loss),
        "relative_hazard_change": _percentile_summary(relative_hazard_change),
        "relative_clock_error_estimate": _percentile_summary(relative_clock_error),
        "cutoff_overshoot_keV": _percentile_summary(cutoff_overshoot),
    }


@dataclass(frozen=True)
class PerElectronTransportConfig:
    """Host-side batching policy for the per-electron cores.

    No field changes results: ``seg_capacity`` only sets how many segment slots
    each electron is given before an overflow forces a replay, and
    ``scratch_budget_bytes`` only sets how many electrons share one launch.
    Both exist because the CUDA path materializes a dense ``(batch, capacity)``
    scratch grid, and that grid must fit in VRAM alongside the case's tables.

    ``seg_capacity`` is only the *initial* guess, and the driver stops trusting it
    as soon as it has a measurement. The cores report the true segment count even
    for electrons that overflowed, so every batch -- including one that overflowed
    and has to be replayed -- says exactly what the material needs, and ``cap`` is
    reset to that running maximum times ``capacity_headroom``. Capacity trades
    against batch size out of a fixed byte budget, so an over-provisioned ``cap``
    costs launches just as an under-provisioned one costs replays.

    ``probe_electrons`` shortens the first batch, which is otherwise most of the
    run and is the one batch sized by the guess. Its segments are kept -- output
    slots are addressed by electron index, so a short first batch is just a short
    batch -- and it bounds what a wrong ``seg_capacity`` can cost to a probe rather
    than a full batch. Set it to 0 to let the first batch run full width, which is
    what a GPU run wants: there a batch costs about one electron lifetime whatever
    its width, so a discarded batch and a probe cost the same launch.
    """

    seg_capacity: int = 512
    scratch_budget_bytes: int = 512 * 1024 * 1024
    capacity_headroom: float = 1.25
    max_batch: int = 65536
    probe_electrons: int = 1024


DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG = PerElectronTransportConfig()

TRANSPORT_CORES = ("auto", "lockstep", "per-electron", "cuda")

# Electron count above which "auto" transports on the device. Measured crossover
# is Ne ~ 500-1000 for both a light (hopg) and a heavy (MoSe2) material on an
# RTX 5080: below it the launch and staging overhead outweighs the kernel, above
# it the CUDA core pulls away monotonically (4.1x at Ne=4000, 4.9-15.9x at
# Ne=16000). 1000 is the conservative end of that band, so no run that would
# have been faster on the CPU core is moved off it. See
# `docs/repo-design/compute/gpu-transport-rawkernel.md`.
CUDA_TRANSPORT_MIN_ELECTRONS = 1000


@cache
def _cuda_transport_available():
    """Whether this process can transport on the device.

    Requires the resolved spectrum backend to BE the CUDA device (a ROCm or SYCL
    backend has no `cupyx.jit` transport kernel, and a CPU backend has nowhere to
    run one) and the kernel module to import. Cached: the backend is fixed at
    import and a failed import will not start succeeding.
    """

    from ._backend import BACKEND

    if BACKEND.name != "cuda":
        return False
    try:
        from .transport_jit_kernel import make_cuda_transport_core  # noqa: F401
    except Exception as error:  # pragma: no cover - needs a broken CuPy install
        logger.warning("CUDA transport kernel unavailable, using the CPU core: %s", error)
        return False
    return True


def resolve_transport_core(requested, Ne, groove=None):
    """Resolve ``transport_core`` for a run of ``Ne`` electrons.

    ``"auto"`` is the default and the only value that resolves: it takes the CUDA
    core when this process has a CUDA device, the run is ungrooved, and
    ``Ne > CUDA_TRANSPORT_MIN_ELECTRONS``; otherwise the lockstep CPU core. Every
    explicit value is returned unchanged so a caller that names a core still gets
    that core or an error, never a silent substitution.

    ``CXR_MC_TRANSPORT_CORE`` overrides the *requested* value for the whole
    process, which is how a run pins the historical CPU core (``=lockstep``)
    without touching call sites -- reproducing a pre-existing result, or
    bisecting a device/host difference.
    """

    pinned = env_value("CXR_MC_TRANSPORT_CORE", "").strip().lower()
    if pinned:
        if pinned not in TRANSPORT_CORES:
            raise ValueError(
                f"PYRITE_MC_TRANSPORT_CORE must be one of {', '.join(TRANSPORT_CORES)}; got {pinned!r}"
            )
        requested = pinned
    if requested not in TRANSPORT_CORES:
        raise ValueError(f"transport_core must be one of {', '.join(TRANSPORT_CORES)}")
    if requested != "auto":
        return requested
    if groove is not None or int(Ne) <= CUDA_TRANSPORT_MIN_ELECTRONS:
        return "lockstep"
    return "cuda" if _cuda_transport_available() else "lockstep"


# float64 mid(3) + dir(3) + len + E + t0 = 9, int64 id, int16 layer.
_SEG_SCRATCH_BYTES = 9 * 8 + 8 + 2


def _batch_size(cap, config):
    per_electron = cap * _SEG_SCRATCH_BYTES
    n = int(config.scratch_budget_bytes // max(per_electron, 1))
    return max(1, min(n, int(config.max_batch)))


def _capacity_for(seen_max, config):
    """Slots per electron given the largest segment count seen so far.

    Never below ``seen_max`` whatever the headroom, so a replay is always given a
    capacity that fits and the retry loop cannot fail to make progress.
    """
    return max(1, seen_max, int(np.ceil(seen_max * config.capacity_headroom)))


def _batch_electrons(e, cap, Ne, config):
    """How many electrons the batch starting at ``e`` covers.

    Non-increasing in ``cap``, which a capacity replay relies on: the retry must
    never need more electrons than the snapshot it restores.
    """
    n = min(_batch_size(cap, config), Ne - e)
    if e == 0 and config.probe_electrons > 0:
        n = min(n, int(config.probe_electrons))
    return n


def _run_per_electron_transport_lut(
    core,
    xp,
    Ne,
    seed,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    alive,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    stragg_layer_tables,
    straggle_on,
    stragg_dE,
    config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_on_device=False,
):
    """Drive the CPU/CUDA LUT per-electron core with capacity replay.

    ``stragg_layer_tables`` is the ``(L_Js, L_Zs, L_ks, L_coeffs, L_E_cross)``
    padded per-element tables (slice D straggling; see
    :func:`_transport_core_ungrooved_perelectron_lut`), ``straggle_on`` gates
    it, and ``stragg_dE`` is the Ne-sized per-electron accumulator the core
    writes into -- downloaded once at the end like the compacted segments,
    since it lives outside the capacity-replay scratch/compaction path.
    """
    on_device = xp is not np
    to_dev = xp.asarray if on_device else (lambda a: a)
    to_host = xp.asnumpy if on_device else (lambda a: a)

    from .runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.upload")
    d_keys = to_dev(stream_keys(seed, Ne))
    d_alive = to_dev(alive)
    d_clock = to_dev(clock)
    d_pos = to_dev(pos)
    d_dirs = to_dev(dirs)
    d_E = to_dev(E_keV)
    d_E_cut = to_dev(E_cut_by_electrons)
    d_bounds = to_dev(internal_bounds)
    d_nel = to_dev(L_nel)
    d_top = to_dev(L_top)
    d_bot = to_dev(L_bot)
    d_total_rate = to_dev(lut.total_rate)
    d_dEds = to_dev(lut.dEds)
    d_inv_beta = to_dev(lut.inv_beta)
    d_cdf = to_dev(lut.cdf)
    d_alpha = to_dev(lut.alpha)
    d_stragg_layers = tuple(to_dev(a) for a in stragg_layer_tables)
    d_stragg = to_dev(stragg_dE)
    _nsys_pop()

    midpoint = energy_model_code == 1
    out_bufs = (seg_dir, seg_mid, seg_len, seg_E, seg_t0, seg_id, seg_lay)
    if midpoint:
        out_bufs += (seg_E_end, seg_t_end, seg_flight, seg_substep)
    batches = []

    cap = max(1, int(config.seg_capacity))
    seen_max = 0
    nseg = 0
    n_back = n_trans = n_side = n_cutoff = n_step_limited = 0
    e = 0
    while e < Ne:
        m = _batch_electrons(e, cap, Ne, config)
        sl = slice(e, e + m)
        snap = (d_pos[sl].copy(), d_dirs[sl].copy(), d_E[sl].copy(), d_clock[sl].copy())

        while True:
            _nsys_push("cxr.transport.scratch")
            scratch = _alloc_scratch(xp, m, cap, midpoint)
            seg_count = xp.zeros(m, dtype=xp.int64)
            exit_code = xp.zeros(m, dtype=xp.int8)
            _nsys_pop()
            _nsys_push("cxr.transport.launch")
            core(
                e,
                m,
                cap,
                d_keys,
                d_alive,
                max_steps,
                n_layers,
                d_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                width_ang,
                height_ang,
                d_clock,
                d_pos,
                d_dirs,
                d_E_cut,
                d_nel,
                d_top,
                d_bot,
                lut.E_min_keV,
                lut.inv_dE_keV,
                lut.n_energy,
                d_total_rate,
                d_dEds,
                d_inv_beta,
                d_cdf,
                d_alpha,
                d_E,
                *scratch,
                seg_count,
                exit_code,
                *d_stragg_layers,
                straggle_on,
                d_stragg,
            )
            _nsys_pop()

            _nsys_push("cxr.transport.capsync")
            needed = int(seg_count.max())
            _nsys_pop()
            seen_max = max(seen_max, needed)
            if needed <= cap:
                break
            cap = _capacity_for(seen_max, config)
            d_pos[sl], d_dirs[sl], d_E[sl], d_clock[sl] = snap
            m = _batch_electrons(e, cap, Ne, config)
            sl = slice(e, e + m)
            snap = tuple(a[:m] for a in snap)

        total = int(seg_count.sum())
        if nseg + total > max_segments:
            raise RuntimeError("segment buffer exhausted")
        _nsys_push("cxr.transport.compact")
        keep = xp.arange(cap)[None, :] < seg_count[:, None]
        s_dir, s_mid, s_len, s_E, s_t0, s_id, s_lay = scratch[:7]
        slots = (
            s_dir.reshape(m, cap, 3),
            s_mid.reshape(m, cap, 3),
            s_len.reshape(m, cap),
            s_E.reshape(m, cap),
            s_t0.reshape(m, cap),
            s_id.reshape(m, cap),
            s_lay.reshape(m, cap),
        )
        if midpoint:
            slots += tuple(a.reshape(m, cap) for a in scratch[7:])
        if keep_on_device:
            batches.append(tuple(a[keep] for a in slots))
        else:
            dst = slice(nseg, nseg + total)
            for buf, a in zip(out_bufs, slots, strict=True):
                buf[dst] = to_host(a[keep])
        nseg += total
        _nsys_pop()

        _nsys_push("cxr.transport.exitcodes")
        n_back += int((exit_code == EXIT_BACKSCATTERED).sum())
        n_trans += int((exit_code == EXIT_TRANSMITTED).sum())
        n_side += int((exit_code == EXIT_SIDE).sum())
        n_cutoff += int((exit_code == EXIT_CUTOFF_STOPPED).sum())
        n_step_limited += int((exit_code == EXIT_STEP_LIMITED).sum())
        _nsys_pop()
        e += m

        if seen_max > 0:
            cap = _capacity_for(seen_max, config)

    joined = None
    if keep_on_device:
        _nsys_push("cxr.transport.join")
        empty = _alloc_scratch(xp, 0, 1, midpoint)
        joined = tuple(
            xp.concatenate([b[i] for b in batches]) if batches else empty[i]
            for i in range(len(out_bufs))
        )
        _nsys_pop()

    return nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, joined, to_host(d_stragg)


def _run_per_electron_transport(
    core,
    xp,
    Ne,
    seed,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    alive,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    layer_tables,
    L_top,
    L_bot,
    mott,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    straggle_on,
    stragg_dE,
    config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_on_device=False,
):
    """Drive ``core`` over electron batches and compact the result.

    ``straggle_on``/``stragg_dE`` are slice D's straggling gate and Ne-sized
    per-electron accumulator; see the LUT driver's docstring
    (:func:`_run_per_electron_transport_lut`) for why it is downloaded
    separately from the compacted segments.

    ``core`` is either :func:`_transport_core_ungrooved_perelectron` or the CUDA
    kernel launcher; ``xp`` is the matching array module. The two share this
    driver so batching, capacity growth, and compaction cannot drift apart.

    Output is electron-major and step-minor, which is a pure function of the
    electron index rather than of execution order, so a run is reproducible
    across batch sizes and (on CUDA) launch geometries. That addressing is also
    what lets the first batch be a short capacity probe: its segments land in the
    same slots they would have anyway, so measuring costs only the electrons it
    transports.

    Per-electron state and the material tables are moved to the device once and
    stay there; only the compacted segments of each batch cross the bus.

    ``keep_on_device`` removes even that crossing. The compacted batches are
    held in ``xp``'s own memory and joined with one ``concatenate``, which is
    returned as an eighth value for the caller to hand out in place of its
    ``seg_*`` buffers (those are then not allocated at all). Joining is what
    sizes the output, because the segment total is not known until the last
    batch has run; it costs one device-to-device pass over the payload -- against
    the pageable host copy it replaces, that is roughly two orders of magnitude
    of bandwidth -- and holds both copies of the payload while it runs.

    Returns ``(nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, joined,
    stragg_dE)``, where ``joined`` is ``None`` unless ``keep_on_device`` and
    ``stragg_dE`` is all-zero unless ``straggle_on``.
    """
    on_device = xp is not np
    to_dev = xp.asarray if on_device else (lambda a: a)
    to_host = xp.asnumpy if on_device else (lambda a: a)

    # NVTX sub-ranges splitting this driver into upload / per-batch launch /
    # capacity sync / compaction / join, so a capture attributes the transport
    # phase instead of leaving it in the unlabelled host remainder. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from .runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.upload")
    d_keys = to_dev(stream_keys(seed, Ne))
    d_alive = to_dev(alive)
    d_clock = to_dev(clock)
    d_pos = to_dev(pos)
    d_dirs = to_dev(dirs)
    d_E = to_dev(E_keV)
    d_E_cut = to_dev(E_cut_by_electrons)
    d_bounds = to_dev(internal_bounds)
    d_top = to_dev(L_top)
    d_bot = to_dev(L_bot)
    d_layers = tuple(to_dev(a) for a in layer_tables)
    d_mott = tuple(to_dev(a) for a in mott)
    d_stragg = to_dev(stragg_dE)
    _nsys_pop()

    midpoint = energy_model_code == 1
    out_bufs = (seg_dir, seg_mid, seg_len, seg_E, seg_t0, seg_id, seg_lay)
    if midpoint:
        out_bufs += (seg_E_end, seg_t_end, seg_flight, seg_substep)
    batches = []

    cap = max(1, int(config.seg_capacity))
    seen_max = 0
    nseg = 0
    n_back = n_trans = n_side = n_cutoff = n_step_limited = 0
    e = 0
    while e < Ne:
        m = _batch_electrons(e, cap, Ne, config)
        sl = slice(e, e + m)
        # The core advances position/direction/energy/clock in place, so a
        # capacity replay has to start from the same state it did.
        snap = (d_pos[sl].copy(), d_dirs[sl].copy(), d_E[sl].copy(), d_clock[sl].copy())

        while True:
            _nsys_push("cxr.transport.scratch")
            scratch = _alloc_scratch(xp, m, cap, midpoint)
            seg_count = xp.zeros(m, dtype=xp.int64)
            exit_code = xp.zeros(m, dtype=xp.int8)
            _nsys_pop()
            _nsys_push("cxr.transport.launch")
            core(
                e,
                m,
                cap,
                d_keys,
                d_alive,
                max_steps,
                n_layers,
                d_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                width_ang,
                height_ang,
                d_clock,
                d_pos,
                d_dirs,
                d_E_cut,
                *d_layers,
                d_top,
                d_bot,
                *d_mott,
                d_E,
                *scratch,
                seg_count,
                exit_code,
                straggle_on,
                d_stragg,
            )
            _nsys_pop()
            # Split from the launch above because on CUDA the launch returns
            # immediately: this read is where the batch's kernel time actually
            # lands, so `launch` is host-side dispatch cost and `capsync` is the
            # device.
            _nsys_push("cxr.transport.capsync")
            needed = int(seg_count.max())
            _nsys_pop()
            seen_max = max(seen_max, needed)
            if needed <= cap:
                break
            # Replay: the streams are counter-addressed, so the retry reproduces
            # the discarded run exactly rather than resampling it. `needed` is the
            # true count, not a truncated one, so one replay always suffices; the
            # headroom is for the batches after this one.
            cap = _capacity_for(seen_max, config)
            d_pos[sl], d_dirs[sl], d_E[sl], d_clock[sl] = snap
            m = _batch_electrons(e, cap, Ne, config)
            sl = slice(e, e + m)
            snap = tuple(a[:m] for a in snap)

        total = int(seg_count.sum())
        if nseg + total > max_segments:
            raise RuntimeError("segment buffer exhausted")
        _nsys_push("cxr.transport.compact")
        keep = xp.arange(cap)[None, :] < seg_count[:, None]
        s_dir, s_mid, s_len, s_E, s_t0, s_id, s_lay = scratch[:7]
        slots = (
            s_dir.reshape(m, cap, 3),
            s_mid.reshape(m, cap, 3),
            s_len.reshape(m, cap),
            s_E.reshape(m, cap),
            s_t0.reshape(m, cap),
            s_id.reshape(m, cap),
            s_lay.reshape(m, cap),
        )
        if midpoint:
            slots += tuple(a.reshape(m, cap) for a in scratch[7:])
        if keep_on_device:
            # No preallocation here: `max_segments` is Ne*max_steps rows, a bound
            # no run comes near and no device would hold. The batch list is the
            # buffer, and the join below sizes the output from what actually ran.
            batches.append(tuple(a[keep] for a in slots))
        else:
            dst = slice(nseg, nseg + total)
            for buf, a in zip(out_bufs, slots, strict=True):
                buf[dst] = to_host(a[keep])
        nseg += total
        _nsys_pop()

        _nsys_push("cxr.transport.exitcodes")
        n_back += int((exit_code == EXIT_BACKSCATTERED).sum())
        n_trans += int((exit_code == EXIT_TRANSMITTED).sum())
        n_side += int((exit_code == EXIT_SIDE).sum())
        n_cutoff += int((exit_code == EXIT_CUTOFF_STOPPED).sum())
        n_step_limited += int((exit_code == EXIT_STEP_LIMITED).sum())
        _nsys_pop()
        e += m

        # Re-size from what electrons actually needed rather than from the guess,
        # in both directions: a tighter `cap` buys a proportionally larger batch
        # out of the same byte budget.
        if seen_max > 0:
            cap = _capacity_for(seen_max, config)

    joined = None
    if keep_on_device:
        _nsys_push("cxr.transport.join")
        # An empty run has no batch to take shapes and dtypes from; borrow them
        # from a zero-length scratch, which is where they came from anyway.
        empty = _alloc_scratch(xp, 0, 1, midpoint)
        joined = tuple(
            xp.concatenate([b[i] for b in batches]) if batches else empty[i]
            for i in range(len(out_bufs))
        )
        _nsys_pop()

    return nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, joined, to_host(d_stragg)


def _alloc_scratch(xp, m, cap, midpoint=False):
    """Slot buffers for one batch, in the segment field order.

    The four flight end-state/identity buffers exist only under the midpoint
    rule; frozen runs allocate them at length zero so the core signature stays
    fixed while the frozen row schema stays exactly seven fields wide.
    """
    n = m * cap
    n_end = n if midpoint else 0
    return (
        xp.empty((n, 3), dtype=xp.float64),
        xp.empty((n, 3), dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.int64),
        xp.empty(n, dtype=xp.int16),
        xp.empty(n_end, dtype=xp.float64),
        xp.empty(n_end, dtype=xp.float64),
        xp.empty(n_end, dtype=xp.int64),
        xp.empty(n_end, dtype=xp.int64),
    )


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
    transverse_distribution=None,
    energy_spread_frac=None,
    crystal_width_mm=None,
    crystal_height_mm=None,
    tilt_polar_rad=0.0,
    tilt_azim_rad=0.0,
    groove=None,
    E_cut_by_electrons=None,
    transport_core="auto",
    per_electron_config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_segments_on_device=False,
    transport_lut_config=DEFAULT_TRANSPORT_LUT_CONFIG,
    collect_diagnostics=False,
    energy_model="frozen",
    max_dE_frac=0.0,
    straggling=False,
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
    effect on the emitted spectrum UNDER THE INCOHERENT EMISSION POLICY, which
    reads only ``|A|^2`` per segment.

    This is NOT true under mc_spectrum(coherent=True). The coherent phase reads
    r_mid directly, so the constant per-electron transverse offset enters every
    cross-electron term as exp[-i(omega n_hat + g).dr]. Turning on a 1 um spot
    drops the coherent peak height by ~43x against the beam_fwhm_mm=None point
    source, and the residual single-n_hat result is one speckle realization
    whose peak scatters 30-41% seed to seed -- a contrast that does NOT fall as
    Ne grows. The transverse form factor that should average those terms away
    is not implemented; see the discrepancy row transverse-bunch-form-factor and
    docs/physics/radiation-physics/coherent-emission.md before using a finite
    spot with emission="coherent"/"both".

    With a finite crystal footprint, the sampled transverse positions classify
    missed entries and can cause side-face exits. Segment positions also affect
    the downstream six-face escape attenuation, so beam size can change the
    emitted radiation spectrum.

    beam_fwhm_y_mm: optional y-plane spot FWHM [mm] for an ELLIPTICAL beam
    (decision 8). None -> equals beam_fwhm_mm (isotropic), which draws
    sigma_x == sigma_y and is bit-for-bit with the historical scalar-spot path.

    transverse_distribution: the resolved Courant-Snyder policy, mutually
    exclusive with the spot FWHMs above (a spot fixes <x^2> alone; a Twiss
    triplet fixes <x^2>, <x x'> and <x'^2>, so accepting both would be
    over-determined). It owns the entry positions AND the per-electron
    directions: the drawn slopes (x', y') are dx/dz, dy/dz about the beam axis,
    turned into unit vectors through geometry.beam_frame_basis, whose transverse
    columns coincide with the lab x / y the spot uses. Until this key is set,
    every direction is the shared beam_dir exactly, as before. Limiting case:
    eps_n -> 0 gives sigma_position -> 0 and slopes -> 0, so positions and
    directions both converge on the collimated point source; leaving the key
    unset reproduces it BIT-FOR-BIT.
    Validation: beam-phase-space-injection

    energy_spread_frac: RMS *relative* energy spread. Each electron starts at
    ``E0_keV * (1 + f * u)``, u ~ Normal(0, 1), drawn uncorrelated with the
    arrival time (decision 3: no chirp model). None/0 -> the monoenergetic beam,
    bit-for-bit. Raises rather than transporting a non-positive energy, which a
    Gaussian permits only for a spread near unity.
    Validation: beam-energy-spread-injection

    Both draws use their own SeedSequence child streams (spawn(5)[4] and
    spawn(6)[5], the indices after the bunch's spawn(4)[3]), so enabling either
    never perturbs the free-path / scattering draws.

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

    groove: optional :class:`~pyrite.montecarlo.groove.GrooveSpec` describing a
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
    ``docs/validation/geometry/blazed-groove-geometry.md``.
    Validation: blazed-groove-geometry

    transport_core: which ungrooved core runs the electrons.
      "auto" (default) -- the CUDA core when this process has a CUDA device, the
          run is ungrooved, and Ne > CUDA_TRANSPORT_MIN_ELECTRONS (1000); the
          lockstep core otherwise. See `resolve_transport_core`; pin the choice
          for a whole process with `CXR_MC_TRANSPORT_CORE`.
      "lockstep" -- the historical core: an outer step loop over an inner
          electron loop, all draws taken from one shared Generator in step-major
          order. BIT-FOR-BIT unchanged from before the other cores existed.
      "per-electron" -- each electron runs to completion against its own
          counter-addressed stream (see `_transport_core_ungrooved_perelectron`).
      "cuda" -- the same algorithm as one CUDA thread per electron.

    The two newer cores are NOT bit-for-bit with "lockstep": they consume
    differently-ordered random streams, so they realize a different sample of
    the same distribution. "cuda" is in turn not bit-for-bit with
    "per-electron", because CUDA's libm differs from the host's by a few ulp and
    transport amplifies that over hundreds of scattering events. What is
    invariant is the physics: identical models, identical draw semantics, and
    aggregate agreement across seeds (backscatter and transmit fractions,
    segments per electron, mean segment length/energy/depth) -- so "auto"
    changes a run's realization, not its distribution.
    Validation: gpu-transport-core
    Grooved transport is rejected for both new cores, so a grooved "auto" run
    stays on the lockstep core rather than failing.

    per_electron_config: batching policy for the two new cores
    (:class:`PerElectronTransportConfig`). Segment capacity and scratch budget
    only bound memory and replay behavior; neither changes results.

    energy_model: how a physical flight's energy and clock advance along it.
      "frozen" (default) -- the historical left-endpoint rule: stopping power
          and beta are evaluated once at the flight's start energy and held
          constant over its whole length. BIT-FOR-BIT unchanged.
      "midpoint" -- second-order predictor-corrector for the implicit midpoint
          rule ``E_end = E_start + (dE/ds)((E_start + E_end)/2) * s``, with the
          transport clock advanced by ``s / beta((E_start + E_end)/2)`` and the
          cutoff truncation distance solved for ``E_end == E_cut``. Adds
          ``E_end_keV``, ``t_end_ang``, and ``E_repr_keV`` to the returned rows
          and leaves one radiating row per physical flight. ``E_repr_keV`` is
          the rule's own representative energy ``(E_start + E_end)/2``, the
          energy at which radiation kernels evaluate the row. Elastic hazard
          stays frozen at the start energy; only stopping and the clock are
          controlled here.
          Currently implemented for the ungrooved lockstep core only -- any
          other core or a grooved run raises rather than returning the frozen
          schema under a midpoint request.
    max_dE_frac: numerical cap on one row's fractional energy loss, splitting a
      physical flight into substeps when the cap binds before any physical
      event. 0.0 (default) disables substepping, leaving one row per flight.
      Requires ``energy_model="midpoint"``. The collision is drawn once per
      physical flight as an optical depth and consumed across its substeps at
      each substep's own hazard, so refining the cap does not resample the
      collision. Rows then carry ``flight_id``/``substep_id``; a substep keeps
      the flight's direction and identity and never scatters.

    Validation: transport-midpoint-stopping, energy-controlled-propagation

    straggling: sample the per-flight Urban energy-loss fluctuation (slice C).
      False (default) is BIT-FOR-BIT with every run before this parameter
      existed: the sampler is skipped entirely on every core, touching neither
      its RNG stream nor any output array. True samples the per-flight/-substep
      Urban compound loss on a counter-addressed stream disjoint from the
      free-path/scattering-angle draws -- keyed on this electron's own stream
      key via a salted rehash, so turning it on can never perturb *those* draws
      -- and returns the summed per-electron SAMPLED loss as
      ``result["straggle_dE_keV"]``.

      Whether that loss is applied to the electron depends on the core, and is
      being staged deliberately:

      - Ungrooved lockstep core, exact stopping (``transport_core="lockstep"``
        with ``transport_lut_config=TransportLUTConfig(enabled=False)``):
        APPLIED. ``E_keV``/``E_end_keV``, the cutoff crossing and
        ``n_cutoff_stopped`` all reflect the sampled loss, so results differ
        from a ``straggling=False`` run. See the module block comment
        "stochastic energy loss in transport (slice E)" above
        ``_transport_core_ungrooved`` for the crossing redefinition and the
        substep-invariance derivation. On a cutoff row the applied loss is
        ``E_start - E_cut``, i.e. less than the sampled loss recorded in the
        diagnostic by the overshoot the truncation discards.
      - Every other core (both LUT variants, grooved, per-electron, CUDA):
        DIAGNOSTIC ONLY, exactly as slice D left it. The sampler is addressed
        and its result returned, but no energy is subtracted, so ``E_keV`` and
        every downstream quantity are unaffected whether this is on or off.
        Applying it there is slice F, which will also decide which cores raise
        instead.

    collect_diagnostics: opt in to fixed-size percentile summaries of the
    per-flight fractional energy loss, relative elastic-hazard change,
    left-endpoint versus midpoint clock estimate, and cutoff overshoot. The
    diagnostic pass runs after transport, consumes no random draws, does not
    alter propagation, and retains no per-flight arrays. A device-resident run
    copies the four required segment arrays to the host only when explicitly
    requested.

    keep_segments_on_device: return the eight per-segment arrays where the CUDA
    core produced them instead of copying them to the host. Requires
    ``transport_core="cuda"``; every other returned array (the incident
    phase-space diagnostics, the groove-gap arrays) and every count stays NumPy,
    since those are per-electron or scalar and no spectrum kernel reads them.
    The VALUES are untouched -- same dtypes, same elements, same order -- so this
    only moves the payload, and a run's segments are identical either way.

    The point is the spectrum phase: a case's kernels then read the transport's
    output in place rather than the transport pushing ~82 B/segment down and the
    kernels pulling it back up. It follows that the caller's spectrum backend has
    to be the same device (``CXR_MC_BACKEND`` resolving to CUDA). NumPy kernels
    cannot consume these arrays; NumPy refuses the implicit conversion rather
    than performing it silently, so the mismatch is an error, not a slow path.

    Costs device memory: the segments are joined with one ``concatenate``, which
    holds two copies of the payload while it runs, and the joined arrays then
    stay resident for as long as the caller keeps the dict.

    Returns dict of per-segment arrays:
      "r_mid" (M,3) [Ang], "v_hat" (M,3), "L_ang" (M,), "E_keV" (M,),
      "t_ang" (M,), "t0_ang" (M,) [per-electron bunch offset], "elec_id" (M,),
      "layer" (M,) [emitting layer index]
    with "E_start_keV"/"t_start_ang" as the canonical spellings of "E_keV"/
    "t_ang", plus "E_end_keV" (M,), "t_end_ang" (M,), and the propagator's
    representative energy "E_repr_keV" (M,) = (E_start + E_end)/2 under
    energy_model="midpoint"
    incident phase-space diagnostics (one row per sampled electron, including
    missed entries): "initial_r_ang" (Ne,3), "initial_v_hat" (Ne,3),
      "initial_E_keV" (Ne,), "initial_t0_ang" (Ne,)
    non-radiating groove-gap flights:
      "vacuum_start_ang" (V,3), "vacuum_end_ang" (V,3),
      "vacuum_E_keV" (V,), "vacuum_t_ang" (V,), "vacuum_t0_ang" (V,),
      "vacuum_elec_id" (V,)
    and diagnostics: "n_backscattered", "n_transmitted", "n_side_exited",
    "n_missed", "n_cutoff_stopped", "n_step_limited", "n_stopped" (the
    compatibility alias of ``n_cutoff_stopped``), and "n_layers". Incomplete
    histories raise ``RuntimeError`` rather than returning these arrays/counts.

    Validation: electron-transport, finite-beam-size, finite-transverse-crystal,
    grazing-beam-projection, multilayer-stack
    """
    if not np.isfinite(E0_keV) or E0_keV <= 0.0:
        raise ValueError("E0_keV must be finite and strictly positive")

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
    if energy_model not in ("frozen", "midpoint"):
        raise ValueError("energy_model must be 'frozen' or 'midpoint'")

    requested_core = transport_core
    transport_core = resolve_transport_core(transport_core, Ne, groove)
    if transport_core != "lockstep" and groove is not None:
        raise ValueError("grooved transport is only implemented for the lockstep core")
    if keep_segments_on_device and transport_core != "cuda":
        raise ValueError(
            "keep_segments_on_device requires transport_core='cuda'; "
            f"{requested_core!r} resolved to {transport_core!r}"
        )
    max_dE_frac = float(max_dE_frac)
    if max_dE_frac < 0.0:
        raise ValueError("max_dE_frac must be non-negative")
    # Substepping a frozen flight is exactly the mis-phased configuration the
    # slice-E convergence study rejected: it multiplies rows without improving
    # the clock, so the two options are not independently selectable.
    if max_dE_frac > 0.0 and energy_model != "midpoint":
        raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")

    if E_cut_by_electrons is None:
        E_cut_by_electrons = np.full(
            Ne,
            float(E_cut_keV),
            dtype=np.float64,
        )
    else:
        E_cut_by_electrons = np.asarray(
            E_cut_by_electrons,
            dtype=np.float64,
        )

        if E_cut_by_electrons.shape != (Ne,):
            raise ValueError(
                f"E_cut_by_electrons must have shape ({Ne},), got {E_cut_by_electrons.shape}"
            )

    if not np.all(np.isfinite(E_cut_by_electrons)) or not np.all(E_cut_by_electrons > 0.0):
        raise ValueError("electron cutoff energies must be finite and strictly positive")

    # NVTX ranges for the transport phase. Everything here is host-side, but a
    # CUDA run's wall clock is not: Round 4 left a 38-52% unattributed remainder
    # around the transport driver, and these are what a capture needs to split it
    # into table build / beam sampling / buffer reservation / core / output.
    # No-op off the profiled GPU path. Lazy import: runner imports this module.
    from .runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.tables")
    z_total = float(layers[-1][1])
    n_layers = len(layers)
    L_Zs = []
    L_Js = []
    L_ncm3 = []
    L_ks = []
    L_coeffs = []
    L_E_cross = []
    mott_tables = []

    for _, _, lc in layers:
        elements = []
        ncm3_arr = []
        Z_arr = []
        J_arr = []
        k_arr = []
        coeff_arr = []
        E_cross_arr = []
        layer_mott_tables = []

        for el, n_i in lc:
            elements.append(el)
            params = TRANSPORT_ELEMENTS[el]
            Z_i = float(params["Z"])
            A_i = float(params["A"])
            J_i = float(params["J_keV"])
            k_i = 0.731 + 0.0688 * np.log10(Z_i)
            coeff_i = (n_i / 0.602214076) * Z_i

            ncm3_arr.append(n_i * 1e24)
            Z_arr.append(Z_i)
            J_arr.append(J_i)
            k_arr.append(k_i)
            coeff_arr.append(coeff_i)
            E_cross_arr.append(_element_crossover_keV(el, Z_i, A_i, J_i))

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
        L_E_cross.append(np.asarray(E_cross_arr, dtype=float))
        mott_tables.append(layer_mott_tables)

    L_sr_rate_numer = []
    L_mott_numer = []
    L_mott_denom1 = []
    L_mott_denom2 = []
    L_sr_joy_numer = []

    for i, Z_i in enumerate(L_Zs):
        n_cm3_i = L_ncm3[i]
        # Rutherford Scattering coefficient hoisted out of hot loop
        L_sr_rate_numer.append(5.21e-21 * Z_i * Z_i * np.float64(4.0) * np.float64(np.pi) * n_cm3_i)

        # Browning fit coefficients to Mott scattering hoisted out of hot loop
        z17 = Z_i ** np.float64(1.7)
        L_mott_numer.append(np.float64(3.0e-18) * z17 * n_cm3_i)
        L_mott_denom1.append(np.float64(0.005) * z17)
        L_mott_denom2.append(np.float64(0.0007) * Z_i * Z_i)

        # Joy-Luo
        L_sr_joy_numer.append(np.float64(3.4e-3) * Z_i ** np.float64(0.67))

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
    _nsys_pop()

    _nsys_push("cxr.transport.sample")
    rng = np.random.default_rng(seed)
    pos = np.zeros((Ne, 3))
    transverse_slopes = None
    if transverse_distribution is not None:
        if beam_fwhm_mm or beam_fwhm_y_mm:
            raise ValueError("transverse_distribution is incompatible with the spot FWHM fields")
        # The Twiss policy owns BOTH transverse moments: positions here and the
        # correlated slopes fed to `dirs` below. Splitting them across two
        # sources would break the <x x'> correlation the emittance encodes.
        # Its own child stream (spawn(5)[4], the next index after the bunch's
        # spawn(4)[3]) keeps the main free-path / scattering draws untouched.
        MM_TO_ANG = 1.0e7
        resolved = resolved_from_mapping(transverse_distribution)
        transverse_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[4])
        x_mm, x_prime, y_mm, y_prime = sample_transverse(resolved, Ne, transverse_rng)
        transverse_slopes = (x_prime, y_prime)
        offsets = np.stack((x_mm * MM_TO_ANG, y_mm * MM_TO_ANG), axis=1)
        pos[:, :2] = project_beam_entry(offsets, tilt_polar_rad, tilt_azim_rad)
    elif beam_fwhm_mm or beam_fwhm_y_mm:
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
    if beam_dir.shape != (3,) or not np.all(np.isfinite(beam_dir)):
        raise ValueError("beam_dir must be a finite three-vector")
    beam_norm = np.linalg.norm(beam_dir)
    if not np.isfinite(beam_norm) or beam_norm == 0.0:
        raise ValueError("beam_dir must be nonzero")
    beam_dir = beam_dir / beam_norm
    if beam_dir[2] <= 1e-6:
        raise ValueError("beam_dir must point into the slab (z component > 0)")
    if transverse_slopes is None:
        dirs = np.tile(beam_dir, (Ne, 1))
    else:
        # Slopes are dx/dz, dy/dz about the beam axis, so the per-electron
        # direction is (x', y', 1) in the beam frame. Zero emittance gives
        # exactly `beam_dir` back, which is what makes the collimated limit
        # bit-for-bit rather than merely close.
        x_prime, y_prime = transverse_slopes
        basis = beam_frame_basis(beam_dir)
        dirs = x_prime[:, None] * basis[:, 0] + y_prime[:, None] * basis[:, 1] + beam_dir
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    E_keV = np.full(Ne, float(E0_keV))
    if energy_spread_frac:
        # RMS *relative* deviation, uncorrelated with arrival time (decision 3:
        # no chirp model). Its own child stream, spawn(6)[5], for the same
        # reason as the transverse draw above.
        spread_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[5])
        E_keV = E_keV * (1.0 + float(energy_spread_frac) * spread_rng.standard_normal(Ne))
        if not np.all(np.isfinite(E_keV)) or not np.all(E_keV > 0.0):
            raise ValueError(
                f"energy_spread_frac={energy_spread_frac} drew a non-finite or non-positive "
                "electron energy; the Gaussian spread model needs spread << 1"
            )
    if not np.all(E_cut_by_electrons < E_keV):
        raise ValueError("each electron cutoff energy must be below its initial energy")
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
    _nsys_pop()

    elastic_model_code = 1 if elastic_model == "mott" else 0
    transport_lut = None
    if groove is None and transport_lut_config.enabled:
        _nsys_push("cxr.transport.lut")
        transport_lut = build_transport_energy_lut(
            float(np.min(E_cut_by_electrons)),
            float(np.max(E_keV)),
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
            config=transport_lut_config,
        )
        _nsys_pop()

    _nsys_push("cxr.transport.alloc")
    # Preallocate fixed-capacity output buffers.  ``max_steps`` is already a
    # conservative safety bound in ordinary runs; groove re-entry events can add
    # material segments without consuming it, so the compiled core raises a
    # clear buffer error if a pathological case exceeds this capacity.
    # A device-resident run fills none of them -- its batches are joined on the
    # device instead -- so it reserves no rows.
    n_rows = 0 if keep_segments_on_device else max_segments
    seg_mid = np.empty((n_rows, 3), dtype=float)
    seg_dir = np.empty((n_rows, 3), dtype=float)
    seg_len = np.empty(n_rows, dtype=float)
    seg_E = np.empty(n_rows, dtype=float)
    seg_t0 = np.empty(n_rows, dtype=float)
    seg_id = np.empty(n_rows, dtype=np.int64)
    seg_lay = np.empty(n_rows, dtype=np.int16)
    # Flight end state exists only under the controlled propagator; the frozen
    # rule keeps the historical seven-field row exactly.
    n_end_rows = n_rows if energy_model == "midpoint" else 0
    seg_E_end = np.empty(n_end_rows, dtype=float)
    seg_t_end = np.empty(n_end_rows, dtype=float)
    seg_flight = np.empty(n_end_rows, dtype=np.int64)
    seg_substep = np.empty(n_end_rows, dtype=np.int64)
    _nsys_pop()

    # Where the segments end up living, and so which array module assembles the
    # output below. NumPy unless the run asked to keep them on the device.
    seg_xp = np
    dev_segs = None
    energy_model_code = 1 if energy_model == "midpoint" else 0

    # Straggling (slice D). ``straggle_on`` is a plain bool -- every core
    # branches on it before touching the Urban sampler's own stream, so False
    # (the default) costs nothing beyond the Ne-sized zero allocations below
    # and is BIT-FOR-BIT with a run compiled before this parameter existed.
    # ``stragg_stream_keys`` feeds the lockstep/grooved cores, which have no
    # per-electron counter stream of their own (see the module-level "counter-
    # based per-electron RNG" comment); the per-electron cores instead reuse
    # their own ``stream_key``/``d_keys`` directly. ``stragg_layer_tables``
    # supplies the per-electron LUT core with the per-element split the LUT
    # itself does not carry (see `_transport_core_ungrooved_perelectron_lut`).
    straggle_on = bool(straggling)
    stragg_dE = np.zeros(Ne) if straggle_on else np.zeros(0)
    stragg_stream_keys = stream_keys(seed, Ne) if straggle_on else np.zeros(1, dtype=np.uint64)
    if straggle_on:
        _stragg_packed = pack_layer_tables(
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            L_sr_rate_numer,
            L_mott_numer,
            L_mott_denom1,
            L_mott_denom2,
            L_sr_joy_numer,
        )
        stragg_layer_tables = _stragg_packed[:5]
    else:
        _stragg_dummy = np.zeros((1, 1), dtype=np.float64)
        stragg_layer_tables = (
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
            _stragg_dummy,
        )

    _nsys_push("cxr.transport.core")
    if groove is None and transport_lut is not None and transport_core != "lockstep":
        if transport_core == "cuda":
            if straggle_on:
                # Slice D wired straggling into the CUDA per-electron *exact*
                # kernel (_transport_kernel/run_transport_kernel) only. The
                # LUT CUDA kernel (_transport_lut_kernel) has no per-element
                # split to sample from (see
                # _transport_core_ungrooved_perelectron_lut's docstring) and
                # slice D did not duplicate the Urban sampler there -- raise
                # rather than silently returning an unstraggled
                # straggle_dE_keV. Use transport_lut_config=TransportLUTConfig
                # (enabled=False) to reach the exact CUDA kernel instead, or
                # transport_core="per-electron" off CUDA.
                raise NotImplementedError(
                    "straggling=True is not implemented on the CUDA LUT core "
                    "(transport_core='cuda' with the LUT enabled); disable the "
                    "LUT (transport_lut_config=TransportLUTConfig(enabled=False)) "
                    "to reach the CUDA exact per-electron core, which slice D "
                    "wired, or run off CUDA"
                )
            from .transport_jit_kernel import make_cuda_transport_lut_core

            core, core_xp = make_cuda_transport_lut_core()
        else:
            core, core_xp = _transport_core_ungrooved_perelectron_lut, np
        if keep_segments_on_device:
            seg_xp = core_xp

        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, dev_segs, stragg_dE = (
            _run_per_electron_transport_lut(
                core,
                core_xp,
                Ne,
                seed,
                max_steps,
                max_segments,
                n_layers,
                internal_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                0.0 if width_ang is None else float(width_ang),
                0.0 if height_ang is None else float(height_ang),
                alive,
                clock,
                pos,
                dirs,
                E_cut_by_electrons,
                transport_lut.n_el,
                L_top,
                L_bot,
                transport_lut,
                E_keV,
                seg_dir,
                seg_mid,
                seg_len,
                seg_E,
                seg_t0,
                seg_id,
                seg_lay,
                seg_E_end,
                seg_t_end,
                seg_flight,
                seg_substep,
                stragg_layer_tables,
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
            )
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None and transport_lut is not None:
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_ungrooved_lut(
            Ne,
            alive,
            max_steps,
            max_segments,
            n_layers,
            internal_bounds,
            elastic_model_code,
            energy_model_code,
            max_dE_frac,
            z_total,
            finite_footprint,
            0.0 if width_ang is None else float(width_ang),
            0.0 if height_ang is None else float(height_ang),
            clock,
            rng,
            pos,
            dirs,
            E_cut_by_electrons,
            transport_lut.n_el,
            L_top,
            L_bot,
            transport_lut.E_min_keV,
            transport_lut.inv_dE_keV,
            transport_lut.n_energy,
            transport_lut.total_rate,
            transport_lut.dEds,
            transport_lut.inv_beta,
            transport_lut.cdf,
            transport_lut.alpha,
            E_keV,
            seg_dir,
            seg_mid,
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None and transport_core != "lockstep":
        # Per-electron streams and run-to-completion ordering. Not bit-for-bit
        # with the lockstep core -- see `_transport_core_ungrooved_perelectron`.
        if transport_core == "cuda":
            from .transport_jit_kernel import make_cuda_transport_core

            core, core_xp = make_cuda_transport_core()
        else:
            core, core_xp = _transport_core_ungrooved_perelectron, np
        if keep_segments_on_device:
            seg_xp = core_xp

        packed_per_layer_tables = pack_layer_tables(
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            L_sr_rate_numer,
            L_mott_numer,
            L_mott_denom1,
            L_mott_denom2,
            L_sr_joy_numer,
        )
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, dev_segs, stragg_dE = (
            _run_per_electron_transport(
                core,
                core_xp,
                Ne,
                seed,
                max_steps,
                max_segments,
                n_layers,
                internal_bounds,
                elastic_model_code,
                energy_model_code,
                max_dE_frac,
                z_total,
                finite_footprint,
                0.0 if width_ang is None else float(width_ang),
                0.0 if height_ang is None else float(height_ang),
                alive,
                clock,
                pos,
                dirs,
                E_cut_by_electrons,
                packed_per_layer_tables,
                L_top,
                L_bot,
                (mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat),
                E_keV,
                seg_dir,
                seg_mid,
                seg_len,
                seg_E,
                seg_t0,
                seg_id,
                seg_lay,
                seg_E_end,
                seg_t_end,
                seg_flight,
                seg_substep,
                straggle_on,
                stragg_dE,
                config=per_electron_config,
                keep_on_device=keep_segments_on_device,
            )
        )
        nvac = 0
        vac_start = np.empty((0, 3), dtype=float)
        vac_end = np.empty((0, 3), dtype=float)
        vac_E = np.empty(0, dtype=float)
        vac_t0 = np.empty(0, dtype=float)
        vac_id = np.empty(0, dtype=np.int64)
    elif groove is None:
        nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_ungrooved(
            Ne,
            alive,
            max_steps,
            max_segments,
            n_layers,
            internal_bounds,
            elastic_model_code,
            energy_model_code,
            max_dE_frac,
            z_total,
            finite_footprint,
            0.0 if width_ang is None else float(width_ang),
            0.0 if height_ang is None else float(height_ang),
            clock,
            rng,
            pos,
            dirs,
            E_cut_by_electrons,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            L_sr_rate_numer,
            L_mott_numer,
            L_mott_denom1,
            L_mott_denom2,
            L_sr_joy_numer,
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
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
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
        nseg, nvac, n_back, n_trans, n_side, n_cutoff, n_step_limited = _transport_core_grooved(
            Ne,
            alive,
            max_steps,
            max_segments,
            max_vac,
            n_layers,
            internal_bounds,
            elastic_model_code,
            energy_model_code,
            max_dE_frac,
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
            E_cut_by_electrons,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            L_sr_rate_numer,
            L_mott_numer,
            L_mott_denom1,
            L_mott_denom2,
            L_sr_joy_numer,
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
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            vac_start_buf,
            vac_end_buf,
            vac_E_buf,
            vac_t0_buf,
            vac_id_buf,
            straggle_on,
            stragg_stream_keys,
            stragg_dE,
        )
        vac_start = vac_start_buf[:nvac]
        vac_end = vac_end_buf[:nvac]
        vac_E = vac_E_buf[:nvac]
        vac_t0 = vac_t0_buf[:nvac]
        vac_id = vac_id_buf[:nvac]
    _nsys_pop()

    if n_step_limited:
        raise RuntimeError(
            "incomplete electron transport: "
            f"n_step_limited={n_step_limited}, Ne={Ne}, max_steps={max_steps}"
        )

    _nsys_push("cxr.transport.output")
    if dev_segs is None:
        r_mid = seg_mid[:nseg]
        v_hat = seg_dir[:nseg]
        L_ang = seg_len[:nseg]
        E_seg = seg_E[:nseg]
        t_ang = seg_t0[:nseg]
        elec_id = seg_id[:nseg]
        layer = seg_lay[:nseg]
    else:
        # Already sized to `nseg` by the join, in the scratch's field order.
        v_hat, r_mid, L_ang, E_seg, t_ang, elec_id, layer = dev_segs[:7]
        if energy_model == "midpoint":
            seg_E_end, seg_t_end, seg_flight, seg_substep = dev_segs[7:]

    vacuum_start_ang = vac_start
    vacuum_end_ang = vac_end
    vacuum_E_keV = vac_E
    vacuum_t_ang = vac_t0
    vacuum_elec_id = vac_id

    # Per-segment, so it follows the segments: gathering on the device costs one
    # Ne-sized upload of `t0_electron` and saves an nseg-sized download.
    t0_by_electron = t0_electron if seg_xp is np else seg_xp.asarray(t0_electron)
    t0_ang = t0_by_electron[elec_id] if elec_id.size else seg_xp.empty(0, dtype=float)
    vacuum_t0_ang = t0_electron[vacuum_elec_id] if vacuum_elec_id.size else np.empty(0, dtype=float)
    _nsys_pop()

    result = {
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
        # `E_keV`/`t_ang` are unscheduled compatibility aliases of the canonical
        # flight-start fields and never become midpoint/representative values.
        "E_keV": E_seg,
        "E_start_keV": E_seg,
        "t_ang": t_ang,  # segment-start age sum(L/beta) [Ang, c=1]
        "t_start_ang": t_ang,
        "t0_ang": t0_ang,  # per-electron longitudinal bunch offset [Ang, c=1]
        # `elec_id` is an unscheduled compatibility alias of `electron_id`.
        "elec_id": elec_id,  # emitting electron index in [0, Ne)
        "electron_id": elec_id,
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
        "n_cutoff_stopped": int(n_cutoff),
        "n_step_limited": 0,
        "n_stopped": int(n_cutoff),
        "Ne": Ne,
        "thickness_ang": z_total,
        "crystal_width_ang": width_ang,
        "crystal_height_ang": height_ang,
        "n_layers": n_layers,
    }
    if straggle_on:
        # The summed per-electron Urban-SAMPLED loss. Applied to the electron's
        # energy only on the ungrooved lockstep exact core (slice E); diagnostic
        # only on every other core until slice F. See the ``straggling``
        # paragraph in this function's docstring.
        result["straggle_dE_keV"] = stragg_dE
    if energy_model == "midpoint":
        result["E_end_keV"] = seg_E_end[:nseg]
        result["t_end_ang"] = seg_t_end[:nseg]
        # The propagator's own representative energy: the implicit midpoint rule
        # evaluates stopping and beta at (E_start + E_end)/2, so radiation and
        # quadrature consumers read that value instead of re-deriving one.
        result["E_repr_keV"] = 0.5 * (E_seg + seg_E_end[:nseg])
        # `(electron_id, flight_id)` is the stable physical key; `substep_id`
        # indexes numerical rows inside one flight and is integration detail.
        result["flight_id"] = seg_flight[:nseg]
        result["substep_id"] = seg_substep[:nseg]
    if collect_diagnostics:
        result["transport_diagnostics"] = _flight_diagnostic_summary(
            E_seg,
            L_ang,
            elec_id,
            layer,
            E_cut_by_electrons,
            L_Js,
            L_Zs,
            L_ks,
            L_coeffs,
            L_E_cross,
            L_ncm3,
            elastic_model,
        )
    return result
