"""
montecarlo.transport

Single-scattering electron transport (Zhai SI S2): element data, elastic
scattering models (Browning free paths + NIST-Mott-calibrated screened-
Rutherford angles, analytic SR fallback), Joy-Luo stopping power, and the
vectorized event-driven trajectory simulator. Pure NumPy -- never touches the
GPU; the spectrum phase consumes the segment arrays it returns.
"""

import logging
import os
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
from numba import jit as njit

from .. import DATA_DIR
from ..materials._transport_data import TRANSPORT_ELEMENTS
from ..materials.attenuation import _normalize_composition
from .geometry import (
    Z_MAX,
    Z_MIN,
    first_prism_exit,
    project_beam_entry,
    validate_transverse_dimensions,
)
from .groove import entry_points, first_surface_event

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


def beta_from_keV(E_keV):
    g = 1.0 + E_keV / 510.99895
    # Exponentiation stays in the input array namespace. dpnp deliberately
    # disables NumPy's ``__array_ufunc__`` bridge, unlike CuPy.
    return (1.0 - 1.0 / g**2) ** 0.5


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


def _alpha_sr_joy(Z, E_keV):
    """Classic analytic screened-Rutherford screening parameter (Joy/Bishop)."""
    return 3.4e-3 * Z**0.67 / E_keV


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
def _sample_cos_theta_sr_numba(Z, E_keV, R):
    out = np.empty_like(E_keV)

    Zfac = 3.4e-3 * Z**0.67

    for i in range(E_keV.size):
        alpha = Zfac / E_keV[i]

        out[i] = 1.0 - 2.0 * alpha * R[i] / (1.0 + alpha - R[i])

    return out


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


# def _dEds_compound(comp, E_keV):
#     """
#     Joy-Luo stopping power [keV/Angstrom] for a compound, additive over
#     elements: dE/ds = -7.85e-4 / E * sum_i (n_i/N_A') Z_i ln(1.166(E+k J)/J)
#     with n_i in atoms/Ang^3 and N_A' = 0.602214 (Avogadro in mol/Ang^3*g
#     bookkeeping units; equals the single-element rho*Z/A form).
#     """
#     total = 0.0
#     for el, n_i in comp:
#         p = TRANSPORT_ELEMENTS[el]
#         k = 0.731 + 0.0688 * np.log10(p["Z"])
#         total = total + (n_i / 0.602214076) * p["Z"] * np.log(
#             1.166 * (E_keV + k * p["J_keV"]) / p["J_keV"]
#         )
#     return -7.85e-4 / E_keV * total


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

    z_total = float(layers[-1][1])
    n_layers = len(layers)
    L_elements = []
    L_Zs = []
    L_Js = []
    L_ncm3 = []
    L_ks = []
    L_coeffs = []
    for _, _, lc in layers:
        elements = []
        ncm3_arr = []
        Z_arr = []
        J_arr = []
        k_arr = []
        coeff_arr = []
        for el, n_i in lc:
            p = TRANSPORT_ELEMENTS[el]
            Z_i = p["Z"]
            J_i = p["J_keV"]
            k_i = 0.731 + 0.0688 * np.log10(Z_i)
            coeff_i = (n_i / 0.602214076) * Z_i
            elements.append(el)
            ncm3_arr.append(n_i * 1e24)
            Z_arr.append(Z_i)
            J_arr.append(J_i)
            k_arr.append(k_i)
            coeff_arr.append(coeff_i)

        L_elements.append(elements)
        L_Zs.append(np.asarray(Z_arr, dtype=float))
        L_Js.append(np.asarray(J_arr, dtype=float))
        L_ncm3.append(np.asarray(ncm3_arr, dtype=float))
        L_ks.append(np.asarray(k_arr, dtype=float))
        L_coeffs.append(np.asarray(coeff_arr, dtype=float))

    L_top = [float(a) for (a, _, _) in layers]
    L_bot = [float(b) for (_, b, _) in layers]
    internal_bounds = np.array(L_bot[:-1], dtype=float)  # between consecutive layers
    EPS = 1e-6  # nudge across an internal boundary so the layer lookup is unambiguous

    def _scatter_rates(Ea, Zs, n_cm3s):
        """Per-element elastic scattering rates [1/cm] at energies Ea (one layer)."""
        rates = []
        for Z_i, n_i in zip(Zs, n_cm3s, strict=False):
            if elastic_model == "mott":
                sig_i = _sigma_browning_cm2(Z_i, Ea)
            elif elastic_model == "sr":
                a = _alpha_sr_joy(Z_i, Ea)
                sig_i = (
                    5.21e-21
                    * Z_i**2
                    / Ea**2
                    * 4.0
                    * np.pi
                    / (a * (1.0 + a))
                    * ((Ea + 511.0) / (Ea + 1024.0)) ** 2
                )
            else:
                raise ValueError("elastic_model must be 'mott' or 'sr'")
            rates.append(n_i * sig_i)
        return np.array(rates)  # (n_elements, m)

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
    E = np.full(Ne, float(E0_keV))
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
    initial_E_keV = E.copy()

    seg_mid, seg_dir, seg_len, seg_E, seg_t0, seg_id, seg_lay = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    vac_start, vac_end, vac_E, vac_t0, vac_id = [], [], [], [], []
    zero_surface_events = np.zeros(Ne, dtype=np.int16) if groove is not None else None
    material_steps = np.zeros(Ne, dtype=np.int32) if groove is not None else None
    surface_events = np.zeros(Ne, dtype=np.int32) if groove is not None else None
    lockstep_step = 0

    # Event-driven loop: ungrooved transport retains the legacy lockstep
    # max_steps iterations bit-for-bit. Grooved transport counts only material
    # collision/layer iterations against that budget; valid exit/re-entry pairs
    # repeat the loop without consuming it. A separate per-electron surface-event
    # bound prevents pathological geometry from becoming an unbounded loop.
    while True:
        if not alive.any():
            break
        if groove is None:
            if lockstep_step >= max_steps:
                break
            lockstep_step += 1
            idx = np.flatnonzero(alive)
        else:
            assert material_steps is not None
            idx = np.flatnonzero(alive & (material_steps < max_steps))
            if idx.size == 0:
                break
        if n_layers == 1:
            lay_all = None  # everyone is in layer 0
        else:
            lay_all = np.clip(
                np.searchsorted(internal_bounds, pos[idx, 2], side="right"),
                0,
                n_layers - 1,
            )
        for L in range(n_layers):
            grp = idx if lay_all is None else idx[lay_all == L]
            if grp.size == 0:
                continue
            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            n_cm3s = L_ncm3[L]
            z_top_L, z_bot_L = L_top[L], L_bot[L]
            Ea = E[grp]  # kinetic energies [keV]

            # -- 1. distance to the next elastic collision (this layer) ---------
            # Exponential free path P(s)=exp(-s/lambda)/lambda, total rate
            # additive over the layer's elements: 1/lambda = sum_i n_i sigma_i(E).
            rates = _scatter_rates(Ea, Z_arr, n_cm3s)  # (n_elements, m) [1/cm]
            lam_ang = 1e8 / rates.sum(axis=0)  # mean free path [Ang]
            step = -lam_ang * np.log(rng.random(grp.size))  # sampled flight [Ang]

            d = dirs[grp]  # current unit direction of each electron
            p = pos[grp]  # current position [Ang]
            # -- 2. truncate at THIS layer's faces ------------------------------
            # The entrance face (z_top==0) and back face (z_bot==z_total) are
            # exits (vacuum -> no re-entry); an INTERNAL boundary instead hands
            # the electron to the neighbor layer with NO collision (it continues
            # straight and re-samples its free path in that layer next iteration).
            if groove is None:
                if finite_footprint:
                    exit_distance, exit_face = first_prism_exit(
                        p,
                        d,
                        z_min_ang=z_top_L,
                        z_max_ang=z_bot_L,
                        width_ang=width_ang,
                        height_ang=height_ang,
                    )
                    crossed_face = step > exit_distance
                    step = np.where(crossed_face, exit_distance, step)
                    cross_up = crossed_face & (exit_face == Z_MIN)
                    cross_dn = crossed_face & (exit_face == Z_MAX)
                    exit_side = crossed_face & (exit_face < Z_MIN)
                    exit_top = cross_up & (z_top_L <= 0.0)  # exited entrance (backscatter)
                    exit_bot = cross_dn & (z_bot_L >= z_total)  # exited back (transmit)
                else:
                    dz = d[:, 2]
                    pz = p[:, 2]
                    cross_up = (dz < 0) & (pz + step * dz < z_top_L)
                    cross_dn = (dz > 0) & (pz + step * dz > z_bot_L)
                    s_up = np.where(dz < 0, (pz - z_top_L) / (-dz + 1e-300), np.inf)
                    s_dn = np.where(dz > 0, (z_bot_L - pz) / (dz + 1e-300), np.inf)
                    step = np.where(cross_up, s_up, step)
                    step = np.where(cross_dn, s_dn, step)
                    exit_top = cross_up & (z_top_L <= 0.0)  # exited entrance (backscatter)
                    exit_bot = cross_dn & (z_bot_L >= z_total)  # exited back (transmit)
                    exit_side = np.zeros(grp.size, dtype=bool)
            else:
                if finite_footprint:
                    exit_distance, exit_face = first_prism_exit(
                        p,
                        d,
                        z_min_ang=z_top_L,
                        z_max_ang=z_bot_L,
                        width_ang=width_ang,
                        height_ang=height_ang,
                    )
                    crossed_face = step > exit_distance
                    step = np.where(crossed_face, exit_distance, step)
                    cross_up = crossed_face & (exit_face == Z_MIN)
                    cross_dn = crossed_face & (exit_face == Z_MAX)
                    exit_side = crossed_face & (exit_face < Z_MIN)
                    exit_top = cross_up & (z_top_L <= 0.0)  # exited entrance (backscatter)
                    exit_bot = cross_dn & (z_bot_L >= z_total)  # exited back (transmit)
                else:
                    dz = d[:, 2]
                    pz = p[:, 2]
                    cross_up = (dz < 0) & (pz + step * dz < z_top_L)
                    cross_dn = (dz > 0) & (pz + step * dz > z_bot_L)
                    s_up = np.where(dz < 0, (pz - z_top_L) / (-dz + 1e-300), np.inf)
                    s_dn = np.where(dz > 0, (z_bot_L - pz) / (dz + 1e-300), np.inf)
                    step = np.where(cross_up, s_up, step)
                    step = np.where(cross_dn, s_dn, step)
                    exit_top = cross_up & (z_top_L <= 0.0)  # exited entrance (backscatter)
                    exit_bot = cross_dn & (z_bot_L >= z_total)  # exited back (transmit)
                    exit_side = np.zeros(grp.size, dtype=bool)
                s_surface = np.array(
                    [
                        first_surface_event(pi, di, groove, transition="exit")
                        for pi, di in zip(p, d, strict=False)
                    ]
                )
                surface_first = s_surface < step
                step = np.where(surface_first, s_surface, step)
                cross_up = cross_up & ~surface_first
                cross_dn = cross_dn & ~surface_first
                exit_top = exit_top & ~surface_first
                exit_bot = exit_bot & ~surface_first
                exit_side = exit_side & ~surface_first

            # -- 3. record the segment (the radiation source list) --------------
            # midpoint -> escape-absorption path; direction -> v.g, v.n in the
            # amplitudes; length -> interaction time t_L; START energy -> beta.
            seg_mid.append(p + 0.5 * step[:, None] * d)
            seg_dir.append(d.copy())
            seg_len.append(step)
            seg_E.append(Ea.copy())
            seg_t0.append(clock[grp].copy())  # age at segment START [Ang, c=1]
            seg_id.append(grp.copy())  # which electron emitted this segment
            seg_lay.append(np.full(grp.size, L, dtype=np.int16))  # emitting layer

            # -- 4. advance: straight line + continuous slowing-down ------------
            # Energy drains deterministically along the flight (CSDA: Joy-Luo
            # modified Bethe for THIS layer's composition; no straggling). The
            # clock advances by L/beta at the segment's start speed.
            pos[grp] = p + step[:, None] * d
            E[grp] = Ea + _dEds_compound(J_arr, k_arr, coeff_arr, Ea) * step
            clock[grp] += step / beta_from_keV(Ea)

            if groove is not None:
                zero_event_counts = zero_surface_events
                assert zero_event_counts is not None
                below_cut = E[grp] < E_cut_keV
                active_surface = surface_first & ~below_cut
                zero_event_counts[grp[~active_surface]] = 0
                reentered_group = np.zeros(grp.size, dtype=bool)
            if groove is not None and active_surface.any():
                surf_local = np.flatnonzero(active_surface)
                surf_global = grp[surf_local]
                surface_points = pos[surf_global].copy()
                entry_distance = np.array(
                    [
                        first_surface_event(pi, di, groove, transition="entry")
                        for pi, di in zip(surface_points, dirs[surf_global], strict=False)
                    ]
                )
                if finite_footprint:
                    side_distance, side_face = first_prism_exit(
                        surface_points,
                        dirs[surf_global],
                        z_min_ang=0.0,
                        z_max_ang=z_total,
                        width_ang=width_ang,
                        height_ang=height_ang,
                    )
                    side_before_entry = (side_face < Z_MIN) & (side_distance < entry_distance)
                    exit_side[surf_local[side_before_entry]] = True
                    entry_distance = np.where(side_before_entry, np.inf, entry_distance)

                reentered = np.isfinite(entry_distance)
                if reentered.any():
                    re_local = surf_local[reentered]
                    re_global = grp[re_local]
                    reentered_group[re_local] = True
                    distance = entry_distance[reentered]
                    start = surface_points[reentered]
                    direction = dirs[re_global]
                    end = start + distance[:, None] * direction
                    vac_start.append(start)
                    vac_end.append(end)
                    vac_E.append(E[re_global].copy())
                    vac_t0.append(clock[re_global].copy())
                    vac_id.append(re_global.copy())
                    clock[re_global] += distance / beta_from_keV(E[re_global])
                    surface_eps = max(
                        64 * np.finfo(float).eps * groove.spacing_ang,
                        2e-12 * groove.spacing_ang,
                    )
                    pos[re_global] = end + surface_eps * direction
                    assert surface_events is not None
                    surface_events[re_global] += 1
                    if np.any(surface_events[re_global] > max_steps):
                        raise RuntimeError("grooved surface event limit exhausted")

                permanent = ~reentered & ~exit_side[surf_local]
                exit_top[surf_local[permanent]] = True
                short = step[surf_local] <= EPS
                zero_event_counts[surf_global] = np.where(
                    short, zero_event_counts[surf_global] + 1, 0
                )
                if np.any(zero_event_counts[surf_global] >= 2):
                    raise RuntimeError("repeated zero-length grooved surface events")

            # -- 5. kill exited / exhausted; pass internal crossers on ----------
            if groove is None:
                died = exit_top | exit_bot | exit_side | (E[grp] < E_cut_keV)
            else:
                died = exit_top | exit_bot | exit_side | below_cut
            n_back += int(exit_top.sum())  # exited the entrance face
            n_trans += int(exit_bot.sum())  # punched through the back face
            n_side += int(exit_side.sum())  # exited a transverse prism face
            alive[grp[died]] = False
            crossed_internal = (cross_up | cross_dn) & ~died  # reached a layer seam
            if crossed_internal.any():
                ci = grp[crossed_internal]
                pos[ci, 2] += np.sign(dirs[ci, 2]) * EPS  # nudge just into neighbor

            # -- 6. elastic collision: scatter the FULL-FLIGHT survivors --------
            # (truncated flights did not collide). The scattering ELEMENT is
            # chosen with probability n_i sigma_i / sum; the polar angle from that
            # element's screened-Rutherford inversion; azimuth uniform; E unchanged.
            if groove is None:
                full = ~(cross_up | cross_dn | exit_side) & ~died
            else:
                full = ~(cross_up | cross_dn | exit_side | surface_first) & ~died

            srv = grp[full]

            if srv.size:
                if Z_arr.size == 1:
                    cos_t = _sample_cos_theta(Z_arr[0], E[srv], rng, elastic_model, elements[0])

                else:
                    cos_t = np.empty(srv.size)

                    p_el = rates[:, full]
                    p_el = p_el / p_el.sum(axis=0)

                    u = rng.random(srv.size)
                    cum = np.cumsum(p_el, axis=0)
                    which = (u[None, :] > cum).sum(axis=0)  # element index

                    for i_el in range(Z_arr.size):
                        m = which == i_el
                        if m.any():
                            srv_i = srv[m]
                            cos_t[m] = _sample_cos_theta(
                                Z_arr[i_el],
                                E[srv_i],
                                rng,
                                elastic_model,
                                elements[i_el],
                            )
                phi = 2.0 * np.pi * rng.random(srv.size)
                dirs[srv] = _rotate_directions(dirs[srv], cos_t, phi)
            if groove is not None:
                assert material_steps is not None
                count_material_step = alive[grp] & ~reentered_group
                material_steps[grp[count_material_step]] += 1

    if finite_footprint and not seg_mid:
        r_mid = np.empty((0, 3), dtype=float)
        v_hat = np.empty((0, 3), dtype=float)
        L_ang = np.empty(0, dtype=float)
        E_keV = np.empty(0, dtype=float)
        t_ang = np.empty(0, dtype=float)
        elec_id = np.empty(0, dtype=np.int64)
        layer = np.empty(0, dtype=np.int16)
    else:
        r_mid = np.concatenate(seg_mid)
        v_hat = np.concatenate(seg_dir)
        L_ang = np.concatenate(seg_len)
        E_keV = np.concatenate(seg_E)
        t_ang = np.concatenate(seg_t0)
        elec_id = np.concatenate(seg_id)
        layer = np.concatenate(seg_lay)

    vacuum_start_ang = np.concatenate(vac_start) if vac_start else np.empty((0, 3))
    vacuum_end_ang = np.concatenate(vac_end) if vac_end else np.empty((0, 3))
    vacuum_E_keV = np.concatenate(vac_E) if vac_E else np.empty(0)
    vacuum_t_ang = np.concatenate(vac_t0) if vac_t0 else np.empty(0)
    vacuum_elec_id = (
        np.concatenate(vac_id).astype(np.int64) if vac_id else np.empty(0, dtype=np.int64)
    )
    # Broadcast the per-electron bunch offset onto every segment / vacuum flight
    # (constant per electron, like its identity), so a coherent sum can read
    # t_abs = t_ang + t0_ang without re-deriving which electron emitted a
    # segment. All-zero when no bunch was sampled (inert; nothing reads it yet).
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
        "E_keV": E_keV,
        "t_ang": t_ang,  # segment-start age sum(L/beta) [Ang, c=1]
        "t0_ang": t0_ang,  # per-electron longitudinal bunch offset [Ang, c=1]
        "elec_id": elec_id,  # emitting electron index in [0, Ne)
        "layer": layer,  # emitting layer index in [0, n_layers)
        "vacuum_start_ang": vacuum_start_ang,
        "vacuum_end_ang": vacuum_end_ang,
        "vacuum_E_keV": vacuum_E_keV,
        "vacuum_t_ang": vacuum_t_ang,
        "vacuum_t0_ang": vacuum_t0_ang,  # per-electron bunch offset [Ang, c=1]
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
