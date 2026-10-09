"""
montecarlo.detector

Detector forward model (Zhai SI S3/S4): soft-X-ray window efficiency, the EDS /
aperture / mosaic line-broadening widths, the representative mosaic geometry
angle, and the Gaussian detector convolution.

Validation: surface-hkl-orientation
"""

import numpy as np

from ..materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from .geometry import _orientation_R, tilted_geometry
from .transport import beta_from_keV


def eds_fwhm_eV(E_eV):
    """Oxford UltimMax 170 resolution fit, SI Eq. (16)."""
    return np.sqrt(2.52 * E_eV + 988.0)


def aperture_fwhm_eV(E_eV, beta, theta_obs_rad, dtheta_obs_rad):
    """Line broadening from the detector polar-angle span, Zhai et al. 2025 SI Eq. (14):

        FWHM_dtheta_obs = (2*sqrt(2*ln2) / 3) * (dEp/dtheta_obs) * dtheta_obs

    dtheta_obs_rad is the FULL polar span (Delta theta_obs = 16.6 deg for the EDS
    aperture used here), and the Gaussian-equivalent width is Zhai's chosen
    stand-in for the (non-Gaussian) uniform angular span. dEp/dtheta_obs (SI Eq.
    (14), second line) reduces, for this beam-axis geometry, to
    E*beta*sin(theta_obs)/(1 - beta*cos(theta_obs)) -- the `dE_dth` term below.
    The prefactor 2*sqrt(2*ln2)/3 ~= 0.785 is adopted as reported by Zhai; a
    naive variance match to a uniform span of full width dtheta_obs would instead
    give sqrt(2*ln2/3) ~= 0.68, so we follow the paper's own convention rather
    than rederive it. Limiting case: FWHM -> 0 as dtheta_obs_rad -> 0 or beta -> 0
    (stationary source / no aperture -> no polar-angle broadening).

    NB: prior to 2026-07-11 this function returned sqrt(3) times the Eq. (14)
    value (the /3 was mistakenly pulled inside the sqrt), broadening every
    detector-convolved line by that factor.

    Validation: detector-line-broadening
    """
    dE_dth = E_eV * beta * np.sin(theta_obs_rad) / (1.0 - beta * np.cos(theta_obs_rad))
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) / 3.0 * dE_dth * dtheta_obs_rad


def mosaic_fwhm_eV(E_eV, psi_rad, mosaic_fwhm_rad):
    """Return first-order analytic line broadening from crystal mosaicity.

    A mosaic crystal is an incoherent ensemble of crystallites whose orientations
    are Gaussian-spread about the mean with a rocking-curve FWHM `mosaic_fwhm_rad`.
    Tilting a crystallite rotates its reciprocal vector g; only the NUMERATOR v.g of
    the resonance E_res = hbar c (v.g)/(1 - v.n) depends on g, so to first order the
    fractional line shift is dE/E = -tan(psi) dtheta, psi = angle(v, g). A Gaussian
    tilt of FWHM `mosaic_fwhm_rad` (in the longitudinal plane) therefore broadens the
    line by a Gaussian of energy FWHM

        FWHM_mosaic = E * |tan(psi)| * mosaic_fwhm_rad,

    added in quadrature with the EDS and aperture widths (results.store_result) and
    applied via the same convolve_detector pass -- the cheap analytic counterpart of
    a full Monte-Carlo average over crystallite orientations.

    Crude by construction: (i) it captures only the resonance-ENERGY shift, holding
    the amplitudes / lineshape weight fixed across the mosaic cone (good while the
    line stays narrow); (ii) the linearization diverges as psi -> 90 deg (g grazing
    the velocity), so the caller should cap the result (store_result clips it at E).
    The exact alternative is the per-orientation Gauss--Hermite sum inside
    mc_spectrum. psi is supplied by mosaic_psi_rad() at the nominal (unscattered)
    geometry. The derivation and model assumptions are documented in
    ``docs/physics/materials/crystal-mosaicity.md``. The width vanishes for zero mosaic spread or
    ``psi = 0`` and is even under ``psi -> -psi``.

    Validation: mosaic-analytic
    """
    return E_eV * np.abs(np.tan(psi_rad)) * mosaic_fwhm_rad


def mosaic_psi_rad(case, E_pk_eV):
    """psi = angle(beam velocity, g) [rad] of the reflection whose NOMINAL
    (unscattered-beam) resonance energy is nearest E_pk -- the representative
    geometry for the analytic mosaic broadening (mosaic_fwhm_eV). Mirrors how
    aperture_fwhm_eV uses the nominal theta_obs rather than the per-segment scattered
    directions. Returns None if no listed reflection radiates a positive line."""
    info = CRYSTALS[case["crystal"]]
    beam_dir, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    beta = beta_from_keV(case["E0_keV"])
    R = _orientation_R(
        info["lattice"],
        case.get("beam_uvw"),
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case.get("surface_hkl"),
    )
    denom = 1.0 - beta * float(beam_dir @ n_hat)  # g-independent (Doppler denominator)
    if denom <= 0.0:
        return None
    best = None
    for hkl in case["hkl_list"]:
        g_vec, g = reciprocal_g_vector(hkl, info["lattice"])
        if R is not None:
            g_vec = R @ g_vec
        bdotg = float(beam_dir @ g_vec)
        E_res = HBARC_EV_ANG * beta * bdotg / denom
        if E_res <= 0.0:
            continue
        psi = np.arccos(np.clip(bdotg / g, -1.0, 1.0))
        d = abs(E_res - E_pk_eV)
        if best is None or d < best[0]:
            best = (d, psi)
    return None if best is None else best[1]
