"""Regression tests for the EDS polar-aperture line-broadening formula
(`aperture_fwhm_eV`, Zhai et al. 2025 SI Eq. (14)).

Pins the corrected prefactor `2*sqrt(2*ln2)/3` after the 2026-07-11 fix for a
transcription bug that had pulled the /3 inside the square root (returning a
FWHM sqrt(3) times too large)."""

import numpy as np

from cxr_mc.montecarlo.detector import aperture_fwhm_eV


def test_aperture_fwhm_prefactor_matches_eq14():
    E_eV = 855.0
    beta = 0.272
    theta = np.deg2rad(119.0)
    dtheta = np.deg2rad(16.6)

    dE_dth = E_eV * beta * np.sin(theta) / (1.0 - beta * np.cos(theta))
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) / 3.0 * dE_dth * dtheta

    result = aperture_fwhm_eV(E_eV, beta, theta, dtheta)
    assert np.isclose(result, expected, rtol=1e-12)


def test_aperture_fwhm_scales_linearly_and_vanishes_as_dtheta_to_zero():
    E_eV = 855.0
    beta = 0.272
    theta = np.deg2rad(119.0)

    dtheta_small = np.deg2rad(1.0)
    dtheta_large = np.deg2rad(16.6)

    fwhm_small = aperture_fwhm_eV(E_eV, beta, theta, dtheta_small)
    fwhm_large = aperture_fwhm_eV(E_eV, beta, theta, dtheta_large)

    # linear in dtheta_obs_rad
    assert np.isclose(fwhm_large / fwhm_small, dtheta_large / dtheta_small, rtol=1e-12)

    # -> 0 as dtheta_obs_rad -> 0
    assert aperture_fwhm_eV(E_eV, beta, theta, 0.0) == 0.0
    assert aperture_fwhm_eV(E_eV, beta, theta, 1e-9) < 1e-6
