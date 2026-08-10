"""Regression coverage for the M4/M7 de-duplication helpers."""

import numpy as np

from pyrite.detectors import eaglexo_response as eag
from pyrite.plots.mpl.detectors import _eag_wide_brem, _eag_wide_charge
from pyrite.results import line_fwhm_eV


def _wide_brem_record():
    return {
        "E_grid_brem": np.array([1000.0, 2000.0, 3000.0]),
        "brem_wide": np.array([1.0, np.nan, 3.0]),
        "scale": 2.0,
    }


def test_eag_wide_brem_applies_qe_to_scaled_brem():
    record = _wide_brem_record()

    E, incident, detected = _eag_wide_brem(record, coating="BN")

    np.testing.assert_array_equal(E, record["E_grid_brem"])
    np.testing.assert_allclose(incident, [2.0, np.nan, 6.0], equal_nan=True)
    np.testing.assert_allclose(detected, incident * eag.qe(E, "BN"), equal_nan=True)


def test_eag_wide_charge_converts_detected_photons_to_charge_density():
    record = _wide_brem_record()

    E, charge_density = _eag_wide_charge(record, coating="BN", beam_current_na=3.0)

    expected_incident = np.nan_to_num(record["brem_wide"] * record["scale"])
    expected = expected_incident * eag.qe(E, "BN") * (E / eag.W_EHP_EV) * 3.0
    np.testing.assert_array_equal(E, record["E_grid_brem"])
    np.testing.assert_allclose(charge_density, expected)


def test_line_fwhm_eV_adds_capped_mosaic_term_in_quadrature():
    case = {
        "E0_keV": 30.0,
        "theta_obs_rad": np.deg2rad(119.0),
        "dtheta_obs_rad": np.deg2rad(1.0),
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "tilt_deg": 30.0,
    }

    perfect = line_fwhm_eV(case, E_pk=5000.0, mosaic_rad=None)
    mosaic = line_fwhm_eV(case, E_pk=5000.0, mosaic_rad=np.deg2rad(3.5))

    assert perfect > 0
    assert mosaic >= perfect


def test_line_fwhm_eV_uses_the_explicit_mosaic_grade():
    case = {
        "E0_keV": 30.0,
        "theta_obs_rad": np.deg2rad(119.0),
        "dtheta_obs_rad": np.deg2rad(1.0),
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "tilt_deg": 30.0,
    }

    assert line_fwhm_eV(case, E_pk=5000.0, mosaic_rad=None) < line_fwhm_eV(
        case, E_pk=5000.0, mosaic_rad=np.deg2rad(0.8)
    )
