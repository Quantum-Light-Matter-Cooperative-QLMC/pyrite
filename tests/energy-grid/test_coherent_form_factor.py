"""Whole-band Gaussian bounds against independently evaluated endpoints."""

from functools import partial

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_form_factor import gaussian_form_factor_bounds


@pytest.mark.parametrize(
    "band,sigma",
    [
        ((0.0, 2000.0), 0.0),
        ((990.0, 1010.0), HBARC_EV_ANG / 1000),
        ((1000.0, np.nextafter(1000.0, np.inf)), 50.0),
        ((0.0, 0.0), 1e300),
        ((0.0, 1e-300), 1e-300),
        ((1.0, 2.0), HBARC_EV_ANG * np.sqrt(745.0)),
        ((1.0, 1.0), HBARC_EV_ANG * np.sqrt(746.0)),
        ((1.0, 1.0), HBARC_EV_ANG * np.sqrt(750.0)),
        ((1.0, 1.0), HBARC_EV_ANG * np.sqrt(744.0)),
    ],
)
def test_gaussian_bounds_enclose_every_energy_in_the_band(band, sigma):
    from mpmath import mp

    lo, hi = gaussian_form_factor_bounds(*band, sigma_z_ang=sigma)
    with mp.workdps(90):
        start, stop = map(mp.mpf, band)
        for fraction in (0, mp.mpf(1) / 7, mp.mpf(1) / 2, 1):
            E = start + fraction * (stop - start)
            value = mp.exp(-((E * mp.mpf(sigma) / mp.mpf(HBARC_EV_ANG)) ** 2))
            assert mp.mpf(lo) <= value <= mp.mpf(hi)
    assert 0 <= lo <= hi <= 1
    if sigma == 0 or band == (0.0, 0.0):
        assert (lo, hi) == (1.0, 1.0)


def test_positive_gaussian_underflow_is_not_certified_zero():
    bounds = gaussian_form_factor_bounds(1000.0, 1010.0, sigma_z_ang=100.0)
    assert bounds == (0.0, np.nextafter(0.0, np.inf))


def test_extreme_finite_gaussian_parameters_do_not_overflow_or_claim_zero():
    assert gaussian_form_factor_bounds(1e300, 1e308, sigma_z_ang=1e308) == (
        0.0,
        np.nextafter(0.0, np.inf),
    )


@pytest.mark.parametrize(
    "band,sigma",
    [((-1, 2), 1), ((2, 1), 1), ((1, np.inf), 1), ((1, 2), -1), ((1, 2), np.nan)],
)
def test_invalid_gaussian_inputs_are_refused(band, sigma):
    with pytest.raises(ValueError, match="Gaussian bounds"):
        gaussian_form_factor_bounds(*band, sigma_z_ang=sigma)


def test_gaussian_band_callback_certifies_a_two_electron_cancellation():
    from mpmath import mp

    from pyrite.montecarlo.spectrum.coherent_band_audit import certify_row_band_power
    from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
    from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField

    row = CoherentRowField(
        label="opposite electron fields",
        energy_eV=np.full(2, 1000.0),
        amplitude=np.array([[1.0, -1.0]]),
        duration_ang=np.ones(2),
        centre_ang=np.zeros(2),
        electron=np.array([0, 1]),
        start_transmission=np.ones(2),
        end_transmission=np.ones(2),
        mean_transmission=np.ones(2),
        attenuation_slope_ang=np.zeros(2),
        escape_mid_ang=np.zeros(2),
        escape_change_ang=np.zeros(2),
        phase_rad=np.zeros(2),
    )
    sigma = float(HBARC_EV_ANG / 1000)
    result = certify_row_band_power(
        row,
        CoherentDispersionLaw("hopg", 990, 1010, use_henke=False),
        form_factor_bounds=partial(gaussian_form_factor_bounds, sigma_z_ang=sigma),
        max_evaluations=1000,
        relative_tolerance=1e-3,
    )
    with mp.workdps(70):
        H = mp.mpf(HBARC_EV_ANG)
        actual = mp.quad(
            lambda E: (
                2
                * mp.sinc((E - 1000) / (2 * H)) ** 2
                * (1 - mp.exp(-((E * mp.mpf(sigma) / H) ** 2)))
            ),
            [990, 1010],
        )
        assert mp.mpf(result.lower) <= actual <= mp.mpf(result.upper)
        assert result.relative_error_upper(float(actual)) <= 1e-3
    assert len(result.intervals) > 1
