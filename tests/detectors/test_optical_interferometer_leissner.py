"""Published Martin--Puplett forward observable (#617 anchor).

Leissner, Berger, Siedling, Tonutti, Geitz, Schmidt and Schmüser, "Bunch
length measurements using a Martin Puplett interferometer at the TESLA Test
Facility linac", Proc. PAC 1999, p. 2172. Their Eq. (4) is the detected
coherent power spectrum of a Gaussian bunch of RMS duration ``sigma`` behind
the empirical high-pass ``g(w) = 1 - exp(-xi^2 w^2)``,

    S(w) = exp(-sigma^2 w^2) * (1 - exp(-xi^2 w^2))^2,

and Eq. (5) its closed-form autocorrelation, the normalized difference
interferogram ``(U1 - U2) / (U1 + U2)`` up to the constant ``S(t=0)``. The
paper reports ``sigma = 2 ps`` for the thermionic-gun scan of Fig. 3 and does
not print ``xi``; Fig. 4 puts the fitted spectral maximum near 0.106 THz,
which fixes ``xi`` through ``d S / d w = 0``.

Scope: a published closed form and figure features of its fit, normalized to
the central maximum, for ideal 45 degree optics. The measured points are not
digitized, the absolute modulation depth (about 0.9 in Fig. 3) is not
modeled, and no detector calibration or inference is involved.

Validation: optical-instrument-observable
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import (
    IdealMartinPuplett,
    OpticalSpectralCoherency,
    polarized_interferogram,
)

_HBAR_EV_PS = 0.6582119569509067e-3
_C_M_PS = 299792458.0e-12
_SIGMA_PS = 2.0  # published fit, Sec. 3.1


def _eq5(t_ps, sigma, xi):
    """Leissner et al. Eq. (5), as printed (not normalized at t = 0)."""
    a, b = sigma**2 + xi**2, sigma**2 + 2 * xi**2
    return (
        np.exp(-(t_ps**2) / (4 * sigma**2))
        - 2 * sigma * np.exp(-(t_ps**2) / (4 * a)) / np.sqrt(a)
        + sigma * np.exp(-(t_ps**2) / (4 * b)) / np.sqrt(b)
    )


def _difference_interferogram(t_ps, sigma, xi, *, nodes=2001, f_max_thz=1.2):
    """Normalized two-port difference from the production forward model."""
    omega = np.linspace(1e-4, 2 * np.pi * f_max_thz, nodes)  # rad/ps
    energy = _HBAR_EV_PS * omega
    power = np.exp(-((sigma * omega) ** 2)) * (1 - np.exp(-((xi * omega) ** 2))) ** 2
    # The detectors integrate energy: photons/eV = (energy spectrum) / E.
    source = OpticalSpectralCoherency.linearly_polarized(
        energy, power / energy, angle_rad=np.pi / 4, provenance="Leissner PAC 1999 Eq. (4)"
    )
    detector = CalibratedPulseEnergyDetector(
        [0.5 * energy[0], 2 * energy[-1]], [1.0, 1.0], 1.0, "flat reference absorber"
    )
    scan = polarized_interferogram(
        source,
        0.5 * _C_M_PS * np.asarray(t_ps),  # mirror travel: OPD = 2 x
        instrument=IdealMartinPuplett("ideal Martin-Puplett, 45 degree grids"),
        detectors=(detector, detector),
        max_spectral_elements=10_000_000,
    )
    v = scan.expected_voltage_v_ports
    return (v[:, 0] - v[:, 1]) / (v[:, 0] + v[:, 1])


def _xi_from_spectral_peak(sigma, f_peak_thz):
    """High-pass parameter placing the maximum of Eq. (4) at ``f_peak_thz``.

    dS/dw = 0 gives exp(-xi^2 w^2) = sigma^2 / (sigma^2 + 2 xi^2).
    """
    w2 = (2 * np.pi * f_peak_thz) ** 2
    return brentq(lambda xi: xi**2 * w2 - np.log1p(2 * xi**2 / sigma**2), 0.3, 5.0)


@pytest.mark.parametrize("xi", [0.5, 1.07, 2.0, 3.0])
def test_two_port_difference_equals_published_closed_form(xi):
    t = np.linspace(-25.0, 25.0, 401)
    expected = _eq5(t, _SIGMA_PS, xi) / _eq5(0.0, _SIGMA_PS, xi)
    np.testing.assert_allclose(
        _difference_interferogram(t, _SIGMA_PS, xi), expected, rtol=0, atol=1e-10
    )


def test_published_fit_features_of_figures_3_and_4():
    # Readings of the solid fit curves (400 dpi render, fresh-context review
    # 2026-11-09): Fig. 4 peak 0.1056 +- 0.005 THz; Fig. 3 first zero
    # 2.24 +- 0.1 ps, side minima at 4.1 +- 0.2 ps with -0.611 +- 0.015 of the
    # central maximum (plotted: -0.547 against a maximum of 0.895).
    xi = _xi_from_spectral_peak(_SIGMA_PS, 0.1056)
    assert xi == pytest.approx(0.753, abs=0.005)
    t = np.linspace(0.0, 25.0, 1251)
    curve = _difference_interferogram(t, _SIGMA_PS, xi)
    assert curve[0] == pytest.approx(1.0, abs=1e-12)  # normalized to the central maximum
    assert t[np.argmax(curve < 0)] == pytest.approx(2.24, abs=0.2)
    assert t[np.argmin(curve)] == pytest.approx(4.1, abs=0.3)
    assert curve.min() == pytest.approx(-0.611, abs=0.03)
    # The side-minimum depth depends only weakly on the spectral-peak reading.
    for f_peak in (0.10, 0.11):
        other = _difference_interferogram(t, _SIGMA_PS, _xi_from_spectral_peak(_SIGMA_PS, f_peak))
        assert abs(other.min() - curve.min()) < 0.01
