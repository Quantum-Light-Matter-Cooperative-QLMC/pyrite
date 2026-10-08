"""Production-law certificates for coherent line windows (#350)."""

import numpy as np
import pytest
from scipy.interpolate import PPoly

from pyrite.materials.crystal import HBARC_EV_ANG, refractive_index
from pyrite.montecarlo.spectrum.coherent_dispersion import (
    CoherentDispersionLaw,
    _polynomial_range,
)


def test_polynomial_enclosure_covers_interior_extrema_and_derivatives():
    # p(x) = x^3 - 3x has extrema at +/-1, absent from the endpoints.
    poly = PPoly(np.array([[1.0], [-6.0], [9.0], [-2.0]]), [-2.0, 2.0])
    x = np.linspace(-2.0, 2.0, 10001)
    for derivative in range(3):
        differentiated = poly.derivative(derivative)
        lo, hi = _polynomial_range(differentiated, -2.0, 2.0)
        values = differentiated(x)
        assert lo <= values.min() <= values.max() <= hi
    lo, hi = _polynomial_range(poly, -0.2, 0.4)
    values = poly(np.linspace(-0.2, 0.4, 1001))
    assert lo <= values.min() <= values.max() <= hi


@pytest.mark.parametrize("crystal", ["hopg", "wse2"])
@pytest.mark.parametrize("use_henke", [False, True])
def test_full_axis_law_matches_production_for_local_and_vector_queries(crystal, use_henke):
    law = CoherentDispersionLaw(crystal, 100.0, 6000.0, use_henke=use_henke)
    # One production call with exactly the full axis endpoints. Calls to law
    # on local subarrays must retain that same interpolation selection.
    energies = np.unique(np.r_[100.0, np.geomspace(100.1, 5999.9, 1001), law.breaks, 6000.0])
    expected = (1 - refractive_index(crystal, energies, use_henke).real) * energies / HBARC_EV_ANG
    np.testing.assert_allclose(law(energies), expected, rtol=2e-11, atol=2e-14)
    for index in (0, energies.size // 2, energies.size - 1):
        assert law(energies[index]) == pytest.approx(expected[index], abs=2e-14, rel=2e-11)


@pytest.mark.parametrize("crystal", ["hopg", "wse2"])
def test_certified_residual_bounds_the_production_phase_across_absorption_edges(crystal):
    law = CoherentDispersionLaw(crystal, 100.0, 6000.0)
    for lo, hi in zip(law.breaks[:-1], law.breaks[1:], strict=True):
        certificate = law.certificate(lo, hi)
        energies = np.linspace(lo, hi, 33)
        # Full endpoints keep xraydb's spline selection unchanged. This
        # expected value uses production, not the reconstructed interpolant.
        full = np.r_[law.start, energies, law.stop]
        actual = (1 - refractive_index(crystal, full).real) * full / HBARC_EV_ANG
        secant = certificate.slope * energies + certificate.intercept
        assert np.abs(actual[1:-1] - secant).max() <= certificate.residual_max
        # A secant cannot exceed the certified first-derivative supremum.
        assert abs(certificate.slope) <= certificate.first_derivative_max


def test_certificates_split_at_knots_and_refuse_unsupported_domains():
    law = CoherentDispersionLaw("hopg", 100.0, 6000.0)
    with pytest.raises(ValueError, match="every interpolation knot"):
        law.certificate(100.0, 6000.0)
    with pytest.raises(ValueError, match="inside its full axis"):
        law.certificate(50.0, 100.0)
    with pytest.raises(ValueError, match="certified full axis"):
        law(6001.0)
    with pytest.raises(ValueError, match="supported C table"):
        CoherentDispersionLaw("hopg", 0.1, 6000.0)
    with pytest.raises(ValueError, match="finite positive energy band"):
        CoherentDispersionLaw("hopg", 100.0, np.inf)


def test_local_production_queries_select_a_different_spline_near_the_carbon_edge():
    law = CoherentDispersionLaw("hopg", 100.0, 6000.0)
    energies = np.array([284.8, 285.0, 285.2])
    local = (1 - refractive_index("hopg", energies).real) * energies / HBARC_EV_ANG
    full = np.r_[law.start, energies, law.stop]
    expected = (1 - refractive_index("hopg", full).real) * full / HBARC_EV_ANG
    np.testing.assert_allclose(law(energies), expected[1:-1], rtol=2e-11, atol=2e-14)
    assert np.max(np.abs(local - expected[1:-1])) > 1e-6


def test_phase_derivative_bound_covers_the_exact_thomson_square_root():
    law = CoherentDispersionLaw("hopg", 1000.0, 1200.0, use_henke=False)
    certificate = law.certificate(1000.0, 1200.0)
    energies = np.linspace(1000.0, 1200.0, 10001)
    plasma_squared = law.prefactor * law.forward_constant
    exact_first = (1 - energies / np.sqrt(energies**2 - plasma_squared)) / HBARC_EV_ANG
    assert np.abs(exact_first).max() <= certificate.first_derivative_max
    exact = (energies - np.sqrt(energies**2 - plasma_squared)) / HBARC_EV_ANG
    affine = certificate.slope * energies + certificate.intercept
    assert np.abs(exact - affine).max() <= certificate.residual_max


def test_material_interpolation_fingerprint_changes_with_axis_and_dispersion_mode():
    signatures = {
        CoherentDispersionLaw("hopg", 100.0, 6000.0).fingerprint,
        CoherentDispersionLaw("hopg", 100.0, 5000.0).fingerprint,
        CoherentDispersionLaw("hopg", 100.0, 6000.0, use_henke=False).fingerprint,
    }
    assert len(signatures) == 3
