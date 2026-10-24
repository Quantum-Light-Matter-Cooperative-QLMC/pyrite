"""Analytic limits of the bulk Frank--Tamm Cherenkov source (#587)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cherenkov import (
    CHERENKOV_MODEL,
    FRANK_TAMM_PER_EV_M,
    ChargedPassage,
    ConstantIndex,
    cherenkov_threshold_beta,
    frank_tamm_yield,
)

# alpha / (hbar c) from CODATA 2018: 1/137.035999084 / 197.3269804 nm eV.
_ALPHA_OVER_HBARC_PER_EV_CM = 369.8101


def test_prefactor_is_alpha_over_hbar_c():
    assert FRANK_TAMM_PER_EV_M / 100.0 == pytest.approx(_ALPHA_OVER_HBARC_PER_EV_CM, rel=1e-6)


@pytest.mark.parametrize(("beta", "n", "charge"), [(0.99, 1.33, -1.0), (0.8, 1.5, 2.0)])
def test_nondispersive_closed_form_and_cone(beta, n, charge):
    energy = np.linspace(1.5, 4.0, 7)
    result = frank_tamm_yield(ChargedPassage(beta, charge), energy, n**2)
    sin2 = 1.0 - 1.0 / (beta * n) ** 2
    np.testing.assert_allclose(
        result.yield_per_m_per_eV, FRANK_TAMM_PER_EV_M * charge**2 * sin2, rtol=1e-14
    )
    np.testing.assert_allclose(np.sin(result.theta_rad) ** 2, sin2, rtol=1e-12)
    np.testing.assert_allclose(result.refractive_index, n, rtol=1e-15)
    assert result.mechanism == CHERENKOV_MODEL
    assert np.all(result.above_threshold)


def test_charge_sign_does_not_enter():
    neg = frank_tamm_yield(ChargedPassage(0.95, -1.0), 2.0, 1.8)
    pos = frank_tamm_yield(ChargedPassage(0.95, +1.0), 2.0, 1.8)
    assert neg.yield_per_m_per_eV == pos.yield_per_m_per_eV


def test_threshold_and_zero_yield_limits():
    n = 1.33
    beta_t = float(cherenkov_threshold_beta(n**2))
    assert beta_t == pytest.approx(1.0 / n, rel=1e-15)
    for beta in (0.0, 0.5, beta_t):
        below = frank_tamm_yield(ChargedPassage(beta), [2.0, 3.0], n**2)
        np.testing.assert_array_equal(below.yield_per_m_per_eV, 0.0)
        assert not below.above_threshold.any()
        assert np.isnan(below.cos_theta).all()
    above = frank_tamm_yield(ChargedPassage(beta_t * (1 + 1e-6)), 2.0, n**2)
    # Linear onset: sin^2 theta = 1 - (beta_t/beta)^2 ~ 2e-6.
    assert above.yield_per_m_per_eV == pytest.approx(FRANK_TAMM_PER_EV_M * 2e-6, rel=1e-5)
    assert above.cos_theta == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("eps", [1.0, 1.0 - 1e-6 + 1e-9j, 0.9])
def test_vacuum_and_sub_unity_index_emit_nothing(eps):
    result = frank_tamm_yield(ChargedPassage(0.999999), [10.0, 1e4], eps)
    np.testing.assert_array_equal(result.yield_per_m_per_eV, 0.0)
    assert np.isinf(cherenkov_threshold_beta(eps))


def test_ultrarelativistic_cone_saturates_at_inverse_index():
    result = frank_tamm_yield(ChargedPassage(1.0 - 1e-12), 2.0, 1.5**2)
    assert result.cos_theta == pytest.approx(1.0 / 1.5, rel=1e-11)


def test_dispersive_band_integral_matches_closed_form():
    # n^2 = a + b E^2: int (1 - 1/(beta^2 n^2)) dE has a closed arctan form.
    a, b, beta = 1.7, 0.01, 0.9
    lo, hi = 1.0, 6.0
    energy = np.linspace(lo, hi, 20001)
    result = frank_tamm_yield(ChargedPassage(beta), energy, a + b * energy**2)

    def primitive(e):
        return e - np.arctan(e * np.sqrt(b / a)) / (beta**2 * np.sqrt(a * b))

    expected = FRANK_TAMM_PER_EV_M * (primitive(hi) - primitive(lo))
    assert np.trapezoid(result.yield_per_m_per_eV, energy) == pytest.approx(expected, rel=1e-8)


def test_dispersive_threshold_is_per_energy():
    # Normal dispersion: only the higher-index (higher-energy) photons radiate.
    energy = np.array([1.0, 2.0, 3.0, 4.0])
    eps = (1.30 + 0.01 * energy) ** 2
    beta = 1.0 / 1.325
    result = frank_tamm_yield(ChargedPassage(beta), energy, eps)
    np.testing.assert_array_equal(result.above_threshold, [False, False, True, True])
    np.testing.assert_array_equal(result.yield_per_m_per_eV[:2], 0.0)
    assert np.all(np.diff(result.yield_per_m_per_eV[2:]) > 0)
    np.testing.assert_allclose(
        cherenkov_threshold_beta(eps), 1.0 / (1.30 + 0.01 * energy), rtol=1e-15
    )


def test_weak_loss_uses_real_index_and_is_continuous():
    eps0 = 1.33**2
    lossless = frank_tamm_yield(ChargedPassage(0.95), 2.0, eps0)
    lossy = frank_tamm_yield(ChargedPassage(0.95), 2.0, eps0 * (1 + 1e-4j))
    assert lossy.refractive_index == pytest.approx(np.sqrt(eps0 * (1 + 1e-4j)).real, rel=1e-15)
    # Re sqrt(eps (1 + i t)) = n (1 + t^2/8 + ...): second order in the loss tangent.
    assert lossy.yield_per_m_per_eV == pytest.approx(lossless.yield_per_m_per_eV, rel=1e-8)


class _Response:
    def epsilon(self, energy_eV):
        return 1.5**2 + 0.0 * np.asarray(energy_eV) + 1e-6j


def test_accepts_explicit_dielectric_response():
    energy = np.array([1.0, 2.0])
    via_object = frank_tamm_yield(ChargedPassage(0.9), energy, _Response())
    via_array = frank_tamm_yield(ChargedPassage(0.9), energy, _Response().epsilon(energy))
    np.testing.assert_array_equal(via_object.yield_per_m_per_eV, via_array.yield_per_m_per_eV)


def test_constant_index_is_the_nondispersive_limit():
    """Validation: frank-tamm-homogeneous-dielectric (constant-n input)."""
    energy = np.linspace(1.5, 4.0, 5)
    medium = ConstantIndex(1.46)
    via_object = frank_tamm_yield(ChargedPassage(0.95), energy, medium)
    via_scalar = frank_tamm_yield(ChargedPassage(0.95), energy, 1.46**2)
    np.testing.assert_array_equal(via_object.yield_per_m_per_eV, via_scalar.yield_per_m_per_eV)
    np.testing.assert_allclose(via_object.cos_theta, 1 / (0.95 * 1.46), rtol=1e-15)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="positive"):
            ConstantIndex(bad)


@pytest.mark.parametrize(
    ("eps", "message"),
    [
        (2.0 - 1e-6j, "passive"),
        (-1.0 + 0.1j, "Re epsilon"),
        (0.0 + 0.0j, "Re epsilon"),
        (2.0 + 0.01j, "loss tangent"),
        (np.nan, "finite"),
    ],
)
def test_out_of_scope_responses_are_refused(eps, message):
    with pytest.raises(ValueError, match=message):
        frank_tamm_yield(ChargedPassage(0.9), 2.0, eps)


def test_loss_gate_is_configurable():
    frank_tamm_yield(ChargedPassage(0.9), 2.0, 2.0 + 0.01j, max_loss_tangent=1e-2)
    with pytest.raises(ValueError, match="max_loss_tangent"):
        frank_tamm_yield(ChargedPassage(0.9), 2.0, 2.0, max_loss_tangent=-1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": 1.0},
        {"beta": -0.1},
        {"beta": 0.5, "charge": np.inf},
        {"beta": 0.5, "path_length_m": -1},
    ],
)
def test_passage_validation(kwargs):
    with pytest.raises(ValueError):
        ChargedPassage(**kwargs)


@pytest.mark.parametrize("energy", [0.0, -1.0, np.inf])
def test_energy_validation(energy):
    with pytest.raises(ValueError, match="energy_eV"):
        frank_tamm_yield(ChargedPassage(0.9), energy, 2.0)


def test_path_scaling_and_angular_histogram():
    energy = np.array([1.0, 2.0, 3.0])
    eps = np.array([1.2, 1.5, 2.0]) ** 2
    result = frank_tamm_yield(ChargedPassage(0.9, path_length_m=0.02), energy, eps)
    np.testing.assert_allclose(result.photons_per_eV(), 0.02 * result.yield_per_m_per_eV)
    edges = np.radians(np.arange(0.0, 91.0, 1.0))
    hist = result.angular_histogram(edges)
    assert hist.shape == (3, 90)
    np.testing.assert_allclose(hist.sum(axis=-1), result.yield_per_m_per_eV, rtol=1e-15)
    expected_bins = np.floor(np.degrees(result.theta_rad[result.above_threshold])).astype(int)
    np.testing.assert_array_equal(np.argmax(hist[result.above_threshold], axis=-1), expected_bins)
    # No cone lies below 1 degree: yield outside the edges is dropped.
    assert result.angular_histogram(np.radians([0.0, 1.0])).sum() == 0.0
    with pytest.raises(ValueError, match="theta_edges_rad"):
        result.angular_histogram([0.1, 0.1])
    with pytest.raises(ValueError, match="path_length_m"):
        frank_tamm_yield(ChargedPassage(0.9), 2.0, 2.0).photons_per_eV()
