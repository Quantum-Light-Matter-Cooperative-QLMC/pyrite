"""Independent closed-form oracles for bounded two-time source research."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.quantum_regression import (
    regression_correlation,
    spectral_quadratic_form,
)

LOWERING = np.array([[0.0, 1.0], [0.0, 0.0]])
EXCITED = np.diag([0.0, 1.0])
GROUND = np.diag([1.0, 0.0])


def _solve(depths, *, gamma=0.0, density=EXCITED, currents=None, extra_jumps=(), **kwargs):
    jumps = (np.sqrt(gamma) * LOWERING, *extra_jumps)
    return regression_correlation(
        np.diag([0.0, 2.0]),
        jumps,
        density,
        [LOWERING] if currents is None else currents,
        depths,
        hbar_v_eV_ang=3.0,
        **kwargs,
    )


def test_damped_two_level_correlation_has_correct_complex_sign_and_decay():
    z = np.array([0.2, 0.7, 1.5, 3.0, 5.0])
    result = _solve(z, gamma=0.4)
    expected = np.exp(-0.2 * (z[:, None] + z[None, :]) - 2j / 3 * (z[:, None] - z[None, :]))
    np.testing.assert_allclose(result.values[:, :, 0, 0], expected, atol=2e-15)
    np.testing.assert_allclose(result.trace, 1, atol=2e-15)


def test_pure_dephasing_changes_delay_not_population():
    z = np.array([0.0, 0.3, 1.0, 2.0])
    eta = 0.7
    # L=sqrt(eta/2)*sigma_z gives coherence decay eta, no population loss.
    result = _solve(z, extra_jumps=(np.sqrt(eta / 2) * np.diag([1.0, -1.0]),))
    delay = z[:, None] - z[None, :]
    np.testing.assert_allclose(
        result.values[:, :, 0, 0], np.exp(-eta * abs(delay) - 2j * delay / 3)
    )


def test_complex_current_cross_terms_are_retained():
    result = _solve([0, 0.5, 1], currents=[LOWERING, 1j * LOWERING])
    scalar = result.values[:, :, 0, 0]
    np.testing.assert_allclose(result.values[:, :, 0, 1], -1j * scalar)
    np.testing.assert_allclose(result.values[:, :, 1, 0], 1j * scalar)
    amplitudes = np.ones((1, 3, 2), complex)
    amplitudes[:, :, 1] = 1j  # J + i*(iJ)=0, complete destructive interference.
    np.testing.assert_allclose(
        spectral_quadratic_form(result, [2 / 3], amplitudes=amplitudes), 0, atol=1e-14
    )


def test_positive_energy_peak_and_finite_time_sinc_limit():
    length = 6.0
    result = _solve(np.linspace(0, length, 129))
    phases = 2 / 3 + np.array([-0.5, 0.0, 0.5])
    expected = length**2 * np.sinc((phases - 2 / 3) * length / (2 * np.pi)) ** 2
    actual = spectral_quadratic_form(result, phases)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-12)
    assert actual[1] > actual[0]
    assert spectral_quadratic_form(result, [-2 / 3])[0] < actual[1] / 10


def test_depth_attenuation_is_applied_at_both_amplitudes():
    length, mu = 3.0, 0.8
    z = np.linspace(0, length, 129)
    result = _solve(z)
    amplitude = np.exp(-mu * (length - z) / 2)[None, :, None]
    actual = spectral_quadratic_form(result, [2 / 3], amplitudes=amplitude)[0]
    expected = ((1 - np.exp(-mu * length / 2)) / (mu / 2)) ** 2
    assert actual == pytest.approx(expected, rel=2e-5)


def test_correlations_are_linear_in_incident_density_and_ground_does_not_emit():
    z = [0.0, 0.5, 2.0]
    ground = _solve(z, density=GROUND)
    empty = _solve(z, density=np.zeros((2, 2)))
    mixed = _solve(z, gamma=0.2, density=0.3 * EXCITED + 0.7 * GROUND)
    reference = _solve(z, gamma=0.2)
    np.testing.assert_allclose(mixed.values, 0.3 * reference.values, atol=1e-15)
    assert np.count_nonzero(ground.values) == 0
    assert np.count_nonzero(empty.values) == 0
    assert spectral_quadratic_form(empty, [2 / 3])[0] == 0


def test_arbitrary_components_and_complex_weights_produce_positive_quadratic_form():
    rng = np.random.default_rng(13)
    currents = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    result = _solve(
        [0.0, 0.3, 1.0, 2.0], gamma=0.4, density=0.6 * EXCITED + 0.4 * GROUND, currents=currents
    )
    amplitudes = rng.normal(size=(20, 4, 2)) + 1j * rng.normal(size=(20, 4, 2))
    actual = spectral_quadratic_form(result, np.linspace(-3, 3, 20), amplitudes=amplitudes)
    assert np.all(actual >= 0)


def test_budgets_refuse_before_exponential_allocation(monkeypatch):
    import pyrite.montecarlo.channeling.quantum_regression as module

    def forbidden(*args, **kwargs):
        raise AssertionError("dense generator allocated before budget check")

    monkeypatch.setattr(module, "_generator", forbidden)
    with pytest.raises(ValueError, match="budget"):
        _solve([0, 1], max_working_bytes=1)
    with pytest.raises(ValueError, match="budget"):
        _solve([0, 1], max_recorded_depths=1)


@pytest.mark.parametrize("depths", [[], [1, 0], [-1, 0], [0, 0], [0, np.nan]])
def test_invalid_depths(depths):
    with pytest.raises(ValueError, match="depths_ang"):
        _solve(depths)


def test_invalid_initial_density_and_angular_weights():
    with pytest.raises(ValueError, match="positive"):
        _solve([0, 1], density=np.diag([-0.1, 1.1]))
    with pytest.raises(ValueError, match="Hermitian"):
        _solve([0, 1], density=LOWERING)
    result = _solve([0, 1], currents=[LOWERING, 1j * LOWERING])
    with pytest.raises(ValueError, match="explicit amplitudes"):
        spectral_quadratic_form(result, [0])
    with pytest.raises(ValueError, match="shape"):
        spectral_quadratic_form(result, [0], amplitudes=np.ones((2, 1)))


def test_spectral_budget_and_bounded_frequency_slices():
    result = _solve([0, 0.5, 1])
    phases = np.linspace(-2, 2, 7)
    with pytest.raises(ValueError, match="working-memory budget"):
        spectral_quadratic_form(result, phases, max_working_bytes=1)
    whole = spectral_quadratic_form(result, phases)
    sliced = np.concatenate([spectral_quadratic_form(result, phase) for phase in phases])
    np.testing.assert_allclose(whole, sliced, atol=1e-15)
