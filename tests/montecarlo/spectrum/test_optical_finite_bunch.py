"""CPU reference limits of finite-screen TR/DR physical bunch scoring (#616)."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.montecarlo.spectrum.optical_finite_bunch import (
    finite_screen_bunch_angular,
    finite_screen_bunch_band,
    finite_screen_realization_field,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    far_field_photons_per_eV_sr,
    spectral_energy_per_eV,
)

_E = 1.602176634e-19
_HBAR = 6.582119569e-16
_GRID = dict(nx=128, ny=128, dx_m=1e-7, dy_m=1e-7)


@dataclass(frozen=True)
class _Source:
    """Focused stand-in for #615 OpticalBunchSource's public source contract."""

    law: str
    physical_electrons: float
    history_weights: np.ndarray | None = None
    covariance_ang2: np.ndarray | None = None
    profile_ang: np.ndarray | None = None
    factor_override: float | None = None

    def form_factor_squared(self, energy_eV, direction):
        if self.factor_override is not None:
            return np.asarray(self.factor_override)
        cov = np.asarray(self.covariance_ang2)
        return np.exp(-((np.asarray(energy_eV) / 1973.269804593025) ** 2) * cov[0, 0])

    def record(self):
        return {"law": self.law, "physical_electrons": self.physical_electrons}


def _screen(**kw):
    return PECCircularScreen(**(dict(outer_radius_m=3.5e-6) | kw))


def _single():
    f = circular_pec_transition_field(1.0, _screen(), **_GRID)
    return far_field_photons_per_eV_sr(f)[2], spectral_energy_per_eV(f)


def _run(screens, source, energy=1.0):
    return finite_screen_bunch_angular(energy, screens, source=source, **_GRID)


def test_explicit_numpy_source_backend_and_unsupported_namespace():
    expected = circular_pec_transition_field(1.0, _screen(), **_GRID)
    actual = circular_pec_transition_field(1.0, _screen(), xp=np, **_GRID)
    np.testing.assert_array_equal(actual.electric_xy_v_s_m, expected.electric_xy_v_s_m)
    with pytest.raises(ValueError, match="backend"):
        circular_pec_transition_field(1.0, _screen(), xp=object(), **_GRID)


def test_n1_has_single_electron_energy_and_angular_yield():
    one, u = _single()
    result = _run([_screen()], _Source("sampled", 1))
    np.testing.assert_allclose(result.self_yield, one, rtol=1e-12, atol=1e-21)
    np.testing.assert_array_equal(result.bunch_yield, result.yield_per_electron)
    assert result.form_factor_squared == 1.0
    assert result.n_histories == 1
    np.testing.assert_allclose(result.collect_cone(theta_max_rad=np.pi / 2) * _E, u, rtol=1e-12)


def test_perfect_coherence_scales_as_n_squared_and_is_sample_count_invariant():
    one, _ = _single()
    for count in (2, 3, 6):
        result = _run([_screen()] * count, _Source("sampled", 100.0))
        np.testing.assert_allclose(result.bunch_yield, 100**2 * one, rtol=2e-12, atol=1e-20)
        np.testing.assert_allclose(result.self_yield, one, rtol=2e-12, atol=1e-20)
        np.testing.assert_allclose(result.pair_yield, one, rtol=2e-12, atol=1e-20)
        assert result.effective_histories == pytest.approx(count)


def test_weighted_distinct_pair_estimator_not_macro_charge_squared():
    # All fields identical: weights of 1, 2, 9 must yield the same physical
    # self and pair moments as unit weights. w² self would fail this limit.
    one, _ = _single()
    result = _run(
        [_screen()] * 3, _Source("sampled", 20.0, history_weights=np.array([1.0, 2.0, 9.0]))
    )
    np.testing.assert_allclose(result.self_yield, one, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(result.pair_yield, one, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(result.bunch_yield, 400 * one, rtol=2e-12, atol=1e-20)
    assert result.effective_histories == pytest.approx(12**2 / (1 + 4 + 81))


def test_out_of_phase_realization_has_zero_pair_total_but_finite_self():
    # Half-cycle apart: the 2 physical electrons radiate equal and opposite
    # complex fields, so self remains >0 while actual bunch yield vanishes.
    half_period = np.pi * _HBAR
    screens = [_screen(), _screen(arrival_time_s=half_period)]
    source = _Source("realization", 2.0)
    result = _run(screens, source)
    one, _ = _single()
    np.testing.assert_allclose(result.self_yield, one, rtol=1e-9, atol=1e-20)
    np.testing.assert_allclose(result.pair_yield, -one, rtol=1e-9, atol=1e-20)
    assert np.max(np.abs(result.bunch_yield)) < np.max(one) * 1e-26
    field = finite_screen_realization_field(1.0, screens, source=source, **_GRID)
    assert spectral_energy_per_eV(field) < _single()[1] * 1e-26


def test_realization_field_preserves_complex_sum_and_detector_plane_propagation():
    from pyrite.montecarlo.spectrum.optical_finite_screen import (
        observe_screen_green,
        propagate_screen_field,
    )

    screens = [_screen(arrival_time_s=0), _screen(arrival_time_s=0)]
    src = _Source("realization", 2.0)
    bunch = finite_screen_realization_field(1.0, screens, source=src, **_GRID)
    one = circular_pec_transition_field(1.0, screens[0], **_GRID)
    np.testing.assert_allclose(bunch.electric_xy_v_s_m, 2 * one.electric_xy_v_s_m)
    propagated = propagate_screen_field(bunch, 0.3e-6)
    sample = observe_screen_green(bunch, np.array([[0.1e-6, 0.1e-6, 0.3e-6]]))
    np.testing.assert_allclose(
        sample, 2 * observe_screen_green(one, np.array([[0.1e-6, 0.1e-6, 0.3e-6]]))
    )
    np.testing.assert_allclose(
        spectral_energy_per_eV(propagated), 4 * spectral_energy_per_eV(one), rtol=1e-12
    )


def test_gaussian_time_form_factor_and_finite_screen_angular_result():
    # Time spread only, computed from the same Gaussian source law as #615:
    # |F|² = exp[-(omega sigma_t)²], with RMS sigma_t rather than FWHM.
    rms_fs = 0.35
    cov = np.diag([(rms_fs * 2997.92458) ** 2, 0.0, 0.0])
    source = _Source("gaussian", 5.0, covariance_ang2=cov)
    result = _run([_screen()] * 3, source)
    one, _ = _single()
    expected = np.exp(-((1.0 * rms_fs * 1e-15 / _HBAR) ** 2))
    assert result.form_factor_squared == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(
        result.bunch_yield, 5 * (1 + 4 * expected) * one, rtol=1e-9, atol=1e-20
    )


def test_aperture_cdr_and_off_axis_histories_have_nonidentical_sources():
    screens = [_screen(inner_radius_m=1e-6), _screen(inner_radius_m=1e-6, electron_x_m=0.3e-6)]
    result = _run(screens, _Source("realization", 2.0))
    assert result.collect_cone(theta_max_rad=np.pi / 2) > 0
    a = circular_pec_transition_field(1.0, screens[0], **_GRID)
    b = circular_pec_transition_field(1.0, screens[1], **_GRID)
    assert not np.allclose(a.electric_xy_v_s_m, b.electric_xy_v_s_m, atol=0, rtol=1e-5)
    np.testing.assert_allclose(
        result.bunch_yield, _run(screens[::-1], _Source("realization", 2.0)).bunch_yield
    )


def test_band_output_feeds_calibrated_optical_energy_channel():
    source = _Source("sampled", 2.0)
    band = finite_screen_bunch_band(
        np.array([0.8, 1.0, 1.2]), [_screen()] * 3, source=source, theta_max_rad=np.pi / 2, **_GRID
    )
    assert band.collected_photons_per_eV_per_pulse.shape == (3,)
    assert np.all(band.collected_photons_per_eV_per_pulse > 0)
    detector = CalibratedPulseEnergyDetector(
        energy_eV=np.array([0.5, 1.5]),
        efficiency=np.ones(2) * 0.3,
        pulse_gain_v_per_j=1e9,
        calibration_id="synthetic test only",
    )
    result = detector.predict(
        band.energy_eV, band.collected_photons_per_eV_per_pulse, repetition_hz=1e6
    )
    assert result.absorbed_joule_per_pulse == pytest.approx(0.3 * result.incident_joule_per_pulse)
    assert result.expected_voltage_v == pytest.approx(1e9 * result.absorbed_joule_per_pulse)


@pytest.mark.parametrize(
    "screens,source,error",
    [
        ([_screen()], _Source("sampled", 8), ">=2"),
        ([_screen()], _Source("realization", 3), "requires N"),
        ([_screen(), _screen(outer_radius_m=1e-6)], _Source("sampled", 2), "same physical screen"),
        (
            [_screen(), _screen(arrival_time_s=1e-15)],
            _Source("gaussian", 3, covariance_ang2=np.diag([1.0, 0.0, 0.0])),
            "offset-free",
        ),
        (
            [_screen()] * 2,
            _Source("gaussian", 3, covariance_ang2=np.diag([1.0, 1.0, 0.0])),
            "temporal RMS only",
        ),
        (
            [_screen()] * 2,
            _Source("empirical", 3, profile_ang=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])),
            "temporal offsets only",
        ),
    ],
)
def test_refuse_unsupported_population_and_analytic_transverse_offsets(screens, source, error):
    with pytest.raises(ValueError, match=error):
        _run(screens, source)


def test_refuse_synthetic_population_field_and_invalid_cone():
    screens = [_screen()] * 2
    src = _Source("sampled", 1e7)
    with pytest.raises(ValueError, match="physical complex bunch field"):
        finite_screen_realization_field(1.0, screens, source=src, **_GRID)
    result = _run(screens, src)
    for kwargs in (
        {"theta_min_rad": 0.4, "theta_max_rad": 0.3},
        {"theta_max_rad": np.pi},
        {"theta_max_rad": np.nan},
    ):
        with pytest.raises(ValueError, match="cone bounds"):
            result.collect_cone(**kwargs)
    with pytest.raises(ValueError, match="increasing"):
        finite_screen_bunch_band([1.0, 0.9], screens, source=src, theta_max_rad=1.0, **_GRID)


def test_native_optical_bunch_source_contract_when_full_pyrite_available():
    """Integration with #615's class; unavailable in source-only test harness."""
    from importlib.util import find_spec

    if find_spec("pyrite.materials") is None:
        pytest.skip("source-only reconstruction lacks full PyRITE materials package")
    from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource

    electron_pc = _E * 1e12
    screens = [_screen()] * 3
    one, _ = _single()
    physical = OpticalBunchSource.sampled(10 * electron_pc, "unit test sample")
    sample = _run(screens, physical)
    np.testing.assert_allclose(sample.bunch_yield, 100 * one, rtol=1e-12, atol=1e-20)
    realized = OpticalBunchSource.realization(3 * electron_pc, "3 electrons")
    real = _run(screens, realized)
    np.testing.assert_allclose(real.bunch_yield, 9 * one, rtol=1e-12, atol=1e-20)
    gaussian = OpticalBunchSource.gaussian(
        4 * electron_pc,
        "temporal Gaussian",
        rms_t_fs=0.35,
        rms_x_um=0,
        rms_y_um=0,
    )
    g = _run(screens, gaussian)
    assert 0 < g.form_factor_squared < 1
    np.testing.assert_allclose(
        g.bunch_yield,
        4 * (1 + 3 * g.form_factor_squared) * one,
        rtol=1e-12,
        atol=1e-20,
    )
