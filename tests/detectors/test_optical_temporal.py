"""Temporal response, pulse trains and digitized counts of an optical channel (#617)."""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_temporal import (
    DetectorTemporalResponse,
    Digitizer,
    predict_pulse_train,
)

ID = "synthetic fixture, not a measured response"


def _brute_train(response, t, f, n_pulses):
    """Direct superposition of shifted impulses (independent of the geometric sum)."""
    period = 1 / f
    out = np.zeros_like(t)
    for k in range(n_pulses):
        out += response.impulse(t - k * period)
    return out


def test_single_pole_limits_and_transfer():
    tau = 2e-3
    h = DetectorTemporalResponse.single_pole(tau, ID)
    assert h.peak_time_s == 0.0
    assert h.impulse(np.array([0.0, tau, -1e-9])) == pytest.approx([1, np.exp(-1), 0])
    assert h.dc_integral_s == pytest.approx(tau)
    assert not h.is_ac_coupled
    f = np.array([0.0, 10.0, 1e3])
    np.testing.assert_allclose(h.transfer(f), tau / (1 + 2j * np.pi * f * tau), rtol=1e-14)
    for rate in (1.0, 500.0, 1e5):
        s = h.steady_state(rate)
        r = np.exp(-1 / (rate * tau))
        assert s.peak == pytest.approx(1 / (1 - r), rel=1e-10)
        assert s.minimum == pytest.approx(r / (1 - r), rel=1e-6)
        assert s.mean == pytest.approx(rate * tau, rel=1e-12)
    # Low rate: isolated pulses; high rate: DC level f*tau with vanishing ripple.
    assert h.steady_state(1.0).peak == pytest.approx(1.0, abs=1e-12)
    fast = h.steady_state(1e6)
    assert fast.peak / fast.mean == pytest.approx(1.0, rel=1e-3)
    assert fast.minimum / fast.mean == pytest.approx(1.0, rel=1e-3)


def test_pyroelectric_closed_form_and_porter_responsivity_shape():
    tau_t, tau_e = 0.1, 1e-3
    h = DetectorTemporalResponse.pyroelectric(tau_t, tau_e, ID)
    t = np.linspace(0, 0.5, 2001)
    closed = (tau_t * np.exp(-t / tau_e) - tau_e * np.exp(-t / tau_t)) / (tau_t - tau_e)
    np.testing.assert_allclose(h.impulse(t), closed, atol=1e-13)
    assert h.is_ac_coupled
    assert quad(
        lambda x: float(h.impulse(np.array([x]))[0]), 0, 50, points=[tau_e, tau_t], limit=400
    )[0] == pytest.approx(0, abs=1e-9)
    w = 2 * np.pi * np.geomspace(0.01, 1e5, 40)
    porter = 1j * w * tau_e * tau_t / ((1 + 1j * w * tau_t) * (1 + 1j * w * tau_e))
    np.testing.assert_allclose(h.transfer(w / (2 * np.pi)), porter, rtol=1e-10)
    # AC coupled: steady-state mean is zero; a lock-in reads the fundamental.
    s = h.steady_state(20.0)
    assert s.mean == pytest.approx(0, abs=1e-12)
    assert s.minimum < 0 < s.peak


@pytest.mark.parametrize("rate", [3.0, 40.0, 900.0])
def test_trains_equal_direct_superposition(rate):
    h = DetectorTemporalResponse.pyroelectric(0.05, 2e-3, ID).with_low_pass(1e-4)
    period = 1 / rate
    t = np.linspace(-0.3 * period, 6.7 * period, 3001)
    np.testing.assert_allclose(h.train(t, rate, 5), _brute_train(h, t, rate, 5), atol=1e-12)
    # Steady state = a long train observed late, in time since its latest pulse.
    n = int(np.ceil(40 * 0.05 * rate)) + 2
    late = (n - 1) * period + np.linspace(0, period, 501, endpoint=False)
    np.testing.assert_allclose(
        h.train(late - (n - 1) * period, rate), _brute_train(h, late, rate, n), atol=1e-10
    )


@pytest.mark.parametrize("rate", [5.0, 200.0])
def test_steady_state_extrema_mean_and_fundamental(rate):
    h = DetectorTemporalResponse.pyroelectric(0.05, 2e-3, ID).with_low_pass(2e-4)
    s = h.steady_state(rate)
    period = 1 / rate
    t = np.linspace(0, period, 200001, endpoint=False)
    wave = h.train(t, rate)
    assert s.peak == pytest.approx(wave.max(), rel=1e-6)
    assert s.minimum == pytest.approx(wave.min(), rel=1e-6, abs=1e-9)
    assert s.mean == pytest.approx(wave.mean(), abs=2e-5 * s.peak)
    c1 = np.mean(wave * np.exp(-2j * np.pi * rate * t))
    assert s.fundamental_amplitude == pytest.approx(2 * abs(c1), rel=1e-4)


def test_low_pass_moves_the_peak_and_keeps_dc():
    base = DetectorTemporalResponse.single_pole(1e-3, ID)
    shaped = base.with_low_pass(1e-4)
    t = np.linspace(0, 5e-3, 200001)
    ref = (np.exp(-t / 1e-3) - np.exp(-t / 1e-4)) / (1 - 0.1)  # analytic convolution
    k = int(np.argmax(ref))
    assert shaped.peak_time_s == pytest.approx(t[k], abs=1e-7)
    np.testing.assert_allclose(shaped.impulse(t), ref / ref[k], atol=1e-6)
    # The DC integral is the shape's, renormalized by its lower peak.
    assert shaped.dc_integral_s == pytest.approx(1e-3 / ref.max(), rel=1e-6)


def test_predict_pulse_train_units_and_pileup():
    det = CalibratedPulseEnergyDetector(
        energy_eV=np.array([0.01, 1.0]),
        efficiency=np.array([0.5, 0.5]),
        pulse_gain_v_per_j=1e6,
        calibration_id=ID,
        saturation_v=3.0,
    )
    bolo = DetectorTemporalResponse.single_pole(1e-3, ID)
    r = predict_pulse_train(det, 1e-6, bolo, 5e3)
    assert r.isolated_peak_v == pytest.approx(1.0)
    assert r.average_absorbed_power_w == pytest.approx(5e-3)
    assert r.dc_responsivity_v_per_w == pytest.approx(1e6 * 1e-3)
    assert r.mean_v == pytest.approx(r.dc_responsivity_v_per_w * r.average_absorbed_power_w)
    assert r.pileup_ratio == pytest.approx(1 / (1 - np.exp(-0.2)))
    assert r.steady_peak_v > 3.0 and r.saturated
    assert r.fundamental_rms_v == pytest.approx(r.fundamental_peak_v / np.sqrt(2))
    assert r.response_sha256 == det.response_sha256
    assert r.temporal_sha256 == bolo.response_sha256
    isolated = predict_pulse_train(det, 1e-6, bolo, 1.0)
    assert isolated.pileup_ratio == pytest.approx(1.0) and not isolated.saturated
    pyro = DetectorTemporalResponse.pyroelectric(0.1, 1e-3, ID)
    ac = predict_pulse_train(det, 1e-6, pyro, 10.0)
    assert ac.dc_responsivity_v_per_w == pytest.approx(0, abs=1e-6)
    assert ac.mean_v == pytest.approx(0, abs=1e-9)


def test_digitizer_counts_clip_and_noise():
    adc = Digitizer(counts_per_v=1000.0, bits=12, calibration_id=ID, offset_counts=100.0)
    counts, clipped = adc.counts(np.array([0.0, 1.0, 5.0, -0.2]))
    np.testing.assert_allclose(counts, [100, 1100, 4095, 0])
    assert clipped.tolist() == [False, False, True, True]
    assert adc.noise_rms_counts(np.array(0.0)) == pytest.approx(np.sqrt(1 / 12))
    assert adc.noise_rms_counts(np.array(2e-3)) == pytest.approx(np.sqrt(4 + 1 / 12))


def test_refusals_and_hash():
    with pytest.raises(ValueError, match="distinct"):
        DetectorTemporalResponse(np.array([1.0, 1.0]), np.array([1e-3, 1e-3]), ID)
    with pytest.raises(ValueError, match="equals"):
        DetectorTemporalResponse.single_pole(1e-3, ID).with_ac_coupling(1e-3)
    with pytest.raises(ValueError, match="positive peak"):
        DetectorTemporalResponse(np.array([-1.0]), np.array([1e-3]), ID)
    with pytest.raises(ValueError, match="calibration_id"):
        DetectorTemporalResponse.single_pole(1e-3, " ")
    with pytest.raises(ValueError, match="n_pulses"):
        DetectorTemporalResponse.single_pole(1e-3, ID).train(np.zeros(2), 1.0, 0)
    with pytest.raises(ValueError, match="bits"):
        Digitizer(1.0, 0, ID)
    a = DetectorTemporalResponse.single_pole(1e-3, ID)
    b = DetectorTemporalResponse(np.array([7.0]), np.array([1e-3]), ID)
    assert a.response_sha256 == b.response_sha256  # amplitude scale is normalized away
    assert a.response_sha256 != DetectorTemporalResponse.single_pole(2e-3, ID).response_sha256


def test_energy_reading_reports_photons_per_pulse():
    det = CalibratedPulseEnergyDetector(
        energy_eV=np.array([1.0, 3.0]),
        efficiency=np.array([0.5, 0.5]),
        pulse_gain_v_per_j=1.0,
        calibration_id=ID,
    )
    energy = np.array([1.0, 1.5, 3.0])
    out = det.predict(energy, np.full(3, 2e6))
    assert out.incident_photons_per_pulse == pytest.approx(4e6)
    assert out.incident_photons.to("dimensionless").magnitude == pytest.approx(4e6)
    batch = det.predict_batch(energy, np.full((2, 3), 2e6))
    np.testing.assert_allclose(batch.incident_photons_per_pulse, [4e6, 4e6])
