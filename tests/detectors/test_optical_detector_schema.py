"""Optical/IR detector-class calibration schema (#617)."""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.detectors.optical_detector_schema import (
    OPTICAL_DETECTOR_CLASSES,
    UNSUPPORTED_DETECTOR_CLASSES,
    OpticalDetectorCalibration,
    pulse_gain_from_power_responsivity,
)
from pyrite.detectors.optical_temporal import DetectorTemporalResponse, predict_pulse_train

ID = "synthetic fixture, not a measured response"
ENERGY = np.array([1e-3, 1e-2, 1.0])
ABSORB = np.array([0.4, 0.6, 0.5])


def _record(cls, **params):
    return OpticalDetectorCalibration(cls, ID, ENERGY, ABSORB, params)


def _sine_fundamental(channel, f, power_w):
    """Lock-in fundamental of V(t) = G h * P(t) for P = P0 (1 + cos 2 pi f t), by quadrature."""
    g, h = channel.detector.pulse_gain_v_per_j, channel.temporal
    w = 2 * np.pi * f

    def hk(t):
        return float(h.impulse(np.array([t]))[0])

    upper = 80 * float(h.time_constants_s.max())
    re = quad(hk, 0, upper, weight="cos", wvar=w, limit=400)[0]
    im = quad(hk, 0, upper, weight="sin", wvar=w, limit=400)[0]
    return g * power_w * np.hypot(re, im)


def test_bolometer_dc_responsivity_reproduced():
    rec = _record(
        "bolometer",
        responsivity_v_per_w=2.5e3,
        thermal_time_constant_s=4e-3,
        nep_w_per_sqrt_hz=1e-11,
        noise_bandwidth_hz=100.0,
        operating_temperature_k=4.2,
    )
    ch = rec.build()
    gain = ch.detector.pulse_gain_v_per_j
    assert gain * ch.temporal.dc_integral_s == pytest.approx(2.5e3, rel=1e-12)
    assert gain == pytest.approx(2.5e3 / 4e-3, rel=1e-12)
    assert ch.detector.noise_rms_v == pytest.approx(2.5e3 * 1e-11 * 10.0, rel=1e-12)
    # A CW-equivalent dense train reproduces mean V = R * absorbed power.
    reading = predict_pulse_train(ch.detector, 1e-9, ch.temporal, 1e5)
    assert reading.mean_v == pytest.approx(2.5e3 * reading.average_absorbed_power_w, rel=1e-10)
    assert reading.dc_responsivity_v_per_w == pytest.approx(2.5e3, rel=1e-12)


def test_golay_modulated_responsivity_reproduced_by_quadrature():
    f_m = 12.5
    rec = _record(
        "golay",
        responsivity_v_per_w=1.0e4,
        responsivity_modulation_hz=f_m,
        time_constant_s=15e-3,
        nep_w_per_sqrt_hz=1.4e-10,
        noise_bandwidth_hz=1.0,
        shaping_time_constant_s=1e-4,
    )
    ch = rec.build()
    assert _sine_fundamental(ch, f_m, 1e-6) == pytest.approx(1.0e4 * 1e-6, rel=1e-5)
    # Lock-in of a square-free pulse train at f_m: 2 f U G |H| = 2 R (U f).
    reading = predict_pulse_train(ch.detector, 1e-8, ch.temporal, f_m)
    assert reading.fundamental_peak_v == pytest.approx(2 * 1.0e4 * 1e-8 * f_m, rel=1e-10)


def test_pyroelectric_and_energy_channel_keep_pulse_gain():
    pyro = _record(
        "pyroelectric",
        pulse_gain_v_per_j=1.2e5,
        thermal_time_constant_s=0.1,
        electrical_time_constant_s=1e-3,
        noise_rms_v=2e-4,
        saturation_v=5.0,
    ).build()
    assert pyro.temporal.is_ac_coupled
    assert pyro.detector.pulse_gain_v_per_j == 1.2e5
    assert pyro.detector.saturation_v == 5.0
    reference = DetectorTemporalResponse.pyroelectric(0.1, 1e-3, ID)
    t = np.linspace(0, 0.3, 50)
    np.testing.assert_allclose(pyro.temporal.impulse(t), reference.impulse(t), atol=1e-14)
    energy = _record("energy_channel", pulse_gain_v_per_j=3.0, time_constant_s=1e-6).build()
    assert energy.detector.noise_rms_v == 0.0


def test_ac_coupled_power_calibration_needs_modulation():
    h = DetectorTemporalResponse.pyroelectric(0.1, 1e-3, ID)
    with pytest.raises(ValueError, match="no DC responsivity"):
        pulse_gain_from_power_responsivity(10.0, h, 0.0)
    gain = pulse_gain_from_power_responsivity(10.0, h, 20.0)
    assert gain * abs(h.transfer(20.0)) == pytest.approx(10.0, rel=1e-12)


def test_schema_refusals_and_hash():
    base = dict(pulse_gain_v_per_j=3.0, time_constant_s=1e-6)
    with pytest.raises(ValueError, match="missing"):
        _record("energy_channel", pulse_gain_v_per_j=3.0)
    with pytest.raises(ValueError, match="unsupported"):
        _record("energy_channel", nep_w_per_sqrt_hz=1e-9, **base)
    with pytest.raises(ValueError, match="positive"):
        _record("energy_channel", pulse_gain_v_per_j=-1.0, time_constant_s=1e-6)
    with pytest.raises(ValueError, match="unknown"):
        _record("photodiode", **base)
    with pytest.raises(ValueError, match="calibration_id"):
        OpticalDetectorCalibration("energy_channel", " ", ENERGY, ABSORB, base)
    for cls in UNSUPPORTED_DETECTOR_CLASSES:
        assert cls not in OPTICAL_DETECTOR_CLASSES
        with pytest.raises(NotImplementedError, match="no forward operator"):
            _record(cls, **base)
    with pytest.raises(ValueError, match="modulation"):
        _record(
            "golay",
            responsivity_v_per_w=1.0,
            responsivity_modulation_hz=0.0,
            time_constant_s=1e-2,
            nep_w_per_sqrt_hz=1e-10,
            noise_bandwidth_hz=1.0,
        )
    with pytest.raises(ValueError, match="efficiency"):
        OpticalDetectorCalibration("energy_channel", ID, ENERGY, [0.4, 1.2, 0.5], base)
    with pytest.raises(ValueError, match="equals a response time constant"):
        _record("energy_channel", shaping_time_constant_s=1e-6, **base)
    a, b = _record("energy_channel", **base), _record("energy_channel", **base)
    assert a.schema_sha256 == b.schema_sha256
    c = _record("energy_channel", pulse_gain_v_per_j=3.0000001, time_constant_s=1e-6)
    assert c.schema_sha256 != a.schema_sha256
    assert a.schema_sha256[:16] in a.build().detector.calibration_id
