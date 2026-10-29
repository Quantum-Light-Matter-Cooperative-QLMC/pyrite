"""Explicit optical calibrated single-pulse channel checks (#617)."""

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector


def _calibrated(**kwargs):
    params = dict(
        energy_eV=np.array([1.0, 2.0, 3.0]),
        efficiency=np.array([0.5, 0.5, 0.5]),
        pulse_gain_v_per_j=1e12,
        calibration_id="measured example / fixture (not a published benchmark)",
    )
    params.update(kwargs)
    return CalibratedPulseEnergyDetector(**params)


def test_physical_joule_pulse_and_average_watts_nonuniform_grid():
    det = _calibrated()
    energy = np.array([1.0, 1.3, 2.2, 3.0])
    photons = np.ones(energy.size) * 1e7
    out = det.predict(energy, photons, repetition_hz=1e6)
    # integral from 1 to 3 eV of E dE = 4 eV^2
    expected = 4 * 1e7 * 1.602176634e-19
    assert out.incident_joule_per_pulse == pytest.approx(expected)
    assert out.absorbed_joule_per_pulse == pytest.approx(expected / 2)
    assert out.expected_voltage_v == pytest.approx(expected / 2 * 1e12)
    assert out.reported_voltage_v == out.expected_voltage_v
    assert out.average_incident_power_w == pytest.approx(expected * 1e6)
    assert out.average_absorbed_power_w == pytest.approx(expected / 2 * 1e6)
    assert not out.saturated
    assert out.response_sha256 == det.response_sha256


def test_interpolation_noise_saturation_and_no_dc_voltage_assumption():
    det = _calibrated(
        efficiency=np.array([0.0, 0.5, 1.0]),
        noise_rms_v=0.002,
        saturation_v=0.001,
    )
    energy = np.linspace(1, 3, 101)
    out = det.predict(energy, np.ones_like(energy) * 1e8, repetition_hz=2e6)
    # eps(E)=(E-1)/2; integral E*eps(E) from 1 to 3 is 7/3 eV^2.
    expected_j = (7 / 3) * 1e8 * 1.602176634e-19
    assert out.absorbed_joule_per_pulse == pytest.approx(expected_j, rel=2e-4)
    assert out.expected_voltage_v > out.reported_voltage_v
    assert out.reported_voltage_v == pytest.approx(0.001)
    assert out.noise_rms_voltage_v == 0.002
    assert out.saturated
    out2 = det.predict(energy, np.ones_like(energy) * 1e8, repetition_hz=3e6)
    assert out2.expected_voltage_v == out.expected_voltage_v  # not V/W
    assert out2.average_absorbed_power_w > out.average_absorbed_power_w


def test_zero_photons_and_response_fingerprint():
    det = _calibrated()
    energy = np.array([1.0, 2.0, 3.0])
    zero = det.predict(energy, np.zeros_like(energy))
    assert zero.expected_voltage_v == 0
    assert zero.average_incident_power_w is None
    assert zero.average_absorbed_power_w is None
    assert _calibrated().response_sha256 == det.response_sha256
    assert _calibrated(efficiency=np.array([0.5, 0.6, 0.5])).response_sha256 != det.response_sha256
    # Calibration arrays are copied, made read-only; external mutation cannot change calibration.
    with pytest.raises(ValueError):
        det.efficiency[0] = 0


@pytest.mark.parametrize(
    "energies,photons",
    [
        ([0.5, 1.0], [0.0, 1.0]),
        ([1.0, 3.1], [0.0, 0.0]),
        ([1.0, 1.0], [1.0, 1.0]),
        ([1.0, 2.0], [1.0, -1.0]),
        ([1.0, np.nan], [0.0, 0.0]),
        ([1.0, 2.0], [np.inf, 0.0]),
    ],
)
def test_bad_grid_or_band_refused(energies, photons):
    with pytest.raises(ValueError):
        _calibrated().predict(energies, photons)


def test_invalid_calibrations_refused():
    with pytest.raises(ValueError, match="efficiency"):
        _calibrated(efficiency=np.array([0, 1.01, 0]))
    with pytest.raises(ValueError, match="provenance"):
        _calibrated(calibration_id=" ")
    with pytest.raises(ValueError, match="pulse_gain"):
        _calibrated(pulse_gain_v_per_j=0)
    with pytest.raises(ValueError, match="noise_rms"):
        _calibrated(noise_rms_v=-1)
    with pytest.raises(ValueError, match="saturation"):
        _calibrated(saturation_v=0)
    with pytest.raises(ValueError, match="repetition_hz"):
        _calibrated().predict([1, 3], [1, 1], repetition_hz=-1)


def test_batched_pulse_reduction_has_exact_nonuniform_grid_units():
    det = _calibrated(saturation_v=0.003)
    energy = np.array([1.0, 1.3, 2.2, 3.0])
    levels = np.array([[0.0, 1e7], [2e7, 3e7]])
    photons = np.broadcast_to(levels[..., None], (2, 2, energy.size))
    out = det.predict_batch(energy, photons, repetition_hz=1e6)
    # Constant spectral density: integral E dE from 1 to 3 = 4 eV^2.
    joules = 4 * levels * 1.602176634e-19
    np.testing.assert_allclose(out.incident_joule_per_pulse, joules, rtol=1e-14)
    np.testing.assert_allclose(out.expected_voltage_v, joules * 0.5e12, rtol=1e-14)
    np.testing.assert_allclose(out.reported_voltage_v, np.minimum(joules * 0.5e12, 0.003))
    np.testing.assert_array_equal(out.saturated, joules * 0.5e12 > 0.003)
    np.testing.assert_allclose(out.average_absorbed_power_w, joules * 0.5e6)


def test_batched_pulse_reduction_checks_every_spectrum():
    det = _calibrated()
    for photons in (np.ones((2, 4)), [[1, 1, 1], [1, -1, 1]], [[1, 1, 1], [1, np.nan, 1]]):
        with pytest.raises(ValueError):
            det.predict_batch([1, 2, 3], photons)


def test_semantic_readings_and_calibration_quantities():
    from pyrite.units import ureg

    det = _calibrated(saturation_v=1.0, noise_rms_v=0.002)
    energy = np.array([1.0, 2.0, 3.0])
    photons = np.ones((2, 3)) * 1e7
    batch = det.predict_batch(energy, photons, repetition_hz=1e3)
    single = det.predict(energy, photons[0])
    assert single.incident_energy.units == ureg.J
    assert single.incident_energy.magnitude == single.incident_joule_per_pulse
    assert single.reported_voltage.to("mV").magnitude == pytest.approx(
        single.reported_voltage_v * 1e3
    )
    assert single.average_incident_power is None
    np.testing.assert_array_equal(batch.absorbed_energy.magnitude, batch.absorbed_joule_per_pulse)
    assert batch.average_absorbed_power.units == ureg.W
    readback = batch.expected_voltage
    readback.magnitude[...] = 0
    assert np.all(batch.expected_voltage_v > 0)
    assert det.energies.units == ureg.eV
    assert det.pulse_gain.to("V/pJ").magnitude == pytest.approx(1.0)
    assert det.noise_rms.magnitude == 0.002
    assert det.saturation.magnitude == 1.0
    assert _calibrated().saturation is None
    knots = det.energies
    knots.magnitude[...] = 0
    assert det.energy_eV[0] == 1.0
