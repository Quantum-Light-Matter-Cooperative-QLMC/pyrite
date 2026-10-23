"""Polarized Martin--Puplett ideal reference limits (#617)."""

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import (
    IdealMartinPuplett,
    OpticalSpectralCoherency,
    load_optical_interferogram,
    polarized_interferogram,
)

_ENERGY = np.linspace(0.4, 1.6, 101)
_DENSITY = np.full(_ENERGY.shape, 3e6)
_HC = 1.2398419843320026e-6


def _det(eta=1.0, gain=2e10, saturation=None):
    return CalibratedPulseEnergyDetector(
        np.array([0.1, 2.0]),
        np.array([eta, eta]),
        pulse_gain_v_per_j=gain,
        noise_rms_v=0.02,
        saturation_v=saturation,
        calibration_id=f"synthetic eta={eta}",
    )


def _src(theta=np.pi / 4):
    return OpticalSpectralCoherency.linearly_polarized(
        _ENERGY, _DENSITY, angle_rad=theta, provenance="synthetic known linear polarization"
    )


def _scan(delays=(0.0,), source=None, instrument=None, detectors=None, repetition=None):
    return polarized_interferogram(
        _src() if source is None else source,
        np.asarray(delays),
        instrument=instrument or IdealMartinPuplett("synthetic ideal test"),
        detectors=detectors or (_det(), _det()),
        repetition_hz=repetition,
    )


def test_single_monochromatic_fringe_max_min_and_mirror_double_pass():
    e = np.array([0.99, 1.01])
    s = OpticalSpectralCoherency.linearly_polarized(
        e, np.ones(2), angle_rad=np.pi / 4, provenance="narrow-bin reference"
    )
    # half wavelength OPD => quarter wavelength *mirror* travel.
    half_period = (_HC / 1.0) / 4
    res = _scan([0, half_period], source=s)
    np.testing.assert_allclose(res.optical_path_difference_m, [0, 2 * half_period])
    np.testing.assert_allclose(
        res.optical_delay_fs, res.optical_path_difference_m / 299792458 * 1e15
    )
    np.testing.assert_allclose(res.spectral_port_photons_per_eV_per_pulse[0, 0], 1.0, rtol=1e-14)
    np.testing.assert_allclose(res.spectral_port_photons_per_eV_per_pulse[0, 1], 0.0, atol=1e-14)
    assert res.spectral_port_photons_per_eV_per_pulse[1, 0].max() < 1e-3
    assert res.spectral_port_photons_per_eV_per_pulse[1, 1].min() > 0.999


def test_port_energy_conservation_over_broadband_scan():
    shift = np.linspace(-10e-6, 10e-6, 23)
    res = _scan(shift)
    expected = np.trapezoid(_ENERGY * _DENSITY, _ENERGY) * 1.602176634e-19
    np.testing.assert_allclose(res.incident_joule_per_pulse_ports.sum(axis=1), expected, rtol=2e-15)
    assert np.all(res.spectral_port_photons_per_eV_per_pulse >= 0)
    assert not np.any(res.saturated_ports)
    np.testing.assert_allclose(
        res.absorbed_joule_per_pulse_ports, res.incident_joule_per_pulse_ports
    )


def test_unpolarized_input_grid_halves_power_but_fringe_contrast_remains():
    eye = np.tile(0.5 * _DENSITY[:, None, None] * np.eye(2), (1, 1, 1))
    source = OpticalSpectralCoherency(_ENERGY, eye, "synthetic unpolarized coherency")
    res = _scan([0, 1e-6], source=source)
    expected = np.trapezoid(_ENERGY * _DENSITY, _ENERGY) * 1.602176634e-19 / 2
    np.testing.assert_allclose(res.incident_joule_per_pulse_ports.sum(axis=1), expected, rtol=1e-15)
    assert res.incident_joule_per_pulse_ports[0, 1] == pytest.approx(0, abs=1e-27)
    assert res.incident_joule_per_pulse_ports[0, 0] > res.incident_joule_per_pulse_ports[1, 0]


def test_input_polarizer_refuses_orthogonal_light():
    source = _src(theta=-np.pi / 4)
    res = _scan([0, 1e-6], source=source)
    assert np.max(np.abs(res.incident_joule_per_pulse_ports)) < 1e-28


def test_jitter_reduces_modulation_but_not_sum():
    no = _scan([0])
    sigma = 1e-6
    jittered = _scan([0], instrument=IdealMartinPuplett("jitter reference", opd_jitter_rms_m=sigma))
    np.testing.assert_allclose(
        jittered.incident_joule_per_pulse_ports.sum(axis=1),
        no.incident_joule_per_pulse_ports.sum(axis=1),
        rtol=1e-15,
    )
    assert jittered.incident_joule_per_pulse_ports[0, 1] > 0
    # A common arrival-time offset multiplies both arms' fields by the same
    # global phase: it cannot be modeled by an OPD jitter envelope.


def test_arm_loss_output_port_efficiency_and_port_imbalance():
    instrument = IdealMartinPuplett(
        "passive nonideal", arm_intensity_transmissions=(0.9, 0.4), port_efficiencies=(0.8, 0.6)
    )
    res = _scan([0], instrument=instrument)
    ideal = _scan([0])
    assert 0 < res.incident_joule_per_pulse_ports.sum() < ideal.incident_joule_per_pulse_ports.sum()
    # Both ports may carry signal even at nominal zero delay when arm losses differ.
    assert res.incident_joule_per_pulse_ports[0, 1] > 0


def test_detector_calibration_noise_saturation_repetition_and_persistence(tmp_path):
    d1 = _det(0.5, gain=1e25, saturation=0.04)
    d2 = _det(0.25, gain=1e10)
    res = _scan([0, 1e-6], detectors=(d1, d2), repetition=1e6)
    np.testing.assert_allclose(
        res.absorbed_joule_per_pulse_ports[:, 0], 0.5 * res.incident_joule_per_pulse_ports[:, 0]
    )
    np.testing.assert_allclose(
        res.absorbed_joule_per_pulse_ports[:, 1], 0.25 * res.incident_joule_per_pulse_ports[:, 1]
    )
    np.testing.assert_allclose(
        res.average_absorbed_power_w_ports, res.absorbed_joule_per_pulse_ports * 1e6
    )
    assert res.saturated_ports[0, 0]
    assert res.reported_voltage_v_ports[0, 0] == 0.04
    np.testing.assert_allclose(res.noise_rms_voltage_v_ports, 0.02)
    assert res.metadata["detector_response_sha256"] == [d1.response_sha256, d2.response_sha256]
    saved = load_optical_interferogram(res.save(tmp_path / "interferogram"))
    np.testing.assert_array_equal(
        saved.spectral_port_photons_per_eV_per_pulse, res.spectral_port_photons_per_eV_per_pulse
    )
    assert saved.metadata == res.metadata


def test_phase_offset_shifts_fringes_and_is_not_delay_jitter():
    a = _scan([0], instrument=IdealMartinPuplett("pi offset", phase_offset_rad=np.pi))
    assert a.incident_joule_per_pulse_ports[0, 0] < 1e-27
    assert a.incident_joule_per_pulse_ports[0, 1] > 0


def test_reject_nonhermitian_negative_coherency_and_ambiguous_intensity():
    j = np.zeros((_ENERGY.size, 2, 2), complex)
    j[:, 0, 0] = 1
    j[:, 0, 1] = 1j
    with pytest.raises(ValueError, match="Hermitian"):
        OpticalSpectralCoherency(_ENERGY, j, "bad")
    j[:, 1, 0] = -1j
    with pytest.raises(ValueError, match="positive semidefinite"):
        OpticalSpectralCoherency(_ENERGY, j, "bad")
    with pytest.raises(TypeError, match="Coherency"):
        _scan([0], source=_DENSITY)
    with pytest.raises(ValueError, match="increasing"):
        OpticalSpectralCoherency(_ENERGY[::-1], np.zeros((_ENERGY.size, 2, 2)), "bad")


def test_reject_invalid_scan_transfer_response_and_missing_calibration():
    with pytest.raises(ValueError, match="mirror_displacement_m"):
        _scan([np.inf])
    with pytest.raises(ValueError, match="transmissions"):
        IdealMartinPuplett("bad", arm_intensity_transmissions=(1.1, 1))
    with pytest.raises(ValueError, match="opd_jitter_rms_m"):
        IdealMartinPuplett("bad", opd_jitter_rms_m=-1)
    with pytest.raises(ValueError, match="calibrated energy band"):
        _scan(
            [0],
            detectors=(_det(), CalibratedPulseEnergyDetector([0.5, 0.8], [1, 1], 1e10, "narrow")),
        )


def test_large_forward_matrix_refused_without_silently_allocating():
    with pytest.raises(ValueError, match="max_spectral_elements"):
        polarized_interferogram(
            _src(),
            np.zeros(10000),
            instrument=IdealMartinPuplett("memory limit"),
            detectors=(_det(), _det()),
            max_spectral_elements=1000,
        )
