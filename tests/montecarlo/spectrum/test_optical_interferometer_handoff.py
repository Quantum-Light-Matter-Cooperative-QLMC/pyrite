"""Complex spectral finite-plane field to calibrated MPI polarization input."""

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import IdealMartinPuplett, polarized_interferogram
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import ScreenSpectralField
from pyrite.montecarlo.spectrum.optical_interferometer_handoff import coherency_from_screen_fields

DX = 0.25e-6
N = 128


def _beam(energy, *, x=1.0, y=1.0, phase=0.0, rapidly_varying=False):
    coords = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(coords, coords)
    envelope = np.exp(-(X * X + Y * Y) / (2 * (5e-6) ** 2))
    if rapidly_varying:
        envelope *= (-1.0) ** np.arange(N)[None, :]
    fields = np.stack([x * envelope, y * envelope * np.exp(1j * phase)]).astype(complex) * 1e-16
    return ScreenSpectralField(energy, DX, DX, fields)


def _coh(fields, **kw):
    return coherency_from_screen_fields(
        fields,
        plane_z_m=1.2e-6,
        provenance="synthetic Gaussian complex E fields, known phase",
        **kw,
    )


def test_paraxial_handoff_preserves_polarization_phase_and_photon_density():
    fields = [_beam(0.9, phase=np.pi / 3), _beam(1.1, phase=np.pi / 3)]
    result = _coh(fields)
    j = result.photons_per_eV_per_pulse
    assert j.shape == (2, 2, 2)
    np.testing.assert_allclose(j[:, 0, 0].real, j[:, 1, 1].real, rtol=1e-12)
    np.testing.assert_allclose(j[:, 0, 1] / j[:, 0, 0], np.exp(-1j * np.pi / 3), rtol=1e-12)
    for i, f in enumerate(fields):
        ref = collect_screen_field(f, plane_z_m=1.2e-6)
        np.testing.assert_allclose(
            result.total_photons_per_eV_per_pulse[i], ref.photons_per_eV, rtol=0.03
        )


def test_derived_coherency_drives_calibrated_interferometer():
    fields = [_beam(0.8, x=1, y=1), _beam(1.2, x=1, y=1)]
    source = _coh(fields)
    detector = CalibratedPulseEnergyDetector([0.5, 1.5], [0.5, 0.5], 1e12, "synthetic detector")
    scan = polarized_interferogram(
        source,
        [0, 1e-6],
        instrument=IdealMartinPuplett("synthetic MPI"),
        detectors=(detector, detector),
    )
    np.testing.assert_allclose(
        scan.incident_joule_per_pulse_ports.sum(axis=1),
        np.trapezoid(source.total_photons_per_eV_per_pulse * source.energy_eV, source.energy_eV)
        * 1.602176634e-19,
        rtol=1e-13,
    )
    assert scan.incident_joule_per_pulse_ports[0, 0] > 0
    np.testing.assert_allclose(scan.incident_joule_per_pulse_ports[0, 1], 0.0, atol=1e-25)


def test_rectangular_detector_plane_aperture_can_handoff():
    plane = PlaneAcceptance("rectangle", width_m=12e-6, height_m=12e-6)
    source = _coh([_beam(0.8), _beam(1.2)], aperture=plane)
    assert np.all(source.total_photons_per_eV_per_pulse > 0)


def test_refuse_nonparaxial_source_and_intensity_only():
    with pytest.raises(ValueError, match="nonparaxial/evanescent"):
        _coh([_beam(0.8, rapidly_varying=True), _beam(1.2, rapidly_varying=True)])
    with pytest.raises(TypeError, match="complex ScreenSpectralField"):
        _coh([np.ones(10), np.ones(10)])
    with pytest.raises(ValueError, match="increasing"):
        _coh([_beam(1.2), _beam(0.8)])
    with pytest.raises(ValueError, match="downstream"):
        coherency_from_screen_fields(
            [_beam(0.8), _beam(1.2)], plane_z_m=0.0, provenance="synthetic"
        )


def test_limited_angular_and_flux_error_settings_are_explicit():
    with pytest.raises(ValueError, match="max_angle_rad"):
        _coh([_beam(0.8), _beam(1.2)], max_angle_rad=np.pi)
    with pytest.raises(ValueError, match="max_flux_relative_error"):
        _coh([_beam(0.8), _beam(1.2)], max_flux_relative_error=0)
