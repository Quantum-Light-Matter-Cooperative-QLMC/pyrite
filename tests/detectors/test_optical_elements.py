"""Measured passive Jones elements on polarized spectral coherency (#616)."""

import numpy as np
import pytest

from pyrite.detectors.optical_elements import CalibratedJonesElement, apply_optical_elements
from pyrite.detectors.optical_interferometer import OpticalSpectralCoherency

E = np.linspace(0.5, 1.5, 11)


def _source(angle=0.3, photons=None):
    photons = np.full(E.size, 2.0e6) if photons is None else photons
    return OpticalSpectralCoherency.linearly_polarized(
        E, photons, angle_rad=angle, provenance="synthetic linear source"
    )


def _element(jones, **kw):
    kw.setdefault("max_half_angle_rad", 0.1)
    return CalibratedJonesElement(
        "synthetic element", [0.4, 1.6], jones, calibration_provenance="synthetic", **kw
    )


def test_ideal_linear_polarizer_obeys_malus_law():
    polarizer = _element(np.broadcast_to(np.diag([1.0, 0.0]), (2, 2, 2)))
    angle = 0.3
    out = apply_optical_elements(_source(angle), [polarizer], collection_half_angle_rad=0.05)
    np.testing.assert_allclose(
        out.total_photons_per_eV_per_pulse, 2.0e6 * np.cos(angle) ** 2, rtol=1e-14
    )
    assert "synthetic element sha256=" in out.provenance


def test_identity_and_unitary_retarder_conserve_flux_and_compose():
    phase = np.diag([1.0, 1j])  # quarter-wave retarder, lossless
    qwp = _element(np.broadcast_to(phase, (2, 2, 2)))
    src = _source()
    once = apply_optical_elements(src, [qwp], collection_half_angle_rad=0.0)
    np.testing.assert_allclose(
        once.total_photons_per_eV_per_pulse, src.total_photons_per_eV_per_pulse, rtol=1e-14
    )
    # Two QWPs = HWP: linear at +0.3 maps to linear at -0.3 (real, flipped off-diagonal).
    twice = apply_optical_elements(src, [qwp, qwp], collection_half_angle_rad=0.0)
    expected = _source(-0.3).photons_per_eV_per_pulse
    np.testing.assert_allclose(twice.photons_per_eV_per_pulse, expected, atol=1e-8)


def test_intensity_transmission_interpolates_and_refuses_extrapolation():
    window = CalibratedJonesElement.from_intensity_transmission(
        "window",
        [0.4, 1.6],
        [0.9, 0.5],
        max_half_angle_rad=0.2,
        calibration_provenance="synthetic two-point scan",
    )
    out = apply_optical_elements(_source(), [window], collection_half_angle_rad=0.1)
    amp = np.interp(E, [0.4, 1.6], np.sqrt([0.9, 0.5]))
    np.testing.assert_allclose(out.total_photons_per_eV_per_pulse, 2.0e6 * amp**2, rtol=1e-14)
    with pytest.raises(ValueError, match="calibrated band"):
        window.transfer(np.array([0.3]))


def test_refuses_gain_wide_angles_and_missing_provenance():
    with pytest.raises(ValueError, match="passive"):
        _element(np.broadcast_to(1.01 * np.eye(2), (2, 2, 2)))
    narrow = _element(np.broadcast_to(np.eye(2), (2, 2, 2)), max_half_angle_rad=0.01)
    with pytest.raises(ValueError, match="half-angle"):
        apply_optical_elements(_source(), [narrow], collection_half_angle_rad=0.02)
    with pytest.raises(ValueError, match="calibration_provenance"):
        CalibratedJonesElement("x", [0.4, 1.6], np.broadcast_to(np.eye(2), (2, 2, 2)), 0.1, " ")
    with pytest.raises(ValueError, match="within"):
        CalibratedJonesElement.from_intensity_transmission(
            "x", [0.4, 1.6], [1.2, 0.5], max_half_angle_rad=0.1, calibration_provenance="s"
        )


def test_interpolated_contractions_stay_passive_for_random_measurements():
    rng = np.random.default_rng(616)
    raw = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
    raw /= np.linalg.svd(raw, compute_uv=False)[:, :1, None]  # unit spectral norm
    element = CalibratedJonesElement("random", np.linspace(0.4, 1.6, 5), raw, 0.1, "seeded")
    t = element.transfer(np.linspace(0.4, 1.6, 301))
    assert np.max(np.linalg.svd(t, compute_uv=False)) <= 1 + 1e-12
