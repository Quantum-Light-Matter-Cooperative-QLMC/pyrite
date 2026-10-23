"""Finite detector-plane polarized CTR/CDR energy/field regression (#616/#617)."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.montecarlo.runner import finite_screen_emission as runner
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
    finite_screen_bunch_plane_band,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    _E_CHARGE_C,
    _HBAR_EV_S,
    PECCircularScreen,
    circular_pec_transition_field,
    spectral_energy_per_eV,
)

_GRID = dict(nx=64, ny=64, dx_m=0.1e-6, dy_m=0.1e-6)
_ENERGY = 1.0
_Z = 0.5e-6


@dataclass(frozen=True)
class _Source:
    law: str
    physical_electrons: float
    history_weights: np.ndarray | None = None
    factor_override: float | None = None
    inter_electron: str = "independent"
    covariance_ang2: np.ndarray | None = None

    def form_factor_squared(self, energy_eV, direction):
        return np.asarray(self.factor_override if self.factor_override is not None else 1.0)

    def record(self):
        return {"law": self.law, "physical_electrons": self.physical_electrons}


def _screen(**kwargs):
    return PECCircularScreen(outer_radius_m=2e-6, **kwargs)


def _field():
    return circular_pec_transition_field(_ENERGY, _screen(), **_GRID)


def _aperture(**kwargs):
    return PlaneAcceptance("rectangle", width_m=2e-6, height_m=2e-6, **kwargs)


def _bunch(screens, src, energies=(0.9, 1.1), **kwargs):
    return finite_screen_bunch_plane_band(
        energies,
        screens,
        source=src,
        plane_z_m=_Z,
        aperture=_aperture(),
        **_GRID,
        **kwargs,
    )


def test_full_plane_energy_is_propagation_invariant_and_has_correct_units():
    field = _field()
    target = spectral_energy_per_eV(field)
    for distance in (0.1e-6, 0.5e-6, 2e-6):
        result = collect_screen_field(field, plane_z_m=distance)
        np.testing.assert_allclose(result.spectral_energy_j_per_eV, target, rtol=2e-13)
        np.testing.assert_allclose(
            result.photons_per_eV * _ENERGY * _E_CHARGE_C, target, rtol=1e-13
        )
    # Important distinction: these are J/eV, not monochromatic W or photons/eV/sr.
    assert 0 < target < 1e-15


def test_subaperture_is_finite_distance_sensitive_and_passive():
    field = _field()
    patch = _aperture()
    near = collect_screen_field(field, plane_z_m=0.1e-6, aperture=patch)
    far = collect_screen_field(field, plane_z_m=2e-6, aperture=patch)
    reference = spectral_energy_per_eV(field)
    assert 0 < near.spectral_energy_j_per_eV < reference
    assert 0 < far.spectral_energy_j_per_eV < reference
    assert abs(near.spectral_energy_j_per_eV - far.spectral_energy_j_per_eV) > 1e-3 * reference


def test_polarization_projectors_are_power_orthogonal_over_full_cell():
    field = _field()
    sp_s = [[1, 0], [0, 0]]
    sp_p = [[0, 0], [0, 1]]
    total = collect_screen_field(field, plane_z_m=_Z)
    s = collect_screen_field(field, plane_z_m=_Z, jones_sp=sp_s)
    p = collect_screen_field(field, plane_z_m=_Z, jones_sp=sp_p)
    np.testing.assert_allclose(
        s.spectral_energy_j_per_eV + p.spectral_energy_j_per_eV,
        total.spectral_energy_j_per_eV,
        rtol=1e-12,
    )
    assert p.spectral_energy_j_per_eV > 1000 * s.spectral_energy_j_per_eV
    # A 45-degree *phase-retarding* Jones element preserves total power.
    rotated = collect_screen_field(field, plane_z_m=_Z, jones_sp=np.diag([1j, -1]))
    np.testing.assert_allclose(
        rotated.spectral_energy_j_per_eV, total.spectral_energy_j_per_eV, rtol=1e-12
    )
    absorbed = collect_screen_field(field, plane_z_m=_Z, jones_sp=np.eye(2) * 0.3)
    np.testing.assert_allclose(
        absorbed.spectral_energy_j_per_eV, 0.09 * total.spectral_energy_j_per_eV, rtol=1e-12
    )


def test_absolute_arrival_phase_does_not_change_single_electron_collection():
    f = _field()
    shifted = circular_pec_transition_field(1.0, _screen(arrival_time_s=0.76 * _HBAR_EV_S), **_GRID)
    ref = collect_screen_field(f, plane_z_m=_Z, aperture=_aperture())
    later = collect_screen_field(shifted, plane_z_m=_Z, aperture=_aperture())
    np.testing.assert_allclose(later.photons_per_eV, ref.photons_per_eV, rtol=1e-12)


def test_detector_aperture_area_is_covered_by_fractional_pixels():
    field = _field()
    # Rectangle is exact for each pixel overlap and the circle approximates
    # its area through a documented fixed subpixel rule.
    patch = _aperture(center_x_m=0.02e-6, center_y_m=-0.05e-6)
    np.testing.assert_allclose(
        np.sum(patch._weights(field)) * field.dx_m * field.dy_m, 4e-12, rtol=1e-12
    )
    circle = PlaneAcceptance("circle", radius_m=0.88e-6)
    area = np.sum(circle._weights(field)) * field.dx_m * field.dy_m
    assert abs(area / (np.pi * circle.radius_m**2) - 1) < 0.007


def test_bunch_one_electron_matches_single_complex_field_collection():
    f = circular_pec_transition_field(0.9, _screen(), **_GRID)
    actual = collect_screen_field(f, plane_z_m=_Z, aperture=_aperture())
    band = _bunch([_screen()], _Source("sampled", 1))
    np.testing.assert_allclose(
        band.collected_photons_per_eV_per_pulse[0], actual.photons_per_eV, rtol=1e-12
    )
    np.testing.assert_allclose(
        band.self_photons_per_eV_per_electron[0], actual.photons_per_eV, rtol=1e-12
    )
    assert band.pair_photons_per_eV_per_electron_pair[0] == 0


def test_fully_coherent_bunch_scales_as_n_squared_independent_of_samples():
    reference = _bunch([_screen()], _Source("sampled", 1))
    for m in (2, 3, 5):
        result = _bunch([_screen()] * m, _Source("sampled", 20))
        np.testing.assert_allclose(
            result.self_photons_per_eV_per_electron,
            reference.collected_photons_per_eV_per_pulse,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.pair_photons_per_eV_per_electron_pair,
            reference.collected_photons_per_eV_per_pulse,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.collected_photons_per_eV_per_pulse,
            400 * reference.collected_photons_per_eV_per_pulse,
            rtol=1e-12,
        )


def test_weights_are_importance_weights_not_macro_charge():
    reference = _bunch([_screen()], _Source("sampled", 1))
    sampled = _bunch(
        [_screen()] * 3, _Source("sampled", 10, history_weights=np.array([1.0, 2.0, 7.0]))
    )
    np.testing.assert_allclose(
        sampled.collected_photons_per_eV_per_pulse,
        100 * reference.collected_photons_per_eV_per_pulse,
        rtol=1e-12,
    )


def test_realization_destructive_interference_is_collected_at_plane():
    # Both E and H cancel before flux integration, at every propagation z.
    screens = [_screen(), _screen(arrival_time_s=np.pi * _HBAR_EV_S)]
    result = _bunch(screens, _Source("realization", 2))
    unit = _bunch([_screen()], _Source("sampled", 1))
    assert (
        abs(result.collected_photons_per_eV_per_pulse[0])
        < 0.1 * unit.collected_photons_per_eV_per_pulse[0]
    )
    assert result.self_photons_per_eV_per_electron[0] > 0


def test_analytic_temporal_only_bunch_form_factor_scales_pairs():
    f = 0.3
    src = _Source("gaussian", 5, factor_override=f, covariance_ang2=np.diag([1.0, 0.0, 0.0]))
    result = _bunch([_screen()] * 3, src)
    unit = _bunch([_screen()], _Source("sampled", 1))
    np.testing.assert_allclose(
        result.collected_photons_per_eV_per_pulse,
        5 * (1 + 4 * f) * unit.collected_photons_per_eV_per_pulse,
        rtol=1e-12,
    )


def test_calibrated_detector_handoff_after_finite_spatial_collection():
    band = _bunch([_screen()] * 2, _Source("sampled", 2))
    det = CalibratedPulseEnergyDetector(
        energy_eV=np.array([0.5, 1.5]),
        efficiency=np.array([0.25, 0.25]),
        pulse_gain_v_per_j=1e10,
        calibration_id="synthetic physics test",
    )
    reading = band.detect(det, repetition_hz=1e6)
    assert reading.incident_joule_per_pulse > 0
    np.testing.assert_allclose(
        reading.absorbed_joule_per_pulse, 0.25 * reading.incident_joule_per_pulse, rtol=1e-12
    )
    np.testing.assert_allclose(
        reading.average_incident_power_w, reading.incident_joule_per_pulse * 1e6, rtol=1e-12
    )


def test_rejections_for_invalid_aperture_optics_and_plane():
    field = _field()
    with pytest.raises(ValueError, match="strictly inside"):
        collect_screen_field(
            field, plane_z_m=_Z, aperture=PlaneAcceptance("circle", radius_m=100e-6)
        )
    with pytest.raises(ValueError, match="passive"):
        collect_screen_field(field, plane_z_m=_Z, jones_sp=np.eye(2) * 2)
    with pytest.raises(ValueError, match="downstream"):
        collect_screen_field(field, plane_z_m=0)
    with pytest.raises(TypeError, match="ScreenSpectralField"):
        collect_screen_field(np.zeros((2, 64, 64)), plane_z_m=_Z)
    with pytest.raises(ValueError, match="rectangle"):
        PlaneAcceptance("rectangle", width_m=1e-6)
    with pytest.raises(ValueError, match="shape"):
        PlaneAcceptance("triangle")
    with pytest.raises(ValueError, match="increasing"):
        _bunch([_screen()], _Source("sampled", 1), energies=[1.0, 0.9])


def test_signed_negative_collection_refuses_detector():
    band = _bunch([_screen()], _Source("sampled", 1))
    from dataclasses import replace

    invalid = replace(
        band, collected_photons_per_eV_per_pulse=-band.collected_photons_per_eV_per_pulse
    )
    with pytest.raises(ValueError, match="signed"):
        invalid.detect(None)


def test_runner_wiring_and_explicit_formation_angle(monkeypatch):
    screens = (_screen(), _screen())
    captured = {}

    def fake_screens(*args, **kwargs):
        captured.update(kwargs)
        return screens

    monkeypatch.setattr(runner, "_case_screens", fake_screens)
    patch = _aperture()
    result = runner.case_finite_screen_plane_band(
        {},
        {},
        [0.9, 1.1],
        source=_Source("sampled", 2),
        outer_radius_m=2e-6,
        aperture=patch,
        plane_z_m=_Z,
        **_GRID,
    )
    direct = _bunch(screens, _Source("sampled", 2))
    np.testing.assert_allclose(
        result.collected_photons_per_eV_per_pulse, direct.collected_photons_per_eV_per_pulse
    )
    assert captured["min_formation_lengths"] is None
    with pytest.raises(ValueError, match="formation_theta_max_rad"):
        runner.case_finite_screen_plane_band(
            {},
            {},
            [0.9, 1.1],
            source=_Source("sampled", 2),
            outer_radius_m=2e-6,
            aperture=patch,
            plane_z_m=_Z,
            min_formation_lengths=3,
            **_GRID,
        )
