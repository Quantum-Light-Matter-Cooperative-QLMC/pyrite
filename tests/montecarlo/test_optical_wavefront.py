"""Bounded angular-spectrum and thin-pupil reference checks (#616)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.optical_wavefront import (
    OpticalWavefront,
    apply_pupil,
    forward_power_w,
    propagate_vacuum,
)


def _plane(*, energy=1.55, n=32, pitch=0.1e-6, ex=1, ey=0):
    field = np.zeros((2, n, n), dtype=complex)
    field[0] = ex
    field[1] = ey
    return OpticalWavefront(energy, pitch, pitch, field)


def test_zero_distance_and_normal_plane_wave():
    plane = _plane()
    same = propagate_vacuum(plane, 0)
    np.testing.assert_array_equal(same.electric_xy_v_m, plane.electric_xy_v_m)
    distance = 4e-6
    shifted = propagate_vacuum(plane, distance)
    k0 = plane.energy_eV / 1.973269804593025e-7
    np.testing.assert_allclose(
        shifted.electric_xy_v_m,
        plane.electric_xy_v_m * np.exp(1j * k0 * distance),
        rtol=5e-14,
        atol=5e-14,
    )
    assert shifted.z_m == distance
    np.testing.assert_allclose(forward_power_w(shifted), forward_power_w(plane), rtol=1e-13)
    expected_w = (plane.dx_m * plane.dy_m * 32**2) / (2 * 376.730313412)
    np.testing.assert_allclose(forward_power_w(plane), expected_w, rtol=1e-13)


def test_oblique_plane_wave_power_and_phase():
    n = 32
    pitch = 0.1e-6
    energy = 1.55
    k0 = energy / 1.973269804593025e-7
    kx = 2 * np.pi * 2 / (n * pitch)
    kz = np.sqrt(k0**2 - kx**2)
    x = np.arange(n) * pitch
    field = np.zeros((2, n, n), complex)
    field[0] = np.exp(1j * kx * x)[None, :]
    oblique = OpticalWavefront(energy, pitch, pitch, field)
    propagated = propagate_vacuum(oblique, 3.4e-6)
    np.testing.assert_allclose(
        propagated.electric_xy_v_m,
        field * np.exp(1j * kz * 3.4e-6),
        atol=3e-14,
    )
    # Oblique Ex carries an Ez component, whereas Ey is perpendicular to
    # the k-z plane. The outgoing Poynting flux differs by the known factor.
    normal_power = (n * pitch) ** 2 / (2 * 376.730313412)
    np.testing.assert_allclose(forward_power_w(oblique), normal_power * (k0 / kz), rtol=1e-13)
    field[0] = 0
    field[1] = np.exp(1j * kx * x)[None, :]
    s_pol = OpticalWavefront(energy, pitch, pitch, field)
    np.testing.assert_allclose(forward_power_w(s_pol), normal_power * (kz / k0), rtol=1e-13)


def test_propagation_composes_for_propagating_and_evanescent_spectrum():
    rng = np.random.default_rng(29)
    f = rng.normal(size=(2, 16, 20)) + 1j * rng.normal(size=(2, 16, 20))
    plane = OpticalWavefront(1.4, 0.12e-6, 0.15e-6, f)
    step1 = propagate_vacuum(plane, 0.11e-6)
    step2 = propagate_vacuum(step1, 0.17e-6)
    direct = propagate_vacuum(plane, 0.28e-6)
    np.testing.assert_allclose(step2.electric_xy_v_m, direct.electric_xy_v_m, atol=1e-13)
    np.testing.assert_allclose(forward_power_w(plane), forward_power_w(step2), rtol=1e-12)


def test_evanescent_field_decays_and_carries_no_forward_power():
    field = np.zeros((2, 16, 16), dtype=complex)
    field[0] = (-1.0) ** np.arange(16)[None, :]
    plane = OpticalWavefront(1.0, 0.1e-6, 0.1e-6, field)
    assert forward_power_w(plane) == pytest.approx(0.0, abs=1e-27)
    shifted = propagate_vacuum(plane, 0.4e-6)
    assert np.max(np.abs(shifted.electric_xy_v_m)) < 1e-4


def test_pupil_phase_polarization_and_absorption():
    plane = _plane(ex=1.0)
    shape = (32, 32)
    out = apply_pupil(
        plane,
        transmission=np.full(shape, 0.5),
        phase_rad=np.full(shape, np.pi / 2),
        jones=np.array([[0, 1], [1, 0]], dtype=complex),
    )
    np.testing.assert_allclose(out.electric_xy_v_m[0], 0, atol=1e-15)
    np.testing.assert_allclose(out.electric_xy_v_m[1], 0.5j, atol=1e-15)
    np.testing.assert_allclose(forward_power_w(out), 0.25 * forward_power_w(plane), rtol=1e-13)
    zero = apply_pupil(plane, transmission=np.zeros(shape))
    assert forward_power_w(zero) == 0


@pytest.mark.parametrize("distance", [-0.01, np.nan, np.inf])
def test_refuse_unphysical_distance(distance):
    with pytest.raises(ValueError, match="distance_m"):
        propagate_vacuum(_plane(), distance)


def test_refuse_invalid_field_aperture_and_gain():
    with pytest.raises(TypeError, match="complex"):
        OpticalWavefront(1.0, 1e-6, 1e-6, np.ones((2, 8, 8)))
    with pytest.raises(ValueError, match="energy_eV"):
        OpticalWavefront(0, 1e-6, 1e-6, np.ones((2, 8, 8), complex))
    plane = _plane()
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_pupil(plane, transmission=np.full((32, 32), 1.2))
    with pytest.raises(ValueError, match="passive"):
        apply_pupil(plane, jones=np.eye(2) * 1.01)
    with pytest.raises(ValueError, match="phase"):
        apply_pupil(plane, phase_rad=np.full((32, 32), np.nan))
