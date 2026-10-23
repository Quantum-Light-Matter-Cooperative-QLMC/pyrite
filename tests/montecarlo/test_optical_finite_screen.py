"""Reference and refusal checks of normal-incidence PEC finite TR / aperture DR (#616)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    ScreenSpectralField,
    circular_pec_transition_field,
    far_field_photons_per_eV_sr,
    ginzburg_frank_photons_per_eV_sr,
    observe_screen_green,
    propagate_screen_field,
    spectral_energy_per_eV,
)


def _source(*, hole=0, outer=4e-6, time=0.0, charge=-1.602176634e-19, n=256):
    screen = PECCircularScreen(outer, hole, beta_z=-0.95, arrival_time_s=time, charge_c=charge)
    return circular_pec_transition_field(
        1.0, screen, nx=n, ny=n, dx_m=12.8e-6 / n, dy_m=12.8e-6 / n
    )


def test_zero_screen_is_zero_in_field_angular_yield_and_energy():
    f = _source(outer=0)
    assert not np.any(f.electric_xy_v_s_m)
    assert spectral_energy_per_eV(f) == 0
    theta, phi, density = far_field_photons_per_eV_sr(f)
    assert not np.any(density)
    assert theta.shape == phi.shape == density.shape == (256, 256)


def test_large_disc_matches_ginzburg_frank_absolute_angular_yield():
    # Independently normalized electromagnetic limit: this comparison checks
    # the factors 2*pi, Z0, hbar and photon energy, not just the angular shape.
    f = _source(n=512)
    theta, _, density = far_field_photons_per_eV_sr(f)
    useful = np.isfinite(theta) & (theta > 0.2) & (theta < 0.9)
    analytic = ginzburg_frank_photons_per_eV_sr(1.0, theta[useful], 0.95)
    assert useful.sum() > 50
    np.testing.assert_allclose(density[useful], analytic, rtol=0.012)
    assert np.all(density >= 0)


def test_aperture_dr_without_crossing_has_nonzero_but_suppressed_emission():
    disc = _source(hole=0)
    aperture = _source(hole=0.5e-6)
    remote_aperture = _source(hole=2e-6)
    disc_energy = spectral_energy_per_eV(disc)
    assert disc_energy > 0
    assert 0 < spectral_energy_per_eV(aperture) < disc_energy
    assert 0 < spectral_energy_per_eV(remote_aperture) < 0.01 * disc_energy
    assert not np.any(aperture.electric_xy_v_s_m[:, 123:133, 123:133])


def test_charge_sign_and_arrival_phase_are_linear_fields_quadratic_yields():
    base = _source()
    opposite = _source(charge=1.602176634e-19)
    np.testing.assert_allclose(opposite.electric_xy_v_s_m, -base.electric_xy_v_s_m, rtol=2e-14)
    shifted_time = _source(time=1.0e-16)
    factor = np.exp(1j * (1.0 / 6.582119569e-16) * 1e-16)
    np.testing.assert_allclose(
        shifted_time.electric_xy_v_s_m, base.electric_xy_v_s_m * factor, rtol=2e-14
    )
    np.testing.assert_allclose(
        spectral_energy_per_eV(base), spectral_energy_per_eV(opposite), rtol=1e-14
    )


def test_finite_distance_propagation_composes_and_conserves_outgoing_energy():
    f = _source()
    before = spectral_energy_per_eV(f)
    first = propagate_screen_field(f, 1e-6)
    later = propagate_screen_field(first, 2e-6)
    direct = propagate_screen_field(f, 3e-6)
    np.testing.assert_allclose(
        later.electric_xy_v_s_m, direct.electric_xy_v_s_m, rtol=1e-12, atol=1e-25
    )
    np.testing.assert_allclose(spectral_energy_per_eV(later), before, rtol=1e-12)
    assert later.z_m == pytest.approx(3e-6)
    assert np.any(np.abs(later.electric_xy_v_s_m - f.electric_xy_v_s_m) > 0)
    assert not f.electric_xy_v_s_m.flags.writeable


def test_far_field_solid_angle_quadrature_matches_spectral_energy():
    f = _source()
    theta, _, dN_dE_dOmega = far_field_photons_per_eV_sr(f)
    k0 = 1.0 / 1.973269804593025e-7
    dkx = 2 * np.pi / (256 * f.dx_m)
    dky = 2 * np.pi / (256 * f.dy_m)
    keep = np.isfinite(theta)
    # dOmega = dkx*dky / (k0^2*cos(theta)); photon->joule uses E*e.
    dOmega = dkx * dky / (k0**2 * np.cos(theta[keep]))
    flux = np.sum(dN_dE_dOmega[keep] * dOmega) * 1.602176634e-19
    np.testing.assert_allclose(flux, spectral_energy_per_eV(f), rtol=1e-13)


def test_mesh_refinement_and_aperture_radius_limits():
    coarse = _source(n=256)
    fine = _source(n=512)
    # At fixed FFT window, the normal-emission energy converges with 2x
    # pixel resolution despite the integrable 1/r source singularity.
    assert abs(spectral_energy_per_eV(coarse) / spectral_energy_per_eV(fine) - 1) < 0.01
    assert spectral_energy_per_eV(_source(hole=4e-6)) == 0


@pytest.mark.parametrize(
    "screen",
    [
        dict(outer_radius_m=-1),
        dict(outer_radius_m=2e-6, inner_radius_m=3e-6),
        dict(outer_radius_m=1e-6, beta_z=0),
        dict(outer_radius_m=1e-6, beta_z=1),
        dict(outer_radius_m=1e-6, charge_c=0),
        dict(outer_radius_m=1e-6, charge_c=-2 * 1.602176634e-19),
        dict(outer_radius_m=1e-6, beta_z=0.8),
        dict(outer_radius_m=2e-6, inner_radius_m=1e-6, electron_x_m=1.2e-6),
    ],
)
def test_refuse_invalid_screen(screen):
    with pytest.raises(ValueError):
        PECCircularScreen(**screen)


def test_refuse_unresolved_and_wrong_units():
    screen = PECCircularScreen(1e-6)
    with pytest.raises(ValueError, match="window"):
        circular_pec_transition_field(1.0, screen, nx=32, ny=32, dx_m=40e-9, dy_m=40e-9)
    with pytest.raises(ValueError, match="nx and ny"):
        circular_pec_transition_field(1.0, screen, nx=33, ny=32, dx_m=0.2e-6, dy_m=0.2e-6)
    with pytest.raises(ValueError, match="electron intercept"):
        circular_pec_transition_field(
            1.0,
            PECCircularScreen(2e-6, electron_x_m=0.5e-6, electron_y_m=0.5e-6),
            nx=32,
            ny=32,
            dx_m=1e-6,
            dy_m=1e-6,
        )
    with pytest.raises(ValueError, match="distance_m"):
        propagate_screen_field(_source(), -1)
    with pytest.raises(ValueError, match="complex"):
        ScreenSpectralField(1.0, 1e-6, 1e-6, np.ones((2, 16, 16)))


def test_direct_green_finite_distance_agrees_with_fft_wavefront():
    # Independent numerical representations of Maxwell's outgoing field:
    # direct retarded Green curl vs periodic angular-spectrum FFT. The finite
    # source is kept well inside the FFT cell to bound periodic-image error.
    screen = PECCircularScreen(outer_radius_m=2e-6, beta_z=-0.95)
    f = circular_pec_transition_field(1.0, screen, nx=256, ny=256, dx_m=25e-9, dy_m=25e-9)
    ix, iy, distance = 135, 128, 0.2e-6
    x = (ix - 127.5) * f.dx_m
    y = (iy - 127.5) * f.dy_m
    green = observe_screen_green(f, np.array([[x, y, distance]]))[0]
    fft = propagate_screen_field(f, distance).electric_xy_v_s_m[:, iy, ix]
    np.testing.assert_allclose(green[:2], fft, rtol=0.012)
    # It is a genuinely vector field, not two independent scalar rays.
    assert np.abs(green[2]) > 0


def test_30kev_nonrelativistic_limit_ginzburg_frank():
    beta = np.sqrt(1 - (511.0 / 541.0) ** 2)
    f = circular_pec_transition_field(
        1.0,
        PECCircularScreen(outer_radius_m=2.5e-6, beta_z=-beta),
        nx=512,
        ny=512,
        dx_m=15e-9,
        dy_m=15e-9,
    )
    theta, _, density = far_field_photons_per_eV_sr(f)
    keep = np.isfinite(theta) & (theta > 0.25) & (theta < 0.9)
    gf = ginzburg_frank_photons_per_eV_sr(1.0, theta[keep], beta)
    assert keep.sum() > 40
    np.testing.assert_allclose(density[keep], gf, rtol=0.015)


def test_green_refuses_upstream_or_malformed_points():
    f = _source()
    with pytest.raises(ValueError, match="downstream"):
        observe_screen_green(f, [[0, 0, 0]])
    with pytest.raises(ValueError, match="array"):
        observe_screen_green(f, [0, 0, 1e-6])
    with pytest.raises(ValueError, match="max_interactions"):
        observe_screen_green(f, [[0, 0, 1e-6]], max_interactions=0)
