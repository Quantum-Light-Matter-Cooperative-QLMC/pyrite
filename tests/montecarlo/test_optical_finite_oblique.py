"""Oblique PEC source vs independently time-integrated moving-charge fields (#616).

Validation target: optical-finite-boundary-field (still unanchored).
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import k1

from pyrite.montecarlo.spectrum.optical_finite_screen import (
    _C_M_S,
    _E_CHARGE_C,
    _EPS0_F_M,
    _HBAR_C_EV_M,
    _HBAR_EV_S,
    PECCircularScreen,
    backward_formation_length_ang,
    circular_pec_transition_field,
    far_field_photons_per_eV_sr,
)
from pyrite.montecarlo.spectrum.optical_finite_transport import (
    entrance_pec_screens_from_crossings,
)


def _params():
    return dict(nx=64, ny=64, dx_m=0.04e-6, dy_m=0.04e-6)


def _incident_field_time_integral(energy_eV, screen, xy):
    """Independent Fourier quadrature of the constant-velocity Lienard field.

    E(t) = q*(1-beta^2)*(r - vt) /
           [4*pi*eps0 *((r-vt)^2 - |beta x (r-vt)|^2)^(3/2)].
    A dimensionless time variable places the integration scale near unity.
    """
    beta = screen.beta_xyz
    beta_abs = np.linalg.norm(beta)
    unit = beta / beta_abs
    gamma = 1 / np.sqrt(1 - beta_abs**2)
    speed = _C_M_S * beta_abs
    omega = energy_eV / _HBAR_EV_S
    u = np.array([*xy, 0.0]) - np.array([screen.electron_x_m, screen.electron_y_m, 0.0])
    along = np.dot(unit, u)
    perp = u - unit * along
    a = np.linalg.norm(perp) / gamma
    freq = omega * a / speed

    def integrand(z):
        # t = t_arrival + along/speed + a*z/speed;
        # R_perp=perp and R_parallel=-a*z.
        R = perp - unit * a * z
        denominator = (a * a * (1 + z * z)) ** 1.5
        E = screen.charge_c / (4 * np.pi * _EPS0_F_M * gamma**2) * R / denominator
        return E[:2] * np.exp(1j * freq * z) * a / speed

    integral, _ = quad_vec(integrand, -1600, 1600, epsabs=1e-22, epsrel=1e-10)
    return -integral * np.exp(1j * omega * (screen.arrival_time_s + along / speed))


def test_oblique_source_matches_independent_exact_moving_charge_fourier_transform():
    screen = PECCircularScreen(
        outer_radius_m=0.9e-6,
        beta_x=0.30,
        beta_y=-0.10,
        beta_z=-0.80,
        arrival_time_s=0.13e-15,
    )
    params = _params()
    source = circular_pec_transition_field(1.5, screen, **params)
    ix, iy = 38, 35
    x = (ix - (params["nx"] - 1) / 2) * params["dx_m"]
    y = (iy - (params["ny"] - 1) / 2) * params["dy_m"]
    reference = _incident_field_time_integral(1.5, screen, (x, y))
    np.testing.assert_allclose(
        source.electric_xy_v_s_m[:, iy, ix],
        reference,
        rtol=3e-5,
        atol=1e-18,
    )


def test_normal_incidence_reduces_to_original_radial_k1_formula():
    screen = PECCircularScreen(outer_radius_m=0.7e-6, beta_z=-0.75)
    params = _params()
    field = circular_pec_transition_field(1.25, screen, **params).electric_xy_v_s_m
    x = (np.arange(params["nx"]) - (params["nx"] - 1) / 2) * params["dx_m"]
    y = (np.arange(params["ny"]) - (params["ny"] - 1) / 2) * params["dy_m"]
    X, Y = np.meshgrid(x, y)
    radius = np.hypot(X, Y)
    gamma = 1 / np.sqrt(1 - screen.beta_z**2)
    kappa = 1.25 / (_HBAR_C_EV_M * gamma * abs(screen.beta_z))
    amplitude = (
        -screen.charge_c
        * kappa
        / (2 * np.pi * _EPS0_F_M * _C_M_S * abs(screen.beta_z))
        * k1(kappa * radius)
    )
    metal = radius <= screen.outer_radius_m
    reference = np.zeros_like(field)
    reference[0, metal] = amplitude[metal] * X[metal] / radius[metal]
    reference[1, metal] = amplitude[metal] * Y[metal] / radius[metal]
    np.testing.assert_allclose(field, reference, rtol=2e-14, atol=1e-18)


def test_oblique_mirror_symmetry_and_complex_charge_phase():
    base = PECCircularScreen(outer_radius_m=0.9e-6, beta_x=0.22, beta_y=0.0, beta_z=-0.73)
    params = _params()
    a = circular_pec_transition_field(0.9, base, **params).electric_xy_v_s_m
    b = circular_pec_transition_field(0.9, replace(base, beta_x=-0.22), **params).electric_xy_v_s_m
    np.testing.assert_allclose(b[0], -a[0, :, ::-1], rtol=1e-13, atol=1e-19)
    np.testing.assert_allclose(b[1], a[1, :, ::-1], rtol=1e-13, atol=1e-19)
    flipped = circular_pec_transition_field(
        0.9, replace(base, charge_c=+_E_CHARGE_C), **params
    ).electric_xy_v_s_m
    np.testing.assert_allclose(flipped, -a, rtol=1e-13, atol=1e-19)
    delay = 0.1e-15
    shifted = circular_pec_transition_field(
        0.9, replace(base, arrival_time_s=delay), **params
    ).electric_xy_v_s_m
    np.testing.assert_allclose(
        shifted, np.exp(1j * 0.9 / _HBAR_EV_S * delay) * a, rtol=1e-13, atol=1e-19
    )


def test_off_axis_angle_changes_far_field_distribution():
    base = PECCircularScreen(outer_radius_m=0.95e-6, beta_z=-0.72)
    tilted = replace(base, beta_x=0.26)
    params = _params()
    a = circular_pec_transition_field(1.0, base, **params)
    b = circular_pec_transition_field(1.0, tilted, **params)
    ya = far_field_photons_per_eV_sr(a)
    yb = far_field_photons_per_eV_sr(b)
    # The oblique incidence shifts and distorts the angular pattern; a
    # beta_z-only approximation would make these identical.
    assert not np.allclose(ya, yb, rtol=1e-3, atol=0)


def test_formation_kinematics_and_rejections():
    normal = PECCircularScreen(outer_radius_m=1e-6, beta_z=-0.7)
    k = 1.0 / _HBAR_C_EV_M
    expected_ang = 0.7 / (k * 1.7) * 1e10
    assert backward_formation_length_ang(1.0, normal, theta_max_rad=0) == pytest.approx(
        expected_ang
    )
    assert backward_formation_length_ang(0.5, normal, theta_max_rad=0) == pytest.approx(
        2 * expected_ang
    )
    assert backward_formation_length_ang(1.0, normal, theta_max_rad=1.1) > expected_ang
    tilt = replace(normal, beta_x=0.2)
    assert backward_formation_length_ang(1.0, tilt, theta_max_rad=1.1) > 0
    with pytest.raises(ValueError, match="beta_z"):
        PECCircularScreen(1e-6, beta_z=-0.8, beta_x=0.8)
    with pytest.raises(ValueError, match="theta_max_rad"):
        backward_formation_length_ang(1.0, normal, theta_max_rad=np.pi / 2)
    with pytest.raises(ValueError, match="energy_eV"):
        backward_formation_length_ang(0.0, normal, theta_max_rad=0)


class _Entry:
    def __init__(self, angle):
        self.electron_id = np.array([1, 0])
        self.interface = np.array([0, 0])
        self.layer_before = np.array([-1, -1])
        self.layer_after = np.array([0, 0])
        self.position_ang = np.array([[0.0, 0.0, 0.0], [100.0, -50.0, 0.0]])
        self.direction = np.tile([np.sin(angle), 0.0, np.cos(angle)], (2, 1))
        self.kinetic_energy_keV = np.array([50.0, 30.0])
        self.time_ang = np.array([0.0, 2997.92458])
        self.charge = np.array([-1, -1])
        self.z_interfaces_ang = np.array([0.0, 1e4])
        self.straight_after_ang = np.array([1e4, 1e4])


def test_transport_handoff_preserves_oblique_velocity_frame_and_history_order():
    entries = _Entry(angle=0.12)
    kw = dict(n_histories=2, outer_radius_m=1e-6)
    with pytest.raises(ValueError, match="max_angle_rad"):
        entrance_pec_screens_from_crossings(entries, **kw)
    screens = entrance_pec_screens_from_crossings(entries, max_angle_rad=0.2, **kw)
    assert screens[0].electron_x_m == pytest.approx(100e-10)
    assert screens[0].electron_y_m == pytest.approx(50e-10)
    b0 = np.sqrt(1 - (1 + 30 / 510.99895069) ** -2)
    np.testing.assert_allclose(
        screens[0].beta_xyz, [b0 * np.sin(0.12), 0.0, -b0 * np.cos(0.12)], rtol=1e-12
    )
    assert screens[1].arrival_time_s < screens[0].arrival_time_s
    # Kinematic guard is opt-in. 1 scale is smaller than 1e4 Å; demanding
    # hundreds of straight formation scales rejects the condensed run.
    guarded = dict(kw, max_angle_rad=0.2, formation_energy_eV=1.0, formation_theta_max_rad=1.0)
    assert (
        len(entrance_pec_screens_from_crossings(entries, min_formation_lengths=1.0, **guarded)) == 2
    )
    with pytest.raises(ValueError, match="formation-scale"):
        entrance_pec_screens_from_crossings(entries, min_formation_lengths=100.0, **guarded)
    with pytest.raises(ValueError, match="photon energy"):
        entrance_pec_screens_from_crossings(entries, min_formation_lengths=2.0, **kw)
    entries.straight_after_ang = None
    with pytest.raises(ValueError, match="missing"):
        entrance_pec_screens_from_crossings(entries, min_formation_lengths=1.0, **guarded)
