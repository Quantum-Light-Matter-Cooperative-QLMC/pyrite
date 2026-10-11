"""Independent ray-plane and Gaussian references for linear bunch-front tilt.

Validation: linear-bunch-front-tilt, transverse-bunch-form-factor
"""

import numpy as np
import pytest

from pyrite import BunchFrontTilt
from pyrite.materials.crystal import HBARC_EV_ANG, beta_from_Ee
from pyrite.montecarlo.geometry import project_beam_entry, sample_to_lab_R
from pyrite.montecarlo.spectrum.coherent_transverse import (
    transverse_envelope,
    transverse_form_factor,
    transverse_form_factor_upper,
    transverse_spot,
)
from pyrite.montecarlo.transport.beam_entry import (
    beam_entry_record,
    bunch_front_delay_ang,
    initial_beam_positions,
    resolve_bunch_times,
)


def _record(front, fwhm=2.0, tilt=0.4, azim=0.7):
    return beam_entry_record(
        transverse_distribution=None,
        beam_fwhm_mm=fwhm * 1e-7,
        beam_fwhm_y_mm=fwhm * 1.5e-7,
        tilt_polar_rad=tilt,
        tilt_azim_rad=azim,
        E0_keV=30.0,
        energy_spread_frac=None,
        groove=None,
        bunch_front_tilt=front,
    )


@pytest.mark.parametrize("angle, azimuth", [(25.0, 0.0), (-25.0, 0.0), (25.0, 90.0)])
def test_signed_arrival_slope_and_separate_face_delay(angle, azimuth):
    front = BunchFrontTilt(angle, azimuth)
    record = _record(front)
    w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 2.0]])
    p = np.zeros((3, 3))
    p[:, :2] = project_beam_entry(w, 0.4, 0.7)
    beta = float(beta_from_Ee(30_000.0))
    unit = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    imposed = (w @ unit) * np.tan(np.deg2rad(angle)) / beta
    np.testing.assert_allclose(bunch_front_delay_ang(record, p), imposed, atol=1e-14)
    R = sample_to_lab_R(0.4, 0.7).T
    origin = w[:, 0:1] * R[:, 0] + w[:, 1:2] * R[:, 1]
    face_delay = -origin[:, 2] / R[2, 2] / beta
    actual, clock, _ = resolve_bunch_times(
        np.full(3, 0.2), np.zeros(3), record, p, np.full(3, 30.0)
    )
    np.testing.assert_allclose(actual, 0.2 + imposed + face_delay, atol=1e-14)
    assert np.array_equal(clock, np.zeros(3))


def test_counter_addressed_spot_draw_and_zero_tilt_are_unchanged():
    kwargs = dict(
        transverse_distribution=None,
        beam_fwhm_mm=0.001,
        beam_fwhm_y_mm=0.002,
        tilt_polar_rad=0.4,
        tilt_azim_rad=0.7,
        groove=None,
        E0_keV=30.0,
    )
    plain, _, record = initial_beam_positions(7, 32, **kwargs)
    zero, _, zero_record = initial_beam_positions(
        7, 32, bunch_front_tilt=BunchFrontTilt(0), **kwargs
    )
    assert np.array_equal(plain, zero) and record == zero_record
    front = BunchFrontTilt(10.0, 20.0)
    full, _, full_record = initial_beam_positions(7, 32, bunch_front_tilt=front, **kwargs)
    block, _, block_record = initial_beam_positions(7, 8, start=9, bunch_front_tilt=front, **kwargs)
    assert np.array_equal(full, plain)
    assert np.array_equal(full[9:17], block) and full_record == block_record
    assert np.array_equal(
        bunch_front_delay_ang(full_record, full)[9:17], bunch_front_delay_ang(block_record, block)
    )


@pytest.mark.parametrize("tilt", [0.0, 30.0, 84.5, -30.0])
@pytest.mark.parametrize("azimuth", [0.0, 75.0, 145.0])
def test_matching_front_and_crystal_angles_make_the_front_parallel_to_the_face(tilt, azimuth):
    """Same angle convention: all parallel rays enter with only residual delay."""
    tp, ta = np.deg2rad([tilt, azimuth])
    record = _record(BunchFrontTilt(tilt, azimuth), tilt=tp, azim=ta)
    w = np.random.default_rng(465).normal(size=(30, 2)) * 100.0
    p = np.zeros((len(w), 3))
    p[:, :2] = project_beam_entry(w, tp, ta)
    epsilon = np.linspace(-0.2, 0.2, len(w))
    t0, _, _ = resolve_bunch_times(epsilon, np.zeros(len(w)), record, p, np.full(len(w), 30.0))
    np.testing.assert_allclose(t0, epsilon, atol=2e-11, rtol=0.0)


def test_joint_form_factor_matches_independent_rays_and_brute_force():
    front = BunchFrontTilt(15.0, 30.0)
    record = _record(front)
    spot = transverse_spot(record)
    n = np.array([0.3, -0.2, 0.93])
    n /= np.linalg.norm(n)
    g = np.array([0.1, 0.4, 1.87])
    omega = 0.5
    beta = float(beta_from_Ee(30_000.0))
    R = sample_to_lab_R(0.4, 0.7).T
    a = np.tan(np.deg2rad(15.0)) / beta * np.array([np.cos(np.deg2rad(30.0)), 0.5])

    def phase(w):
        origin = w[:, 0:1] * R[:, 0] + w[:, 1:2] * R[:, 1]
        s = -origin[:, 2] / R[2, 2]
        p = origin + s[:, None] * R[:, 2]
        return omega * (w @ a + s / beta) - p @ (omega * n + g)

    kappa = phase(np.eye(2))
    covariance = np.asarray(record["spot_covariance_ang2"])
    expected = np.exp(-kappa @ covariance @ kappa)
    actual = float(transverse_form_factor(spot, np.array([omega]), n, g)[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    w = np.random.default_rng(465).multivariate_normal([0.0, 0.0], covariance, size=100_000)
    sampled = abs(np.mean(np.exp(1j * phase(w)))) ** 2
    np.testing.assert_allclose(sampled, expected, atol=0.005)


def test_phase_matching_and_window_bounds_keep_the_cross_term():
    n = np.array([0.3, 0.0, np.sqrt(0.91)])
    beta = float(beta_from_Ee(30_000.0))
    matched = BunchFrontTilt(np.rad2deg(np.arctan(beta * 0.3)))
    spot = transverse_spot(_record(matched, fwhm=1e5, tilt=0.0, azim=0.0))
    omega = np.array([0.1, 0.5, 1.0])
    g = np.array([0.0, 0.0, 1.87])
    np.testing.assert_allclose(transverse_form_factor(spot, omega, n, g), 1.0, atol=1e-12)
    assert transverse_form_factor_upper(spot, (0.1, 1.0), n, g) >= 1.0 - 1e-12
    bound = transverse_envelope(spot, n, g)(omega * HBARC_EV_ANG)
    np.testing.assert_allclose(bound, 1.0, atol=1e-12)
    reversed_spot = transverse_spot(
        _record(BunchFrontTilt(-matched.tilt_deg), fwhm=1e5, tilt=0.0, azim=0.0)
    )
    assert np.max(transverse_form_factor(reversed_spot, omega, n, g)) == 0.0


def test_conditional_and_marginal_duration_have_distinct_variances():
    front = BunchFrontTilt(30.0, 45.0)
    record = _record(front, fwhm=100.0, tilt=0.0, azim=0.0)
    rng = np.random.default_rng(465)
    w = rng.multivariate_normal([0.0, 0.0], record["spot_covariance_ang2"], 100_000)
    epsilon = rng.normal(0.0, 2.0, len(w))
    p = np.column_stack([w, np.zeros(len(w))])
    delay = bunch_front_delay_ang(record, p)
    gradient = np.tan(np.deg2rad(30.0)) / float(beta_from_Ee(30_000.0)) * np.array([2**-0.5] * 2)
    expected = 4.0 + gradient @ np.asarray(record["spot_covariance_ang2"]) @ gradient
    np.testing.assert_allclose(np.var(epsilon + delay), expected, rtol=0.02)
    np.testing.assert_allclose(np.std((epsilon + delay) - delay), 2.0, rtol=0.02)


@pytest.mark.parametrize("finite", [False, True])
def test_complete_reducer_phase_matching_recovers_physical_population_enhancement(finite):
    """Identical tracks recover N times the self density at either footprint."""
    from types import SimpleNamespace

    from pyrite.montecarlo import mc_cxr_spectrum
    from pyrite.montecarlo.spectrum.cxr._setup import SpectrumRequest, _prepare_spectrum
    from pyrite.montecarlo.spectrum.cxr._temporal import coherent_offset_chi

    n = np.array([0.3, 0.0, np.sqrt(0.91)])
    beta = float(beta_from_Ee(30_000.0))
    front = BunchFrontTilt(np.rad2deg(np.arctan(beta * 0.3)))
    record = _record(front, fwhm=200.0, tilt=0.0, azim=0.0)
    count = 6
    w = np.random.default_rng(465).normal(size=(count, 2)) * 50.0
    p = np.column_stack([w, np.zeros(count)])
    t0 = bunch_front_delay_ang(record, p)
    segments = dict(
        r_mid=np.array([4.0, 0.0, 5.0]) + p,
        v_hat=np.tile([0.0, 0.0, 1.0], (count, 1)),
        L_ang=np.full(count, 10.0),
        E_keV=np.full(count, 30.0),
        t_ang=np.zeros(count),
        t0_ang=t0,
        elec_id=np.arange(count),
        layer=np.zeros(count, dtype=int),
        Ne=count,
        thickness_ang=10.0,
        n_backscattered=0,
        n_missed=0,
        n_layers=1,
        initial_r_ang=p,
        initial_t0_ang=t0,
        beam_entry=record,
    )
    if finite:
        segments.update(crystal_width_ang=1e6, crystal_height_ang=1e6)
    single = dict(segments)
    for key in ("v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        single[key] = segments[key][:1]
    single.update(
        r_mid=np.array([[4.0, 0.0, 5.0]]),
        t0_ang=np.zeros(1),
        initial_r_ang=np.zeros((1, 3)),
        initial_t0_ang=np.zeros(1),
        Ne=1,
    )
    energy = np.arange(1500.0, 2200.0, 2.0)
    kwargs = dict(crystal="hopg", hkl_list=[(0, 0, 2)], B_ang2=0.8, n_hat=n)
    self_term = mc_cxr_spectrum(single, energy, coherent=True, **kwargs)
    population = 50.0
    actual = mc_cxr_spectrum(
        segments,
        energy,
        coherent=True,
        longitudinal_rms_fs=0.0,
        physical_electrons=population,
        **kwargs,
    )
    assert self_term.max() > 0
    np.testing.assert_allclose(
        actual, population * self_term, rtol=1e-6, atol=1e-12 * self_term.max()
    )
    # Temporal characteristic function must be the amplitude (not intensity)
    # factor on the same grid, with no extra sampled-offset factor.
    request = SpectrumRequest(
        segments=segments,
        E_grid_eV=energy,
        coherent=True,
        longitudinal_rms_fs=0.0,
        physical_electrons=population,
        **kwargs,
    )
    state = _prepare_spectrum(request)
    profile = SimpleNamespace(energy_grid=lambda: energy, n=len(energy))
    chi = coherent_offset_chi(state, profile, np.array([0.0, 0.0, 1.87]))
    np.testing.assert_allclose(chi, 1.0, atol=1e-12)
