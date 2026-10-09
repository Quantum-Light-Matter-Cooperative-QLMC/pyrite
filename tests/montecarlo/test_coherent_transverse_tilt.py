"""Analytic transverse bunch form factor on a tilted face (issue #370).

Validation: transverse-bunch-form-factor
"""

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.geometry import project_beam_entry, sample_to_lab_R
from pyrite.montecarlo.spectrum.coherent_transverse import (
    transverse_envelope,
    transverse_form_factor,
    transverse_spot,
)
from pyrite.montecarlo.transport.beam_entry import beam_entry_record, face_arrival_delay_ang
from pyrite.montecarlo.transport.kinematics import beta_from_keV_scalar

TILT, AZIM = np.deg2rad(45.0), np.deg2rad(135.0)
E0_KEV = 30.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _record(fwhm_ang, fwhm_y_ang=None, *, tilt=TILT, azim=AZIM):
    return beam_entry_record(
        transverse_distribution=None,
        beam_fwhm_mm=fwhm_ang * 1e-7,
        beam_fwhm_y_mm=None if fwhm_y_ang is None else fwhm_y_ang * 1e-7,
        tilt_polar_rad=tilt,
        tilt_azim_rad=azim,
        E0_keV=E0_KEV,
        energy_spread_frac=None,
        groove=None,
    )


def _phase(w, omega, q_vec, tilt, azim):
    """Independent ray-plane construction of the offset phase phi(w).

    Lab ray ``(u, v, 0) + s z_lab``; sample frame by ``R^T``; hits ``z = 0``
    after ``s*``, arriving ``s*/beta`` late. ``phi = omega s*/beta - Q . p0``.
    """
    Rt = sample_to_lab_R(tilt, azim).T
    origin = w[:, 0:1] * Rt[:, 0] + w[:, 1:2] * Rt[:, 1]
    beam = Rt[:, 2]
    s_star = -origin[:, 2] / beam[2]
    p0 = origin + s_star[:, None] * beam
    beta = beta_from_keV_scalar(E0_KEV)
    return omega * s_star / beta - p0 @ q_vec


def _brute_force(cov, omega, q_vec, tilt=TILT, azim=AZIM, draws=400_000):
    w = np.random.default_rng(370).multivariate_normal(np.zeros(2), cov, size=draws)
    return float(np.abs(np.mean(np.exp(1j * _phase(w, omega, q_vec, tilt, azim)))) ** 2)


def _kappa(omega, q_vec, tilt=TILT, azim=AZIM):
    """phi is linear in w: read kappa off by finite differences."""
    unit = np.eye(2)
    zero = _phase(np.zeros((1, 2)), omega, q_vec, tilt, azim)[0]
    return np.array([_phase(unit[i : i + 1], omega, q_vec, tilt, azim)[0] - zero for i in (0, 1)])


@pytest.mark.parametrize(
    ("fwhm", "fwhm_y", "n_hat", "g"),
    [
        (2.0, None, (0.3, -0.2, -0.93), (0.1, 0.4, 1.87)),
        (1.5, 3.0, (0.8, 0.1, 0.59), (-0.2, 0.0, -1.87)),
    ],
)
def test_tilted_gaussian_spot_matches_brute_force_offset_average(fwhm, fwhm_y, n_hat, g):
    spot = transverse_spot(_record(fwhm, fwhm_y))
    sigma_x = fwhm * FWHM_TO_SIGMA
    sigma_y = (fwhm if fwhm_y is None else fwhm_y) * FWHM_TO_SIGMA
    cov = np.diag([sigma_x**2, sigma_y**2])
    n_hat, g = np.asarray(n_hat) / np.linalg.norm(n_hat), np.asarray(g)
    omega = 0.5
    q_vec = omega * n_hat + g
    kappa = _kappa(omega, q_vec)
    closed = float(np.exp(-kappa @ cov @ kappa))
    assert 0.05 < closed < 0.95  # nondegenerate test point
    actual = float(transverse_form_factor(spot, np.array([omega]), n_hat, g)[0])
    np.testing.assert_allclose(actual, closed, rtol=1e-10)
    # Monte Carlo of the complete-field phase: standard error ~ 1/sqrt(draws).
    np.testing.assert_allclose(_brute_force(cov, omega, q_vec), closed, atol=5e-3)


def test_phase_matched_tilt_is_fully_coherent_despite_transverse_momentum():
    """K = (Q - omega b / beta)_xy = 0 gives F_perp = 1 although Q_perp != 0."""
    spot = transverse_spot(_record(4.0e4))
    omega = 0.5
    n_hat = np.array([0.2, -0.3, 0.93])
    n_hat /= np.linalg.norm(n_hat)
    beam = sample_to_lab_R(TILT, AZIM).T[:, 2]
    g = np.zeros(3)
    g[:2] = -omega * (n_hat[:2] - beam[:2] / beta_from_keV_scalar(E0_KEV))
    g[2] = 1.87
    q_vec = omega * n_hat + g
    q_perp = q_vec - (q_vec @ beam) * beam
    sigma = 4.0e4 * FWHM_TO_SIGMA
    assert np.linalg.norm(q_perp) * sigma > 1e3  # zero-tilt formula would give 0
    np.testing.assert_allclose(_kappa(omega, q_vec), 0.0, atol=1e-9)
    actual = float(transverse_form_factor(spot, np.array([omega]), n_hat, g)[0])
    np.testing.assert_allclose(actual, 1.0, rtol=1e-12)
    cov = np.diag([sigma**2, sigma**2])
    np.testing.assert_allclose(_brute_force(cov, omega, q_vec, draws=20_000), 1.0, rtol=1e-9)


def test_zero_tilt_recovers_q_perp_and_point_spot_is_coherent():
    spot = transverse_spot(_record(2.0, tilt=0.0, azim=0.0))
    sigma = 2.0 * FWHM_TO_SIGMA
    n_hat, g = np.array([0.6, 0.0, 0.8]), np.array([0.3, -0.1, 1.0])
    omega = np.array([0.2, 0.7])
    q_perp = omega[:, None] * n_hat[:2] + g[:2]
    expected = np.exp(-((q_perp**2).sum(axis=1)) * sigma**2)
    np.testing.assert_allclose(transverse_form_factor(spot, omega, n_hat, g), expected, rtol=1e-12)
    point = transverse_spot(_record(1e-30))
    np.testing.assert_allclose(transverse_form_factor(point, omega, n_hat, g), 1.0, rtol=1e-12)


def test_window_envelope_is_nonincreasing_and_bounds_the_factor():
    spot = transverse_spot(_record(2.0))
    n_hat, g = np.array([0.3, -0.2, -0.93]), np.array([0.1, 0.4, 1.87])
    envelope = transverse_envelope(spot, n_hat, g)
    energy = np.linspace(1.0, 4000.0, 4001)
    bound = envelope(energy)
    exact = transverse_form_factor(spot, energy / HBARC_EV_ANG, n_hat, g)
    assert np.all(np.diff(bound) <= 0.0)
    assert np.all(bound >= exact * (1.0 - 1e-12))


def test_face_arrival_delay_is_the_vacuum_flight_and_vanishes_untilted():
    record = _record(2.0)
    w = np.random.default_rng(1).normal(size=(50, 2)) * 1e3
    p0 = np.zeros((50, 3))
    p0[:, :2] = project_beam_entry(w, TILT, AZIM)
    Rt = sample_to_lab_R(TILT, AZIM).T
    origin = w[:, 0:1] * Rt[:, 0] + w[:, 1:2] * Rt[:, 1]
    s_star = -origin[:, 2] / Rt[2, 2]
    energies = np.full(50, E0_KEV)
    np.testing.assert_allclose(
        face_arrival_delay_ang(record, p0, energies),
        s_star / beta_from_keV_scalar(E0_KEV),
        rtol=1e-12,
        atol=1e-9,
    )
    flat = _record(2.0, tilt=0.0, azim=0.0)
    assert np.array_equal(face_arrival_delay_ang(flat, p0, energies), np.zeros(50))


def _tilted_identical_tracks(count, fwhm_ang):
    """Identical straight tracks translated by sampled face points and delays."""
    record = _record(fwhm_ang)
    sigma = fwhm_ang * FWHM_TO_SIGMA
    w = np.random.default_rng(7).normal(size=(count, 2)) * sigma
    p0 = np.zeros((count, 3))
    p0[:, :2] = project_beam_entry(w, TILT, AZIM)
    t0 = face_arrival_delay_ang(record, p0, np.full(count, E0_KEV))
    reference = np.array([4.0, 0.0, 5.0])
    return {
        "r_mid": reference + p0,
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, E0_KEV),
        "t_ang": np.zeros(count),
        "t0_ang": t0,
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 1.0e4,
        "crystal_height_ang": 1.0e4,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
        "initial_r_ang": p0,
        "initial_t0_ang": t0,
        "beam_entry": record,
    }, record


def test_tilted_finite_footprint_reducer_applies_the_analytic_transverse_factor():
    """Offset-free identical fields: Y = |S|^2 [1 + (N-1) F_z F_perp(kappa)]."""
    energy = np.arange(700.0, 1500.0, 2.0)
    kwargs = {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "n_hat": np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1]),
    }
    segments, record = _tilted_identical_tracks(6, 2.0)
    single = {key: value for key, value in segments.items()}
    single.update(
        {
            key: segments[key][:1]
            for key in ("v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer")
        },
        r_mid=np.array([[4.0, 0.0, 5.0]]),
        t0_ang=np.zeros(1),
        initial_r_ang=np.zeros((1, 3)),
        initial_t0_ang=np.zeros(1),
        Ne=1,
    )
    self_term = mc_spectrum(single, energy, coherent=True, **kwargs)
    population = 50.0
    actual = mc_spectrum(
        segments,
        energy,
        coherent=True,
        longitudinal_rms_fs=1e-12,
        physical_electrons=population,
        **kwargs,
    )
    g_vec, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    g = np.asarray(g_vec, dtype=float)
    omega = energy / HBARC_EV_ANG
    sigma = 2.0 * FWHM_TO_SIGMA
    cov = np.diag([sigma**2, sigma**2])
    F_perp = np.array(
        [np.exp(-k @ cov @ k) for k in (_kappa(om, om * kwargs["n_hat"] + g) for om in omega)]
    )
    assert 0.01 < F_perp.max() and F_perp.min() < 0.99  # nondegenerate
    F_z = np.exp(-((omega * 1e-12 * 2997.92458) ** 2))
    expected = self_term * (1.0 + (population - 1.0) * F_z * F_perp)
    peak = float(np.max(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12 * peak)
    assert record["face_arrival_delay"]


def test_finite_footprint_refuses_a_spot_reaching_the_edge():
    segments, _ = _tilted_identical_tracks(4, 2.0)
    segments.update(crystal_width_ang=100.0, crystal_height_ang=100.0)
    with pytest.raises(ValueError, match="well inside the footprint"):
        mc_spectrum(
            segments,
            np.arange(700.0, 1500.0, 2.0),
            "hopg",
            [(0, 0, 2)],
            B_ang2=0.8,
            n_hat=np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1]),
            coherent=True,
            longitudinal_rms_fs=1e-12,
            physical_electrons=50.0,
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"position_slope_correlated": True}, "correlated"),
        ({"energy_spread": True}, "energy spread"),
        ({"face_arrival_delay": False}, "face-arrival delay"),
    ],
)
def test_unsupported_beams_are_refused(change, match):
    with pytest.raises(ValueError, match=match):
        transverse_spot({**_record(2.0), **change})
