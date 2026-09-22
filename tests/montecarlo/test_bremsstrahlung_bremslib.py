"""BremsLib direction-resolved bremsstrahlung: staging, interpolation, spectrum.

Validation: bremslib-angular-model
"""

import warnings

import numpy as np
import pytest

from pyrite._backend import REAL, xp
from pyrite.montecarlo.spectrum import brem
from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    prepare_bremslib_table,
    solid_angle_integral,
    stage_bremslib_table,
    top_reduced_energy,
)
from tests.helpers import scaled_rtol, to_host
from tests.helpers.bremslib import (
    SYNTHETIC_T1_MEV,
    synthetic_bremslib_arrays,
    synthetic_scaled_sdcs,
    synthetic_shape,
)


def _zero_mu(_composition, energy):
    return xp.zeros_like(xp.asarray(energy, dtype=REAL))


@pytest.fixture
def carbon_table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6, key="k", digest="d")


def _evaluate(table, T_keV, photon_eV, cos_theta=None):
    staged = stage_bremslib_table(table)
    T = np.atleast_1d(np.asarray(T_keV, dtype=float))
    cos = None if cos_theta is None else np.atleast_1d(np.asarray(cos_theta, dtype=float))
    state = bremslib_segment_state(staged, T, cos)
    return to_host(evaluate_bremslib(staged, state, np.asarray(photon_eV, dtype=float)))


def _segments(T_keV, v_hat, length_ang=100.0):
    v = np.atleast_2d(np.asarray(v_hat, dtype=float))
    n = v.shape[0]
    return {
        "r_mid": np.tile([0.0, 0.0, 0.5 * length_ang], (n, 1)),
        "v_hat": v,
        "L_ang": np.full(n, length_ang),
        "E_keV": np.full(n, float(T_keV)),
        "elec_id": np.zeros(n, dtype=int),
        "Ne": 1,
        "thickness_ang": length_ang,
    }


def test_top_reduced_energy_follows_the_manual():
    np.testing.assert_allclose(
        top_reduced_energy([1.0e-3, 5.0e-3, 1.0e-2, 0.5, 0.6, 30.0]),
        [0.99, 0.99, 1.0 - 50.0e-6 / 1.0e-2, 1.0 - 50.0e-6 / 0.5, 0.9999, 0.9999],
    )


def test_every_staged_node_integrates_to_its_sdcs(carbon_table):
    integral = solid_angle_integral(carbon_table.theta_rad, carbon_table.scaled_ddcs_mb_sr)
    np.testing.assert_allclose(integral, carbon_table.scaled_sdcs_mb, rtol=1.0e-12)


def test_refinement_keeps_library_nodes_and_inserts_geometric_sub_nodes(carbon_table):
    arrays = synthetic_bremslib_arrays()
    theta = carbon_table.theta_rad
    raw = arrays["ddcs_mb_sr"].astype(float).reshape(SYNTHETIC_T1_MEV.size, 13, theta.size)
    shape = raw / solid_angle_integral(theta, raw)[..., None]

    np.testing.assert_allclose(carbon_table.incident_energy_keV[::4], SYNTHETIC_T1_MEV * 1.0e3)
    np.testing.assert_allclose(
        carbon_table.scaled_ddcs_mb_sr[::4], shape * arrays["sdcs_mb"][..., None], rtol=1e-12
    )
    # The midpoint sub-node of the first interval: geometric mean, renormalized.
    midpoint = np.sqrt(shape[0] * shape[1])
    midpoint /= solid_angle_integral(theta, midpoint)[..., None]
    chi = np.sqrt(arrays["sdcs_mb"][0] * arrays["sdcs_mb"][1])
    np.testing.assert_allclose(carbon_table.incident_energy_keV[2], np.sqrt(10.0 * 20.0))
    np.testing.assert_allclose(
        carbon_table.scaled_ddcs_mb_sr[2], midpoint * chi[:, None], rtol=1e-12
    )


def test_evaluation_reproduces_a_library_node(carbon_table):
    T_keV, reduced, theta_deg = 50.0, 0.4, 30.0
    k_eV = reduced * T_keV * 1.0e3
    got = _evaluate(carbon_table, T_keV, [k_eV], np.cos(np.radians(theta_deg)))[0, 0]

    theta = carbon_table.theta_rad
    shape = synthetic_shape(0.05, theta)
    expected_scaled = (
        synthetic_scaled_sdcs(0.05, reduced)
        * np.interp(np.radians(theta_deg), theta, shape)
        / solid_angle_integral(theta, shape)
    )
    expected = expected_scaled * 1.0e-27 * 36.0 / k_eV
    # The stored DDCS is float32, as in a released table.
    np.testing.assert_allclose(got, expected, rtol=1.0e-6)


def test_solid_angle_integral_of_ddcs_recovers_sdcs_off_grid(carbon_table):
    # Evaluated on the table's own theta grid the interpolant is piecewise
    # linear with its breakpoints there, so the exact linear-interpolant
    # integral is the whole integral.
    T_keV = np.array([13.7, 71.0, 188.0])
    k_eV = np.array([0.5e3, 3.3e3, 9.0e3, 12.9e3])
    sdcs = _evaluate(carbon_table, T_keV, k_eV)
    theta = carbon_table.theta_rad
    ddcs = np.stack(
        [_evaluate(carbon_table, T_keV, k_eV, np.full(T_keV.size, np.cos(t))) for t in theta],
        axis=-1,
    )
    integral = solid_angle_integral(theta, ddcs)
    positive = sdcs > 0.0
    assert positive.sum() > 6
    np.testing.assert_allclose(
        integral[positive], sdcs[positive], rtol=scaled_rtol(1.0e-10, eps_multiple=4096.0)
    )
    np.testing.assert_array_equal(integral[~positive], 0.0)


def test_cross_section_vanishes_above_the_kinematic_tip(carbon_table):
    got = _evaluate(carbon_table, 20.0, [19.99e3, 20.0e3, 20.01e3, 25.0e3])[0]
    assert got[0] > 0.0 and got[1] > 0.0
    np.testing.assert_array_equal(got[2:], 0.0)


def test_prepare_rejects_an_incomplete_node_grid():
    arrays = synthetic_bremslib_arrays()
    arrays["node_k_index"] = arrays["node_k_index"][::-1].copy()
    with pytest.raises(ValueError, match="every"):
        prepare_bremslib_table(arrays, atomic_number=6)


def test_mc_brem_weights_each_segment_by_its_emission_angle(monkeypatch, carbon_table):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    photon_eV = np.array([2.0e3, 20.0e3, 45.0e3])
    theta_obs = np.radians(119.0)
    density, length, T_keV = 0.1, 100.0, 60.0
    spectrum = brem.mc_brem_spectrum(
        _segments(T_keV, [0.0, 0.0, 1.0], length),
        photon_eV,
        composition=[("C", density)],
        theta_obs_rad=theta_obs,
        cross_section_model="bremslib",
        bremslib_tables={"C": carbon_table},
    )
    ddcs = _evaluate(carbon_table, T_keV, photon_eV, np.cos(theta_obs))[0]
    expected = density * 1.0e24 * length * 1.0e-8 * ddcs
    np.testing.assert_allclose(spectrum, expected, rtol=scaled_rtol(1.0e-12, eps_multiple=64.0))


def test_direction_average_matches_the_isotropic_sdcs_yield(monkeypatch, carbon_table):
    """Averaging the directional estimate over electron directions gives SDCS/(4 pi)."""
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    photon_eV = np.array([2.0e3, 20.0e3, 45.0e3])
    n_dirs = 4000
    mu = -1.0 + (np.arange(n_dirs) + 0.5) * (2.0 / n_dirs)  # equal solid-angle cells
    v_hat = np.stack([np.sqrt(1.0 - mu**2), np.zeros(n_dirs), mu], axis=1)
    segments = _segments(60.0, v_hat)
    segments["Ne"] = n_dirs
    segments["elec_id"] = np.arange(n_dirs)
    spectrum = brem.mc_brem_spectrum(
        segments,
        photon_eV,
        composition=[("C", 0.1)],
        theta_obs_rad=0.0,
        cross_section_model="bremslib",
        bremslib_tables={"C": carbon_table},
    )
    sdcs = _evaluate(carbon_table, 60.0, photon_eV)[0]
    isotropic = 0.1 * 1.0e24 * 100.0 * 1.0e-8 * sdcs / (4.0 * np.pi)
    np.testing.assert_allclose(
        spectrum, isotropic, rtol=max(2.0e-5, 1.0e3 * float(np.finfo(REAL).eps))
    )


def test_forward_emission_exceeds_backward_at_high_energy(monkeypatch, carbon_table):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    photon_eV = np.array([50.0e3])

    def toward(v):
        return brem.mc_brem_spectrum(
            _segments(180.0, v),
            photon_eV,
            composition=[("C", 0.1)],
            n_hat=[0.0, 0.0, 1.0],
            cross_section_model="bremslib",
            bremslib_tables={"C": carbon_table},
        )[0]

    forward, backward = np.radians(30.0), np.radians(150.0)
    assert toward([np.sin(forward), 0.0, np.cos(forward)]) > 5.0 * toward(
        [np.sin(backward), 0.0, np.cos(backward)]
    )


def test_bremslib_requires_tables():
    with pytest.raises(ValueError, match="bremslib_tables"):
        brem.mc_brem_spectrum(
            _segments(60.0, [0.0, 0.0, 1.0]),
            np.array([1.0e3]),
            composition=[("C", 0.1)],
            cross_section_model="bremslib",
        )


def test_bremslib_requires_segment_directions(carbon_table):
    segments = _segments(60.0, [0.0, 0.0, 1.0])
    del segments["v_hat"]
    with pytest.raises(ValueError, match="v_hat"):
        brem.mc_brem_spectrum(
            segments,
            np.array([1.0e3]),
            composition=[("C", 0.1)],
            cross_section_model="bremslib",
            bremslib_tables={"C": carbon_table},
        )


def test_element_without_table_warns_and_emits_isotropic_eedl(monkeypatch):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    photon_eV = np.array([2.0e3, 20.0e3])
    segments = _segments(60.0, [0.0, 0.0, 1.0])
    with pytest.warns(RuntimeWarning, match="no BremsLib table supplied for C"):
        got = brem.mc_brem_spectrum(
            segments,
            photon_eV,
            composition=[("C", 0.1)],
            cross_section_model="bremslib",
            bremslib_tables={},
        )
    expected = brem.mc_brem_spectrum(segments, photon_eV, composition=[("C", 0.1)])
    np.testing.assert_allclose(got, expected, rtol=scaled_rtol(1.0e-12, eps_multiple=8.0))


def test_segments_outside_the_table_warn_and_emit_isotropic_eedl(monkeypatch, carbon_table):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    photon_eV = np.array([2.0e3, 20.0e3])
    inside = _segments(60.0, [0.0, 0.0, 1.0])
    outside = _segments(400.0, [0.0, 0.0, 1.0])  # the synthetic table stops at 200 keV
    both = {
        key: np.concatenate([inside[key], outside[key]])
        for key in ("r_mid", "v_hat", "L_ang", "E_keV")
    }
    both.update(elec_id=np.zeros(2, dtype=int), Ne=1, thickness_ang=100.0)

    kwargs = dict(composition=[("C", 0.1)], cross_section_model="bremslib")
    tables = {"C": carbon_table}
    with pytest.warns(RuntimeWarning, match="outside the BremsLib range"):
        got = brem.mc_brem_spectrum(both, photon_eV, bremslib_tables=tables, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        directional = brem.mc_brem_spectrum(inside, photon_eV, bremslib_tables=tables, **kwargs)
    isotropic = brem.mc_brem_spectrum(outside, photon_eV, composition=[("C", 0.1)])
    np.testing.assert_allclose(
        got, directional + isotropic, rtol=scaled_rtol(1.0e-12, eps_multiple=16.0)
    )


def test_bremslib_spectrum_is_chunk_invariant(monkeypatch, carbon_table):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)
    rng = np.random.default_rng(87)
    n = 37
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    segments = _segments(60.0, v)
    segments["E_keV"] = rng.uniform(12.0, 190.0, n)
    photon_eV = np.linspace(0.5e3, 150.0e3, 23)
    kwargs = dict(
        composition=[("C", 0.1)],
        cross_section_model="bremslib",
        bremslib_tables={"C": carbon_table},
    )
    whole = brem.mc_brem_spectrum(segments, photon_eV, **kwargs)
    pieces = brem.mc_brem_spectrum(segments, photon_eV, chunk=5, **kwargs)
    np.testing.assert_allclose(pieces, whole, rtol=scaled_rtol(1.0e-12, eps_multiple=64.0))


def test_angle_integrated_selector_returns_the_bremslib_sdcs(carbon_table):
    T_keV = np.array([30.0, 150.0])
    photon_eV = np.array([1.0e3, 25.0e3])
    got = to_host(
        brem._bremsstrahlung_dsigma_dk(
            "C",
            T_keV,
            photon_eV,
            cross_section_model="bremslib",
            bremslib_tables={"C": carbon_table},
        )
    )
    np.testing.assert_allclose(got, _evaluate(carbon_table, T_keV, photon_eV), rtol=1e-12)
