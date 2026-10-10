"""Escape attenuation averaged along each segment (issue #176).

Validation: segment-escape-average
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite._backend import REAL, xp
from pyrite.montecarlo.groove import blazed_groove_spec, escape_distance_ang
from pyrite.montecarlo.spectrum import brem, characteristic
from pyrite.montecarlo.spectrum.segment_escape import mean_transmission, segment_escape_paths
from tests.helpers import scaled_rtol

MU_INV_ANG = 1.0 / 300.0  # C K-like: attenuation length comparable to a segment
# Endpoint escape paths round at REAL, so exp(-tau) carries a relative error
# ~tau eps(REAL); the box/layer oracles below reach tau ~25.
ESCAPE_RTOL = scaled_rtol(1e-10, eps_multiple=30.0)


def _absorbing_mu(_composition, energy):
    return xp.full_like(xp.asarray(energy, dtype=REAL), MU_INV_ANG)


def _track(lengths, z0=0.0, direction=(0.0, 0.0, 1.0)):
    """One straight electron track cut into collinear segments."""
    lengths = np.asarray(lengths, dtype=float)
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    s_mid = np.cumsum(lengths) - 0.5 * lengths
    return {
        "r_mid": np.column_stack(
            (s_mid * v[0], s_mid * v[1], z0 + s_mid * v[2]),
        ),
        "v_hat": np.tile(v, (lengths.size, 1)),
        "L_ang": lengths,
        "E_keV": np.full(lengths.size, 30.0),
        "elec_id": np.zeros(lengths.size, dtype=int),
        "layer": np.zeros(lengths.size, dtype=int),
        "Ne": 1,
    }


def _spectrum(segments, thickness, **kwargs):
    segments = {**segments, "thickness_ang": thickness}
    return characteristic.mc_characteristic_spectrum(
        segments,
        np.arange(250.0, 291.0, 1.0),
        composition=[("C", 0.1)],
        electron_limit=1,
        **kwargs,
    )


def test_mean_transmission_matches_quadrature_and_its_limits():
    tau0 = np.array([0.0, 0.3, 2.0, 5.0, 40.0, 1.0])
    tau1 = np.array([0.0, 0.3 + 1e-6, 0.5, 9.0, 800.0, 1.0 + 3e-5])
    reference = np.array(
        [
            quad(
                lambda s, a=a, b=b: np.exp(-(a + (b - a) * s)), 0.0, 1.0, epsabs=0.0, epsrel=1e-13
            )[0]
            for a, b in zip(tau0, tau1, strict=True)
        ]
    )

    got = mean_transmission(tau0, tau1)

    np.testing.assert_allclose(got, reference, rtol=1e-9, atol=0.0)
    assert got[0] == 1.0  # no absorption
    assert np.all(np.isfinite(got)) and np.all(got <= np.exp(-np.minimum(tau0, tau1)))
    np.testing.assert_allclose(mean_transmission(tau1, tau0), got, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("pieces", [2, 8, 32])
def test_absorbing_slab_yield_is_invariant_under_segment_splitting(monkeypatch, pieces):
    """A long segment and its collinear pieces escape identically."""
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _absorbing_mu)
    direction = (0.3, 0.0, 1.0)

    whole = _spectrum(_track([2000.0], z0=100.0, direction=direction), 3000.0)
    split = _spectrum(_track([2000.0 / pieces] * pieces, z0=100.0, direction=direction), 3000.0)

    np.testing.assert_allclose(split, whole, rtol=1e-6, atol=0.0)


def test_absorbed_yield_is_the_transparent_yield_times_the_segment_mean(monkeypatch):
    """Issue #176: the yield carries the segment mean, which exceeds the midpoint value."""
    track = _track([2000.0], z0=100.0)  # depth 100 -> 2100 A, escaping backwards
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", lambda c, e: 0.0 * _absorbing_mu(c, e))
    transparent = _spectrum(track, 3000.0).sum()
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _absorbing_mu)
    absorbed = _spectrum(track, 3000.0).sum()

    n_z = abs(np.cos(np.deg2rad(119.0)))
    tau = MU_INV_ANG / n_z * np.array([100.0, 2100.0])
    mean_factor = mean_transmission(tau[:1], tau[1:])[0]
    midpoint_factor = np.exp(-tau.mean())

    np.testing.assert_allclose(absorbed, transparent * mean_factor, rtol=1e-6)
    assert mean_factor > 2.0 * midpoint_factor


def test_layered_stack_yield_is_invariant_under_segment_splitting(monkeypatch):
    """Escape through an overlying layer keeps tau linear inside the emitter."""
    monkeypatch.setattr(characteristic, "_mu_total_inv_ang", _absorbing_mu)
    layers = [(0.0, 500.0, [("C", 0.1)]), (500.0, 4000.0, [("C", 0.1)])]
    track = dict(z0=800.0, direction=(0.2, 0.0, 1.0))

    whole = {**_track([2400.0], **track), "layer": np.array([1])}
    split = {**_track([300.0] * 8, **track), "layer": np.ones(8, dtype=int)}

    np.testing.assert_allclose(
        _spectrum(split, 4000.0, layers=layers),
        _spectrum(whole, 4000.0, layers=layers),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize("pieces", [1, 8, 32])
def test_bremsstrahlung_escape_is_invariant_under_segment_splitting(monkeypatch, pieces):
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _absorbing_mu)
    kwargs = dict(
        E_grid_eV=np.array([1000.0, 2000.0, 3000.0]),
        composition=[("C", 0.1)],
        cross_section_model="bethe-heitler",
    )
    whole = brem.mc_brem_spectrum({**_track([2000.0], z0=100.0), "thickness_ang": 3000.0}, **kwargs)
    split = brem.mc_brem_spectrum(
        {**_track([2000.0 / pieces] * pieces, z0=100.0), "thickness_ang": 3000.0},
        **kwargs,
    )
    np.testing.assert_allclose(split, whole, rtol=1e-6)


@pytest.mark.parametrize("pieces", [1, 4, 20])
def test_finite_side_exit_crosses_absorber_layer_at_exact_breakpoint(pieces):
    """A thin W crossing near a box exit is resolved independently of track splitting."""
    layers = [(0.0, 10.0, [("C", 0.1)]), (10.0, 100.0, [("W", 0.1)])]
    segments = _track([10.0 / pieces] * pieces)
    segments["r_mid"][:, 0] = 4.0
    segments.update(thickness_ang=100.0, crystal_width_ang=10.0, crystal_height_ang=10.0)
    n_hat = np.array([1.0, 0.0, 0.01])
    n_hat /= np.linalg.norm(n_hat)
    mu = np.array([0.1, 10.0])

    owner, fraction, start, end = segment_escape_paths(
        segments, np.arange(pieces), n_hat, layers=layers
    )
    actual = (
        np.sum(segments["L_ang"][owner] * fraction * mean_transmission(start @ mu, end @ mu)) / 10.0
    )

    side_distance = 1.0 / n_hat[0]
    z_cross = 10.0 - n_hat[2] * side_distance

    def transmission(z):
        carbon_path = min(side_distance, max((10.0 - z) / n_hat[2], 0.0))
        return np.exp(-0.1 * carbon_path - 10.0 * (side_distance - carbon_path))

    expected = (
        quad(transmission, 0.0, z_cross, epsrel=1e-12)[0]
        + quad(transmission, z_cross, 10.0, epsrel=1e-12)[0]
    ) / 10.0
    np.testing.assert_allclose(actual, expected, rtol=ESCAPE_RTOL)
    assert np.all(np.bincount(owner, minlength=pieces) > 0)


@pytest.mark.parametrize("layered", [False, True])
def test_finite_box_track_starting_on_entrance_face_escapes_through_it(layered):
    """Every track starts on z = 0; escape back through that face starts at zero path.

    The exit distance is degenerate on that face (the compiled prism exit once
    skipped it for the far side face), so endpoints come from interior samples.
    """
    segments = _track([400.0], direction=(0.5, 0.0, 1.0))
    segments.update(thickness_ang=2000.0, crystal_width_ang=5.0e7, crystal_height_ang=5.0e7)
    n_hat = np.array([np.cos(0.05), 0.0, -np.sin(0.05)])
    layers = [(0.0, 100.0, [("C", 0.1)]), (100.0, 2000.0, [("C", 0.1)])] if layered else None
    mu = np.full(2 if layered else 1, MU_INV_ANG)

    owner, fraction, start, end = segment_escape_paths(segments, np.arange(1), n_hat, layers=layers)
    actual = np.sum(fraction * mean_transmission(start @ mu, end @ mu))

    v_z = 1.0 / np.hypot(0.5, 1.0)
    expected = (
        quad(lambda s: np.exp(-MU_INV_ANG * s * v_z / n_hat[2] * -1.0), 0.0, 400.0)[0] / 400.0
    )
    np.testing.assert_allclose(actual, expected, rtol=ESCAPE_RTOL)
    np.testing.assert_allclose(start.sum(axis=1)[owner == 0][0], 0.0, atol=1e-6)


def test_groove_escape_is_invariant_across_periodic_discontinuities():
    groove = blazed_groove_spec(100.0, np.pi / 2.0, np.pi / 4.0, np.pi)
    n_hat = np.array([np.cos(np.pi / 4.0), 0.0, -np.sin(np.pi / 4.0)])

    def average(pieces):
        segments = _track([200.0 / pieces] * pieces, z0=200.0, direction=(1.0, 0.0, 0.0))
        segments["thickness_ang"] = 1000.0
        owner, fraction, start, end = segment_escape_paths(
            segments, np.arange(pieces), n_hat, groove=groove
        )
        return (
            np.sum(
                segments["L_ang"][owner]
                * fraction
                * mean_transmission(start[:, 0] / 100.0, end[:, 0] / 100.0)
            )
            / 200.0
        )

    whole = average(1)
    split = average(16)
    x_mid = (np.arange(200_000) + 0.5) * (200.0 / 200_000)
    reference = np.mean(np.exp(-escape_distance_ang(x_mid, 200.0, groove) / 100.0))

    np.testing.assert_allclose(whole, split, rtol=1e-10)
    np.testing.assert_allclose(whole, reference, rtol=2e-4)
