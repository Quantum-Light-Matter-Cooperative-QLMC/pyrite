"""Coherent transition radiation (CTR) between transported histories (#103)."""

import dataclasses

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.spectrum.coherent_population import (
    CoherentSamplingError,
    CoherentSamplingWarning,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    BoundaryCrossings,
    crossing_transition_radiation,
)

_V = np.array([0.0, 0.0, 0.995])
_EPS = [1 - (31.0 / 300.0) ** 2 * (1 - 0.05j)]
_THETA = 0.08
_N = np.array([np.sin(_THETA), 0.0, np.cos(_THETA)])


def _film(foil=2e4):
    """Entry and exit crossings of one straight charge through a film at t = 0."""
    planes = np.array([0.0, foil])
    t = planes / _V[2]
    return BoundaryCrossings(
        electron_id=np.zeros(2, int), track_id=np.zeros(2, int), charge=-np.ones(2, np.int8),
        position_ang=_V * t[:, None], direction=np.tile([0.0, 0.0, 1.0], (2, 1)),
        kinetic_energy_keV=np.full(2, M_E_EV * (1 / np.sqrt(1 - _V @ _V) - 1) / 1e3),
        time_ang=t, interface=np.arange(2), normal=np.tile([0.0, 0.0, 1.0], (2, 1)),
        layer_before=np.array([-1, 0]), layer_after=np.array([0, -1]), z_interfaces_ang=planes,
        n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _bunch(t0_ang, dx_ang):
    """Copies of the film history, delayed by ``t0`` and shifted along ``x``."""
    one = _film()
    m, k = len(t0_ang), len(one.interface)
    position = np.repeat(one.position_ang[None], m, 0)
    position[:, :, 0] += np.asarray(dx_ang)[:, None]
    fields = {
        f.name: np.concatenate([getattr(one, f.name)] * m)
        for f in dataclasses.fields(one)
        if isinstance(getattr(one, f.name), np.ndarray) and f.name != "z_interfaces_ang"
    }
    fields.update(
        electron_id=np.repeat(np.arange(m), k), track_id=np.repeat(np.arange(m), k),
        position_ang=position.reshape(-1, 3),
        time_ang=(one.time_ang[None] + np.asarray(t0_ang)[:, None]).reshape(-1),
    )  # fmt: skip
    return dataclasses.replace(one, **fields)


def test_identical_histories_scale_by_one_plus_n_minus_one_times_f():
    energy = np.geomspace(200.0, 2000.0, 7)
    single = crossing_transition_radiation(_film(), energy, _N, _EPS, n_electrons=1)
    m = 5
    same = _bunch(np.zeros(m), np.zeros(m))
    kw = dict(n_electrons=m, coherence="bunch")
    # The sampled histories as the bunch, in phase: N = M, N^2 per bunch.
    point = crossing_transition_radiation(same, energy, _N, _EPS, **kw)
    np.testing.assert_allclose(point, m * single, rtol=1e-12)
    # A physical bunch of 1e4 electrons with an analytic form factor.
    factor = np.exp(-((energy * 0.3 / HBARC_EV_ANG) ** 2))
    bunch = crossing_transition_radiation(
        same, energy, _N, _EPS, physical_electrons=1e4, form_factor=factor, **kw
    )
    np.testing.assert_allclose(bunch, (1 + (1e4 - 1) * factor) * single, rtol=1e-12)
    called = crossing_transition_radiation(
        same, energy, _N, _EPS, physical_electrons=1e4,
        form_factor=lambda e: np.exp(-((e * 0.3 / HBARC_EV_ANG) ** 2)), **kw,
    )  # fmt: skip
    np.testing.assert_array_equal(called, bunch)
    # One physical electron keeps the per-history (self) term alone.
    alone = crossing_transition_radiation(same, energy, _N, _EPS, physical_electrons=1, **kw)
    np.testing.assert_allclose(alone, single, rtol=1e-12)
    history = crossing_transition_radiation(same, energy, _N, _EPS, n_electrons=m)
    np.testing.assert_allclose(history, single, rtol=1e-12)


def test_sampled_offsets_give_the_gaussian_form_factors():
    rng = np.random.default_rng(7)
    m, sigma_t, sigma_x = 4000, 2.0, 30.0
    t0, dx = rng.normal(0.0, sigma_t, m), rng.normal(0.0, sigma_x, m)
    energy = np.array([300.0, 500.0, 800.0])
    single = crossing_transition_radiation(_film(), energy, _N, _EPS, n_electrons=1)
    population = 1e6
    ctr = crossing_transition_radiation(
        _bunch(t0, dx), energy, _N, _EPS, n_electrons=m, coherence="bunch",
        physical_electrons=population,
    )  # fmt: skip
    omega = energy / HBARC_EV_ANG
    # |<exp(i(w t0 - k_x dx))>|^2 for independent Gaussian offsets, k_x = w sin(theta).
    factor = np.exp(-((omega * sigma_t) ** 2) - (omega * _N[0] * sigma_x) ** 2)
    expected = (1 + (population - 1) * factor) * single
    # Three standard deviations of the sampled |<exp(i phase)>|^2 about F.
    sigma = (1.0 - factor) * np.sqrt(2.0 / (m * factor))
    assert np.all(np.abs(ctr / expected - 1.0) < 3.0 * sigma)
    assert np.all((factor > 0.15) & (factor < 0.85))


def test_a_long_bunch_returns_the_incoherent_sum():
    rng = np.random.default_rng(3)
    m = 400
    t0 = rng.normal(0.0, 5e3, m)  # omega sigma_t ~ 10^3: F ~ 0
    energy = np.linspace(300.0, 900.0, 600)
    bunch = _bunch(t0, np.zeros(m))
    history = crossing_transition_radiation(bunch, energy, _N, _EPS, n_electrons=m)
    ctr = crossing_transition_radiation(bunch, energy, _N, _EPS, n_electrons=m, coherence="bunch")
    # N = M sampled electrons with random phases: speckle about the incoherent sum,
    # decorrelated over an energy step ~ hbar c / sigma_t (0.4 eV).
    assert np.mean(ctr / history) == pytest.approx(1.0, abs=4.0 / np.sqrt(energy.size))


def test_bunch_rejections_and_the_signed_estimate():
    same = _bunch(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="coherence='bunch'"):
        crossing_transition_radiation(same, 300.0, _N, _EPS, n_electrons=2, physical_electrons=10)
    with pytest.raises(CoherentSamplingError, match="two incident samples"):
        crossing_transition_radiation(
            _film(), 300.0, _N, _EPS, n_electrons=1, coherence="bunch", physical_electrons=10
        )
    # Two histories in antiphase at 300 eV: P = 0 < G, an unresolved negative excess.
    half = np.pi * HBARC_EV_ANG / 300.0
    opposed = _bunch(np.array([0.0, half]), np.zeros(2))
    with pytest.warns(CoherentSamplingWarning, match="returned unclipped"):
        y = crossing_transition_radiation(
            opposed, 300.0, _N, _EPS, n_electrons=2, coherence="bunch", physical_electrons=10
        )
    assert y < 0.0
