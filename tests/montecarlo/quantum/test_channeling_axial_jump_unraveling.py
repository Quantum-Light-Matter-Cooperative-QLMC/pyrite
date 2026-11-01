"""Unraveling vs the independent dense absorbing-model oracle (finite algebra only)."""

import numpy as np
import pytest
from scipy import stats

from pyrite.montecarlo.channeling.axial_jump_instrument import absorbing_axial_model
from pyrite.montecarlo.channeling.axial_jump_unraveling import (
    event_tallies,
    unravel_absorbing_outgoing,
)
from pyrite.montecarlo.channeling.axial_phase_space import density_phase_space_weights

WINDOW = np.array([[0.8, 0.0], [0.6j, 0.0]])


def _inputs():
    # Recycling strong enough (~4 per incident) that dropping retained-jump
    # phases shifts the outgoing tally by ~8 sigma.
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    psi = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi /= np.linalg.norm(psi)
    mixed = 0.7 * np.outer(psi, psi.conj()) + 0.2 * np.diag([0.0, 1.0, 0.0])
    return dict(
        hamiltonian_eV=0.5 * (x + x.conj().T),
        initial_density=mixed,
        retained_jumps_per_sqrt_ang=(
            0.625 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))),
        ),
        outgoing_jumps_per_sqrt_ang=tuple(
            0.2 * (rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))) for _ in range(2)
        ),
        unresolved_effect_inv_ang=np.diag([0.05, 0.0, 0.1]).astype(complex),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=2.0,
    )


@pytest.fixture(scope="module")
def events():
    return unravel_absorbing_outgoing(
        **_inputs(),
        window=WINDOW,
        max_depth_ang=12.0,
        incident_ids=range(3000),
        seed=4,
        search_step_ang=2.0,
    )


def test_tallies_match_dense_oracle_within_binomial_error(events):
    model = absorbing_axial_model(**_inputs())
    depths = np.array([1.0, 3.0, 6.0, 12.0])
    sampled = event_tallies(events, depths)
    weight = model.initial_weight
    for i, z in enumerate(depths):
        snapshot = model.at_depth(z)
        expected = {
            "outgoing": np.trace(snapshot.outgoing_density).real,
            "unresolved": snapshot.unresolved_weight,
            "retained": np.trace(snapshot.retained_density).real,
        }
        for kind, value in expected.items():
            p = value / weight
            sigma = weight * np.sqrt(max(p * (1 - p), 1e-4) / len(events))
            assert abs(sampled[kind][i] - value) < 4.5 * sigma, (kind, z)


def test_joint_outgoing_outcomes_follow_integrated_weyl_weights(events):
    final = absorbing_axial_model(**_inputs()).at_depth(12.0).outgoing_density
    weights = density_phase_space_weights(final, WINDOW)
    probabilities = (weights / weights.sum()).ravel()
    counts = np.zeros(probabilities.size)
    for e in events:
        if e.kind == "outgoing":
            counts[np.ravel_multi_index((*e.position_index, *e.momentum_index), weights.shape)] += 1
    live = probabilities > 1e-12
    assert counts[~live].sum() == 0
    expected = probabilities[live] * counts.sum()
    assert stats.chisquare(counts[live], expected).pvalue > 1e-3


def test_retained_recycling_occurs_and_depths_are_not_step_quantized(events):
    assert any(e.retained_jumps > 0 for e in events)
    depths = np.array([e.depth_ang for e in events if e.kind != "retained"])
    assert np.unique(np.round(depths % 2.0, 6)).size > 100


def test_zero_jumps_leave_every_incident_retained():
    events = unravel_absorbing_outgoing(
        np.diag([0.0, 1.0]),
        np.eye(2) / 2,
        outgoing_shape=(1, 1),
        hbar_v_eV_ang=1.0,
        window=np.ones((1, 1)),
        max_depth_ang=50.0,
        incident_ids=range(5),
    )
    assert {e.kind for e in events} == {"retained"}
    assert all(e.depth_ang == 50.0 and e.weight == pytest.approx(1.0) for e in events)


def test_scalar_outgoing_depths_follow_the_exponential_law():
    events = unravel_absorbing_outgoing(
        np.array([[2.0]]),
        np.array([[1.0]]),
        outgoing_jumps_per_sqrt_ang=(np.array([[0.5]]),),
        outgoing_shape=(1, 1),
        hbar_v_eV_ang=3.0,
        window=np.ones((1, 1)),
        max_depth_ang=1e4,
        incident_ids=range(800),
        seed=2,
    )
    depths = [e.depth_ang for e in events]
    assert all(e.kind == "outgoing" for e in events)
    assert stats.kstest(depths, "expon", args=(0, 1 / 0.25)).pvalue > 1e-3


def test_events_depend_on_incident_identity_not_processing_order():
    kwargs = dict(**_inputs(), window=WINDOW, max_depth_ang=12.0, seed=9)
    forward = unravel_absorbing_outgoing(**kwargs, incident_ids=[0, 1, 2, 7])
    reverse = unravel_absorbing_outgoing(**kwargs, incident_ids=[7, 2, 1, 0])
    assert forward == sorted(reverse, key=lambda e: [0, 1, 2, 7].index(e.incident_id))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"initial_density": np.zeros((3, 3))}, "trace"),
        ({"hbar_v_eV_ang": 0.0}, "hbar_v"),
        ({"outgoing_shape": (2, 3)}, "outgoing jump"),
        ({"unresolved_effect_inv_ang": -np.eye(3)}, "positive"),
    ],
)
def test_invalid_models_refuse(change, message):
    kwargs = {**_inputs(), **change}
    with pytest.raises(ValueError, match=message):
        unravel_absorbing_outgoing(**kwargs, window=WINDOW, max_depth_ang=1.0, incident_ids=[0])


def test_expm_fallback_reproduces_the_eigen_propagation(monkeypatch):
    import pyrite.montecarlo.channeling.axial_jump_unraveling as unraveling

    kwargs = dict(**_inputs(), window=WINDOW, max_depth_ang=12.0, incident_ids=range(40))
    eigen = unravel_absorbing_outgoing(**kwargs)
    monkeypatch.setattr(unraveling, "_MAX_CONDITION", 0.0)
    fallback = unravel_absorbing_outgoing(**kwargs)
    assert [(e.kind, e.retained_jumps, e.position_index) for e in eigen] == [
        (e.kind, e.retained_jumps, e.position_index) for e in fallback
    ]
    np.testing.assert_allclose(
        [e.depth_ang for e in eigen], [e.depth_ang for e in fallback], atol=1e-5
    )


def test_tallies_refuse_depths_beyond_the_unraveling_horizon(events):
    with pytest.raises(ValueError, match="max_depth"):
        event_tallies(events, [13.0])
