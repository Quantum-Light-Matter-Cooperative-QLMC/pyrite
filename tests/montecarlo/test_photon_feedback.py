"""Charged vacancy/positron emission return to the CPU photon queue (#341)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.transport.annihilation import (
    ANNIHILATION_AT_REST,
    ANNIHILATION_IN_FLIGHT,
    ELECTRON_REST_KEV,
)
from pyrite.montecarlo.transport.photon_coupling import coupled_ledger_residual_eV
from pyrite.montecarlo.transport.photon_event_sources import (
    emission_sources,
    primary_annihilation_sources,
    primary_relaxation_sources,
)
from pyrite.montecarlo.transport.photon_feedback import (
    FeedbackChargedStage,
    annihilation_emissions,
    impact_relaxation_emissions,
)
from pyrite.montecarlo.transport.photon_histories import ELECTRON, POSITRON
from tests.montecarlo.test_photon_coupling import silicon_stage as silicon_stage

REST = 2 * ELECTRON_REST_KEV * 1e3


def _rows(energy, species=ELECTRON, z=1e7):
    energy = np.atleast_1d(energy).astype(float)
    n = energy.size
    return dict(
        kinetic_eV=energy,
        species=np.full(n, species),
        key=np.arange(n, dtype=np.uint64) + 31,
        position_ang=np.tile([0.0, 0.0, z], (n, 1)),
        direction=np.tile([0.0, 0.0, 1.0], (n, 1)),
    )


def _closed(stage, rows):
    result = stage(rows, 5)
    expected = rows["kinetic_eV"] + np.where(rows["species"] == POSITRON, REST, 0.0)
    np.testing.assert_allclose(
        result.escaped_eV + result.deposited_eV + result.binding_eV + result.radiated_eV,
        expected,
        rtol=2e-12,
        atol=1e-8,
    )
    photons = result.photons
    actual = np.bincount(photons["parent"], weights=photons["energy_eV"], minlength=expected.size)
    np.testing.assert_allclose(actual, result.radiated_eV, rtol=2e-12, atol=1e-8)
    return result


def test_unattenuated_annihilation_sources_close_energy_and_momentum():
    kinetic = np.array([0.0, 1000.0])
    photons = annihilation_emissions(
        [ANNIHILATION_AT_REST, ANNIHILATION_IN_FLIGHT],
        kinetic,
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.zeros((2, 3)),
        [31, 32],
    )
    energies = photons["energy_eV"].reshape(2, 2)
    directions = photons["direction"].reshape(2, 2, 3)
    np.testing.assert_allclose(energies.sum(axis=1), kinetic * 1e3 + REST, rtol=1e-14)
    np.testing.assert_array_equal(energies[0], [REST / 2, REST / 2])
    np.testing.assert_allclose(directions[0, 0], -directions[0, 1], atol=1e-14)
    momentum = np.sum(energies[:, :, None] * directions, axis=1)
    expected = np.sqrt(kinetic * (kinetic + REST * 1e-3)) * 1e3
    np.testing.assert_allclose(
        momentum, np.column_stack([np.zeros(2), np.zeros(2), expected]), atol=1e-8
    )
    assert "absorbed" not in photons and "distance_ang" not in photons


def test_impact_vacancies_emit_photons_and_auger_with_one_binding_debit():
    n = 100
    result = dict(
        hard_channel=np.zeros(n, dtype=int),
        electron_id=np.arange(n),
        track_id=np.arange(n),
        flight_id=np.zeros(n, dtype=int),
        layer=np.zeros(n, dtype=int),
        r_mid=np.zeros((n, 3)),
        v_hat=np.tile([0.0, 0.0, 1.0], (n, 1)),
        L_ang=np.full(n, 2.0),
        inelastic={"channels": [[dict(code=0, atomic_number=29, shell="K", inner=True)]]},
    )
    keys = np.arange(n, dtype=np.uint64)
    particles, released = impact_relaxation_emissions(result, keys, 50.0)
    assert (particles["species"] == -1).any()
    assert (particles["species"] == ELECTRON).any()
    np.testing.assert_allclose(np.linalg.norm(particles["direction"], axis=1), 1, atol=1e-14)
    np.testing.assert_array_equal(particles["position_ang"][:, 2], 1.0)
    actual = np.bincount(particles["parent"], weights=particles["energy_eV"], minlength=n)
    np.testing.assert_allclose(actual, released, rtol=1e-14)
    again, _ = impact_relaxation_emissions(result, keys, 50.0)
    for name in particles:
        np.testing.assert_array_equal(again[name], particles[name])
    result["inelastic"]["channels"][0][0]["inner"] = False
    empty, reserved = impact_relaxation_emissions(result, keys, 50.0)
    assert empty["energy_eV"].size == 0 and not reserved.any()


def test_subthreshold_positrons_annihilate_without_transport():
    stage = FeedbackChargedStage(SimpleNamespace(threshold_eV=5e4), 1e3)
    rows = _rows([0, 1e3, 5e4], POSITRON)
    result = _closed(stage, rows)
    np.testing.assert_array_equal(result.deposited_eV, rows["kinetic_eV"])
    np.testing.assert_allclose(result.rest_energy_eV, REST, rtol=1e-15)
    assert result.photons["energy_eV"].size == 6


def test_photon_queue_releases_pair_rest_energy_into_annihilation():
    from pyrite.montecarlo.transport.photon_coupling import track_coupled_histories
    from tests.montecarlo.test_photon_coupling import _beam

    # Large threshold isolates the at-rest handoff and needs no charged tables.
    stage = FeedbackChargedStage(SimpleNamespace(threshold_eV=1e8), 1e3)
    out = track_coupled_histories(
        *_beam(100, 4e6),
        [(0.0, 1e8, (("Cu", 0.0849),))],
        seed=7,
        photon_cutoff_eV=1e3,
        charged_stage=stage,
    )
    assert (out.charged["species"] == POSITRON).any()
    np.testing.assert_allclose(coupled_ledger_residual_eV(out.ledger), 0, atol=1e-7)
    np.testing.assert_allclose(out.ledger["positron_rest_eV"], 0, atol=1e-8)
    assert not out.ledger["positron_pending_eV"].any()
    assert out.rounds > 0


@pytest.mark.parametrize("species", [ELECTRON, POSITRON])
def test_real_charged_feedback_closes(silicon_stage, species):
    _, base = silicon_stage
    rows = _rows([3e5], species)
    stage = FeedbackChargedStage(base, 1e3)
    result = _closed(stage, rows)
    assert result.deposited_eV[0] > 0
    repeated = stage(rows, 5)
    for name in result.photons:
        np.testing.assert_array_equal(result.photons[name], repeated.photons[name])


def test_escaping_positron_keeps_rest_reservoir(silicon_stage):
    layers, base = silicon_stage
    result = _closed(FeedbackChargedStage(base, 1e3), _rows([3e5], POSITRON, layers[-1][1] - 1.0))
    assert result.escaped_eV[0] >= REST


def test_event_sources_keep_root_history_weight_and_disjoint_event_ids():
    photons = annihilation_emissions(
        [ANNIHILATION_AT_REST, ANNIHILATION_AT_REST],
        [0, 0],
        np.zeros((2, 3)),
        np.array([[0, 0, 1], [0, 0, 2]]),
        [31, 32],
    )
    batch = emission_sources(photons, [7, 7], weight=[2, 3], source_kind=2, event_offset=100)
    np.testing.assert_array_equal(batch.history_id, [7, 7, 7, 7])
    np.testing.assert_array_equal(batch.weight, [2, 2, 3, 3])
    np.testing.assert_array_equal(batch.event_id, [100, 101, 102, 103])
    np.testing.assert_array_equal(batch.source_kind, 2)
    assert len(batch.track_args) == 6


def test_event_source_cutoff_retains_weighted_local_energy():
    photons = dict(
        parent=np.array([0, 0]),
        energy_eV=np.array([500.0, 1500.0]),
        position_ang=np.zeros((2, 3)),
        direction=np.tile([0, 0, 1.0], (2, 1)),
    )
    batch = emission_sources(photons, [7], weight=[3.0], photon_cutoff_eV=1000.0)
    np.testing.assert_array_equal(batch.energy_eV, [1500.0])
    np.testing.assert_array_equal(batch.weight, [3.0])
    np.testing.assert_array_equal(batch.local_history_id, [7])
    np.testing.assert_array_equal(batch.local_energy_eV, [1500.0])


def test_primary_annihilation_adapter_keeps_originally_absorbed_photons():
    raw = dict(
        origin=np.zeros((4, 3)),
        direction=np.tile([0, 0, 1.0], (4, 1)),
        k_keV=np.full(4, ELECTRON_REST_KEV),
        event=np.array([0, 0, 1, 1]),
        absorbed=np.array([True, True, False, False]),
    )
    result = {
        "pair_production": {
            "events": {"electron_id": np.array([3, 3])},
            "annihilation_photons": raw,
        }
    }
    sources = primary_annihilation_sources(result, weight=[1, 1, 1, 2], event_offset=8)
    assert sources.energy_eV.size == 4
    np.testing.assert_array_equal(sources.history_id, 3)
    np.testing.assert_array_equal(sources.weight, 2)
    np.testing.assert_array_equal(sources.event_id, [8, 9, 10, 11])


def test_primary_relaxation_source_budget_closes_without_primary_debit(silicon_stage):
    from pyrite.montecarlo.transport.photon_feedback import _launch_cascade

    _, base = silicon_stage
    stage = FeedbackChargedStage(base, 1e3)
    rows = _rows([3e5])
    transport, _ = _launch_cascade(base, rows, 5)
    initial = transport["E_start_keV"].copy()
    sources = primary_relaxation_sources(transport, rows["key"], stage, seed=9)
    photons = sources.photons
    emitted = np.bincount(photons["parent"], weights=photons["energy_eV"], minlength=1)
    np.testing.assert_allclose(
        sources.escaped_eV + sources.deposited_eV + sources.binding_eV + emitted,
        sources.reserved_eV,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(transport["E_start_keV"], initial)
    assert sources.reserved_eV[0] > 0


@pytest.mark.parametrize(
    "kind,energy", [(-1, 0), (ANNIHILATION_AT_REST, 1), (ANNIHILATION_IN_FLIGHT, -1)]
)
def test_invalid_annihilation_sources_rejected(kind, energy):
    with pytest.raises(ValueError):
        annihilation_emissions([kind], [energy], [[0, 0, 1]], [[0, 0, 0]], [1])


def test_auger_electrons_reenter_charged_transport_and_caps_fail_closed(monkeypatch):
    from pyrite.montecarlo.transport import photon_feedback as module
    from pyrite.montecarlo.transport import secondaries

    calls = []
    empty = dict(
        position_ang=np.empty((0, 3)),
        direction=np.empty((0, 3)),
        energy_eV=np.empty(0),
        parent=np.empty(0, dtype=int),
        ordinal=np.empty(0, dtype=int),
    )

    def launch(base, rows, seed, **kw):
        calls.append((rows["kinetic_eV"].copy(), rows["key"].copy(), seed))
        return {"first": rows["kinetic_eV"][0] == 10000.0}, {}

    def balance(result, **kw):
        first = result["first"]
        return dict(
            escaped_keV=np.zeros(1),
            deposited_keV=np.array([6.5 if first else 2.5]),
            binding_reserved_keV=np.array([3.5 if first else 0]),
            radiated_keV=np.zeros(1),
        )

    def relax(result, keys, cutoff):
        if not result["first"]:
            return dict(
                position_ang=np.empty((0, 3)),
                direction=np.empty((0, 3)),
                energy_eV=np.empty(0),
                parent=np.empty(0, dtype=int),
                species=np.empty(0, dtype=int),
                key=np.empty(0, dtype=np.uint64),
            ), np.zeros(1)
        return dict(
            position_ang=np.zeros((2, 3)),
            direction=np.tile([0, 0, 1.0], (2, 1)),
            energy_eV=np.array([1000.0, 2500.0]),
            parent=np.zeros(2, dtype=int),
            species=np.array([-1, ELECTRON]),
            key=np.array([500, 501], dtype=np.uint64),
        ), np.array([3500.0])

    monkeypatch.setattr(module, "_launch_cascade", launch)
    monkeypatch.setattr(secondaries, "secondary_energy_balance", balance)
    monkeypatch.setattr(module, "_hard_photons", lambda result: empty)
    monkeypatch.setattr(module, "impact_relaxation_emissions", relax)
    base = SimpleNamespace(threshold_eV=500.0, transport={})
    result = _closed(FeedbackChargedStage(base, 50.0), _rows([10000.0]))
    np.testing.assert_array_equal(result.deposited_eV, [9000.0])
    np.testing.assert_array_equal(result.radiated_eV, [1000.0])
    np.testing.assert_array_equal(calls[1][0], [2500.0])
    np.testing.assert_array_equal(calls[1][1], [501])
    assert calls[0][2] != calls[1][2]
    with pytest.raises(RuntimeError, match="max_relaxation_rounds"):
        FeedbackChargedStage(base, 50.0, max_relaxation_rounds=1)(_rows([10000.0]), 5)


def test_complete_feedback_refuses_filter_boxes_before_tracking():
    from pyrite.montecarlo.transport.photon_coupling import track_coupled_histories
    from tests.montecarlo.test_photon_coupling import _beam

    stage = FeedbackChargedStage(SimpleNamespace(threshold_eV=1e8), 1e3)
    with pytest.raises(ValueError, match="filter boxes"):
        track_coupled_histories(
            *_beam(1, 4e6), [], seed=7, photon_cutoff_eV=1e3, charged_stage=stage, boxes=[object()]
        )


@pytest.mark.parametrize(
    "layers",
    [[], [(1.0, 2.0, (("Si", 0.05),))], [(0.0, 1.0, (("Si", 0.05),)), (2.0, 3.0, (("Si", 0.05),))]],
)
def test_feedback_factory_refuses_untransportable_gaps(layers):
    from pyrite.montecarlo.transport.photon_feedback import cascade_feedback_stage

    with pytest.raises(ValueError, match="contiguous"):
        cascade_feedback_stage(layers, photon_cutoff_eV=1e3)


def test_unknown_charged_species_cannot_disappear_from_energy_account():
    rows = _rows([1000.0])
    rows["species"][0] = 99
    with pytest.raises(ValueError, match="species"):
        FeedbackChargedStage(SimpleNamespace(threshold_eV=500.0), 50.0)(rows, 5)
