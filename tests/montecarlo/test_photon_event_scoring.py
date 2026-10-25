"""Explicit physical photon sources use the ordinary aperture normalization."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.runner.photon_emission import score_photon_sources
from pyrite.montecarlo.transport.photon_event_sources import (
    emission_sources,
    primary_annihilation_sources,
)
from pyrite.montecarlo.transport.photon_histories import PhotonAperture

APERTURE = PhotonAperture((0, 0, 100), (1, 0, 0), (0, 1, 0), 10, 10)


def _sources():
    return emission_sources(
        dict(
            position_ang=np.zeros((4, 3)),
            direction=np.array([[0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 1]]),
            energy_eV=np.array([2000.0, 2000.0, 8000.0, 500.0]),
            parent=np.array([0, 0, 1, 1]),
        ),
        np.array([1, 3]),
        weight=np.array([2.0, 3.0]),
        photon_cutoff_eV=1000.0,
    )


def _score(sources, **options):
    kwargs = dict(n_primaries=10, seed=17, photon_cutoff_eV=1000.0, electron_cutoff_eV=1000.0)
    kwargs.update(options)
    return score_photon_sources(sources, [], APERTURE, [1000.0, 3000.0, 5000.0], **kwargs)


def test_event_hits_normalize_by_all_incident_electrons_and_keep_local_energy():
    sources = _sources()
    score = _score(sources)
    # One forward 2 keV photon (weight 2); one 8 keV hit lies outside bins.
    np.testing.assert_array_equal(score.total_per_primary_eV, [2 / (10 * 2000), 0])
    np.testing.assert_array_equal(score.direct_per_primary_eV, score.total_per_primary_eV)
    np.testing.assert_array_equal(score.redistributed_per_primary_eV, [0, 0])
    assert score.detected_per_primary == 0.5
    assert score.unbinned_detected_per_primary == 0.3
    assert score.n_proposals == 3 and score.n_primaries == 10
    assert score.sources is sources
    np.testing.assert_array_equal(sources.local_energy_eV, [0.0, 1500.0])
    assert np.sum(score.histories.ledger["incident_eV"]) == 32000.0
    again = _score(sources, n_proposals=100)
    np.testing.assert_array_equal(again.total_per_primary_eV, score.total_per_primary_eV)


def test_recorded_absorbed_annihilation_photons_reach_aperture():
    raw = dict(
        origin=np.zeros((2, 3)),
        direction=np.array([[0, 0, 1], [0, 0, -1]]),
        k_keV=np.array([511.0, 511.0]),
        event=np.array([0, 0]),
        absorbed=np.array([True, True]),
    )
    sources = primary_annihilation_sources(
        dict(pair_production=dict(events=dict(electron_id=np.array([3])), annihilation_photons=raw))
    )
    score = score_photon_sources(
        sources,
        [],
        APERTURE,
        [500000.0, 520000.0],
        n_primaries=10,
        seed=17,
        photon_cutoff_eV=1000.0,
        electron_cutoff_eV=1000.0,
    )
    assert score.detected_per_primary == 0.1
    np.testing.assert_array_equal(score.total_per_primary_eV, [1 / (10 * 20000)])
    np.testing.assert_array_equal(raw["absorbed"], [True, True])


def test_empty_event_batch_has_zero_score():
    sources = _sources()
    empty = replace(
        sources,
        **{
            name: getattr(sources, name)[:0]
            for name in (
                "position_ang",
                "direction",
                "energy_eV",
                "weight",
                "history_id",
                "event_id",
                "source_kind",
                "segment_index",
            )
        },
    )
    score = _score(empty)
    assert score.n_proposals == 0
    assert score.detected_per_primary == 0
    np.testing.assert_array_equal(score.total_per_primary_eV, [0, 0])


@pytest.mark.parametrize(
    "owners",
    [np.array([1, 10, 3]), np.array([-1, 1, 3]), np.array([1.0, 1.0, 3.0]), np.array([[1, 1, 3]])],
)
def test_source_ownership_must_match_denominator(owners):
    with pytest.raises(ValueError, match="history ids"):
        _score(replace(_sources(), history_id=owners))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"photon_cutoff_eV": 500.0}, "cutoffs must match"),
        ({"n_proposals": 2}, "n_proposals"),
        ({"n_proposals": True}, "n_proposals"),
        ({"n_primaries": True}, "n_primaries"),
        ({"electron_cutoff_eV": None}, "electron_cutoff_eV"),
        ({"max_rounds": 2}, "max_rounds"),
    ],
)
def test_invalid_scoring_contract_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        _score(_sources(), **options)


def test_local_only_source_ownership_must_match_denominator():
    with pytest.raises(ValueError, match="history ids"):
        _score(replace(_sources(), local_history_id=np.array([1, 10])))
