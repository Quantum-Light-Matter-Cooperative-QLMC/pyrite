"""Pair positrons transported through the secondary cascade (#276).

Validation: bhabha-close, positron-brems-scaling, sbethe-positron-stopping
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV
from tests.helpers.positron_tables import require_positron_tables

from .test_pair_production import _cascade


@pytest.fixture(autouse=True)
def _require_positron_tables():
    require_positron_tables()


def _positrons(monkeypatch, **kw):
    return _cascade(monkeypatch, pair_scale=1e9, positron_transport=True, **kw)


def test_positrons_are_launched_transported_and_energy_closes(monkeypatch):
    from pyrite.montecarlo.transport.events import EVENT_CUTOFF, check_segment_event_contract
    from pyrite.montecarlo.transport.secondaries import (
        LAUNCH_POSITRON,
        secondary_energy_balance,
    )

    result = _positrons(monkeypatch)
    pair = result["pair_production"]
    events, tracks = pair["events"], result["secondary_tracks"]
    assert pair["positrons_transported"] is True
    launched = events["positron_launched"]
    np.testing.assert_array_equal(launched, events["positron_keV"] > 100.0)
    assert launched.any()
    track = events["positron_track"][launched]
    np.testing.assert_array_equal(events["positron_track"][~launched], -1)
    np.testing.assert_array_equal(tracks["launch_kind"][track], LAUNCH_POSITRON)
    np.testing.assert_array_equal(tracks["launch_E_keV"][track], events["positron_keV"][launched])
    np.testing.assert_array_equal(tracks["parent_id"][track], events["parent_track"][launched])
    np.testing.assert_array_equal(tracks["electron_id"][track], events["electron_id"][launched])
    assert np.count_nonzero(tracks["launch_kind"] == LAUNCH_POSITRON) == launched.sum()
    # Every launched positron track has rows, starting at its launch energy.
    rows = np.isin(result["track_id"], track)
    assert np.unique(result["track_id"][rows]).size == track.size
    check_segment_event_contract(dict(result, electron_id=result["track_id"]))

    terms = secondary_energy_balance(result)
    assert "positron_keV" not in terms and "pair_rest_mass_keV" not in terms
    # Each pair's 2 m_e c^2 leaves with an escaping positron or as annihilation
    # photons, which also carry the kinetic energy of an in-flight annihilation.
    leaving = (
        terms["positron_escaped_rest_keV"]
        + terms["annihilation_escaped_keV"]
        + terms["annihilation_absorbed_keV"]
    )
    expected = events["row"].size * PAIR_THRESHOLD_EV * 1e-3 + np.nansum(events["annihilation_keV"])
    assert leaving == pytest.approx(expected, rel=1e-12)
    stopped = ~launched
    assert terms["positron_subthreshold_keV"] == pytest.approx(
        float(np.sum(events["positron_keV"][stopped]))
    )
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    # A positron's hard loss never overdraws its row-end energy (W_max = E).
    absorbed = rows & (result["event_kind"] == int(EVENT_CUTOFF)) & (result["hard_channel"] >= 0)
    photon = result["hard_radiative_k_eV"] * 1e-3
    assert np.all((result["E_end_keV"] - result["hard_W_keV"] - photon)[absorbed] >= -1e-12)
    per_history = secondary_energy_balance(result, per_history=True)
    np.testing.assert_allclose(per_history["residual_keV"], 0.0, atol=1e-9 * 3000.0)


def test_positron_hard_collisions_reach_beyond_the_electron_limit(monkeypatch):
    """Bhabha W_max = E: a positron can lose more than half its energy in one collision."""
    from pyrite.montecarlo.transport.secondaries import LAUNCH_POSITRON

    result = _positrons(monkeypatch)
    tracks = result["secondary_tracks"]
    positron_tracks = np.flatnonzero(tracks["launch_kind"] == LAUNCH_POSITRON)
    rows = np.isin(result["track_id"], positron_tracks) & (result["hard_channel"] >= 0)
    assert rows.any()
    assert np.all(result["hard_W_keV"][rows] <= result["E_end_keV"][rows] + 1e-9)


def test_positron_mode_is_reproducible_and_off_is_unchanged(monkeypatch):
    first = _positrons(monkeypatch)
    again = _positrons(monkeypatch)
    for name in ("E_end_keV", "r_mid", "track_id", "hard_W_keV"):
        np.testing.assert_array_equal(first[name], again[name])
    with pytest.warns(UserWarning, match="not transported"):
        off = _cascade(monkeypatch, pair_scale=1e9)
    with pytest.warns(UserWarning, match="not transported"):
        explicit = _cascade(monkeypatch, pair_scale=1e9, positron_transport=False)
    for name in ("E_end_keV", "r_mid", "v_hat", "track_id", "parent_id", "hard_W_keV"):
        np.testing.assert_array_equal(off[name], explicit[name])
    # Electron tracks and pair events up to the first positron launch match the mode off.
    n_off = off["secondary_tracks"]["generation"]
    assert first["secondary_tracks"]["generation"].size > n_off.size or np.any(
        first["secondary_tracks"]["launch_kind"] == 3
    )
    gen0 = first["generation"] == 0
    np.testing.assert_array_equal(
        first["E_end_keV"][gen0], off["E_end_keV"][off["generation"] == 0]
    )


def test_positron_transport_requires_pair_mode():
    from pyrite.montecarlo.transport import simulate_trajectories

    with pytest.raises(ValueError, match="requires pair_production_model"):
        simulate_trajectories(50.0, 2, 1.0e4, element="Si", positron_transport=True)
