"""Post-transport hard-photon scoring uses the conditional BremsLib DDCS.

Validation: bremslib-radiative-event-spectrum
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import brem_events
from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    prepare_bremslib_table,
    stage_bremslib_table,
)
from pyrite.montecarlo.transport.events import (
    EVENT_CUTOFF,
    EVENT_ELASTIC,
    EVENT_HARD_RADIATIVE,
    check_segment_event_contract,
)
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture
def table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)


def _segments():
    return {
        "event_kind": np.array([EVENT_HARD_RADIATIVE, EVENT_ELASTIC]),
        "hard_radiative_k_eV": np.array([20_000.0, 0.0]),
        "hard_radiative_Z": np.array([6, 0], dtype=np.int16),
        "E_end_keV": np.array([50.0, 30.0]),
        "r_mid": np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 60.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "L_ang": np.array([10.0, 10.0]),
        "electron_id": np.array([0, 0]),
        "Ne": 1,
        "thickness_ang": 100.0,
    }


def test_hard_event_spectrum_uses_conditional_angle_without_reweighting_rate(monkeypatch, table):
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", lambda _comp, energy: np.zeros_like(energy)
    )
    grid = np.array([10_000.0, 20_000.0, 30_000.0])
    got = brem_events.mc_hard_brem_event_spectrum(
        _segments(),
        grid,
        cutoff_eV=15_000.0,
        bremslib_tables={"C": table},
        element="C",
        n_atoms_per_ang3=0.1,
        n_hat=[0.0, 0.0, 1.0],
    )
    staged = stage_bremslib_table(table)
    sdcs = evaluate_bremslib(staged, bremslib_segment_state(staged, [50.0]), [20_000.0])[0, 0]
    ddcs = evaluate_bremslib(staged, bremslib_segment_state(staged, [50.0], [1.0]), [20_000.0])[
        0, 0
    ]
    np.testing.assert_allclose(got, [0.0, ddcs / sdcs / 10_000.0, 0.0])


def test_soft_track_and_hard_event_bins_do_not_overlap(monkeypatch, table):
    monkeypatch.setattr(
        brem_events, "mc_brem_spectrum", lambda *_args, **_kwargs: np.array([1.0, 2.0, 3.0])
    )
    got = brem_events.mc_soft_brem_spectrum(
        {}, [10_000.0, 20_000.0, 30_000.0], cutoff_eV=15_000.0, bremslib_tables={"C": table}
    )
    np.testing.assert_array_equal(got, [1.0, 0.0, 0.0])


def test_hard_event_requires_matching_bremslib_coverage(monkeypatch, table):
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", lambda _comp, energy: np.zeros_like(energy)
    )
    segments = _segments()
    segments["hard_radiative_Z"][0] = 14
    with pytest.raises(ValueError, match="Z=14"):
        brem_events.mc_hard_brem_event_spectrum(
            segments,
            [10_000.0, 20_000.0, 30_000.0],
            cutoff_eV=15_000.0,
            bremslib_tables={"C": table},
            element="C",
            n_atoms_per_ang3=0.1,
        )
    with pytest.raises(ValueError, match="bin edge"):
        brem_events.mc_soft_brem_spectrum(
            {},
            [10_000.0, 20_000.0, 30_000.0],
            cutoff_eV=16_000.0,
            bremslib_tables={"C": table},
        )


def test_radiative_event_payload_closes_flight_and_matches_energy_jump():
    segments = _segments()
    segments.update(
        event_kind=np.array([EVENT_HARD_RADIATIVE, EVENT_CUTOFF]),
        E_start_keV=np.array([50.0, 29.9]),
        E_end_keV=np.array([49.9, 29.8]),
        t_start_ang=np.array([0.0, 10.0]),
        t_end_ang=np.array([10.0, 20.0]),
        flight_id=np.array([0, 1]),
        substep_id=np.array([0, 0]),
    )
    check_segment_event_contract(segments)
    segments["hard_radiative_k_eV"][0] = 19_000.0
    with pytest.raises(ValueError, match="photon energy disagrees"):
        check_segment_event_contract(segments)


def test_terminal_cutoff_photon_is_still_scored(monkeypatch, table):
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", lambda _comp, energy: np.zeros_like(energy)
    )
    regular = _segments()
    terminal = _segments()
    terminal["event_kind"][0] = EVENT_CUTOFF
    kwargs = dict(
        cutoff_eV=15_000.0,
        bremslib_tables={"C": table},
        element="C",
        n_atoms_per_ang3=0.1,
    )
    expected = brem_events.mc_hard_brem_event_spectrum(
        regular, [10_000.0, 20_000.0, 30_000.0], **kwargs
    )
    got = brem_events.mc_hard_brem_event_spectrum(
        terminal, [10_000.0, 20_000.0, 30_000.0], **kwargs
    )
    np.testing.assert_array_equal(got, expected)
