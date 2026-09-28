"""Post-transport hard-photon scoring uses the conditional BremsLib DDCS.

Validation: bremslib-radiative-event-spectrum
"""

import numpy as np
import pytest

from pyrite._grid_semantics import node_bin_edges_and_widths
from pyrite.montecarlo.spectrum import brem_events
from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    evaluate_bremslib_rows,
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


def test_cutoff_inside_a_bin_splits_it_between_soft_width_and_hard_events(monkeypatch, table):
    # Production brem grids put k_c = 1 keV on a node, mid-bin.
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", lambda _comp, energy: np.zeros_like(energy)
    )
    monkeypatch.setattr(
        brem_events, "mc_brem_spectrum", lambda *_args, **_kwargs: np.array([1.0, 2.0, 3.0])
    )
    grid = [10_000.0, 20_000.0, 30_000.0]  # edges 5, 15, 25, 35 keV
    soft = brem_events.mc_soft_brem_spectrum(
        {}, grid, cutoff_eV=20_000.0, bremslib_tables={"C": table}
    )
    np.testing.assert_allclose(soft, [1.0, 1.0, 0.0])
    on_edge = brem_events.mc_soft_brem_spectrum(
        {}, grid, cutoff_eV=np.nextafter(15_000.0, 16_000.0), bremslib_tables={"C": table}
    )
    np.testing.assert_allclose(on_edge, [1.0, 0.0, 0.0], atol=1e-12)
    first_bin = brem_events.mc_soft_brem_spectrum(
        {}, grid, cutoff_eV=10_000.0, bremslib_tables={"C": table}
    )
    np.testing.assert_allclose(first_bin, [0.5, 0.0, 0.0])

    segments = _segments()  # one hard photon at 20 keV, in the straddling bin
    hard = brem_events.mc_hard_brem_event_spectrum(
        segments,
        grid,
        cutoff_eV=20_000.0,
        bremslib_tables={"C": table},
        element="C",
        n_atoms_per_ang3=0.1,
        n_hat=[0.0, 0.0, 1.0],
    )
    assert hard[0] == 0.0 and hard[1] > 0.0 and hard[2] == 0.0
    segments["hard_radiative_k_eV"][0] = 19_000.0
    with pytest.raises(ValueError, match="cutoff"):
        brem_events.mc_hard_brem_event_spectrum(
            segments,
            grid,
            cutoff_eV=20_000.0,
            bremslib_tables={"C": table},
            element="C",
            n_atoms_per_ang3=0.1,
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


def _mu_by_composition(values):
    def mu(comp, energy):
        return np.full(np.shape(energy), values[comp[0][0]])

    return mu


def _score(segments, table, **kwargs):
    return brem_events.mc_hard_brem_event_spectrum(
        segments,
        [10_000.0, 20_000.0, 30_000.0],
        cutoff_eV=15_000.0,
        bremslib_tables={"C": table},
        n_hat=[0.0, 0.0, 1.0],
        **kwargs,
    )


def test_hard_event_escape_reduces_to_slab_for_one_layer_and_wide_footprint(monkeypatch, table):
    comp = [("C", 0.1)]
    monkeypatch.setattr(brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.0}))
    unattenuated = _score(_segments(), table, composition=comp)
    monkeypatch.setattr(brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.01}))
    slab = _score(_segments(), table, composition=comp)
    # Endpoint z = 55 Ang, forward exit through z = 100: exp(-0.01 * 45).
    np.testing.assert_allclose(slab, unattenuated * np.exp(-0.45))
    np.testing.assert_allclose(
        _score(_segments(), table, layers=[(0.0, 100.0, comp)]), slab, rtol=1e-12
    )
    wide = _segments()
    wide.update(crystal_width_ang=1e9, crystal_height_ang=1e9)
    np.testing.assert_allclose(_score(wide, table, composition=comp), slab, rtol=1e-12)


def test_hard_event_escape_sums_optical_depth_over_crossed_layers(monkeypatch, table):
    layers = [(0.0, 80.0, [("C", 0.1)]), (80.0, 100.0, [("Si", 0.05)])]
    layered = _segments()
    layered["n_layers"] = 2
    monkeypatch.setattr(brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.0, "Si": 0.0}))
    reference = _score(layered, table, layers=layers)
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.01, "Si": 0.03})
    )
    got = _score(layered, table, layers=layers)
    # z = 55: 25 Ang of C, then 20 Ang of Si.
    np.testing.assert_allclose(got, reference * np.exp(-(0.01 * 25.0 + 0.03 * 20.0)))
    with pytest.raises(ValueError, match="absorbing layers"):
        _score(layered, table, composition=[("C", 0.1)])


def test_hard_event_escape_uses_nearest_footprint_face(monkeypatch, table):
    monkeypatch.setattr(brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.01}))
    comp = [("C", 0.1)]
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    kwargs = dict(composition=comp, n_hat=direction)
    slab = brem_events.mc_hard_brem_event_spectrum(
        _segments(),
        [10_000.0, 20_000.0, 30_000.0],
        cutoff_eV=15_000.0,
        bremslib_tables={"C": table},
        **kwargs,
    )
    narrow = _segments()
    narrow.update(crystal_width_ang=20.0, crystal_height_ang=20.0)
    got = brem_events.mc_hard_brem_event_spectrum(
        narrow,
        [10_000.0, 20_000.0, 30_000.0],
        cutoff_eV=15_000.0,
        bremslib_tables={"C": table},
        **kwargs,
    )
    # From x = 0 the +x face at 10 Ang is 10*sqrt(2) away, before z = 100.
    ratio = np.exp(-0.01 * 10.0 * np.sqrt(2.0)) / np.exp(-0.01 * 45.0 * np.sqrt(2.0))
    np.testing.assert_allclose(got, slab * ratio)


def test_cutoff_below_grid_scores_every_bin_as_hard(monkeypatch, table):
    monkeypatch.setattr(
        brem_events, "_mu_total_inv_ang", lambda _comp, energy: np.zeros_like(energy)
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("no soft bin lies on the grid")

    monkeypatch.setattr(brem_events, "mc_brem_spectrum", unexpected)
    grid = [30_000.0, 40_000.0, 50_000.0]
    segments = _segments()
    segments["hard_radiative_k_eV"][0] = 30_000.0
    segments["E_end_keV"][0] = 60.0
    np.testing.assert_array_equal(
        brem_events.mc_soft_brem_spectrum(
            {}, grid, cutoff_eV=1_000.0, bremslib_tables={"C": table}
        ),
        0.0,
    )
    got = brem_events.mc_hard_brem_event_spectrum(
        segments,
        grid,
        cutoff_eV=1_000.0,
        bremslib_tables={"C": table},
        element="C",
        n_atoms_per_ang3=0.1,
        n_hat=[0.0, 0.0, 1.0],
    )
    assert got[0] > 0.0 and np.all(got[1:] == 0.0)
    above = brem_events._cutoff_edge(grid, 60_000.0)[2]
    assert above == 3


def _random_events(n, seed=3):
    rng = np.random.default_rng(seed)
    energy_keV = rng.uniform(20.0, 150.0, n)
    flight = rng.normal(size=(n, 3))
    flight /= np.linalg.norm(flight, axis=1)[:, None]
    return energy_keV, rng.uniform(15_000.0, 0.95 * energy_keV * 1e3), flight


def test_row_evaluation_is_the_diagonal_of_the_grid_evaluation(table):
    staged = stage_bremslib_table(table)
    energy_keV, photon_eV, flight = _random_events(40)
    for cos_theta in (None, flight[:, 2]):
        state = bremslib_segment_state(staged, energy_keV, cos_theta)
        np.testing.assert_allclose(
            evaluate_bremslib_rows(staged, state, photon_eV),
            np.diag(evaluate_bremslib(staged, state, photon_eV)),
            rtol=1e-12,
            atol=0.0,
        )


def test_vectorized_event_weights_match_per_event_evaluation(monkeypatch, table):
    """The per-element vector pass reproduces the per-event DDCS/SDCS loop."""
    monkeypatch.setattr(brem_events, "_mu_total_inv_ang", _mu_by_composition({"C": 0.0}))
    silicon = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    tables = {"C": table, "Si": silicon}
    n = 60
    energy_keV, photon_eV, flight = _random_events(n)
    element = np.where(np.arange(n) % 3 == 0, 14, 6)
    segments = {
        "event_kind": np.full(n, EVENT_HARD_RADIATIVE),
        "hard_radiative_k_eV": photon_eV,
        "hard_radiative_Z": element.astype(np.int16),
        "E_end_keV": energy_keV,
        "r_mid": np.tile([0.0, 0.0, 50.0], (n, 1)),
        "v_hat": flight,
        "L_ang": np.zeros(n),
        "Ne": n,
        "thickness_ang": 100.0,
    }
    grid = np.arange(10_000.0, 150_001.0, 5_000.0)
    got = brem_events.mc_hard_brem_event_spectrum(
        segments,
        grid,
        cutoff_eV=1_000.0,
        bremslib_tables=tables,
        composition=[("C", 0.1)],
        n_hat=[0.0, 0.0, 1.0],
    )
    staged = {6: stage_bremslib_table(table), 14: stage_bremslib_table(silicon)}
    weights = []
    for Z, T, k, v in zip(element, energy_keV, photon_eV, flight, strict=True):
        s = staged[int(Z)]
        sdcs = evaluate_bremslib(s, bremslib_segment_state(s, [T]), [k])[0, 0]
        ddcs = evaluate_bremslib(s, bremslib_segment_state(s, [T], [v[2]]), [k])[0, 0]
        weights.append(ddcs / sdcs)
    edges, widths = node_bin_edges_and_widths(grid)
    expected = np.histogram(photon_eV, bins=edges, weights=weights)[0] / (n * widths)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
