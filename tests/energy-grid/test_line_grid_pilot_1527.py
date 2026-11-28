"""Pilot-guided bandwidth certification and exact fallback (#1527).

Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite.montecarlo.runner._pilot_tail import (
    certify_tail,
    pilot_block,
    pilot_candidates,
    project_summary,
    resolve_pilot_tail,
)
from pyrite.montecarlo.spectrum import _lattice_tail as lt
from pyrite.montecarlo.spectrum.cxr_seeds import _lattice_stop


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ceiling", [54000.0, 54321.5])
@pytest.mark.parametrize("shift", [0.0, 15000.0])
def test_pilot_success_and_miss_match_full(dtype, ceiling, shift, monkeypatch):
    monkeypatch.setattr(lt, "REAL", dtype)
    full = lt.stop_lattice(ceiling)
    limit = 5e-5
    energy = np.concatenate((np.full(20, 1000.0), np.full(100, 1000.0 + shift)))
    width = np.full(energy.size, 0.5, dtype=dtype)
    weight = np.ones(energy.size, dtype=dtype)
    energy = energy.astype(dtype)

    def collect(stops, selection=slice(None)):
        return lt.summarize_lines(energy[selection], width[selection], weight[selection], stops)

    pilot = collect(full, slice(0, 20))
    stops = pilot_candidates(ceiling, limit, pilot)
    summary = lt.merge_tail(project_summary(pilot, full, stops), collect(stops, slice(20, None)))
    calls = []

    def replay(stops):
        calls.append(len(stops))
        return collect(stops)

    tail, certified = resolve_pilot_tail(ceiling, limit, stops, summary, replay)
    expected = lt.LatticeTail(full, collect(full))
    actual, record = _lattice_stop(tail, ceiling, limit, 100.0)
    wanted, reference = _lattice_stop(expected, ceiling, limit, 100.0)
    assert actual == wanted
    assert record == pytest.approx(reference)
    assert certified is (shift == 0)
    assert bool(calls) is (shift != 0)
    assert summary.raw_lines == energy.size


def test_distant_passing_ceiling_is_not_an_adjacent_bracket():
    stops = np.array([100.0, 200.0, 1000.5])
    summary = lt.summarize_lines(np.array([550.0]), np.array([0.1]), np.ones(1), stops)
    assert certify_tail(1000.5, 0.01, stops, summary) is None


def test_non_lattice_ceiling_adjacent_to_predecessor():
    ceiling = 1000.5
    stops = np.array([900.0, 1000.0, ceiling])
    summary = lt.summarize_lines(np.array([1000.0]), np.array([0.01]), np.ones(1), stops)
    tail = certify_tail(ceiling, 0.01, stops, summary)
    assert tail is not None
    assert _lattice_stop(tail, ceiling, 0.01, 100.0)[0] == ceiling


def test_empty_and_capped_populations_need_no_fallback():
    stops = np.array([100.0, 1000.5])
    assert certify_tail(1000.5, 0.01, stops, None).n_lines == 0
    summary = lt.summarize_lines(np.array([1000.0]), np.array([100.0]), np.ones(1), stops)
    tail = certify_tail(1000.5, 0.01, stops, summary)
    assert tail is not None
    assert _lattice_stop(tail, 1000.5, 0.01, 100.0)[0] == 1000.5


def test_pilot_block_partitions_rows_once_and_preserves_electrons():
    # Deliberately split near the middle of an electron's three rows.
    ids = np.repeat(np.arange(10), 3)
    segments = {"elec_id": ids, "L_ang": np.ones(ids.size), "Ne": 10}
    seen = []
    sizes = []

    def collect(part, stops):
        assert part["Ne"] == 10
        seen.extend(part["elec_id"].tolist())
        sizes.append(len(stops))
        assert np.all(np.unique(part["elec_id"], return_counts=True)[1] == 3)
        n = len(part["elec_id"])
        return lt.summarize_lines(np.full(n, 1000.0), np.full(n, 0.1), np.ones(n), stops)

    stops, summary, pilot_rows = pilot_block(segments, 54000.0, 0.001, collect, pilot_rows=8)
    assert pilot_rows == 6
    assert seen == ids.tolist()
    assert sizes == [541, len(stops)]
    assert len(stops) <= 22
    assert summary.raw_lines == ids.size
