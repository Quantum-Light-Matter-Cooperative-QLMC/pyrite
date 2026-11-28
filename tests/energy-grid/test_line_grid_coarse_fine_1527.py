"""Exact sparse-stop search and deterministic replay (#1527).

Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import _lattice_tail as lt
from pyrite.montecarlo.spectrum._lattice_search import coarse_stops, refine_tail
from pyrite.montecarlo.spectrum.cxr_seeds import _lattice_stop


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ceiling", [54000.0, 54321.5])
@pytest.mark.parametrize("limit", [1e-8, 0.001, 0.05, 0.9])
def test_coarse_fine_matches_all_stops(dtype, ceiling, limit, monkeypatch):
    monkeypatch.setattr(lt, "REAL", dtype)
    rng = np.random.default_rng(1527)
    energy = rng.uniform(100, 18000, 500).astype(dtype)
    width = rng.uniform(0.1, 15, 500).astype(dtype)
    weight = rng.lognormal(0, 1, 500).astype(dtype)
    calls = []

    def collect(stops):
        calls.append(len(stops))
        return lt.summarize_lines(energy, width, weight, stops, 1 / 3)

    full = lt.stop_lattice(ceiling)
    reference = lt.LatticeTail(full, collect(full))
    calls.clear()
    tail = refine_tail(ceiling, limit, collect)
    expected, record = _lattice_stop(reference, ceiling, limit, 100.0)
    actual, measured = _lattice_stop(tail, ceiling, limit, 100.0)
    assert actual == expected
    assert measured == pytest.approx(record)
    assert len(calls) <= 2
    assert sum(calls) < 60


def test_empty_population_needs_no_replay():
    calls = []

    def collect(stops):
        calls.append(stops)
        return None

    tail = refine_tail(54321.5, 0.01, collect)
    assert tail.n_lines == 0 and len(calls) == 1


def test_coarse_lattice_keeps_exact_ceiling():
    stops = coarse_stops(54321.5)
    assert stops[0] == 0 and stops[-1] == 54321.5
    assert np.all(stops[:-1] % 100 == 0)
    assert np.all(np.diff(stops) > 0)


def test_replay_population_change_is_refused():
    calls = []

    def collect(stops):
        calls.append(stops)
        n = len(calls)
        return lt.summarize_lines(np.full(n, 1000.0), np.ones(n), np.ones(n), stops)

    with pytest.raises(RuntimeError, match="changed the production-line population"):
        refine_tail(54321.5, 0.01, collect)
