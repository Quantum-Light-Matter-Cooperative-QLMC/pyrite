"""Streamed pilot transactions: no double-counted evidence on a failed block.

Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite.montecarlo.runner import cxr_grid, streamed_line_grid
from pyrite.montecarlo.spectrum import _lattice_tail as lt


def test_streamed_pilot_retry_and_certification(monkeypatch):
    band = {"policy": "resonance-population", "start_eV": 100, "stop_eV": 54000,
            "truncation_limit": 0.001}
    payload = {"bandwidth": band, "resolution": {"policy": "sinc-oversampled"}}
    case = {"Ne": 20, "Ne_brem": 20, "cxr_fiber": {}, "line_grid_policy": payload}
    direction = np.array([0.0, 0.0, 1.0])
    monkeypatch.setattr(streamed_line_grid, "_block_sinc_samples", lambda *args: (np.ones(1), np.ones(1)))
    evidence = streamed_line_grid.BlockGridEvidence(case, payload, direction, 20)
    evidence._tail_replay = lambda stops: pytest.fail("unexpected replay")
    block = {"L_ang": np.ones(10), "Ne": 10}

    def broken(*args, **kwargs):
        raise MemoryError("injected")

    monkeypatch.setattr(cxr_grid, "_direction_tail", broken)
    with pytest.raises(MemoryError, match="injected"):
        evidence.add(block)
    assert evidence._seen == [0, 0, 0, 0]
    assert evidence._tail_stops is None and evidence._tail_summary is None
    assert evidence._pilot_rows == 0

    def collect(*args, stops):
        return lt.summarize_lines(np.full(10, 1000.0), np.full(10, 0.1), np.ones(10), stops)

    monkeypatch.setattr(cxr_grid, "_direction_tail", collect)
    evidence.add(block)
    evidence.add(block)
    tail = evidence.tail(direction, *evidence.band, None)
    assert tail.n_lines == 20
    assert evidence.line_evidence_record()["pilot_certified"] is True
    assert evidence.line_evidence_record()["line_population_bytes"] == 0
    assert evidence._seen[0] == 20
    evidence.close()
