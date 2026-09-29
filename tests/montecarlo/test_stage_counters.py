"""Per-case stage counters reach performance telemetry next to phase times."""

import sys

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.timing import _TimingAgg
from pyrite.montecarlo.spectrum.lines import _setup as line_setup


def _tp():
    return {
        "E_grid": np.linspace(50.0, 5000.0, 1234),
        "E_brem": np.arange(0.0, 30000.0, 25.0),
        "segs": {"L_ang": np.ones(777), "E_keV": np.ones(777)},
    }


def test_stage_counters_report_axes_segments_and_tabulation(monkeypatch):
    monkeypatch.setattr(line_setup, "SETUP_STATS", {"line_tab_points": 5000, "line_table_mib": 1.5})
    counters = runner._stage_counters(_tp())
    assert counters["_line_axis_nodes"] == 1234
    assert counters["_brem_axis_nodes"] == 1200
    assert counters["_segments"] == 777
    assert counters["_line_tab_points"] == 5000
    assert counters["_line_table_mib"] == 1.5


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RSS is read from /proc")
def test_stage_counters_include_host_rss():
    counters = runner._stage_counters(_tp())
    assert 0.0 < counters["_host_rss_mib"] <= counters["_host_rss_peak_mib"] * 1.01


def test_timing_collect_forwards_stage_counters_and_strips_them():
    out = {"_t_spectrum": 0.5, **runner._stage_counters(_tp())}
    update = _TimingAgg().collect(out)
    assert update["line_axis_nodes"] == 1234
    assert update["segments"] == 777
    assert not any(key.startswith("_") for key in out)
