"""The multi-quantity sweep drivers compute the per-record metrics map ONCE.

plot_heatmaps / plot_scan (matplotlib) and scan_charts (Altair) render one
figure per quantity, but the expensive per-record ``results.line_metrics``
reduction is quantity-independent -- these guard that the drivers build the
map once and share it across quantities, instead of once per quantity
(the pre-fix behaviour: 8 default quantities -> 8x the metric extraction).
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import cxr_mc.plots._common as _common
from cxr_mc.plots.altair_sweeps import scan_charts
from cxr_mc.plots.sweeps import plot_heatmaps, plot_scan

N_RECORDS = 8  # 2 tilts x 2 azimuths x 2 beam energies (see _store)

# two quantities are enough to distinguish once (N_RECORDS) from
# once-per-quantity (2 * N_RECORDS)
_TRIPLES = [("peak_flux", "peak", "viridis"), ("line_flux", "line", "viridis")]
_KEYS = ["peak_flux", "line_flux"]


def _settings():
    return SimpleNamespace(beam_current_na=1.0)


def _record(name, E0, tilt, azim, amp, n=80):
    E = np.linspace(1000.0, 5000.0, n)
    spec = amp * np.exp(-(((E - 2500.0) / 40.0) ** 2))
    brem = np.linspace(0.5, 0.1, n)
    return {
        "E_grid": E,
        "spec": spec,
        "brem": brem,
        "scale": 1.0,
        "case": {
            "name": name,
            "crystal": "HOPG",
            "E0_keV": E0,
            "tilt_deg": tilt,
            "tilt_azim_deg": azim,
            "thickness_ang": 5.0e4,
        },
    }


def _store():
    store = {}
    for tilt in (20.0, 10.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + 0.1 * E0 + 0.01 * azim)
                for E0 in (30.0, 60.0)
            }
    return store


@pytest.fixture
def line_metrics_calls(monkeypatch):
    """Count every per-record line_metrics computation (_metrics_map resolves
    the name from plots._common at call time). Clears `_LINE_METRICS_CACHE`
    first: it persists across calls (and so across tests, within one process)
    by design -- every test here uses the same deterministic `_store()` +
    default rel_prominence/line_metric, so without a reset the second test
    onward would cache-hit the first test's entries and see zero calls."""
    _common._LINE_METRICS_CACHE.clear()
    real = _common.line_metrics
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_common, "line_metrics", counting)
    return calls


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_plot_heatmaps_computes_metrics_once(line_metrics_calls):
    figs = plot_heatmaps(_store(), _settings(), quantities=_TRIPLES)
    assert len(figs) == 2
    assert line_metrics_calls["n"] == N_RECORDS


def test_plot_scan_line_mode_computes_metrics_once(line_metrics_calls):
    figs = plot_scan(_store(), _settings(), quantities=_KEYS, force="lines")
    assert len(figs) == 2
    assert line_metrics_calls["n"] == N_RECORDS


def test_plot_scan_heatmap_mode_computes_metrics_once(line_metrics_calls):
    figs = plot_scan(_store(), _settings(), quantities=_KEYS, force="heatmap")
    assert len(figs) == 2
    assert line_metrics_calls["n"] == N_RECORDS


def test_scan_charts_line_mode_computes_metrics_once(line_metrics_calls):
    charts = scan_charts(_store(), _settings(), quantities=_KEYS, force="lines")
    assert len(charts) == 2
    assert line_metrics_calls["n"] == N_RECORDS


def test_scan_charts_heatmap_mode_computes_metrics_once(line_metrics_calls):
    charts = scan_charts(_store(), _settings(), quantities=_KEYS, force="heatmap")
    assert len(charts) == 2
    assert line_metrics_calls["n"] == N_RECORDS
