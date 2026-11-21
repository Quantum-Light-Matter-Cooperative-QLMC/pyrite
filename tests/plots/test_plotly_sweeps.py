"""Guard tests for the Plotly sweep renderers (pyrite.plots.plotly.sweeps).

Synthetic records only (no GPU, no checkpoint). The metric prep
(results.cxr_metrics / selection_score) is shared with the Matplotlib path; the
click-selectable Map is covered by ``test_plotly_map_pilot.py``.
"""

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from pyrite.plots._frames import _value_label, heatmap_frame, metric_vs_frame
from pyrite.plots.plotly.sweeps import heatmap_chart, metric_vs_chart, scan_charts


def _settings():
    return SimpleNamespace(beam_current_na=1.0)


def test_thickness_value_labels_switch_to_millimetres():
    assert _value_label("thickness_ang", 10_000_000.0) == "1mm"


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
    # 2 tilts x 2 azimuths (distinct configs) x 2 beam energies = 8 records.
    store = {}
    for tilt in (20.0, 10.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + 0.1 * E0 + 0.01 * azim)
                for E0 in (30.0, 60.0)
            }
    return store


# ---- metric_vs ---------------------------------------------------------------
def test_metric_vs_frame_shape():
    df = metric_vs_frame(_store(), _settings(), x="tilt_deg", metric="peak_flux", hue="E0_keV")
    assert list(df.columns) == ["x", "metric", "hue"]
    # 2 beam energies x 2 tilt values (azimuth reduced to best)
    assert len(df) == 4
    assert set(df["hue"]) == {"30 keV", "60 keV"}


def test_metric_vs_frame_single_valued_x_falls_back():
    # thickness has one value -> guard substitutes a swept knob (tilt_deg)
    df = metric_vs_frame(_store(), _settings(), x="thickness_ang", metric="peak_flux", hue="E0_keV")
    assert len(df) == 4  # 2 energies x 2 tilts, not a single stacked x


def test_metric_vs_chart_one_line_per_hue_from_shared_frame():
    fig = metric_vs_chart(_store(), _settings(), x="tilt_deg", metric="peak_flux")
    frame = metric_vs_frame(_store(), _settings(), x="tilt_deg", metric="peak_flux", hue="E0_keV")
    assert isinstance(fig, go.Figure)
    assert {trace.name for trace in fig.data} == set(frame["hue"])
    for trace in fig.data:
        part = frame[frame["hue"] == trace.name].sort_values("x")
        np.testing.assert_array_equal(trace.y, part["metric"].to_numpy(dtype=float))
    fig.to_json()


def test_metric_vs_chart_none_on_empty():
    assert metric_vs_chart({}, _settings()) is None


# ---- heatmap -----------------------------------------------------------------
def test_heatmap_frame_shape():
    df = heatmap_frame(
        _store(), _settings(), quantity="peak_flux", x="tilt_azim_deg", y="tilt_deg", panel="E0_keV"
    )
    assert list(df.columns) == ["x", "y", "panel", "value", "name", "panel_raw"]
    # 2 panels x (2 azimuths x 2 tilts) = 8 cells (peak_flux is never gated)
    assert len(df) == 8
    assert set(df["panel"]) == {"30 keV", "60 keV"}
    # name/panel_raw carry per-cell identity for click-to-select back-mapping
    assert set(df["name"]) == {
        "HOPG t20.0 a0.0",
        "HOPG t20.0 a30.0",
        "HOPG t10.0 a0.0",
        "HOPG t10.0 a30.0",
    }
    assert set(df["panel_raw"]) == {30.0, 60.0}


def test_heatmap_chart_one_heatmap_per_panel():
    fig = heatmap_chart(_store(), _settings(), quantity="peak_flux")
    assert [trace.type for trace in fig.data] == ["heatmap", "heatmap"]
    assert [a.text for a in fig.layout.annotations] == ["30 keV", "60 keV"]
    fig.to_json()


def test_heatmap_chart_accepts_quantity_triple():
    fig = heatmap_chart(_store(), _settings(), quantity=("peak_flux", "my label", "magma"))
    assert fig.layout.title.text == "my label"
    assert fig.layout.coloraxis.colorscale == go.layout.Coloraxis(colorscale="magma").colorscale


def test_heatmap_chart_none_on_empty():
    assert heatmap_chart({}, _settings()) is None


# ---- scan (auto-pick) --------------------------------------------------------
def test_scan_charts_force_lines_one_per_quantity():
    charts = scan_charts(_store(), _settings(), force="lines")
    assert len(charts) == 7  # spacing-dependent peak density is opt-in
    assert all(trace.type == "scatter" for c in charts for trace in c.data)


def test_scan_charts_force_heatmap():
    charts = scan_charts(_store(), _settings(), force="heatmap")
    assert len(charts) == 7
    assert all(trace.type == "heatmap" for c in charts for trace in c.data)


def test_scan_charts_empty():
    assert scan_charts({}, _settings()) == []
