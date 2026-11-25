"""Zoom-driven spectral re-sampling in the analysis app (#1442).

Pins the marimo contract the app relies on: Plotly relayout axis keys reach
``mo.ui.plotly``'s raw payload (``_selection_data``) and fire ``on_change``.
A marimo upgrade that changes either fails here instead of silently freezing
the zoomed sampling.
"""

from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import numpy as np
import plotly.graph_objects as go
import pytest

from pyrite.apps.analysis_ui.zoom import UNCHANGED, viewport_from_relayout, zoomable_chart
from pyrite.plots.plotly.spectra import prepare_compare_spectra, render_prepared_spectra


def _record(tilt, n=60_000):
    E = np.linspace(50.0, 2000.0, n)
    return {
        "E_grid": E,
        "spec": np.sinc((E - 1000.0) / 2.0) ** 2,
        "brem": np.full(E.size, 0.01),
        "scale": 1.0,
        "fwhm": 2.0,
        "case": {
            "name": "HOPG bulk",
            "E0_keV": 300.0,
            "tilt_deg": float(tilt),
            "tilt_azim_deg": 180.0,
            "thickness_ang": 5.0e4,
        },
    }


def _settings():
    return SimpleNamespace(apply_detector_qe=False, convolve_with_det=False, brem_source="mc")


@pytest.mark.parametrize(
    ("raw", "x_type", "expected"),
    [
        ({"xaxis": {"range": [990.0, 1010.0]}}, "linear", (990.0, 1010.0)),
        ({"xaxis": {"range": [1010.0, 990.0]}}, "linear", (990.0, 1010.0)),
        ({"xaxis": {"range": [2.0, 3.0]}}, "log", (100.0, 1000.0)),
        ({"xaxis": {"autorange": True}}, "linear", None),
        ({}, "linear", None),
        (None, "linear", None),
        ({"yaxis": {"range": [0.0, 1.0]}}, "linear", UNCHANGED),
        ({"dragmode": "pan"}, "linear", UNCHANGED),
        ({"xaxis": {"range": [5.0, 5.0]}}, "linear", UNCHANGED),
    ],
)
def test_viewport_from_relayout(raw, x_type, expected):
    assert viewport_from_relayout(raw, x_type=x_type) == expected


def test_marimo_plotly_forwards_relayout_payload():
    calls = []
    element = mo.ui.plotly(go.Figure(go.Scatter(x=[0, 1], y=[0, 1])), on_change=calls.append)
    payload = {"xaxis": {"range": [0.2, 0.4]}, "dragmode": "zoom"}
    element._update(payload)

    assert calls, "mo.ui.plotly no longer fires on_change for relayout payloads"
    assert element._selection_data == payload


def test_marimo_frontend_forwards_xaxis_relayout_keys():
    assets = Path(mo.__file__).parent / "_static" / "assets"
    bundles = [p.read_text(errors="replace") for p in assets.glob("index-*.js")]
    assert any("onRelayout" in text and "includes(`xaxis`)" in text for text in bundles)


def test_zoom_stores_viewport_and_resamples_natively():
    state = {"viewport": None}
    prepared = prepare_compare_spectra(
        {"HOPG bulk": {15.0: _record(15.0)}},
        _settings(),
        hue="tilt_deg",
        include_brem=False,
        band="broad",
    )
    figure = render_prepared_spectra(prepared, resolution="fast")
    element = zoomable_chart(
        mo,
        figure,
        x_type="linear",
        get_viewport=lambda: state["viewport"],
        set_viewport=lambda value: state.update(viewport=value),
    )
    element._update({"xaxis": {"range": [990.0, 1010.0]}})
    assert state["viewport"] == (990.0, 1010.0)

    zoomed = render_prepared_spectra(prepared, resolution="fast", viewport=state["viewport"])
    (trace,) = zoomed.data
    x = np.asarray(trace.x)
    native = prepared.frame["energy_eV"].to_numpy()
    visible = native[(native >= 990.0) & (native <= 1010.0)]
    assert np.isin(visible, x).all()
    assert list(zoomed.layout.xaxis.range) == [990.0, 1010.0]
    assert len(figure.data[0].x) <= 2_000

    element._update({"xaxis": {"autorange": True}})
    assert state["viewport"] is None
