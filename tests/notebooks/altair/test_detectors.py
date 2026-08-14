"""Guard tests for the Altair detector renderers (pyrite.plots.altair.detectors).

Like test_altair_plots.py, these exercise only the NEW rendering layer on
synthetic records (no GPU, no checkpoint), small enough to stay under Vega-Lite's
default 5000-row cap so ``chart.to_dict()`` serializes without touching global
config. The detector physics (tpx/eag response models) is shared with the
matplotlib path; a small ``n_mc`` keeps the Timepix Monte Carlo cheap.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from pyrite.detectors import Timepix3
from pyrite.plots.altair.detectors import (
    eaglexo_charge_chart,
    eaglexo_charge_frame,
    eaglexo_detected_chart,
    eaglexo_detected_frame,
    timepix_detected_chart,
    timepix_detected_frame,
)

_TPX_KW = {"n_mc": 500, "seed": 0}


def _settings():
    return SimpleNamespace(beam_current_na=1.0)


def _record(E0_keV, tilt_deg, azim_deg, n=120, wide_brem=False):
    E = np.linspace(1000.0, 6000.0, n)  # eV
    peak = np.exp(-(((E - 2500.0) / 50.0) ** 2))  # a single sharp line
    brem = np.linspace(1.0, 0.2, n)  # a smooth falling continuum
    rec = {
        "E_grid": E,
        "spec": peak,
        "brem": brem,
        "scale": 2.0,
        "case": {
            "name": "HOPG bulk",
            "E0_keV": E0_keV,
            "tilt_deg": tilt_deg,
            "tilt_azim_deg": azim_deg,
            "thickness_ang": 5.0e4,
        },
    }
    if wide_brem:
        Eb = np.linspace(0.0, E0_keV * 1e3, n)  # full range out to the beam energy
        rec["E_grid_brem"] = Eb
        rec["brem_wide"] = np.linspace(1.0, 0.05, n)
    return rec


def _store(wide_brem=False):
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0, wide_brem=wide_brem),
            60.0: _record(60.0, -20.0, 0.0, wide_brem=wide_brem),
        }
    }


def _dataset(spec):
    name = spec.get("data", {}).get("name")
    if name is None:
        name = spec["layer"][0]["data"]["name"]
    return spec["datasets"][name]


# ---- Timepix3 ----------------------------------------------------------------
def test_timepix_frame_has_incident_and_detected():
    df = timepix_detected_frame([_record(30.0, -20.0, 0.0)], _settings(), **_TPX_KW)
    assert list(df.columns) == ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "kind", "band"]
    assert set(df["kind"]) == {"incident", "detected"}
    assert set(df["band"]) == {"line"}


def test_timepix_frame_uses_one_response_across_line_grid_boundary(monkeypatch):
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["spec"] = np.zeros_like(rec["spec"])
    rec["brem_wide"] = np.ones_like(rec["brem_wide"])

    calls = []

    def _score(_self, E, spec, **_kwargs):
        calls.append(np.asarray(E, dtype=float))
        return np.asarray(spec, dtype=float)

    monkeypatch.setattr(Timepix3, "score", _score)
    df = timepix_detected_frame([rec], _settings(), **_TPX_KW)

    # Broadband Timepix detection must use one response over the summed
    # spectrum.  Splitting it at the line-grid endpoint loses redistributed
    # photons and makes the detected trace jump there.
    assert set(df["band"]) == {"broad"}
    assert df["energy_eV"].max() == 30000.0
    incident = df.loc[df["kind"] == "incident"]
    assert incident["energy_eV"].is_monotonic_increasing
    np.testing.assert_allclose(incident["intensity"], 2.0)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], incident["energy_eV"])


def test_timepix_chart_builds_valid_spec():
    chart = timepix_detected_chart(_store(), _settings(), **_TPX_KW)
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()  # raises if malformed / over the row cap
    # incident + detected + threshold rule
    assert len(spec["layer"]) == 3


def test_timepix_chart_broadband_log_y_floors_domain_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    store = {"HOPG bulk": {30.0: rec}}

    chart = timepix_detected_chart(store, _settings(), **_TPX_KW, y_type="log")

    spec = chart.to_dict()
    scale = spec["layer"][0]["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert scale.get("domainMin", 0) > 0
    assert scale.get("clamp") is True
    assert any(row["intensity"] == 0.0 for row in _dataset(spec))


def test_timepix_chart_none_on_empty():
    assert timepix_detected_chart({}, _settings(), **_TPX_KW) is None


# ---- Eagle XO detected -------------------------------------------------------
def test_eaglexo_detected_frame_line_only_without_wide_brem():
    df = eaglexo_detected_frame([_record(30.0, -20.0, 0.0)], _settings())
    assert set(df["band"]) == {"line"}
    assert set(df["kind"]) == {"incident", "detected"}


def test_eaglexo_detected_frame_includes_wide_brem():
    df = eaglexo_detected_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings())
    assert set(df["band"]) == {"line", "brem"}


def test_eaglexo_detected_chart_with_qe_resolves_independent_y():
    chart = eaglexo_detected_chart(_store(), _settings(), show_qe=True)
    spec = chart.to_dict()
    # incident + detected + Si-K rule + QE envelope
    assert len(spec["layer"]) == 4
    assert spec["resolve"]["scale"]["y"] == "independent"


def test_eaglexo_detected_chart_without_qe():
    chart = eaglexo_detected_chart(_store(), _settings(), show_qe=False)
    spec = chart.to_dict()
    assert len(spec["layer"]) == 3  # no QE layer


def test_eaglexo_detected_chart_broadband_log_y_floors_domain_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    store = {"HOPG bulk": {30.0: rec}}

    chart = eaglexo_detected_chart(store, _settings(), show_qe=False, y_type="log")

    spec = chart.to_dict()
    scale = spec["layer"][0]["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert scale.get("domainMin", 0) > 0
    assert scale.get("clamp") is True
    assert any(row["intensity"] == 0.0 for row in _dataset(spec))


def test_eaglexo_detected_chart_none_on_empty():
    assert eaglexo_detected_chart({}, _settings()) is None


# ---- Eagle XO charge ---------------------------------------------------------
def test_eaglexo_charge_frame_bands():
    df = eaglexo_charge_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings())
    assert list(df.columns) == ["energy_eV", "charge_density", "E0_keV", "azimuth_deg", "band"]
    assert set(df["band"]) == {"line", "brem"}


def test_eaglexo_charge_chart_builds_valid_spec():
    chart = eaglexo_charge_chart(_store(wide_brem=True), _settings())
    spec = chart.to_dict()
    # line + brem + Si-K rule
    assert len(spec["layer"]) == 3


def test_eaglexo_charge_chart_none_on_empty():
    assert eaglexo_charge_chart({}, _settings()) is None


def test_eaglexo_charge_chart_x_domain_fixes_scale():
    chart = eaglexo_charge_chart(_store(wide_brem=True), _settings(), x_domain=(500.0, 7000.0))
    spec = chart.to_dict()
    x_scale = spec["layer"][0]["encoding"]["x"]["scale"]
    assert x_scale["domain"] == [500.0, 7000.0]


def test_eaglexo_charge_chart_broadband_log_y_floors_domain_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    store = {"HOPG bulk": {30.0: rec}}

    chart = eaglexo_charge_chart(store, _settings(), y_type="log")

    spec = chart.to_dict()
    scale = spec["layer"][0]["encoding"]["y"]["scale"]
    assert scale["type"] == "log"
    assert scale.get("domainMin", 0) > 0
    assert scale.get("clamp") is True
    assert any(row["charge_density"] == 0.0 for row in _dataset(spec))
