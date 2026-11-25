"""Guard tests for the Plotly detector renderers (pyrite.plots.plotly.detectors)
and their renderer-neutral frames (pyrite.plots._detector_frames).

Synthetic records only (no GPU, no checkpoint). The detector response models
are shared with the Matplotlib path; a small ``n_mc`` keeps Timepix cheap.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pyrite.detectors import Timepix3
from pyrite.plots._detector_frames import (
    _NA_PER_UA,
    _chart_frame,
    eaglexo_charge_frame,
    eaglexo_detected_frame,
    timepix_detected_frame,
)
from pyrite.plots.plotly.detectors import (
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    timepix_detected_chart,
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


def _curves(fig):
    return [t for t in fig.data if t.name != "Quantum efficiency"]


def _vlines(fig):
    return [s for s in fig.layout.shapes if s.x0 == s.x1]


def _assert_log_y_masks_zero(fig, field_values):
    assert fig.layout.yaxis.type == "log"
    assert np.isfinite(fig.layout.yaxis.range[0])
    assert (np.asarray(field_values) == 0.0).any()  # the source really hits 0
    shown = np.concatenate([np.asarray(t.y, dtype=float) for t in _curves(fig)])
    source = np.concatenate([np.asarray(t.customdata, dtype=float)[:, 1] for t in _curves(fig)])
    assert (source == 0.0).any() and not (shown == 0.0).any()


def test_detector_chart_sampling_bounds_payload_and_keeps_both_peaks():
    energy = np.arange(20000, dtype=float)
    incident = np.ones(energy.size)
    detected = np.ones(energy.size) * 0.01
    incident[4321] = 100.0
    detected[15678] = 5.0
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "energy_eV": energy,
                    "intensity": intensity,
                    "E0_keV": 30.0,
                    "azimuth_deg": 0.0,
                    "kind": kind,
                    "band": "line",
                }
            )
            for kind, intensity in (("incident", incident), ("detected", detected))
        ],
        ignore_index=True,
    )
    sampled = _chart_frame(frame, value_column="intensity")
    assert 4_500 <= len(sampled) <= 5000
    curves = {kind: part for kind, part in sampled.groupby("kind")}
    assert set(curves["incident"]["energy_eV"]) == set(curves["detected"]["energy_eV"])
    assert 4321.0 in set(curves["incident"]["energy_eV"])
    assert 15678.0 in set(curves["detected"]["energy_eV"])


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
    np.testing.assert_allclose(incident["intensity"], 2.0 * _NA_PER_UA)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], incident["energy_eV"])


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


def test_eaglexo_detected_chart_none_on_empty():
    assert eaglexo_detected_chart({}, _settings()) is None


# ---- Eagle XO charge ---------------------------------------------------------


def test_eaglexo_charge_frame_bands():
    df = eaglexo_charge_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings())
    assert list(df.columns) == ["energy_eV", "charge_density", "E0_keV", "azimuth_deg", "band"]
    assert set(df["band"]) == {"line", "brem"}


def test_eaglexo_charge_chart_none_on_empty():
    assert eaglexo_charge_chart({}, _settings()) is None


def test_dense_eagle_chart_bounds_points():
    store = {"HOPG bulk": {30.0: _record(30.0, -20.0, 0.0, n=8651)}}
    fig = eaglexo_detected_chart(store, _settings(), show_qe=False)
    # Per-curve display budget (#1440): standard caps each curve at 5000.
    assert max(len(t.x) for t in _curves(fig)) <= 5000
    fast = eaglexo_detected_chart(store, _settings(), show_qe=False, resolution="fast")
    assert max(len(t.x) for t in _curves(fast)) <= 2000
    assert "of 8,651 native points" in fast.layout.title.subtitle.text
    full = eaglexo_detected_chart(store, _settings(), show_qe=False, resolution="full")
    assert max(len(t.x) for t in _curves(full)) == 8651


def test_timepix_chart_incident_detected_and_threshold():
    fig = timepix_detected_chart(_store(), _settings(), **_TPX_KW)
    dashes = {t.line.dash for t in fig.data}
    assert {"dot", "solid"} <= dashes  # incident dotted, detected solid
    assert {t.legendgroup for t in fig.data} == {"30 keV", "60 keV"}
    assert len(_vlines(fig)) == 1
    fig.to_json()


def test_timepix_chart_broadband_log_y_floors_range_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    fig = timepix_detected_chart({"HOPG bulk": {30.0: rec}}, _settings(), **_TPX_KW, y_type="log")
    frame = timepix_detected_frame([rec], _settings(), **_TPX_KW)
    _assert_log_y_masks_zero(fig, frame["intensity"])


def test_eaglexo_detected_chart_with_qe_on_secondary_axis():
    fig = eaglexo_detected_chart(_store(), _settings(), show_qe=True)
    qe = [t for t in fig.data if t.name == "Quantum efficiency"]
    assert len(qe) == 1 and qe[0].yaxis == "y2"
    assert fig.layout.yaxis2.overlaying == "y"
    assert len(_vlines(fig)) == 1  # Si-K edge


def test_eaglexo_detected_chart_without_qe():
    fig = eaglexo_detected_chart(_store(), _settings(), show_qe=False)
    assert not [t for t in fig.data if t.name == "Quantum efficiency"]


def test_eaglexo_detected_chart_broadband_log_y_floors_range_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    fig = eaglexo_detected_chart({"HOPG bulk": {30.0: rec}}, _settings(), show_qe=False)
    frame = eaglexo_detected_frame([rec], _settings())
    _assert_log_y_masks_zero(fig, frame["intensity"])


def test_eaglexo_charge_chart_line_and_brem_traces():
    fig = eaglexo_charge_chart(_store(wide_brem=True), _settings())
    assert {t.line.dash for t in fig.data} == {"solid", "dash"}  # line, brem
    assert len(_vlines(fig)) == 1
    assert fig.layout.yaxis.title.text == "charge density (e-/eV/s)"


def test_eaglexo_charge_chart_x_domain_fixes_range():
    fig = eaglexo_charge_chart(_store(wide_brem=True), _settings(), x_domain=(500.0, 7000.0))
    np.testing.assert_allclose(fig.layout.xaxis.range, np.log10([500.0, 7000.0]))


def test_eaglexo_charge_chart_broadband_log_y_floors_range_away_from_zero():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.0, 0.0, rec["E_grid_brem"].size)
    fig = eaglexo_charge_chart({"HOPG bulk": {30.0: rec}}, _settings(), y_type="log")
    frame = eaglexo_charge_frame([rec], _settings())
    _assert_log_y_masks_zero(fig, frame["charge_density"])
