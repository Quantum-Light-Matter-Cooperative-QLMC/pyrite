"""Detector backend parity guards for issue #622, independent of GPU/checkpoints."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyrite.plots import _detector_frames as frames
from pyrite.plots.plotly import detectors as plotly_detectors


def _record(energy=30.0, n=50, *, wide=True):
    E = np.linspace(1000.0, 5000.0, n)
    out = dict(
        case=dict(
            name="silicon detector test",
            E0_keV=energy,
            thickness_ang=20_000.0,
            tilt_deg=2.0,
            tilt_azim_deg=0.0,
        ),
        E_grid=E,
        spec=np.ones(n),
        brem=np.zeros(n),
        scale=2.0,
    )
    if wide:
        out["E_grid_brem"] = np.linspace(0.0, 30000.0, 100)
        out["brem_wide"] = np.ones(100)
    return out


@pytest.fixture
def detector_stub(monkeypatch):
    """Deterministic response data; this tests plotting, not detector physics."""
    monkeypatch.setattr(frames, "_best_azimuth", lambda recs, collapse: recs)
    monkeypatch.setattr(
        frames,
        "_eag_detected",
        lambda r, s, c, blur: (
            np.asarray(r["spec"]) * 2,
            np.asarray(r["spec"]) * 0.02,
        ),
    )
    monkeypatch.setattr(
        frames,
        "_eag_wide_brem",
        lambda r, c: (
            r["E_grid_brem"],
            np.ones(len(r["E_grid_brem"])),
            np.full(len(r["E_grid_brem"]), 0.001),
        ),
    )
    monkeypatch.setattr(
        frames,
        "_eag_wide_charge",
        lambda r, c, current: (
            r["E_grid_brem"],
            np.ones(len(r["E_grid_brem"])) * current,
        ),
    )
    monkeypatch.setattr(frames, "incident_spectrum", lambda r: np.asarray(r["spec"]))
    monkeypatch.setattr(frames, "cxr_spectrum", lambda r: np.asarray(r["spec"]))
    monkeypatch.setattr(frames, "beam_current_na", lambda r, s: 2.5)
    monkeypatch.setattr(
        frames.eag,
        "get_response",
        lambda E, coating: SimpleNamespace(
            charge_density=lambda incident: np.asarray(incident) * 3.0,
        ),
    )
    monkeypatch.setattr(plotly_detectors.eag, "qe", lambda E, coating: np.full_like(E, 0.8))
    monkeypatch.setattr(plotly_detectors, "_thr_keV", lambda: 1.1)


def test_eagle_photon_traces_share_frame_values_and_units(detector_stub):
    record = _record()
    frame = frames.eaglexo_detected_frame([record], object())
    plot = plotly_detectors.eaglexo_detected_chart([record], object(), show_qe=False)
    assert plot.layout.yaxis.title.text == "Phs/s/eV/uA"
    assert len(plot.data) == 4  # incident/detected on line/brem grids
    for trace in plot.data:
        key = "incident" if trace.line.dash == "dot" else "detected"
        band = "brem" if len(trace.x) == 100 else "line"
        raw = frame[(frame["kind"] == key) & (frame["band"] == band)]
        np.testing.assert_allclose(trace.customdata[:, 0], raw["energy_eV"])
        np.testing.assert_allclose(trace.customdata[:, 1], raw["intensity"])
    assert plot.layout.shapes[0].x0 == pytest.approx(plotly_detectors.SI_K_EDGE_EV)


def test_eagle_qe_on_independent_linear_axis(detector_stub):
    fig = plotly_detectors.eaglexo_detected_chart([_record()], object(), show_qe=True)
    assert len(fig.data) == 5
    qe = fig.data[-1]
    assert qe.yaxis == "y2"
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis2.range == (0, 1.05)
    assert fig.layout.yaxis2.overlaying == "y"
    np.testing.assert_allclose(qe.y, 0.8)


def test_eagle_charge_keeps_charge_units_separate_from_photon_units(detector_stub):
    record = _record()
    frame = frames.eaglexo_charge_frame([record], object())
    fig = plotly_detectors.eaglexo_charge_chart([record], object())
    assert fig.layout.yaxis.title.text == "charge density (e-/eV/s)"
    assert len(fig.data) == 2
    for trace in fig.data:
        band = "brem" if trace.line.dash == "dash" else "line"
        orig = frame.loc[frame["band"] == band]
        np.testing.assert_allclose(trace.customdata[:, 1], orig.charge_density)


def test_timepix_one_scoring_pass_on_shared_wide_grid(detector_stub, monkeypatch):
    calls = []

    class DummyTimepix:
        def __init__(self, **kwargs):
            pass

    class DummyDetector:
        def __init__(self, response):
            pass

        def score(self, energies, incident):
            calls.append((np.asarray(energies), np.asarray(incident)))
            return np.asarray(incident) / 2

    monkeypatch.setattr(frames, "Timepix3", DummyTimepix)
    monkeypatch.setattr(frames, "Detector", DummyDetector)
    record = _record()
    fig = plotly_detectors.timepix_detected_chart([record], object())
    assert len(calls) == 1
    assert len(fig.data) == 2
    assert min(calls[0][0]) >= record["E_grid"].min()
    assert max(calls[0][0]) == pytest.approx(30000)
    np.testing.assert_allclose(fig.data[0].customdata[:, 1], calls[0][1] * 1000)
    assert fig.layout.shapes[0].x0 == pytest.approx(1100)


def test_log_zeros_masked_only_in_display_not_numeric_source(detector_stub):
    record = _record(wide=False)
    record["spec"] = np.array([0.0, 1.0] + [0.0] * 48)
    fig = plotly_detectors.eaglexo_detected_chart([record], object(), show_qe=False)
    assert np.isnan(fig.data[0].y[0])
    assert fig.data[0].customdata[0, 1] == 0.0
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.range[0] is not None


def test_peak_aware_sampling_keeps_weak_detector_peak():
    E = np.arange(20000, dtype=float)
    rows = []
    for kind in ("incident", "detected"):
        y = np.ones(E.size) * (1 if kind == "incident" else 0.01)
        y[4321 if kind == "incident" else 15678] = 500 if kind == "incident" else 9
        rows.append(
            pd.DataFrame(
                dict(
                    energy_eV=E,
                    intensity=y,
                    E0_keV=30.0,
                    azimuth_deg=0.0,
                    kind=kind,
                    band="line",
                )
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    subset = frames._chart_frame(frame, value_column="intensity")
    assert len(subset) == 5000
    incident = subset.loc[subset.kind == "incident"].energy_eV.tolist()
    detected = subset.loc[subset.kind == "detected"].energy_eV.tolist()
    assert incident == detected
    assert 4321 in incident and 15678 in detected


def test_bad_log_domain_rejected_and_empty_view(detector_stub):
    assert plotly_detectors.timepix_detected_chart([], object()) is None
    with pytest.raises(ValueError, match="positive"):
        plotly_detectors.eaglexo_charge_chart([_record()], object(), x_domain=(0, 1))
    with pytest.raises(ValueError, match="increasing"):
        plotly_detectors.eaglexo_charge_chart([_record()], object(), y_domain=(2, 1))
