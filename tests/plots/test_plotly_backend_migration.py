"""Regression contracts for Plotly-first interactive 2D migration (#622).

Tests deliberately use prepared frames, so no radiation transport, detectors,
file I/O or GPU is involved.  They protect the viewer from changing numeric
sample values, detector bin edges and missing-versus-zero cell semantics.
"""

import numpy as np
import pandas as pd
import pytest

from pyrite.plots.plotly import pixels, spectra, sweeps, temporal


def _spectrum_frame():
    return pd.DataFrame(
        {
            "energy_eV": [10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
            "intensity": [1.0, 0.0, 8.0, 2.0, 0.0, 5.0],
            "E0_keV": [30.0] * 6,
            "component": ["total"] * 3 + ["brem"] * 3,
        }
    )


def test_spectrum_components_preserve_coordinates_and_units():
    frame = _spectrum_frame()
    original = frame.copy(deep=True)
    figure = spectra._trace_series(
        frame,
        hue="E0_keV",
        components=("total", "brem"),
        width=640,
        height=320,
        title="test",
        x_type="linear",
        y_type="linear",
        x_domain=None,
    )
    assert len(figure.data) == 2
    assert list(figure.data[0].x) == [10, 11, 12]
    assert list(figure.data[0].y) == [1, 0, 8]
    assert list(figure.data[1].y) == [2, 0, 5]
    assert figure.layout.yaxis.title.text == "Intensity (Phs/s/eV/uA)"
    pd.testing.assert_frame_equal(frame, original)


def test_log_axes_mask_only_display_values():
    frame = _spectrum_frame()
    figure = spectra._trace_series(
        frame,
        hue="E0_keV",
        components=("total",),
        width=500,
        height=200,
        title="test",
        x_type="log",
        y_type="log",
        x_domain=(10, 12),
    )
    assert figure.layout.xaxis.type == "log"
    assert figure.layout.xaxis.range[0] == pytest.approx(1.0)
    assert np.isnan(figure.data[0].y[1])
    assert frame["intensity"].iloc[1] == 0.0
    assert figure.layout.yaxis.range[0] is not None


def test_log_axis_rejects_nonpositive_bounds():
    with pytest.raises(ValueError, match="positive"):
        spectra._axis_range((0, 10), "log")


def test_sweep_heatmap_preserves_missing_cell_vs_zero(monkeypatch):
    frame = pd.DataFrame(
        {
            "x": [0, 1, 0],
            "y": [0, 0, 1],
            "panel": ["30 keV"] * 3,
            "value": [0.0, 4.0, 2.0],
            "name": ["a", "b", "c"],
            "panel_raw": [30.0] * 3,
        }
    )
    monkeypatch.setattr(sweeps, "records_for_cases", lambda *_: [object()])
    monkeypatch.setattr(sweeps, "heatmap_frame", lambda *args, **kwargs: frame)
    figure = sweeps.heatmap_chart({}, object(), quantity="line_flux")
    z = np.asarray(figure.data[0].z)
    assert z[0, 0] == 0
    assert np.isnan(z[1, 1])
    assert list(figure.data[0].x) == [0, 1]


def test_pixel_map_orientation_selected_cell_and_block_bounds():
    frame = pd.DataFrame(
        {
            "row": [0, 0, 2, 2],
            "column": [0, 2, 0, 2],
            "row0": [0, 0, 2, 2],
            "row1": [1, 1, 2, 2],
            "column0": [0, 2, 0, 2],
            "column1": [1, 2, 1, 2],
            "block": [2] * 4,
            "value": [0, 1, 2, 3],
        }
    )
    figure = pixels.detector_image_chart(
        frame,
        title="detector",
        value_title="counts",
        selected=(2, 2),
    )
    assert np.asarray(figure.data[0].z).shape == (2, 2)
    assert figure.layout.xaxis.range == (0, 3)
    assert figure.layout.yaxis.range == (3, 0)
    selection = figure.layout.shapes[0]
    assert (selection.x0, selection.x1, selection.y0, selection.y1) == (2, 3, 2, 3)


def test_pixel_histogram_respects_measured_edges_and_raw_counts():
    frame = pd.DataFrame({"low_eV": [10.0, 13.0], "high_eV": [13.0, 20.0], "counts": [0.0, 6.0]})
    figure = pixels.histogram_chart(
        frame, title="measured", count_title="expected counts", y_type="log"
    )
    assert list(figure.data[0].width) == [3, 7]
    assert list(figure.data[0].customdata[:, 2]) == [0, 6]
    assert figure.layout.yaxis.tickmode == "array"


def test_temporal_is_built_from_exact_shared_frame(monkeypatch):
    frame = pd.DataFrame(
        {
            "time_fs": [-1.0, 0.0, 1.0],
            "intensity": [0.0, 4.0, 0.0],
            "E0_keV": [30.0] * 3,
            "component": ["incoherent"] * 3,
        }
    )
    monkeypatch.setattr(temporal, "temporal_frame", lambda *args, **kwargs: frame)
    figure = temporal.temporal_chart({})
    assert list(figure.data[0].x) == [-1, 0, 1]
    assert list(figure.data[0].y) == [0, 4, 0]


def test_material_comparison_keeps_source_values():
    fig = spectra.material_comparison_chart(
        [("A", 1234.0, 56.0, 0.85, {"E0_keV": 30.0, "tilt_deg": 5.0, "tilt_azim_deg": 10.0})],
        {},
    )
    assert fig.data[0].x[0] == pytest.approx(1.234)
    assert fig.data[0].y[0] == pytest.approx(56.0)
    assert fig.layout.yaxis.type == "log"
