"""Display-only spectral resolution and lineshape decimation (#1440, #1442).

Synthetic records only. Decimation selects stored nodes; it never
interpolates, smooths or mutates the record arrays.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.plots._decimation import (
    PLOT_RESOLUTIONS,
    lineshape_indices,
    resolve_plot_resolution,
    trace_budget,
    trace_indices,
)
from pyrite.plots._spectrum_frames import _NA_PER_UA, spectrum_frame
from pyrite.plots.plotly.spectra import compare_spectrum_chart, spectrum_chart


def _settings():
    return SimpleNamespace(apply_detector_qe=False, convolve_with_det=False, brem_source="mc")


def _lines(x):
    """sinc^2 line with sidelobes, an overlapping pair (central minimum),
    a 1 % weak line and a shoulder on a falling continuum."""
    lor = lambda c, g: 1.0 / (1.0 + ((x - c) / g) ** 2)  # noqa: E731
    return (
        0.05 * np.exp(-x / 800.0)
        + np.sinc((x - 1000.0) / 2.0) ** 2
        + 0.6 * lor(1200.0, 1.5)
        + 0.6 * lor(1205.0, 1.5)
        + 0.01 * np.exp(-(((x - 1500.0) / 0.5) ** 2))
        + 0.4 * np.exp(-(((x - 1700.0) / 4.0) ** 2))
        + 0.15 * np.exp(-(((x - 1709.0) / 2.0) ** 2))
    )


def _record(tilt, n=20_000, E=None):
    E = np.linspace(50.0, 2000.0, n) if E is None else E
    return {
        "E_grid": E,
        "spec": _lines(E),
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


def _old_max_only(y, budget):
    """The pre-#1442 rule: one maximum per index bucket."""
    keep = {0, y.size - 1}
    for bucket in np.array_split(np.arange(1, y.size - 1), budget - 2):
        if bucket.size:
            keep.add(int(bucket[np.argmax(y[bucket])]))
    return np.fromiter(sorted(keep), dtype=int)


def _raster_error(x, y, idx, columns=800):
    """Worst per-pixel-column envelope error of the drawn polyline (fraction of peak)."""
    drawn = np.interp(x, x[idx], y[idx])
    column = np.minimum(((x - x[0]) / (x[-1] - x[0]) * columns).astype(int), columns - 1)
    err = 0.0
    for values in (y, drawn):
        hi = np.full(columns, -np.inf)
        lo = np.full(columns, np.inf)
        np.maximum.at(hi, column, values)
        np.minimum.at(lo, column, values)
        if values is y:
            native = (hi, lo)
        else:
            err = max(np.abs(hi - native[0]).max(), np.abs(lo - native[1]).max())
    return err / y.max()


# ---- resolution presets ---------------------------------------------------------
def test_presets_and_legacy_max_points_resolve():
    assert resolve_plot_resolution("full").is_full
    assert resolve_plot_resolution(None).name == "standard"
    legacy = resolve_plot_resolution("standard", max_points=40)
    assert legacy.max_total_points == 40
    with pytest.raises(ValueError, match="resolution must be one of"):
        resolve_plot_resolution("ultra")
    assert set(PLOT_RESOLUTIONS) == {"fast", "standard", "high", "publication", "full"}


def test_shared_budget_has_a_per_trace_floor():
    standard = PLOT_RESOLUTIONS["standard"]
    assert trace_budget(standard, 1) == 5_000
    assert trace_budget(standard, 15) == 5_000
    squeezed = resolve_plot_resolution("standard", max_points=5_000)
    assert trace_budget(squeezed, 15) == 1_000  # was 333 before #1440
    assert trace_budget(resolve_plot_resolution("full", max_points=40), 1) == 40
    assert trace_budget(PLOT_RESOLUTIONS["full"], 15) is None


def test_many_spectra_do_not_fall_to_a_few_hundred_points():
    recs = {"HOPG bulk": {t: _record(t) for t in range(15, 90, 5)}}
    fig = compare_spectrum_chart(
        recs, _settings(), hue="tilt_deg", include_brem=False, band="broad"
    )

    assert len(fig.data) == 15
    assert min(len(trace.x) for trace in fig.data) >= 4_500


# ---- full resolution ------------------------------------------------------------
def test_full_mode_passes_every_native_coordinate_to_the_renderer():
    rec = _record(15.0)
    store = {"HOPG bulk": {300.0: rec}}

    fig = spectrum_chart(store, _settings(), include_brem=False, resolution="full", band="broad")

    (trace,) = fig.data
    np.testing.assert_array_equal(trace.x, rec["E_grid"])
    expected = rec["spec"] * rec["scale"] * _NA_PER_UA
    np.testing.assert_allclose(trace.y, expected, rtol=0, atol=0)
    assert trace.line.simplify is False
    assert trace.meta == {
        "label": "300.0",
        "native_points": rec["E_grid"].size,
        "displayed_points": rec["E_grid"].size,
    }
    assert "every native point" in fig.layout.title.subtitle.text


def test_decimated_figure_and_export_say_so():
    fig = spectrum_chart({"HOPG bulk": {300.0: _record(15.0)}}, _settings(), resolution="fast")

    note = fig.layout.title.subtitle.text
    assert "fast" in note and "of 20,000 native points" in note and "not interpolated" in note
    assert fig.layout.meta["display_points"]["traces"][0]["native"] == 20_000
    assert "display:" in fig.data[0].hovertemplate


def test_display_never_mutates_source_arrays():
    rec = _record(15.0)
    before = {k: np.copy(v) for k, v in rec.items() if isinstance(v, np.ndarray)}
    for resolution in PLOT_RESOLUTIONS:
        spectrum_chart({"HOPG bulk": {300.0: rec}}, _settings(), resolution=resolution)
    for key, values in before.items():
        np.testing.assert_array_equal(rec[key], values)


# ---- lineshape preservation -----------------------------------------------------
def test_selection_is_stored_nodes_only():
    x = np.linspace(0.0, 2000.0, 50_000)
    y = _lines(x)
    idx = lineshape_indices(x, y, 1_200)

    assert idx.size <= 1_200 and idx[0] == 0 and idx[-1] == x.size - 1
    assert np.all(np.diff(idx) > 0)


@pytest.mark.parametrize("budget", [2_000, 5_000])
def test_minmax_preserves_lineshape_better_than_max_only(budget):
    x = np.linspace(0.0, 2000.0, 100_000)
    y = _lines(x)
    new = lineshape_indices(x, y, budget)
    old = _old_max_only(y, budget)

    assert new.size <= budget
    assert _raster_error(x, y, new) < 0.5 * _raster_error(x, y, old)
    # Overlapping-pair minimum and weak line keep their native extremes.
    dip = (x > 1201.0) & (x < 1204.0)
    assert y[new][dip[new]].min() == pytest.approx(y[dip].min(), rel=0.02)
    weak = (x > 1498.0) & (x < 1502.0)
    assert y[new][weak[new]].max() == y[weak].max()


@pytest.mark.parametrize("log_x", [False, True])
def test_nonuniform_grid_fills_budget_and_keeps_global_peak(log_x):
    x = np.geomspace(10.0, 2000.0, 60_000)
    y = _lines(x)
    idx = lineshape_indices(x, y, 3_000, log_x=log_x)

    assert 2_400 <= idx.size <= 3_000
    assert np.argmax(y) in idx


# ---- viewport-aware windowing (#1442) -------------------------------------------
def test_window_keeps_every_visible_node_when_it_fits():
    x = np.linspace(0.0, 2000.0, 100_000)
    y = _lines(x)
    window = (990.0, 1010.0)
    idx = trace_indices(x, y, np.ones(x.size, bool), 5_000, x_window=window)

    visible = np.flatnonzero((x >= window[0]) & (x <= window[1]))
    assert np.isin(visible, idx).all()
    assert idx.size <= 5_000
    # Coarse context remains on both sides for a client-side zoom-out.
    assert x[idx].min() == 0.0 and x[idx].max() == 2000.0


def test_zooming_progressively_restores_native_sampling():
    x = np.linspace(0.0, 2000.0, 100_000)
    y = _lines(x)
    mask = np.ones(x.size, bool)
    spacing = []
    for half in (1000.0, 100.0, 10.0):
        window = (1000.0 - half, 1000.0 + half)
        idx = trace_indices(x, y, mask, 2_000, x_window=window)
        seen = x[idx][(x[idx] >= window[0]) & (x[idx] <= window[1])]
        spacing.append(np.median(np.diff(seen)))
    assert spacing[0] > spacing[1] > spacing[2] == pytest.approx(x[1] - x[0])


def test_narrowband_chart_spends_budget_on_the_displayed_range():
    rec = _record(15.0, n=10_000)
    frame_window = (900.0, 1100.0)
    fig = spectrum_chart(
        {"HOPG bulk": {300.0: rec}},
        _settings(),
        include_brem=False,
        x_domain=frame_window,
        resolution="fast",
    )
    (trace,) = fig.data
    x = np.asarray(trace.x)
    native_visible = np.count_nonzero(
        (rec["E_grid"] >= frame_window[0]) & (rec["E_grid"] <= frame_window[1])
    )
    shown_visible = np.count_nonzero((x >= frame_window[0]) & (x <= frame_window[1]))
    assert native_visible <= 2_000 and shown_visible == native_visible


def test_frame_default_stays_native():
    rec = _record(15.0)
    frame = spectrum_frame([rec], _settings(), include_brem=False)
    assert len(frame) == rec["E_grid"].size


def test_window_between_two_samples_keeps_the_bracketing_pair():
    x = np.arange(0.0, 2000.0, 5.0)
    y = _lines(x)
    idx = trace_indices(x, y, np.ones(x.size, bool), 50, x_window=(1001.0, 1003.0))

    assert {200, 201} <= set(idx.tolist())  # 1000 and 1005 eV
    assert idx.size <= 50


def test_log_axis_after_linear_zoom_below_zero_does_not_raise():
    from pyrite.plots.plotly.spectra import prepare_compare_spectra, render_prepared_spectra

    prepared = prepare_compare_spectra(
        {"HOPG bulk": {15.0: _record(15.0)}}, _settings(), hue="tilt_deg", band="broad"
    )
    fig = render_prepared_spectra(prepared, viewport=(-40.0, 600.0), x_type="log")

    low, high = fig.layout.xaxis.range
    assert low == pytest.approx(np.log10(50.0)) and high == pytest.approx(np.log10(600.0))


def test_decimated_traces_disable_plotly_simplify_and_note_is_per_trace():
    recs = [_record(15.0), _record(15.0)]
    recs[1]["case"] = {**recs[1]["case"], "tilt_azim_deg": 90.0}
    fig = spectrum_chart(
        {"HOPG bulk": {300.0: recs[0], 301.0: recs[1]}},
        _settings(),
        include_brem=False,
        collapse_azimuth=False,
        resolution="fast",
        band="broad",
    )
    assert all(trace.line.simplify is False for trace in fig.data)
    assert "of 20,000 native points per trace" in fig.layout.title.subtitle.text
