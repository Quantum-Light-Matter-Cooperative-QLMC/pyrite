"""Guard tests for the Plotly spectrum renderer (pyrite.plots.plotly.spectra).

The renderer shares its physics/data prep (``pyrite.plots._spectrum_frames``)
with the Matplotlib path. Synthetic records only (no GPU, no checkpoint).
"""

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from pyrite.plots._spectrum_frames import _NA_PER_UA, spectrum_frame
from pyrite.plots.plotly.spectra import spectrum_chart


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _cxr_brem/detected_background on the pure-numpy path.
    return SimpleNamespace(
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",
    )


def _record(E0_keV, tilt_deg, azim_deg, n=200, wide_brem=False):
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - 2500.0) / 50.0) ** 2))  # a single sharp line
    brem = np.linspace(1.0, 0.2, n)  # a smooth falling continuum
    rec = {
        "E_grid": E,
        "spec": peak,
        "brem": brem,
        "scale": 2.0,
        "fwhm": 30.0,
        "case": {
            "name": "HOPG bulk",
            "E0_keV": E0_keV,
            "tilt_deg": tilt_deg,
            "tilt_azim_deg": azim_deg,
            "thickness_ang": 5.0e4,
        },
    }
    if wide_brem:
        Eb = np.linspace(0.0, E0_keV * 1000.0, 120)
        rec["E_grid_brem"] = Eb
        rec["brem_wide"] = np.linspace(1.2, 0.02, Eb.size)
    return rec


def _store():
    # {name: {E0: record}} -- one polar tilt, two beam energies.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0),
            60.0: _record(60.0, -20.0, 0.0),
        }
    }


def _components(fig):
    return {trace.name.split(" · ")[-1] for trace in fig.data}


def _trace(fig, label, component):
    return next(t for t in fig.data if t.name == f"{label} · {component}")


# ---- renderer-neutral frames ---------------------------------------------------
def test_spectrum_frame_shapes_and_components():
    df = spectrum_frame([_record(30.0, -20.0, 0.0), _record(60.0, -20.0, 0.0)], _settings())
    assert list(df.columns) == ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
    # 2 energies x 2 components x 200 grid points
    assert len(df) == 2 * 2 * 200
    assert set(df["component"]) == {"total", "brem"}
    assert sorted(df["E0_keV"].unique()) == [30.0, 60.0]
    # total = line + brem >= brem everywhere (line is non-negative)
    assert (
        df[df.component == "total"]["intensity"].to_numpy()
        >= df[df.component == "brem"]["intensity"].to_numpy() - 1e-9
    ).all()


def test_spectrum_frame_broadband_uses_wide_brem_tail():
    df = spectrum_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings(), band="broad")

    assert df["energy_eV"].max() == 30000.0
    tail = df[(df.component == "total") & (df.energy_eV > 5000.0)]
    brem_tail = df[(df.component == "brem") & (df.energy_eV > 5000.0)]
    assert not tail.empty
    assert np.allclose(tail["intensity"].to_numpy(), brem_tail["intensity"].to_numpy())


def test_spectrum_frame_broadband_clips_tail_to_beam_energy():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["E_grid_brem"] = np.array([0.0, 5000.0, 30000.0, 30500.0])
    rec["brem_wide"] = np.array([1.0, 0.5, 0.2, 99.0])

    df = spectrum_frame([rec], _settings(), band="broad")

    assert df["energy_eV"].max() == 30000.0
    assert 30500.0 not in set(df["energy_eV"])


def test_spectrum_frame_narrow_extends_with_wide_brem_tail():
    # A narrow chart must not cut off abruptly at the line grid's own edge --
    # it should keep drawing the stored continuum past that point too, same
    # as "broad" (with zero line contribution there, since the line grid
    # doesn't extend that far).
    df = spectrum_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings())

    assert df["energy_eV"].max() == 30000.0
    tail = df[(df.component == "total") & (df.energy_eV > 5000.0)]
    brem_tail = df[(df.component == "brem") & (df.energy_eV > 5000.0)]
    assert not tail.empty
    assert np.allclose(tail["intensity"].to_numpy(), brem_tail["intensity"].to_numpy())


def test_spectrum_frame_peak_preserving_decimation_keeps_line_peak():
    rec = _record(30.0, -20.0, 0.0, n=1000)
    df = spectrum_frame([rec], _settings(), include_brem=False, max_points=40)

    total = df[df.component == "total"]
    assert len(total) <= 40
    # frames report per-uA: the stored per-nA scale times _NA_PER_UA
    assert np.isclose(total["intensity"].max(), rec["spec"].max() * rec["scale"] * _NA_PER_UA)


def test_spectrum_frame_brem_toggle_preserves_total_sampling_budget():
    rec = _record(30.0, -20.0, 0.0, n=1200)

    cxr_only = spectrum_frame([rec], _settings(), include_brem=False, max_points=120)
    with_brem = spectrum_frame([rec], _settings(), include_brem=True, max_points=120)

    line_energy = cxr_only.loc[cxr_only.component == "total", "energy_eV"].to_numpy()
    total_energy = with_brem.loc[with_brem.component == "total", "energy_eV"].to_numpy()
    np.testing.assert_array_equal(total_energy, line_energy)


def test_spectrum_frame_wide_brem_preserves_line_window_resolution():
    rec = _record(30.0, -20.0, 0.0, n=12000, wide_brem=True)

    cxr_only = spectrum_frame([rec], _settings(), include_brem=False, max_points=500)
    with_brem = spectrum_frame([rec], _settings(), include_brem=True, max_points=500)

    def _line_window(df):
        total = df[df.component == "total"]
        return total.loc[total.energy_eV.between(2400.0, 2600.0), "energy_eV"].to_numpy()

    line_energy = _line_window(cxr_only)
    total_energy = _line_window(with_brem)
    line_shape = np.interp(line_energy, rec["E_grid"], rec["spec"])
    total_shape = np.interp(total_energy, rec["E_grid"], rec["spec"])

    def _sampled_fwhm(energy, intensity):
        above_half = energy[intensity >= 0.5 * intensity.max()]
        return above_half[-1] - above_half[0]

    assert np.min(np.abs(total_energy - 2500.0)) == np.min(np.abs(line_energy - 2500.0))
    assert total_shape.max() == line_shape.max()
    # The tail keeps a tenth of the budget, so line buckets differ slightly in
    # width; the sampled FWHM may move by at most one sampled step.
    assert (
        abs(_sampled_fwhm(total_energy, total_shape) - _sampled_fwhm(line_energy, line_shape))
        <= np.diff(line_energy).max()
    )
    assert np.mean(np.diff(total_energy)) <= 1.15 * np.mean(np.diff(line_energy))
    assert np.diff(total_energy).max() <= 1.15 * np.diff(line_energy).max()


def test_spectrum_frame_keeps_full_line_grid_before_decimating_wide_tail():
    rec = _record(30.0, -20.0, 0.0, n=200, wide_brem=True)

    df = spectrum_frame([rec], _settings(), include_brem=True, max_points=250)

    total = df[df.component == "total"]
    line_region = total[total.energy_eV <= rec["E_grid"].max()]
    np.testing.assert_array_equal(line_region.energy_eV.to_numpy(), rec["E_grid"])
    assert len(total) == 250
    assert total.energy_eV.max() == 30000.0


def test_spectrum_frame_excludes_brem_when_disabled():
    df = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings(), include_brem=False)
    assert set(df["component"]) == {"total"}


def test_spectrum_frame_excludes_characteristic_radiation_when_disabled():
    rec = _record(30.0, -20.0, 0.0)
    characteristic = 0.25 * rec["spec"]
    rec["spec_characteristic"] = characteristic
    rec["spec"] = rec["spec"] + characteristic

    with_characteristic = spectrum_frame(
        [rec],
        _settings(),
        include_brem=False,
        include_characteristic=True,
    )
    without_characteristic = spectrum_frame(
        [rec],
        _settings(),
        include_brem=False,
        include_characteristic=False,
    )

    with_total = with_characteristic.loc[
        with_characteristic.component == "total", "intensity"
    ].to_numpy()
    without_total = without_characteristic.loc[
        without_characteristic.component == "total", "intensity"
    ].to_numpy()
    np.testing.assert_allclose(
        with_total - without_total,
        characteristic * rec["scale"] * _NA_PER_UA,
    )


def test_spectrum_frame_adds_line_component_without_touching_brem():
    df = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings(), include_cxr=True)
    assert set(df["component"]) == {"total", "brem", "line"}
    line = df[df.component == "line"]["intensity"].to_numpy()
    total_no_line = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings())
    brem_only = total_no_line[total_no_line.component == "brem"]["intensity"].to_numpy()
    total_only = total_no_line[total_no_line.component == "total"]["intensity"].to_numpy()
    # total == line + brem, so the line-only trace equals total minus brem
    # (atol covers float64 cancellation noise near the ~0 tail of the line).
    np.testing.assert_allclose(line, total_only - brem_only, atol=1e-12)


def test_spectrum_frame_coherent_overlay_absent_without_spec_coherent():
    df = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings(), include_coherent=True)
    assert "coherent" not in set(df["component"])


def test_spectrum_frame_coherent_overlay_present_with_spec_coherent():
    rec = _record(30.0, -20.0, 0.0)
    rec["spec_coherent"] = rec["spec"] * 2.0
    df = spectrum_frame([rec], _settings(), include_coherent=True)
    coherent = df[df.component == "coherent"]["intensity"].to_numpy()
    line = spectrum_frame([rec], _settings(), include_cxr=True)
    line = line[line.component == "line"]["intensity"].to_numpy()
    # spec_coherent is exactly 2x spec, so its detected line is 2x too.
    np.testing.assert_allclose(coherent, line * 2.0)


# ---- Plotly chart ----------------------------------------------------------------
def test_narrow_chart_auto_domain_uses_widest_line_grid():
    narrow = _record(30.0, -20.0, 0.0, wide_brem=True)
    widest = _record(60.0, -20.0, 0.0, wide_brem=True)
    widest["E_grid"] = np.linspace(500.0, 7500.0, 200)
    store = {"HOPG bulk": {30.0: narrow, 60.0: widest}}

    narrow_chart = spectrum_chart(store, _settings(), band="narrow")
    broad_chart = spectrum_chart(store, _settings(), band="broad")

    assert list(narrow_chart.layout.xaxis.range) == [500.0, 7500.0]
    assert broad_chart.layout.xaxis.range is None


def test_spectrum_chart_one_trace_per_energy_and_component():
    fig = spectrum_chart(_store(), _settings())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 2 beam energies x (total, brem)
    assert _components(fig) == {"total", "brem"}
    assert fig.layout.xaxis.title.text == "Photon energy (eV)"
    assert fig.layout.yaxis.title.text == "Intensity (Phs/s/eV/uA)"
    assert fig.layout.legend.title.text == "beam energy (keV)"
    fig.to_json()


def test_spectrum_chart_traces_equal_the_shared_frame():
    records = [_store()["HOPG bulk"][30.0], _store()["HOPG bulk"][60.0]]
    frame = spectrum_frame(records, _settings(), max_points=5000)
    fig = spectrum_chart(_store(), _settings())
    for (energy, component), part in frame.groupby(["E0_keV", "component"]):
        part = part.sort_values("energy_eV")
        trace = _trace(fig, energy, component)
        np.testing.assert_array_equal(trace.x, part["energy_eV"].to_numpy())
        np.testing.assert_array_equal(trace.y, part["intensity"].to_numpy())


def test_spectrum_chart_components_share_coordinates_within_budget():
    rec = _record(30.0, -20.0, 0.0, n=1200)
    fig = spectrum_chart({"HOPG bulk": {30.0: rec}}, _settings(), include_brem=True, max_points=120)
    total, brem = _trace(fig, 30.0, "total"), _trace(fig, 30.0, "brem")
    assert len(total.x) <= 120
    np.testing.assert_array_equal(total.x, brem.x)


def test_spectrum_chart_accepts_log_x_scale_and_broadband():
    store = {"HOPG bulk": {30.0: _record(30.0, -20.0, 0.0, wide_brem=True)}}
    fig = spectrum_chart(
        store,
        _settings(),
        band="broad",
        x_domain=(50.0, 30000.0),
        x_type="log",
        y_type="log",
    )
    assert fig.layout.xaxis.type == "log"
    assert list(fig.layout.xaxis.range) == pytest.approx([np.log10(50.0), np.log10(30000.0)])
    assert fig.layout.yaxis.type == "log"
    assert max(max(trace.x) for trace in fig.data) == 30000.0


def test_spectrum_chart_broadband_log_y_floors_range_and_masks_zero_only_in_display():
    # The wide brem tail tapers to exactly 0 at the radiative endpoint; a log
    # axis cannot show it. The range floor must stay positive and the zero must
    # be masked in the displayed trace, never in the source frame.
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["brem_wide"] = np.linspace(1.2, 0.0, rec["E_grid_brem"].size)  # tapers to 0
    store = {"HOPG bulk": {30.0: rec}}

    fig = spectrum_chart(store, _settings(), band="broad", y_type="log")

    assert fig.layout.yaxis.type == "log"
    assert np.isfinite(fig.layout.yaxis.range[0])
    frame = spectrum_frame([rec], _settings(), band="broad", max_points=5000)
    assert (frame["intensity"] == 0.0).any()  # the source really hits 0
    brem = np.asarray(_trace(fig, 30.0, "brem").y, dtype=float)
    assert np.isnan(brem).any() and not (brem == 0.0).any()


def test_spectrum_chart_log_y_floor_rises_for_narrowed_x_window():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    store = {"HOPG bulk": {30.0: rec}}

    full = spectrum_chart(store, _settings(), band="broad", y_type="log")
    narrow = spectrum_chart(
        store, _settings(), band="broad", y_type="log", x_domain=(2400.0, 2600.0)
    )
    assert narrow.layout.yaxis.range[0] > full.layout.yaxis.range[0]


def test_spectrum_chart_linear_y_range_matches_visible_window():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    store = {"HOPG bulk": {30.0: rec}}
    x_domain = (2400.0, 2600.0)

    fig = spectrum_chart(store, _settings(), band="broad", y_type="linear", x_domain=x_domain)

    df = spectrum_frame([rec], _settings(), band="broad", max_points=5000)
    windowed = df[(df.energy_eV >= x_domain[0]) & (df.energy_eV <= x_domain[1])]
    assert fig.layout.yaxis.range[0] == 0.0
    assert fig.layout.yaxis.range[1] == pytest.approx(windowed["intensity"].max() * 1.05)


def test_spectrum_chart_linear_y_range_auto_when_no_x_domain():
    # Broadband without an x-domain leaves the y range to Plotly; the narrow band
    # always pins x to the line window, so its y range fits that window instead.
    assert spectrum_chart(_store(), _settings(), band="broad").layout.yaxis.range is None
    narrow = spectrum_chart(_store(), _settings())
    assert narrow.layout.xaxis.range is not None and narrow.layout.yaxis.range is not None


def test_spectrum_chart_total_only_without_brem():
    fig = spectrum_chart(_store(), _settings(), include_brem=False)
    assert [trace.name for trace in fig.data] == ["30.0", "60.0"]


def test_spectrum_chart_none_on_empty():
    assert spectrum_chart({}, _settings()) is None


def test_spectrum_chart_adds_line_component():
    assert _components(spectrum_chart(_store(), _settings(), include_cxr=True)) == {
        "total",
        "brem",
        "line",
    }


def test_spectrum_chart_adds_coherent_component_when_requested():
    rec = _record(30.0, -20.0, 0.0)
    rec["spec_coherent"] = rec["spec"] * 2.0
    store_with_coherent = {"HOPG bulk": {30.0: rec}}

    fig = spectrum_chart(store_with_coherent, _settings(), include_coherent=True)
    assert _components(fig) == {"total", "brem", "coherent"}
    # Without the flag, no coherent trace even when the data has it.
    assert _components(spectrum_chart(store_with_coherent, _settings())) == {"total", "brem"}
