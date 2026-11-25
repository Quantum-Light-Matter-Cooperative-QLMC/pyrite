"""Guard tests for the hue-generalized Plotly overlays
(pyrite.plots.plotly.spectra.compare_spectrum_chart and
multi_case_spectrum_chart) backing the analysis app's comparison tabs and case
basket. Frame-level selection rules are checked on the shared
``pyrite.plots._spectrum_frames`` frames. Synthetic records; no GPU.
"""

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest

from pyrite.plots._spectrum_frames import _compare_frame
from pyrite.plots.plotly.spectra import (
    compare_spectrum_chart,
    multi_case_spectrum_chart,
    spectrum_chart,
)


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _cxr_brem/detected_background on the pure-numpy path.
    return SimpleNamespace(
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",
    )


def _record(E0_keV, tilt_deg, azim_deg, peak_center=2500.0, n=200, wide_brem=False):
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - peak_center) / 50.0) ** 2))  # a single sharp line
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
        rec["brem_wide"] = np.linspace(1.2, 0.0, Eb.size)
    return rec


def _store_single_tilt():
    # {name: {E0: record}} -- one polar tilt, two beam energies (mirrors
    # test_plotly_spectra._store()).
    return {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0),
            60.0: _record(60.0, 20.0, 0.0),
        }
    }


def _store_multi_tilt(wide_brem=False):
    # One beam energy, three polar tilts (single azimuth each) -- the shape a
    # "Polar-angle comparison" tab slice looks like.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, 40.0, 0.0, wide_brem=wide_brem),
            30.1: _record(30.0, 20.0, 0.0, wide_brem=wide_brem),
            30.2: _record(30.0, 0.0, 0.0, wide_brem=wide_brem),
        }
    }


def _store_multi_azim(wide_brem=False):
    # One beam energy, one polar tilt, three azimuths -- the "Azimuthal
    # comparison" tab slice shape.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0, wide_brem=wide_brem),
            30.1: _record(30.0, 20.0, 45.0, wide_brem=wide_brem),
            30.2: _record(30.0, 20.0, 90.0, wide_brem=wide_brem),
        }
    }


def _labels(fig):
    return {trace.meta["label"] for trace in fig.data}


def _components(fig):
    return {trace.name.split(" · ")[-1] if " · " in trace.name else "total" for trace in fig.data}


def _frame(results, **kw):
    recs = [r for by_energy in results.values() for r in by_energy.values()]
    kw.setdefault("include_brem", True)
    kw.setdefault("include_cxr", False)
    kw.setdefault("include_characteristic", True)
    kw.setdefault("include_channeling", True)
    kw.setdefault("include_transition", True)
    kw.setdefault("include_coherent", False)
    kw.setdefault("sum_components", False)
    kw.setdefault("band", "narrow")
    kw.setdefault("max_points", 5000)
    return _compare_frame(recs, _settings(), **kw)


def _total_y(fig):
    (trace,) = fig.data
    return np.asarray(trace.y, dtype=float)


def test_compare_chart_one_line_per_hue_value_tilt():
    fig = compare_spectrum_chart(_store_multi_tilt(), _settings(), hue="tilt_deg")
    assert isinstance(fig, go.Figure)
    assert _labels(fig) == {"40", "20", "0"}
    assert fig.layout.legend.title.text == r"$\theta_{\mathrm{tilt}}\ (^\circ)$"
    # The title pins only what is common: no single tilt for a tilt comparison.
    title = fig.layout.title.text
    assert title.startswith("$") and title.endswith("$")
    assert r"\theta" not in title and r"\phi_{\mathrm{tilt}} = 0^\circ" in title
    assert fig.layout.title.subtitle.text is None


def test_compare_chart_one_line_per_hue_value_azimuth():
    fig = compare_spectrum_chart(_store_multi_azim(), _settings(), hue="tilt_azim_deg")
    assert _labels(fig) == {"0", "45", "90"}


def test_compare_frame_carries_every_hue_field():
    df = _frame(_store_multi_tilt(), hue="tilt_deg")
    for col in ("energy_eV", "intensity", "E0_keV", "tilt_deg", "tilt_azim_deg", "component"):
        assert col in df.columns


@pytest.mark.parametrize(
    ("store", "hue", "values"),
    [
        (_store_multi_tilt, "tilt_deg", {"40", "20", "0"}),
        (_store_multi_azim, "tilt_azim_deg", {"0", "45", "90"}),
    ],
)
def test_compare_chart_broadband_uses_wide_brem_tail(store, hue, values):
    fig = compare_spectrum_chart(store(wide_brem=True), _settings(), hue=hue, band="broad")
    totals = [t for t in fig.data if t.name.endswith("· total")]
    assert {t.meta["label"] for t in totals if max(t.x) > 5000.0} == values
    assert max(max(t.x) for t in fig.data) == 30000.0


def test_compare_chart_none_on_empty():
    assert compare_spectrum_chart({}, _settings(), hue="E0_keV") is None


def test_selected_components_do_not_expand_dense_chart_payload():
    """Selecting Line changes the total without adding a trace."""
    store = {"HOPG bulk": {30.0: _record(30.0, 20.0, 0.0, n=8651)}}
    for include_cxr in (False, True):
        fig = compare_spectrum_chart(
            store,
            _settings(),
            hue="E0_keV",
            include_brem=True,
            include_cxr=include_cxr,
            include_characteristic=False,
            sum_components=True,
        )
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 5000


def test_selected_components_make_one_additive_spectrum():
    record = _record(30.0, 20.0, 0.0, n=80)
    record["spec_characteristic"] = np.full(80, 0.4)
    store = {"HOPG bulk": {30.0: record}}

    def total(*, brem: bool, line: bool, characteristic: bool):
        return _total_y(
            compare_spectrum_chart(
                store,
                _settings(),
                hue="E0_keV",
                include_brem=brem,
                include_cxr=line,
                include_characteristic=characteristic,
                sum_components=True,
            )
        )

    brem = total(brem=True, line=False, characteristic=False)
    line = total(brem=False, line=True, characteristic=False)
    characteristic = total(brem=False, line=False, characteristic=True)
    combined = total(brem=True, line=True, characteristic=True)

    assert np.max(characteristic) > 0
    np.testing.assert_allclose(combined, brem + line + characteristic)


def test_case_basket_selected_components_have_one_trace():
    record = _record(30.0, 20.0, 0.0, n=80)
    fig = multi_case_spectrum_chart(
        [(record, "HOPG")],
        _settings(),
        include_brem=True,
        include_cxr=True,
        include_characteristic=False,
        sum_components=True,
    )
    assert len(fig.data) == 1


def test_both_emissions_sum_the_same_selected_components():
    record = _record(30.0, 20.0, 0.0, n=80)
    record["spec_coherent"] = record["spec"] * 2.0
    record["spec_characteristic"] = np.full(80, 0.4)
    store = {"HOPG bulk": {30.0: record}}

    both = compare_spectrum_chart(
        store,
        _settings(),
        hue="E0_keV",
        include_brem=True,
        include_cxr=True,
        include_characteristic=True,
        include_coherent=True,
        sum_components=True,
    )
    line = compare_spectrum_chart(
        store,
        _settings(),
        hue="E0_keV",
        include_brem=False,
        include_cxr=True,
        include_characteristic=False,
        sum_components=True,
    )
    assert len(both.data) == 2
    by_component = {t.name.split(" · ")[-1]: np.asarray(t.y, dtype=float) for t in both.data}
    np.testing.assert_allclose(
        by_component["coherent"] - by_component["total"], _total_y(line), atol=1e-12
    )


def test_compare_chart_single_hue_value_still_renders_one_line():
    fig = compare_spectrum_chart(_store_single_tilt(), _settings(), hue="tilt_deg")
    assert _labels(fig) == {"20"}


def test_compare_chart_rejects_unknown_hue():
    with pytest.raises(ValueError):
        compare_spectrum_chart(_store_single_tilt(), _settings(), hue="not_a_field")


def test_compare_duplicate_hue_values_collapse_to_max_peak():
    # Two records share tilt_deg=20 (different azimuths); the stronger peak wins.
    store = {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0, peak_center=2500.0),
            30.1: _record(30.0, 20.0, 45.0, peak_center=2500.0),
        }
    }
    store["HOPG bulk"][30.1]["spec"] = store["HOPG bulk"][30.1]["spec"] * 5.0
    df = _frame(store, hue="tilt_deg")
    assert set(df["tilt_deg"]) == {20.0}
    assert set(df["tilt_azim_deg"]) == {45.0}
    fig = compare_spectrum_chart(store, _settings(), hue="tilt_deg")
    assert len([t for t in fig.data if t.name.endswith("· total")]) == 1


def test_compare_characteristic_peak_can_reverse_duplicate_hue_winner():
    store = {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0, peak_center=2500.0),
            30.1: _record(30.0, 20.0, 45.0, peak_center=2500.0),
        }
    }
    first = store["HOPG bulk"][30.0]
    second = store["HOPG bulk"][30.1]
    second["spec"] = second["spec"] * 0.5
    first["spec_characteristic"] = np.zeros_like(first["spec"])
    second["spec_characteristic"] = second["spec"] * 4.0

    with_char = _frame(store, hue="tilt_deg", include_characteristic=True)
    without_char = _frame(store, hue="tilt_deg", include_characteristic=False)
    assert set(with_char["tilt_azim_deg"]) == {45.0}
    assert set(without_char["tilt_azim_deg"]) == {0.0}


# ---- correctness oracle -------------------------------------------------------
def test_compare_chart_matches_spectrum_chart_for_hue_E0_single_tilt():
    """compare_spectrum_chart(hue="E0_keV") on a single-tilt slice reproduces the
    same traces as spectrum_chart on that slice."""
    store = _store_single_tilt()
    orig = spectrum_chart(store, _settings())
    cmp = compare_spectrum_chart(store, _settings(), hue="E0_keV")
    assert [t.name for t in orig.data] == [t.name for t in cmp.data]
    for a, b in zip(orig.data, cmp.data, strict=True):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.y, b.y)


# ---- multi_case_spectrum_chart ------------------------------------------------
def test_multi_case_chart_none_on_empty():
    assert multi_case_spectrum_chart([], _settings()) is None


def test_multi_case_chart_single_case():
    fig = multi_case_spectrum_chart([(_record(30.0, 20.0, 0.0), "HOPG - 30 keV")], _settings())
    assert isinstance(fig, go.Figure)
    assert _labels(fig) == {"HOPG - 30 keV"}


def test_multi_case_chart_distinct_entries_survive_duplicate_fields():
    """Two entries with identical case fields both survive (no collapse to the
    strongest peak) -- what distinguishes the basket from compare_spectrum_chart."""
    cases = [
        (_record(30.0, 20.0, 0.0), "copy A"),
        (_record(30.0, 20.0, 0.0), "copy B"),
    ]
    assert _labels(multi_case_spectrum_chart(cases, _settings())) == {"copy A", "copy B"}


def test_multi_case_chart_repeated_labels_stay_distinct():
    cases = [(_record(30.0, 20.0, 0.0), "same"), (_record(30.0, 40.0, 0.0), "same")]
    assert _labels(multi_case_spectrum_chart(cases, _settings())) == {"same", "same (#2)"}


def test_multi_case_chart_legend_titles_cases():
    cases = [
        (_record(30.0, 20.0, 0.0), "HOPG - 30 keV"),
        (_record(60.0, 20.0, 0.0), "HOPG - 60 keV"),
    ]
    fig = multi_case_spectrum_chart(cases, _settings())
    assert fig.layout.legend.title.text == "case"


def test_multi_case_chart_brem_toggle():
    cases = [(_record(30.0, 20.0, 0.0), "HOPG - 30 keV")]
    assert len(multi_case_spectrum_chart(cases, _settings(), include_brem=True).data) == 2
    assert len(multi_case_spectrum_chart(cases, _settings(), include_brem=False).data) == 1


def test_multi_case_chart_broadband_wide_brem_tail():
    cases = [
        (_record(30.0, 20.0, 0.0, wide_brem=True), "HOPG - 30 keV broad"),
        (_record(30.0, 40.0, 0.0, wide_brem=True), "HOPG - 40 deg broad"),
    ]
    fig = multi_case_spectrum_chart(cases, _settings(), band="broad")
    totals = [t for t in fig.data if t.name.endswith("· total") and max(t.x) > 5000.0]
    assert {t.meta["label"] for t in totals} == {"HOPG - 30 keV broad", "HOPG - 40 deg broad"}
    assert max(max(t.x) for t in fig.data) == 30000.0


def test_multi_case_chart_mixed_material_labels():
    rec_hopg = _record(30.0, 20.0, 0.0)
    rec_sic = _record(30.0, 20.0, 0.0, peak_center=3000.0)
    rec_sic["case"]["name"] = "SiC bulk"
    rec_sic["spec"] = rec_sic["spec"] * 1.5
    fig = multi_case_spectrum_chart(
        [(rec_hopg, "HOPG - 30 keV"), (rec_sic, "SiC - 30 keV")], _settings()
    )
    assert _labels(fig) == {"HOPG - 30 keV", "SiC - 30 keV"}
