"""Guard tests for cxr_mc.plots.altair_spectra.compare_spectrum_chart -- the
hue-generalized overlay chart backing the notebook's Polar-angle/Azimuthal
comparison tabs (Energy comparison reuses the original spectrum_chart).

Same synthetic-record approach as test_altair_plots.py: small charts, no GPU,
no checkpoint. pyarrow-before-cxr_mc DLL ordering is handled globally by
tests/conftest.py.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from cxr_mc.plots.altair_spectra import compare_spectrum_chart, spectrum_chart


def _dataset(spec):
    """The layered chart's shared row data (a list of dict rows) from a
    ``Chart.to_dict()`` spec -- layers share one inline dataset referenced by
    name at the TOP level (``spec["data"]["name"]``), not per-layer."""
    return spec["datasets"][spec["data"]["name"]]


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _line_brem/detected_background on the pure-numpy path.
    return SimpleNamespace(
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",
    )


def _record(E0_keV, tilt_deg, azim_deg, peak_center=2500.0, n=200):
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - peak_center) / 50.0) ** 2))  # a single sharp line
    brem = np.linspace(1.0, 0.2, n)  # a smooth falling continuum
    return {
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


def _store_single_tilt():
    # {name: {E0: record}} -- one polar tilt, two beam energies (mirrors
    # test_altair_plots._store()).
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0),
            60.0: _record(60.0, -20.0, 0.0),
        }
    }


def _store_multi_tilt():
    # One beam energy, three polar tilts (single azimuth each) -- the shape a
    # "Polar-angle comparison" tab slice looks like.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -40.0, 0.0),
            30.1: _record(30.0, -20.0, 0.0),
            30.2: _record(30.0, 0.0, 0.0),
        }
    }


def _store_multi_azim():
    # One beam energy, one polar tilt, three azimuths -- the "Azimuthal
    # comparison" tab slice shape.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0),
            30.1: _record(30.0, -20.0, 45.0),
            30.2: _record(30.0, -20.0, 90.0),
        }
    }


def test_compare_chart_one_line_per_hue_value_tilt():
    chart = compare_spectrum_chart(_store_multi_tilt(), _settings(), hue="tilt_deg")
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    df = _dataset(spec)
    tilts = {row["tilt_deg"] for row in df}
    assert tilts == {-40.0, -20.0, 0.0}
    enc = spec["layer"][0]["encoding"]
    assert enc["color"]["field"] == "tilt_deg"


def test_compare_chart_one_line_per_hue_value_azimuth():
    chart = compare_spectrum_chart(_store_multi_azim(), _settings(), hue="tilt_azim_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    azims = {row["tilt_azim_deg"] for row in df}
    assert azims == {0.0, 45.0, 90.0}
    enc = spec["layer"][0]["encoding"]
    assert enc["color"]["field"] == "tilt_azim_deg"


def test_compare_chart_hue_column_present_for_every_field():
    chart = compare_spectrum_chart(_store_multi_tilt(), _settings(), hue="tilt_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    row = df[0]
    for col in ("energy_eV", "intensity", "E0_keV", "tilt_deg", "tilt_azim_deg", "component"):
        assert col in row


def test_compare_chart_none_on_empty():
    assert compare_spectrum_chart({}, _settings(), hue="E0_keV") is None


def test_compare_chart_single_hue_value_still_renders_one_line():
    chart = compare_spectrum_chart(_store_single_tilt(), _settings(), hue="tilt_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    assert {row["tilt_deg"] for row in df} == {-20.0}


def test_compare_chart_rejects_unknown_hue():
    import pytest

    with pytest.raises(ValueError):
        compare_spectrum_chart(_store_single_tilt(), _settings(), hue="not_a_field")


def test_compare_chart_duplicate_hue_values_collapse_to_max_peak():
    # Two records sharing tilt_deg=-20 (different azimuths) -- the stronger-peak
    # one should win, giving exactly ONE line for that hue value.
    store = {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0, peak_center=2500.0),
            30.1: _record(30.0, -20.0, 45.0, peak_center=2500.0),
        }
    }
    # Boost the second record's peak so the selection is unambiguous.
    store["HOPG bulk"][30.1]["spec"] = store["HOPG bulk"][30.1]["spec"] * 5.0
    chart = compare_spectrum_chart(store, _settings(), hue="tilt_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    tilts = {row["tilt_deg"] for row in df}
    assert tilts == {-20.0}
    azims = {row["tilt_azim_deg"] for row in df}
    assert azims == {45.0}  # the boosted (stronger-peak) record's azimuth won


# ---- correctness oracle -------------------------------------------------------
def test_compare_chart_matches_spectrum_chart_for_hue_E0_single_tilt():
    """compare_spectrum_chart(hue="E0_keV") on a single-tilt slice must reproduce
    the SAME data as spectrum_chart on that slice -- the generalization is
    faithful, not just structurally similar."""
    store = _store_single_tilt()
    settings = _settings()

    orig = spectrum_chart(store, settings)
    cmp = compare_spectrum_chart(store, settings, hue="E0_keV")

    orig_spec = orig.to_dict()
    cmp_spec = cmp.to_dict()
    orig_df = _dataset(orig_spec)
    cmp_df = _dataset(cmp_spec)

    def _by_energy_component(rows):
        out = {}
        for r in rows:
            out[(r["E0_keV"], r["component"], round(r["energy_eV"], 6))] = r["intensity"]
        return out

    orig_map = _by_energy_component(orig_df)
    cmp_map = _by_energy_component(cmp_df)
    assert set(orig_map) == set(cmp_map)
    for key, val in orig_map.items():
        assert val == cmp_map[key]
