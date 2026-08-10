"""Guard tests for pyrite.plots.altair.spectra.compare_spectrum_chart -- the
hue-generalized overlay chart backing the notebook's Polar-angle/Azimuthal
comparison tabs (Energy comparison reuses the original spectrum_chart).

Same synthetic-record approach as test_altair_plots.py: small charts, no GPU,
no checkpoint. pyarrow-before-pyrite DLL ordering is handled globally by
tests/conftest.py.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from pyrite.plots.altair.spectra import (
    compare_spectrum_chart,
    multi_case_spectrum_chart,
    spectrum_chart,
)


def _dataset(spec):
    """The layered chart's shared row data (a list of dict rows) from a
    ``Chart.to_dict()`` spec -- layers share one inline dataset referenced by
    name at the TOP level (``spec["data"]["name"]``), not per-layer."""
    rows = spec["datasets"][spec["data"]["name"]]
    if not rows or "component" in rows[0]:
        return rows
    components = [name for name in ("total", "brem") if name in rows[0]]
    return [
        {
            **{key: value for key, value in row.items() if key not in components},
            "component": component,
            "intensity": row[component],
        }
        for row in rows
        for component in components
    ]


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _line_brem/detected_background on the pure-numpy path.
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
    # test_altair_plots._store()).
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


def test_compare_chart_one_line_per_hue_value_tilt():
    chart = compare_spectrum_chart(_store_multi_tilt(), _settings(), hue="tilt_deg")
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    df = _dataset(spec)
    tilts = {row["tilt_deg"] for row in df}
    assert tilts == {40.0, 20.0, 0.0}
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


def test_compare_chart_broadband_uses_wide_brem_tail_for_tilts():
    chart = compare_spectrum_chart(
        _store_multi_tilt(wide_brem=True),
        _settings(),
        hue="tilt_deg",
        band="broad",
    )
    spec = chart.to_dict()
    df = _dataset(spec)
    tail = [row for row in df if row["component"] == "total" and row["energy_eV"] > 5000.0]

    assert tail
    assert max(row["energy_eV"] for row in df) == 30000.0
    assert {row["tilt_deg"] for row in tail} == {40.0, 20.0, 0.0}


def test_compare_chart_broadband_uses_wide_brem_tail_for_azimuths():
    chart = compare_spectrum_chart(
        _store_multi_azim(wide_brem=True),
        _settings(),
        hue="tilt_azim_deg",
        band="broad",
    )
    spec = chart.to_dict()
    df = _dataset(spec)
    tail = [row for row in df if row["component"] == "total" and row["energy_eV"] > 5000.0]

    assert tail
    assert max(row["energy_eV"] for row in df) == 30000.0
    assert {row["tilt_azim_deg"] for row in tail} == {0.0, 45.0, 90.0}


def test_compare_chart_none_on_empty():
    assert compare_spectrum_chart({}, _settings(), hue="E0_keV") is None


def test_compare_chart_single_hue_value_still_renders_one_line():
    chart = compare_spectrum_chart(_store_single_tilt(), _settings(), hue="tilt_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    assert {row["tilt_deg"] for row in df} == {20.0}


def test_compare_chart_rejects_unknown_hue():
    import pytest

    with pytest.raises(ValueError):
        compare_spectrum_chart(_store_single_tilt(), _settings(), hue="not_a_field")


def test_compare_chart_duplicate_hue_values_collapse_to_max_peak():
    # Two records sharing tilt_deg=20 (different azimuths) -- the stronger-peak
    # one should win, giving exactly ONE line for that hue value.
    store = {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0, peak_center=2500.0),
            30.1: _record(30.0, 20.0, 45.0, peak_center=2500.0),
        }
    }
    # Boost the second record's peak so the selection is unambiguous.
    store["HOPG bulk"][30.1]["spec"] = store["HOPG bulk"][30.1]["spec"] * 5.0
    chart = compare_spectrum_chart(store, _settings(), hue="tilt_deg")
    spec = chart.to_dict()
    df = _dataset(spec)
    tilts = {row["tilt_deg"] for row in df}
    assert tilts == {20.0}
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


# ---- multi_case_spectrum_chart tests -----------------------------------------------
def test_multi_case_chart_none_on_empty():
    """Empty cases list returns None, matching compare_spectrum_chart behavior."""
    assert multi_case_spectrum_chart([], _settings()) is None


def test_multi_case_chart_single_case():
    """Single (record, label) tuple produces exactly one label in the dataset."""
    cases = [(_record(30.0, 20.0, 0.0), "HOPG - 30 keV")]
    chart = multi_case_spectrum_chart(cases, _settings())
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    df = _dataset(spec)
    labels = {row["label"] for row in df}
    assert labels == {"HOPG - 30 keV"}


def test_multi_case_chart_distinct_labels_survive_duplicate_fields():
    """Core regression: two entries with identical case fields but different
    labels must BOTH survive as distinct lines (no collapse to strongest peak).
    This is the anti-collapse behavior that distinguishes multi_case_spectrum_chart
    from compare_spectrum_chart."""
    cases = [
        (_record(30.0, 20.0, 0.0, peak_center=2500.0), "copy A"),
        (_record(30.0, 20.0, 0.0, peak_center=2500.0), "copy B"),
    ]
    chart = multi_case_spectrum_chart(cases, _settings())
    spec = chart.to_dict()
    df = _dataset(spec)
    labels = {row["label"] for row in df}
    assert labels == {"copy A", "copy B"}


def test_multi_case_chart_color_field_is_label():
    """Color encoding must be literally the 'label' field (the passed label string),
    NOT any case field like tilt_deg or E0_keV."""
    cases = [
        (_record(30.0, 20.0, 0.0), "HOPG - 30 keV"),
        (_record(60.0, 20.0, 0.0), "HOPG - 60 keV"),
    ]
    chart = multi_case_spectrum_chart(cases, _settings())
    spec = chart.to_dict()
    enc = spec["layer"][0]["encoding"]
    assert enc["color"]["field"] == "label"
    assert enc["color"]["title"] == "case"


def test_multi_case_chart_brem_layer_toggle():
    """include_brem=True produces 2 layers (total + brem);
    include_brem=False produces 1 layer (total only)."""
    cases = [(_record(30.0, 20.0, 0.0), "HOPG - 30 keV")]

    chart_with_brem = multi_case_spectrum_chart(cases, _settings(), include_brem=True)
    spec_with = chart_with_brem.to_dict()
    assert len(spec_with["layer"]) == 2

    chart_no_brem = multi_case_spectrum_chart(cases, _settings(), include_brem=False)
    spec_no = chart_no_brem.to_dict()
    assert len(spec_no["layer"]) == 1


def test_multi_case_chart_broadband_wide_brem_tail():
    """band='broad' with wide_brem records extends energies past the narrow
    1000-5000 eV band, reaching up to beam energy (30 keV for the test record)."""
    cases = [
        (_record(30.0, 20.0, 0.0, wide_brem=True), "HOPG - 30 keV broad"),
        (_record(30.0, 40.0, 0.0, wide_brem=True), "HOPG - 40 deg broad"),
    ]
    chart = multi_case_spectrum_chart(cases, _settings(), band="broad")
    spec = chart.to_dict()
    df = _dataset(spec)
    tail = [row for row in df if row["component"] == "total" and row["energy_eV"] > 5000.0]

    assert tail
    assert max(row["energy_eV"] for row in df) == 30000.0
    assert {row["label"] for row in tail} == {"HOPG - 30 keV broad", "HOPG - 40 deg broad"}


def test_multi_case_chart_mixed_material_labels():
    """Different material names and labels in a case basket must render without error,
    with each label surviving as a distinct line (multi_case_spectrum_chart does not
    enforce material consistency, only reads case fields for physics calculations)."""
    rec_hopg = _record(30.0, 20.0, 0.0)
    rec_hopg["case"]["name"] = "HOPG bulk"
    rec_sic = _record(30.0, 20.0, 0.0, peak_center=3000.0)
    rec_sic["case"]["name"] = "SiC bulk"
    rec_sic["spec"] = rec_sic["spec"] * 1.5  # Boost SiC's peak to make it visually distinct

    cases = [
        (rec_hopg, "HOPG - 30 keV"),
        (rec_sic, "SiC - 30 keV"),
    ]
    chart = multi_case_spectrum_chart(cases, _settings())
    spec = chart.to_dict()
    df = _dataset(spec)
    labels = {row["label"] for row in df}
    assert labels == {"HOPG - 30 keV", "SiC - 30 keV"}
