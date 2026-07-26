"""Guard tests for cxr_mc.results.selection -- case label formatting,
case-picker table rows, and basket-record slimming. Same synthetic-record
approach as test_altair_spectra_compare.py: small records, no GPU, no
checkpoint. Tests case_label(), case_table_rows(), and slim_case_record().
"""

import numpy as np

from cxr_mc.results.selection import _peak, case_label, case_table_rows, slim_case_record


def _record(
    E0_keV, tilt_deg, azim_deg, peak_center=2500.0, n=200, wide_brem=False, name="HOPG bulk"
):
    """Synthetic record builder: E_grid, spec (sharp peak), brem (smooth
    continuum), case dict, and optionally wide_brem arrays."""
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - peak_center) / 50.0) ** 2))  # single sharp line
    brem = np.linspace(1.0, 0.2, n)  # smooth falling continuum
    rec = {
        "E_grid": E,
        "spec": peak,
        "brem": brem,
        "scale": 2.0,
        "fwhm": 30.0,
        "case": {
            "name": name,
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


# ---- case_label tests -------------------------------------------------------


def test_case_label_all_fields_varying_none():
    """case_label with all fields present and varying=None (default)
    includes material, energy, thickness, tilt, azimuth."""
    rec = _record(30.0, 20.0, 0.0)
    result = case_label(rec["case"], material_label="HOPG", varying=None)
    # Expect: "HOPG - 30 keV - 5um - tilt 20deg - az 0deg"
    assert result == "HOPG - 30 keV - 5um - tilt 20deg - az 0deg"


def test_case_label_varying_subset():
    """case_label with varying={tilt_deg} elides thickness and azimuth,
    keeps material, energy, tilt."""
    rec = _record(30.0, 20.0, 0.0)
    result = case_label(rec["case"], material_label="HOPG", varying={"tilt_deg"})
    # Expect: "HOPG - 30 keV - tilt 20deg" (no thickness or azimuth)
    assert result == "HOPG - 30 keV - tilt 20deg"


def test_case_label_face_blazed_appends_suffix():
    """case_label with face='blazed' appends ' (blazed)' suffix."""
    rec = _record(30.0, 20.0, 0.0)
    result = case_label(rec["case"], material_label="HOPG", face="blazed")
    assert result.endswith(" (blazed)")
    assert "HOPG - 30 keV - 5um - tilt 20deg - az 0deg" in result


def test_case_label_face_flat_no_suffix():
    """case_label with face='flat' or face=None does not append suffix."""
    rec = _record(30.0, 20.0, 0.0)
    result_flat = case_label(rec["case"], material_label="HOPG", face="flat")
    result_none = case_label(rec["case"], material_label="HOPG", face=None)
    assert not result_flat.endswith(" (blazed)")
    assert not result_none.endswith(" (blazed)")
    assert result_flat == "HOPG - 30 keV - 5um - tilt 20deg - az 0deg"


def test_case_label_no_material_label():
    """case_label with material_label=None omits the material segment,
    starts directly with energy."""
    rec = _record(30.0, 20.0, 0.0)
    result = case_label(rec["case"], material_label=None)
    # Expect: "30 keV - 5um - tilt 20deg - az 0deg" (no HOPG prefix)
    assert result == "30 keV - 5um - tilt 20deg - az 0deg"
    assert "HOPG" not in result


# ---- case_table_rows tests --------------------------------------------------


def test_case_table_rows_two_records():
    """case_table_rows on a 2-record store returns 2 rows with correct keys
    and values matching the source records."""
    results = {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0),
            60.0: _record(60.0, 20.0, 0.0),
        }
    }
    rows = case_table_rows(results)
    assert len(rows) == 2
    # Check that both records are present
    energies = {row["E0_keV"] for row in rows}
    assert energies == {30.0, 60.0}
    # Check keys exist and are correct type
    for row in rows:
        assert "name" in row
        assert "E0_keV" in row
        assert "thickness" in row
        assert "thickness_ang" in row
        assert "tilt_deg" in row
        assert "tilt_azim_deg" in row
        assert "peak_flux" in row
        assert row["name"] == "HOPG bulk"
        assert row["thickness"] == "5um"
        assert row["thickness_ang"] == 5.0e4
        assert row["tilt_deg"] == 20.0
        assert row["tilt_azim_deg"] == 0.0
        assert isinstance(row["peak_flux"], float)


def test_case_table_rows_peak_flux_matches_peak():
    """case_table_rows peak_flux value equals _peak(record) for each record."""
    results = {
        "HOPG bulk": {
            30.0: _record(30.0, 20.0, 0.0),
            60.0: _record(60.0, 20.0, 0.0),
        }
    }
    rows = case_table_rows(results)
    # Verify peak_flux matches _peak for each row
    for row in rows:
        rec = results[row["name"]][row["E0_keV"]]
        expected_peak = _peak(rec)
        assert row["peak_flux"] == expected_peak


def test_case_table_rows_empty_store():
    """case_table_rows on an empty {} store returns []."""
    rows = case_table_rows({})
    assert rows == []


# ---- slim_case_record tests -------------------------------------------------


def test_slim_case_record_without_wide_brem():
    """slim_case_record on record WITHOUT wide_brem (no E_grid_brem/brem_wide)
    returns dict with keys exactly from _BASKET_RECORD_FIELDS that exist in
    source, plus case."""
    rec = _record(30.0, 20.0, 0.0, wide_brem=False)
    slim = slim_case_record(rec, material="HOPG", label="test label")
    # Should have E_grid, spec, brem, fwhm, scale, case
    # Should NOT have E_grid_brem, brem_wide (absent from source)
    assert "E_grid" in slim
    assert "spec" in slim
    assert "brem" in slim
    assert "fwhm" in slim
    assert "scale" in slim
    assert "case" in slim
    assert "E_grid_brem" not in slim
    assert "brem_wide" not in slim
    # Verify it has exactly the expected keys
    expected_keys = {"E_grid", "spec", "brem", "fwhm", "scale", "case"}
    assert set(slim.keys()) == expected_keys


def test_slim_case_record_with_wide_brem():
    """slim_case_record on record WITH wide_brem=True includes E_grid_brem
    and brem_wide in output."""
    rec = _record(30.0, 20.0, 0.0, wide_brem=True)
    slim = slim_case_record(rec, material="HOPG", label="test label")
    # Should have E_grid_brem and brem_wide since they exist in source
    assert "E_grid_brem" in slim
    assert "brem_wide" in slim
    # Verify arrays are the same objects (copied from record)
    assert np.array_equal(slim["E_grid_brem"], rec["E_grid_brem"])
    assert np.array_equal(slim["brem_wide"], rec["brem_wide"])


def test_slim_case_record_case_dict_has_material_face_label():
    """slim_case_record returned case dict has material, face, label set
    to passed-in values, and retains original case fields."""
    rec = _record(30.0, 20.0, 0.0)
    slim = slim_case_record(rec, material="HOPG", label="my label", face="blazed")
    # Check that case dict has the new fields
    assert slim["case"]["material"] == "HOPG"
    assert slim["case"]["face"] == "blazed"
    assert slim["case"]["label"] == "my label"
    # Check that original case fields are still there
    assert slim["case"]["E0_keV"] == 30.0
    assert slim["case"]["tilt_deg"] == 20.0
    assert slim["case"]["tilt_azim_deg"] == 0.0
    assert slim["case"]["thickness_ang"] == 5.0e4


def test_slim_case_record_does_not_mutate_input():
    """slim_case_record does not mutate the input record's case dict."""
    rec = _record(30.0, 20.0, 0.0)
    original_case_keys = set(rec["case"].keys())
    # Verify "material" is not in original
    assert "material" not in rec["case"]
    # Call slim_case_record
    slim = slim_case_record(rec, material="HOPG", label="test")
    assert slim["case"]["material"] == "HOPG"
    # Verify original record's case dict is unchanged
    assert "material" not in rec["case"]
    assert set(rec["case"].keys()) == original_case_keys


def test_slim_case_record_default_face_flat():
    """slim_case_record default face='flat' when not passed explicitly."""
    rec = _record(30.0, 20.0, 0.0)
    # Call without face parameter (should default to "flat")
    slim = slim_case_record(rec, material="HOPG", label="test")
    assert slim["case"]["face"] == "flat"


def test_slim_case_record_arrays_are_references():
    """slim_case_record copies (references) the spectral arrays from source."""
    rec = _record(30.0, 20.0, 0.0, wide_brem=True)
    slim = slim_case_record(rec, material="HOPG", label="test")
    # Arrays should be the same objects (not deep-copied)
    assert slim["E_grid"] is rec["E_grid"]
    assert slim["spec"] is rec["spec"]
    assert slim["brem"] is rec["brem"]
    assert slim["E_grid_brem"] is rec["E_grid_brem"]
    assert slim["brem_wide"] is rec["brem_wide"]
