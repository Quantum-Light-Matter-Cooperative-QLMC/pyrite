"""TOML profile quantities share Python configuration normalization semantics."""

import numpy as np

from pyrite.materials._beam_detector_parse import (
    _parse_physical_detector,
    _parse_profile_beam,
)
from pyrite.materials._catalog_decode import _Errors, _grid


def test_scan_grid_values_accept_explicit_units_and_keep_canonical_numbers() -> None:
    errors = _Errors()
    grid = _grid(
        {"values": ["60 keV", "0.06 MeV"]},
        "energy_keV",
        errors,
        unit="keV",
        expected="energy",
    )
    assert errors.items == []
    np.testing.assert_array_equal(grid, np.array([60.0, 60.0]))


def test_beam_profile_and_physical_detector_parse_unit_strings() -> None:
    errors = _Errors()
    beam = _parse_profile_beam(
        {
            "transverse_fwhm_x_mm": "500 um",
            "bunch_length_fs": "0.2 ps",
            "rep_rate_hz": "5 kHz",
            "bunch_charge_pc": "1 pC",
        },
        "profiles.demo.beam",
        errors,
    )
    assert beam == {
        "transverse_fwhm_x_mm": 0.5,
        "bunch_length_fs": 200.0,
        "rep_rate_hz": 5000.0,
        "bunch_charge_pc": 1.0,
    }

    detector = _parse_physical_detector(
        {
            "distance_mm": "10 cm",
            "polar_deg": "0.5 rad",
            "pitch_mm": ["55 um", "55 um"],
            "acquisition": {
                "exposure_s": "500 ms",
                "measured_edges_eV": ["0 eV", "0.1 keV", "200 eV"],
            },
        },
        "profiles.demo.physical_detector",
        errors,
    )
    assert errors.items == []
    assert detector["distance_mm"] == 100.0
    assert detector["polar_deg"] == np.rad2deg(0.5)
    assert detector["pitch_mm"] == (0.055, 0.055)
    assert detector["acquisition"]["exposure_s"] == 0.5
    assert detector["acquisition"]["measured_edges_eV"] == (0.0, 100.0, 200.0)


def test_invalid_profile_dimensions_report_field_and_expected_unit() -> None:
    errors = _Errors()
    _parse_physical_detector({"distance_mm": "5 keV"}, "profiles.demo.physical_detector", errors)
    assert any(
        "profiles.demo.physical_detector.distance_mm" in item
        and "electron_volt" in item
        and "length" in item
        and "mm" in item
        for item in errors.items
    )
