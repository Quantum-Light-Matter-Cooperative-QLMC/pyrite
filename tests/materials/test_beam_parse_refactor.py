"""Regression coverage for profile beam field decoding and error ordering."""

from copy import deepcopy
from types import MappingProxyType

from pyrite.materials._beam_detector_parse import _parse_beams, _parse_profile_beam
from pyrite.materials._catalog_decode import _Errors

_PATH = "profiles.demo.beam"


def test_gdf_fields_preserve_false_zero_signed_coordinates_and_order() -> None:
    raw = {
        "source": "gpt_gdf",
        "gdf_path": "beam.gdf",
        "gdf_z_origin_m": -0.1,
        "gdf_shape_only": False,
        "gdf_screen_position_m": -0.2,
        "gdf_screen_tolerance_m": 0.0,
        "gdf_normalization": "gdf_charge",
        "rep_rate_hz": "5 kHz",
    }
    original = deepcopy(raw)
    errors = _Errors()
    parsed = _parse_profile_beam(raw, _PATH, errors)
    assert errors.items == []
    assert parsed is not None
    assert list(parsed) == list(raw)
    assert parsed == {**raw, "rep_rate_hz": 5000.0}
    assert parsed["gdf_shape_only"] is False
    assert raw == original


def test_invalid_gdf_fields_keep_partial_results_and_ordered_errors() -> None:
    raw = {
        "source": "gpt_gdf",
        "gdf_repetition_rate_hz": 1,
        "unknown": 2,
        "long_shape": "gaussian",
        "gdf_time_s": -1.0,
        "gdf_screen_position_m": 0.0,
        "gdf_shape_only": "false",
        "gdf_normalization": "invalid",
    }
    errors = _Errors()
    parsed = _parse_profile_beam(raw, _PATH, errors)
    assert parsed == {"source": "gpt_gdf", "long_shape": "gaussian", "gdf_screen_position_m": 0.0}
    assert errors.items == [
        f"{_PATH}.gdf_repetition_rate_hz: removed; replace gdf_repetition_rate_hz with rep_rate_hz",
        f"{_PATH}.unknown: unknown key",
        f"{_PATH}: gdf_time_s and gdf_screen_position_m are mutually exclusive",
        f"{_PATH}: gpt_gdf is incompatible with analytic fields: long_shape",
        f"{_PATH}: gpt_gdf requires gdf_path",
        f"{_PATH}: gpt_gdf requires gdf_z_origin_m",
        f"{_PATH}.gdf_time_s: must be finite and non-negative "
        "(origin/screen coordinates may be signed)",
        f"{_PATH}.gdf_shape_only: must be a boolean",
        f"{_PATH}.gdf_normalization: must be pyrite_current or gdf_charge",
    ]


def test_analytic_beam_nested_values_and_named_beam_parity() -> None:
    raw = {
        "long_offsets_fs": [-10.0, 0.0, 10.0],
        "bunch_front_tilt": {"tilt_deg": -15.0},
        "long_shape": "gaussian",
        "bunch_length_fs": "0.2 ps",
    }
    errors = _Errors()
    parsed = _parse_profile_beam(raw, _PATH, errors)
    named = _parse_beams({"demo": raw}, errors)
    assert errors.items == []
    assert parsed is not None
    assert parsed == named["demo"]
    assert parsed["long_offsets_fs"] == (-10.0, 0.0, 10.0)
    assert parsed["bunch_length_fs"] == 200.0
    front = parsed["bunch_front_tilt"]
    assert isinstance(front, MappingProxyType)
    assert front == {"tilt_deg": -15.0, "azimuth_deg": 0.0}


def test_invalid_analytic_values_do_not_discard_valid_fields() -> None:
    errors = _Errors()
    parsed = _parse_profile_beam(
        {
            "bunch_front_tilt": {"tilt_deg": 90.0},
            "long_offsets_fs": [],
            "energy_spread_frac": 0.0,
            "long_shape": "uniform",
        },
        _PATH,
        errors,
    )
    assert parsed == {"long_shape": "uniform"}
    assert errors.items == [
        f"{_PATH}.bunch_front_tilt: requires finite -90 < tilt_deg < 90 and finite azimuth_deg",
        f"{_PATH}.long_offsets_fs: must be a non-empty array of finite times in fs",
        f"{_PATH}.energy_spread_frac: must be a finite positive number",
    ]
