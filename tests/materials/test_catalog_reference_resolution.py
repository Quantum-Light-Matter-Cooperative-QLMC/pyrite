"""Regression tests for catalog profile reference resolution stages."""

from pathlib import Path
from types import MappingProxyType

from pyrite.materials._catalog_decode import _Errors
from pyrite.materials.catalog import (
    _resolve_profile_beams,
    _resolve_profile_detector_sets,
    _resolve_profile_detectors,
    _resolve_profile_transport_numerics,
)


def test_beam_resolution_removes_labels_and_anchors_gdf_paths(tmp_path: Path) -> None:
    relative = "data/beam.gdf"
    absolute = tmp_path / "other.gdf"
    named = {"source": "gpt_gdf", "gdf_path": relative, "label": "Display only"}
    inline = {"source": "gpt_gdf", "gdf_path": str(absolute)}
    profiles = {"named": {"beam": "gpt"}, "inline": {"beam": inline}, "empty": {}}
    errors = _Errors()

    resolved = _resolve_profile_beams(profiles, {"gpt": named}, tmp_path, errors)

    assert errors.items == []
    assert list(resolved) == ["named", "inline"]
    assert resolved["named"] == {"source": "gpt_gdf", "gdf_path": str(tmp_path / relative)}
    assert resolved["inline"] == inline
    assert all(isinstance(spec, MappingProxyType) for spec in resolved.values())
    assert named["label"] == "Display only"
    assert named["gdf_path"] == relative
    assert inline["gdf_path"] == str(absolute)


def test_beam_and_legacy_detector_reference_errors_are_ordered(tmp_path: Path) -> None:
    profiles = {
        "first": {"beam": "missing-a", "detector": "missing-b"},
        "second": {"beam": "missing-c", "detector": "missing-d"},
    }
    errors = _Errors()

    assert _resolve_profile_beams(profiles, {}, tmp_path, errors) == {}
    assert errors.items == [
        "profiles.first.beam: unknown beam 'missing-a'",
        "profiles.second.beam: unknown beam 'missing-c'",
    ]
    assert _resolve_profile_detectors(profiles, {}, errors) == {}
    assert errors.items[2:] == [
        "profiles.first.detector: unknown detector 'missing-b'",
        "profiles.second.detector: unknown detector 'missing-d'",
    ]


def test_detector_sets_preserve_explicit_order_and_legacy_defaults() -> None:
    scalar = MappingProxyType({"observation_angle_deg": 45.0})
    pixel = MappingProxyType({"distance_mm": 100.0})
    inline = MappingProxyType({"observation_angle_deg": 90.0})
    profiles = {
        "named": {"detectors": {"right": "scalar", "left": inline}},
        "legacy": {"detector": "scalar"},
        "empty": {},
        "explicit_empty": {"detectors": {}},
        "bad": {"detectors": {"unknown": "missing", "ok": "scalar"}},
    }
    detectors = {"scalar": scalar}
    errors = _Errors()

    legacy = _resolve_profile_detectors(profiles, detectors, errors)
    sets = _resolve_profile_detector_sets(profiles, detectors, legacy, {"legacy": pixel}, errors)

    assert errors.items == ["profiles.bad.detectors.unknown: unknown detector 'missing'"]
    assert list(sets["named"]) == ["right", "left"]
    assert sets["named"]["right"] is scalar
    assert sets["named"]["left"] is inline
    assert list(sets["legacy"]) == ["physical", "default"]
    assert sets["legacy"]["physical"] is pixel
    assert sets["legacy"]["default"] is scalar
    assert sets["empty"] == {"default": {}}
    assert isinstance(sets["empty"]["default"], MappingProxyType)
    assert sets["explicit_empty"] == {}
    assert sets["bad"] == {"ok": scalar}
    assert all(isinstance(spec, MappingProxyType) for spec in sets.values())


def test_transport_numerics_freeze_electron_count_sweeps() -> None:
    errors = _Errors()
    profiles = {
        "sweep": {"n_electrons": {"values": [10, 20]}, "n_electrons_brem": 50},
        "empty": {},
    }

    resolved = _resolve_profile_transport_numerics(profiles, errors)

    assert errors.items == []
    assert resolved == {"sweep": {"n_electrons": (10, 20), "n_electrons_brem": (50,)}}
    assert isinstance(resolved["sweep"], MappingProxyType)
