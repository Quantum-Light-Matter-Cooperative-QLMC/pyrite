"""Regression coverage for profile parsing and grouped diagnostics."""

from copy import deepcopy
from types import MappingProxyType

import pytest

from pyrite.materials._catalog_decode import _Errors
from pyrite.materials._parse import _parse_profiles

_SCAN = {
    "thickness_ang": 5000,
    "energy_keV": 30,
    "tilt_deg": 15,
    "tilt_azim_deg": 180,
}


def test_profile_named_instruments_and_artifact_refs() -> None:
    row = {
        **_SCAN,
        "materials": ["hopg", "hbn"],
        "beam": "example",
        "detectors": {"left": "first", "right": "second"},
        "energy_grid_refs": {"hopg": "a" * 64},
        "emission": "incoherent",
        "temporal_profile": False,
    }
    original = deepcopy(row)
    errors = _Errors()

    parsed = _parse_profiles({"study": row}, errors)["study"]

    assert errors.items == []
    assert row == original
    assert parsed == original
    assert parsed is not row
    detectors = parsed["detectors"]
    assert isinstance(detectors, MappingProxyType)
    assert list(detectors) == ["left", "right"]


def test_profile_keeps_error_order_and_original_legacy_conflicts() -> None:
    row = {
        **_SCAN,
        "tilt_deg": 90,
        "materials": ["hopg", "hopg"],
        "emission": "invalid",
        "temporal_profile": 1,
        "energy_grid_refs": {"": "BAD"},
        "beam": "",
        "detector": "",
        "filters": False,
        "physical_detector": False,
        "detectors": {"Upper": "", "good": False},
    }
    original = deepcopy(row)
    errors = _Errors()

    parsed = _parse_profiles({"study": row}, errors)["study"]

    prefix = "profiles.study"
    assert errors.items == [
        f"{prefix}.tilt_deg: values must satisfy 0 <= tilt_deg < 90",
        f"{prefix}.materials: must not contain duplicate material keys",
        f"{prefix}.emission: must be one of ('incoherent', 'coherent', 'both')",
        f"{prefix}.temporal_profile: must be a boolean",
        f"{prefix}.energy_grid_refs: material keys must be nonempty strings",
        f"{prefix}.energy_grid_refs.: must be a 64-character lowercase SHA-256 digest",
        f"{prefix}.beam: must be a nonempty beam name",
        f"{prefix}.detector: must be a nonempty detector name",
        f"{prefix}.filters: must be an array of tables",
        f"{prefix}.physical_detector: must be a table",
        f"{prefix}.detectors: cannot be combined with legacy detector or physical_detector",
        f"{prefix}.detectors.Upper: ID must start with a lowercase letter and contain only "
        "lowercase letters, digits, '_' or '-'",
        f"{prefix}.detectors.Upper: must be a nonempty detector reference",
        f"{prefix}.detectors.good: must be a table",
        f"{prefix}.detectors: must contain at least one detector",
    ]
    assert row == original
    assert not {"beam", "detector", "filters", "physical_detector"} & parsed.keys()
    assert isinstance(parsed["detectors"], MappingProxyType)
    assert parsed["detectors"] == {}


@pytest.mark.parametrize("bad", [None, False, []])
def test_profile_invalid_tables_do_not_discard_valid_siblings(bad) -> None:
    errors = _Errors()
    parsed = _parse_profiles({"bad": bad, "study": {**_SCAN, "detectors": bad}}, errors)

    assert errors.items == [
        "profiles.bad: must be a table",
        "profiles.study.detectors: must be a table",
    ]
    assert list(parsed) == ["study"]
    assert parsed["study"]["detectors"] is bad
