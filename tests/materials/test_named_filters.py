"""Issue #665: named plate resolution changes no physical observation values."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from pyrite.campaign.observation import filter_from_config
from pyrite.materials._catalog_decode import _Errors
from pyrite.materials._named_filters import load_filter, resolve_named_filters

PLATE = {
    "material": "silicon",
    "thickness_mm": 0.1,
    "size_mm": [20.0, 10.0],
    "distance_mm": 50.0,
    "polar_deg": 60.0,
}


def test_named_filter_matches_inline_and_does_not_inject_name():
    errors = _Errors()
    profiles = {
        "reference": {"filters": [{"ref": "window", "distance_mm": 70.0}]},
        "inline": {"filters": [{**PLATE, "distance_mm": 70.0}]},
    }
    definitions, resolved, refs = resolve_named_filters(
        {"window": {**PLATE, "description": "a window"}}, profiles, errors
    )
    assert not errors.items
    assert refs["reference"] == ("window",)
    assert resolved["reference"]["filters"] == resolved["inline"]["filters"]
    assert "ref" not in resolved["reference"]["filters"][0]
    assert "description" not in definitions["window"]
    # Existing material/catalog parsing normalizes the raw mm arrays.
    from pyrite.materials._beam_detector_parse import _parse_filter_rows

    lhs = _parse_filter_rows(resolved["reference"]["filters"], "lhs", errors)[0]
    rhs = _parse_filter_rows(resolved["inline"]["filters"], "rhs", errors)[0]
    assert filter_from_config(lhs) == filter_from_config(rhs)


def test_unknown_reference_and_invalid_definition_fail_with_paths():
    errors = _Errors()
    resolve_named_filters(
        {"window": {**PLATE, "thickness_mm": -1}},
        {"p": {"filters": [{"ref": "missing"}]}},
        errors,
    )
    assert any("filters.window" in item for item in errors.items)
    assert any("profiles.p.filters[0].ref" in item for item in errors.items)


def test_legacy_filter_passes_through_and_layer_is_readonly():
    errors = _Errors()
    defs, profiles, refs = resolve_named_filters({}, {"p": {"filters": [dict(PLATE)]}}, errors)
    assert not errors.items
    assert not defs
    assert profiles["p"]["filters"] == [PLATE]
    assert refs["p"] == (None,)
    with pytest.raises(TypeError):
        cast(Any, defs)["new"] = {}


def test_python_loader_returns_existing_typed_plate():
    errors = _Errors()
    definitions, _, _ = resolve_named_filters({"window": PLATE}, {}, errors)
    assert not errors.items
    catalog = SimpleNamespace(named_filters=definitions)
    loaded = load_filter("window", catalog=cast(Any, catalog))
    assert loaded == filter_from_config(definitions["window"])
    assert loaded.name is None
    with pytest.raises(KeyError, match="unknown filter preset"):
        load_filter("missing", catalog=cast(Any, catalog))


def test_shared_reference_preserves_position_and_instance_names():
    errors = _Errors()
    definitions, profiles, refs = resolve_named_filters(
        {"window": PLATE},
        {
            "one": {
                "filters": [
                    {"ref": "window", "name": "upstream"},
                    {**PLATE, "name": "middle"},
                    {"ref": "window", "name": "downstream", "distance_mm": 80.0},
                ]
            },
            "two": {"filters": [{"ref": "window", "name": "separate"}]},
        },
        errors,
    )
    assert not errors.items
    assert refs["one"] == ("window", None, "window")
    assert refs["two"] == ("window",)
    assert [f["name"] for f in profiles["one"]["filters"]] == ["upstream", "middle", "downstream"]
    assert profiles["one"]["filters"][2]["distance_mm"] == 80.0
    assert "name" not in definitions["window"]


def test_case_insensitive_duplicate_definitions_rejected():
    errors = _Errors()
    resolve_named_filters({"Window": PLATE, "window": PLATE}, {}, errors)
    assert any("only by case" in item for item in errors.items)
