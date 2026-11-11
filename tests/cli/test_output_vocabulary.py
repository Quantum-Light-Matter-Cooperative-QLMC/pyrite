"""Issue #1078: every human list/show uses one source/origin vocabulary."""

import re

import pytest

from pyrite.cli import command
from pyrite.console.tables import OBJECT_ORIGINS, VALUE_SOURCES, object_origin, value_source
from tests.helpers.cli import assert_clean_result, invoke

#: Raw provenance spellings resolvers emit (and JSON keeps) -> human label.
_RAW_SOURCES = {
    "profile": "profile",
    "explicit": "profile",
    "inherited": "profile",
    "overridden": "material",
    "preset fast": "preset:fast",
    "preset:fast": "preset:fast",
    "run": "run",
    "built-in": "default",
    "built-in default": "default",
    "stored": "default",
    "default": "default",
    "command line": "option",
    "config store": "config",
    "PYRITE_PROFILE": "env:PYRITE_PROFILE",
}


@pytest.mark.parametrize(("raw", "human"), sorted(_RAW_SOURCES.items()))
def test_value_sources_map_into_the_documented_vocabulary(raw, human):
    assert value_source(raw) == human
    assert human.split(":")[0] in VALUE_SOURCES


@pytest.mark.parametrize("raw", ["fixed", "yours", "", "preset"])
def test_unknown_value_sources_are_rejected(raw):
    with pytest.raises(ValueError):
        value_source(raw)


def test_object_origins_are_bundled_user_or_catalog():
    assert {object_origin(raw) for raw in ("bundled", "user", "yours", "catalog")} == OBJECT_ORIGINS
    with pytest.raises(ValueError):
        object_origin("[bundled]")


_LEGACY = re.compile(
    r"\((?:built-in|fidelity|yours|stored|explicit|bundled|user|inherited|overridden)\)"
    r"|\[(?:bundled|user)\]|explicit: unset|\bfidelity\b|\b1 profiles\b|\bNone\b|\bTrue\b|\bFalse\b"
)

_COMMANDS = [
    ["profile", "list"],
    ["profile", "show", "quickstart"],
    ["profile", "numerics", "show", "quickstart"],
    ["profile", "physics", "show", "quickstart"],
    ["profile", "numerics", "adaptive-sampling", "show", "quickstart"],
    ["profile", "numerics", "cxr-grid", "show", "quickstart"],
    ["profile", "ranges", "show", "quickstart", "hopg"],
    ["profile", "detector", "list", "two_detector_demo"],
    ["numerics", "list"],
    ["numerics", "show", "default"],
    ["physics", "list"],
    ["physics", "show", "electron-standard"],
    ["beam", "list"],
    ["beam", "show", "example"],
    ["detector", "list"],
    ["detector", "show", "reference"],
    ["filter", "list"],
    ["target", "list"],
    ["material", "list"],
    ["material", "show", "hopg"],
]


@pytest.mark.parametrize("argv", _COMMANDS, ids=" ".join)
def test_human_list_and_show_use_only_the_documented_labels(argv, _isolate_user_catalog):
    result = invoke(command, argv)
    assert_clean_result(result)
    assert not _LEGACY.search(result.stdout), result.stdout
