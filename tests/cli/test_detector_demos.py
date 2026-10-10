"""Detector demo profiles are inspectable and clone their detector references."""

import json

import pytest
import tomlkit

from pyrite.cli.commands import profile
from tests.helpers.cli import assert_clean_result, invoke


@pytest.mark.parametrize(
    ("name", "references"),
    [
        ("two_detector_demo", {"at_90": "reference", "at_135": "reference_135"}),
        ("pixelated_detector_demo", {"camera": "pixelated_reference"}),
    ],
)
def test_demo_profile_display_resolves_named_detector_collection(name, references):
    human = invoke(profile.command, ["show", name])
    machine = invoke(profile.command, ["show", name, "-o", "json"])
    assert_clean_result(human)
    assert_clean_result(machine)
    assert "unspecified" not in human.stdout
    detectors = json.loads(machine.stdout)["payload"]["detectors"]
    assert {key: row["reference"] for key, row in detectors.items()} == references
    assert all(
        value is not None for row in detectors.values() for value in row["acceptance"].values()
    )
    if name == "two_detector_demo":
        assert detectors["at_90"]["acceptance"]["observation_angle_deg"] == 90.0
        assert detectors["at_135"]["acceptance"]["observation_angle_deg"] == 135.0
    else:
        assert detectors["camera"]["kind"] == "pixel"
        assert detectors["camera"]["settings"]["shape"] == [256, 256]
        assert detectors["camera"]["settings"]["response"] == {"kind": "ideal"}


@pytest.mark.parametrize("source", ["two_detector_demo", "pixelated_detector_demo"])
def test_demo_profile_clone_retains_detector_collection(_isolate_user_catalog, source):
    created = invoke(profile.command, ["create", "mine", "--from", source])
    assert_clean_result(created)
    assert "inherited: beam, detector collection" in created.stdout
    stored = tomlkit.parse((_isolate_user_catalog / "profiles" / "mine.toml").read_text())
    assert "detectors" in stored
    assert "detector" not in stored
    source_show = invoke(profile.command, ["show", source, "-o", "json"])
    clone_show = invoke(profile.command, ["show", "mine", "-o", "json"])
    assert_clean_result(source_show)
    assert_clean_result(clone_show)
    assert (
        json.loads(clone_show.stdout)["payload"]["detectors"]
        == json.loads(source_show.stdout)["payload"]["detectors"]
    )
