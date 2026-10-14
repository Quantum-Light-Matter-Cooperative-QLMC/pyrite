"""CLI-only detector collection editing; no transport or response Monte Carlo."""

import json
import shutil

import pytest
import tomlkit

from pyrite import DATA_DIR
from pyrite._catalog_layout import read_text
from pyrite.campaign import profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli.commands import detector, profile
from tests.helpers.cli import assert_clean_result, invoke
from tests.helpers.user_catalog import copy_full_catalog


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    copy_full_catalog(data / "catalog")
    path = data / "materials.toml"
    path.write_text(read_text(data / "catalog"))
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", path)
    return path


def call(*args):
    result = invoke(profile.command, ["detector", *args])
    assert_clean_result(result)
    return result


def row(catalog):
    return tomlkit.parse(catalog.read_text())["profiles"]["standard"]


def test_named_scalar_and_pixel_collection_entirely_cli(catalog):
    for args in (
        ["create", "scalar", "--observation-angle", "100"],
        [
            "create",
            "camera",
            "--distance-mm",
            "400",
            "--shape",
            "4",
            "8",
            "--angular-shape",
            "2",
            "2",
            "--response",
            "ideal",
            "--exposure-s",
            "1",
            "--measured-edges-ev",
            "0,100,200",
        ],
        ["set", "camera", "--roll-deg", "2", "-y"],
    ):
        assert_clean_result(invoke(detector.command, args))
    assert_clean_result(invoke(profile.command, ["set", "standard", "--detector", "scalar", "-y"]))
    original = catalog.read_text()
    preview = call("migrate", "standard", "--dry-run")
    assert "legacy stem -> detector-default" in preview.stdout
    assert catalog.read_text() == original
    call("migrate", "standard", "-y")
    call("add", "standard", "reference", "--detector", "reference", "-y")
    call("add", "standard", "sensor", "--detector", "camera", "-y")
    payload = json.loads(call("list", "standard", "-o", "json").stdout)
    assert payload["schema"] == "cxr.profile.detector.list"
    assert list(payload["payload"]["detectors"]) == ["default", "reference", "sensor"]
    shown = json.loads(call("show", "standard", "sensor", "-o", "json").stdout)["payload"]
    assert shown["kind"] == "pixel"
    assert shown["settings"]["acquisition"]["exposure_s"] == 1
    assert shown["settings"]["shape"] == [4, 8]
    assert shown["acceptance"]["solid_angle_sr"] > 0
    baseline = row(catalog).unwrap()
    call("set", "standard", "sensor", "--inline-copy", "--roll-deg", "3", "-y")
    assert tomlkit.parse(catalog.read_text())["detectors"]["camera"]["roll_deg"] == 2
    call("reset", "standard", "sensor", "acquisition", "-y")
    assert "acquisition" not in row(catalog)["detectors"]["sensor"]
    before_remove = catalog.read_text()
    call("rm", "standard", "sensor", "--dry-run")
    assert catalog.read_text() == before_remove
    call("rm", "standard", "sensor", "-y")
    after = row(catalog).unwrap()
    assert after["detectors"] == {k: v for k, v in baseline["detectors"].items() if k != "sensor"}
    assert {k: v for k, v in after.items() if k != "detectors"} == {
        k: v for k, v in baseline.items() if k != "detectors"
    }


def test_legacy_edits_do_not_convert_storage(catalog):
    call(
        "set",
        "standard",
        "physical",
        "--distance-mm",
        "100",
        "--response",
        "ideal",
        "--exposure-s",
        "1",
        "--measured-edges-ev",
        "0,100,200",
        "-y",
    )
    assert "detectors" not in row(catalog)
    original = row(catalog).unwrap()
    before = profile_edit.profile_payload(tomlkit.parse(catalog.read_text()), "standard")[
        "detectors"
    ]
    call("set", "standard", "physical", "--distance-mm", "100", "-y")
    assert row(catalog).unwrap() == original
    call("migrate", "standard", "-y")
    after = profile_edit.profile_payload(tomlkit.parse(catalog.read_text()), "standard")[
        "detectors"
    ]
    assert before == after
    assert "detector" not in row(catalog)
    assert "physical_detector" not in row(catalog)


def test_migrate_implicit_default_keeps_resolved_values(catalog):
    document = tomlkit.parse(catalog.read_text())
    document["profiles"]["standard"].pop("detector", None)
    catalog.write_text(tomlkit.dumps(document))
    call("migrate", "standard", "-y")
    assert row(catalog)["detectors"]["default"]["observation_angle_deg"] == 90


@pytest.mark.parametrize(
    "args, message",
    [
        (["add", "standard", "new", "--detector", "reference"], "migrate"),
        (["set", "standard", "missing", "--detector", "reference"], "unknown detector ID"),
        (["set", "standard", "default", "--roll-deg", "2"], "shared detector"),
        (["set", "standard", "physical", "--shape", "4", "4"], "unknown detector ID"),
        (
            ["set", "standard", "default", "--detector", "reference", "--roll-deg", "2"],
            "separately",
        ),
    ],
)
def test_legacy_invalid_edits_write_nothing(catalog, args, message):
    original = catalog.read_text()
    result = invoke(profile.command, ["detector", *args])
    assert result.exit_code == 2
    assert message in result.stderr
    assert catalog.read_text() == original


def test_collection_validation_and_shortcut(catalog):
    call("migrate", "standard", "-y")
    assert_clean_result(
        invoke(profile.command, ["set", "standard", "--detector", "reference", "-y"])
    )
    assert row(catalog)["detectors"]["default"] == "reference"
    call("add", "standard", "other", "--detector", "reference", "-y")
    for args, message in (
        (["add", "standard", "other", "--detector", "reference"], "duplicate"),
        (["add", "standard", "Bad.ID", "--detector", "reference"], "ID must"),
        (["add", "standard", "new", "--detector", "missing"], "unknown detector"),
        (["set", "standard", "default", "--inline-copy", "--shape", "4", "4"], "scalar acceptance"),
    ):
        original = catalog.read_text()
        result = invoke(profile.command, ["detector", *args])
        assert result.exit_code == 2
        assert message in result.stderr
        assert catalog.read_text() == original
    original = catalog.read_text()
    result = invoke(profile.command, ["set", "standard", "--detector", "reference", "-y"])
    assert result.exit_code != 0
    assert "profile detector set PROFILE ID" in result.stderr
    assert catalog.read_text() == original
    call("rm", "standard", "other", "-y")
    original = catalog.read_text()
    result = invoke(profile.command, ["detector", "rm", "standard", "default", "-y"])
    assert result.exit_code == 2
    assert "at least one detector" in result.stderr
    assert catalog.read_text() == original


def test_named_pixel_reset_replaces_response_and_axis(catalog):
    for args in (
        [
            "create",
            "camera",
            "--distance-mm",
            "100",
            "--response",
            "timepix3",
            "--timepix-seed",
            "2",
            "--exposure-s",
            "1",
            "--measured-edges-ev",
            "0,100,200",
        ],
        [
            "set",
            "camera",
            "--response",
            "ideal",
            "--measured-range-ev",
            "0",
            "400",
            "--measured-bin-width-ev",
            "100",
            "-y",
        ],
    ):
        assert_clean_result(invoke(detector.command, args))
    target = tomlkit.parse(catalog.read_text())["detectors"]["camera"]
    assert target["response"] == {"kind": "ideal"}
    assert "measured_edges_eV" not in target["acquisition"]
    assert_clean_result(
        invoke(detector.command, ["reset", "camera", "response", "acquisition", "-y"])
    )
    target = tomlkit.parse(catalog.read_text())["detectors"]["camera"]
    assert "response" not in target and "acquisition" not in target


def test_legacy_noop_keeps_case_identity_migration_changes_only_variant(catalog, monkeypatch):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.profiles import named_profile_identity, named_profile_stem
    from pyrite.campaign.sweep import build_cases

    assert_clean_result(
        invoke(
            profile.command,
            ["create", "identity_probe", "--from", "quickstart", "--detector", "reference"],
        )
    )
    monkeypatch.setenv("PYRITE_CATALOG", str(catalog))

    def resolved():
        identity = named_profile_identity(
            "hopg", catalog_profile="identity_probe", detector_id="default"
        )
        stem = named_profile_stem("hopg", catalog_profile="identity_probe", detector_id="default")
        cases = build_cases(
            material_sweep("hopg", catalog_profile="identity_probe", detector_id="default"), 4, 4
        )
        return identity, stem, cases

    before = resolved()
    call("set", "identity_probe", "default", "--detector", "reference")
    unchanged = resolved()
    assert unchanged == before
    call("migrate", "identity_probe", "-y")
    migrated = resolved()
    assert migrated[2] == before[2]
    assert migrated[1] != before[1]
    assert "detector-default" in migrated[1]
    assert migrated[0]["parameter_sha256"] != before[0]["parameter_sha256"]
    assert before[0]["variant"] is None
    assert migrated[0]["variant"] == "detector-default"
    assert {k: v for k, v in migrated[0]["resolved_parameters"].items() if k != "variant"} == {
        k: v for k, v in before[0]["resolved_parameters"].items() if k != "variant"
    }


def test_deprecated_json_warns_once_and_keeps_schema(catalog):
    from pyrite.cli import command

    result = invoke(command, ["profile", "physical-detector", "show", "standard", "-o", "json"])
    assert result.exit_code == 0
    assert len([line for line in result.stderr.splitlines() if "is deprecated" in line]) == 1
    assert "profile detector show PROFILE physical" in result.stderr
    assert json.loads(result.stdout)["schema"] == "cxr.profile.physical-detector.show"


def test_native_reporting_on_named_detector_and_inline_copy(catalog):
    from pyrite.campaign.observation import resolve_profile_observation
    from pyrite.instrument import NativeAcquisition
    from pyrite.materials import load_material_catalog

    assert_clean_result(
        invoke(
            detector.command,
            [
                "create",
                "native_camera",
                "--distance-mm",
                "400",
                "--shape",
                "2",
                "3",
                "--response",
                "ideal",
                "--exposure-s",
                "1",
                "--reporting-bins",
                "native",
            ],
        )
    )
    call("migrate", "standard", "-y")
    call("set", "standard", "default", "--detector", "native_camera", "-y")
    resolved = resolve_profile_observation(load_material_catalog(catalog), "standard")
    assert isinstance(resolved.acquisition, NativeAcquisition)
    shown = json.loads(invoke(detector.command, ["show", "native_camera", "-o", "json"]).stdout)
    assert shown["payload"]["acquisition"]["reporting_bins"] == "native"
    call("set", "standard", "default", "--inline-copy", "--measured-edges-ev", "0,100,200", "-y")
    resolved = resolve_profile_observation(load_material_catalog(catalog), "standard")
    assert resolved.acquisition.measured_edges_eV == (0, 100, 200)
    assert "reporting_bins" not in row(catalog)["detectors"]["default"]["acquisition"]
