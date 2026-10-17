"""Named beams/detectors live outside installed package data (#417)."""

import json
import shutil

import pytest

from pyrite import _catalog_layout as layout
from pyrite.cli import _catalog_io
from pyrite.cli.commands import beam, detector
from pyrite.devtools import dev_cli
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    root = tmp_path / "site-packages" / "pyrite" / "data" / "catalog"
    shutil.copytree(layout.bundled_catalog(), root)
    monkeypatch.setattr(layout, "bundled_catalog", lambda: root)
    monkeypatch.setattr(_catalog_io, "active_catalog_path", lambda: root)
    return root


def snapshot(root):
    return {
        str(file.relative_to(root)): file.read_bytes() for file in root.rglob("*") if file.is_file()
    }


@pytest.mark.parametrize(
    "kind,module,source,option",
    [
        ("beam", beam, "example", "--rep-rate-hz"),
        ("detector", detector, "reference", "--observation-angle"),
    ],
)
def test_user_instrument_lifecycle_preserves_package(
    bundled, _isolate_user_catalog, kind, module, source, option
):
    before = snapshot(bundled)
    command = module.command
    assert_clean_result(
        invoke(
            command,
            ["create", *(["clone"] if module is detector else []), "mine", "--from", source],
        )
    )
    table = kind + "s"
    original = layout.read_raw(bundled)[table][source]
    assert layout.read_raw(bundled)[table]["mine"] == original
    assert (_isolate_user_catalog / table / "mine.toml").is_file()
    assert_clean_result(invoke(command, ["set", "mine", option, "45", "--yes"]))
    assert_clean_result(invoke(command, ["rename", "mine", "renamed"]))
    listed = invoke(command, ["list", "-o", "json"])
    assert_clean_result(listed)
    origins = {row["name"]: row["source"] for row in json.loads(listed.stdout)["payload"][table]}
    assert origins["renamed"] == "user"
    assert origins[source] == "bundled"
    assert "[user]" in invoke(command, ["list"]).stdout
    assert_clean_result(invoke(command, ["delete", "renamed", "--yes"]))
    assert not (_isolate_user_catalog / table / "renamed.toml").exists()
    assert snapshot(bundled) == before


@pytest.mark.parametrize("module,source", [(beam, "example"), (detector, "reference")])
@pytest.mark.parametrize("operation", ["set", "rename", "delete"])
def test_bundled_instrument_edits_fail_with_copy_hint(bundled, module, source, operation):
    before = snapshot(bundled)
    args = {
        "set": ["set", source, "--label", "changed"],
        "rename": ["rename", source, "new-name"],
        "delete": ["delete", source, "--yes"],
    }[operation]
    result = invoke(module.command, args)
    assert result.exit_code == 1
    assert "read-only" in result.stderr
    assert f"create NAME --from {source}" in result.stderr
    assert snapshot(bundled) == before


@pytest.mark.parametrize("module,source", [(beam, "example"), (detector, "reference")])
def test_copy_rejects_case_collision_and_dry_run_writes_nothing(
    bundled, module, source, _isolate_user_catalog
):
    before = snapshot(bundled), snapshot(_isolate_user_catalog)
    result = invoke(
        module.command,
        ["create", *(["clone"] if module is detector else []), source.upper(), "--from", source],
    )
    assert result.exit_code == 1
    assert "already exists" in result.stderr
    result = invoke(
        module.command,
        [
            "create",
            *(["clone"] if module is detector else []),
            "mine",
            "--from",
            source,
            "--dry-run",
        ],
    )
    assert_clean_result(result)
    assert "mine" in result.stdout
    assert (snapshot(bundled), snapshot(_isolate_user_catalog)) == before


@pytest.mark.parametrize("kind,source", [("beam", "example"), ("detector", "reference")])
def test_maintainer_edits_bundle_and_restores_selection(
    bundled, kind, source, monkeypatch, _isolate_user_catalog
):
    monkeypatch.setenv("PYRITE_CATALOG", "/elsewhere")
    dev_cli.main([kind, "set", source, "--label", "maintained", "--yes"])
    with layout.bundled_profile_writes():
        assert layout.read_raw(bundled)[kind + "s"][source]["label"] == "maintained"
    assert not (_isolate_user_catalog / (kind + "s") / f"{source}.toml").exists()
    import os

    assert os.environ["PYRITE_CATALOG"] == "/elsewhere"


@pytest.mark.parametrize(
    "module,source,options,key",
    [
        (beam, "example", ["--rep-rate-hz", "123"], "rep_rate_hz"),
        (detector, "pixelated_reference", ["--distance-mm", "123"], "distance_mm"),
    ],
)
def test_copy_preserves_nested_fields_and_applies_overrides(bundled, module, source, options, key):
    table = "beams" if module is beam else "detectors"
    original = layout.read_raw(bundled)[table][source]
    result = invoke(
        module.command,
        ["create", *(["planar"] if module is detector else []), "mine", "--from", source, *options],
    )
    assert_clean_result(result)
    copied = layout.read_raw(bundled)[table]["mine"]
    assert copied == {**original, key: 123.0}
    assert layout.read_raw(bundled)[table][source] == original


def test_bundled_detector_reset_and_machine_delete_report_copy_hint(bundled):
    result = invoke(detector.command, ["reset", "pixelated_reference", "response", "--yes"])
    assert result.exit_code == 1
    assert "create NAME --from pixelated_reference" in result.stderr
    result = invoke(beam.command, ["delete", "example", "--yes", "-o", "json"])
    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert "create NAME --from example" in str(document)
    assert result.stderr == ""
