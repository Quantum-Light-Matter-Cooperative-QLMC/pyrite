"""Ownership, compatibility and no-write contracts for CLI cleanup (#551)."""

import json

import pytest

from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite._numerics import PHYSICS_FIELDS
from pyrite.catalog_edit import CatalogDraft
from pyrite.cli import command
from pyrite.detectors.editing import create_detector
from tests.helpers.cli import assert_clean_result, invoke


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_detector_creation_help_and_incompatible_types():
    scalar = invoke(command, ["detector", "create", "scalar", "--help"])
    planar = invoke(command, ["detector", "create", "planar", "--help"])
    assert_clean_result(scalar)
    assert_clean_result(planar)
    assert "--distance-mm" not in scalar.stdout
    assert "--observation-angle" not in planar.stdout
    assert "--solid-angle" in scalar.stdout
    assert "--distance-mm" in planar.stdout
    for kind, args in (("scalar", ["--distance-mm", "10"]), ("planar", ["--solid-angle", "1"])):
        result = invoke(command, ["detector", "create", kind, "bad", *args])
        assert result.exit_code == 2
        assert "No such option" in result.stderr


def test_legacy_detector_create_parses_options_and_warns_only_on_stderr(_isolate_user_catalog):
    result = invoke(command, ["detector", "create", "legacy", "--observation-angle", "91"])
    assert result.exit_code == 0
    assert result.stdout == "created detector legacy\n"
    assert result.stderr.count("is deprecated") == 1
    assert "0.17.0" in result.stderr
    assert read_raw(bundled_catalog())["detectors"]["legacy"]["observation_angle_deg"] == 91


def test_first_use_dry_run_does_not_create_user_catalog(tmp_path, monkeypatch):
    user = tmp_path / "new" / "catalog"
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(user))
    args = ["detector", "create", "scalar", "first", "--observation-angle", "90"]
    preview = invoke(command, [*args, "--dry-run"])
    assert_clean_result(preview)
    assert not user.exists()
    assert_clean_result(invoke(command, args))
    assert (user / "detectors" / "first.toml").is_file()


def test_cli_and_service_detector_drafts_match_and_cancel_is_inert(_isolate_user_catalog):
    selected = bundled_catalog()
    before = snapshot(_isolate_user_catalog)
    draft = CatalogDraft.open(selected)
    create_detector(draft.document, "service", scalar={"observation_angle_deg": 92}, kind="scalar")
    draft.validate()
    assert "+[detectors.service]" in draft.diff().replace(" ", "")
    assert snapshot(_isolate_user_catalog) == before
    draft.save()
    result = invoke(command, ["detector", "create", "scalar", "batch", "--observation-angle", "92"])
    assert_clean_result(result)
    rows = read_raw(selected)["detectors"]
    assert rows["service"] == rows["batch"]


def test_draft_failure_readonly_and_concurrent_edit_never_save(_isolate_user_catalog):
    selected = bundled_catalog()
    before = snapshot(_isolate_user_catalog)
    invalid = CatalogDraft.open(selected)
    create_detector(invalid.document, "invalid", scalar={"solid_angle_sr": -1})
    with pytest.raises(ValueError):
        invalid.save()
    assert snapshot(_isolate_user_catalog) == before
    readonly = CatalogDraft.open(selected)
    readonly.document["detectors"]["reference"]["observation_angle_deg"] = 42
    with pytest.raises(ValueError, match="read-only"):
        readonly.save()
    assert snapshot(_isolate_user_catalog) == before
    stale = CatalogDraft.open(selected)
    create_detector(stale.document, "stale", source="reference")
    assert_clean_result(
        invoke(command, ["detector", "create", "clone", "newer", "--from", "reference"])
    )
    newer = snapshot(_isolate_user_catalog)
    with pytest.raises(ValueError, match="changed while editing"):
        stale.save()
    assert snapshot(_isolate_user_catalog) == newer


def test_calculation_groups_have_api_field_ownership_and_scoped_resets(_isolate_user_catalog):
    assert_clean_result(invoke(command, ["profile", "create", "campaign", "--from", "quickstart"]))
    assert_clean_result(invoke(command, ["physics", "set", "campaign", "--straggling"]))
    assert_clean_result(invoke(command, ["numerics", "set", "campaign", "--mosaic-nodes", "5"]))
    for noun in ("physics", "numerics"):
        shown = invoke(command, [noun, "show", "campaign", "-o", "json"])
        assert_clean_result(shown)
        fields = json.loads(shown.stdout)["payload"]["fields"]
        assert fields
        assert all((field["key"] in PHYSICS_FIELDS) == (noun == "physics") for field in fields)
    before = snapshot(_isolate_user_catalog)
    frozen = invoke(command, ["numerics", "set", "campaign", "--mosaic-nodes", "7", "--dry-run"])
    assert_clean_result(frozen)
    assert snapshot(_isolate_user_catalog) == before
    assert_clean_result(invoke(command, ["physics", "reset", "campaign"]))
    saved = read_raw(bundled_catalog())["profiles"]["campaign"]
    assert "straggling" not in saved
    assert saved["mosaic_nodes"] == 5
    bad = invoke(command, ["numerics", "set", "campaign", "--straggling"])
    assert bad.exit_code == 2
    assert "No such option" in bad.stderr


def test_calculation_cross_model_errors_and_dry_run_do_not_write(_isolate_user_catalog):
    assert_clean_result(invoke(command, ["profile", "create", "campaign", "--from", "quickstart"]))
    assert_clean_result(
        invoke(command, ["physics", "set", "campaign", "--radiative-model", "uncoupled"])
    )
    assert_clean_result(
        invoke(command, ["numerics", "set", "campaign", "--energy-model", "frozen"])
    )
    before = snapshot(_isolate_user_catalog)
    rejected = invoke(
        command,
        [
            "physics",
            "set",
            "campaign",
            "--inelastic-model",
            "shell-soft-hard",
            "--inelastic-cutoff-ev",
            "50",
        ],
    )
    assert rejected.exit_code == 1
    assert "midpoint" in rejected.stderr
    assert snapshot(_isolate_user_catalog) == before


def test_material_inspection_is_intrinsic_and_creation_help_is_typed():
    shown = invoke(command, ["material", "show", "hopg", "-o", "json"])
    assert_clean_result(shown)
    document = json.loads(shown.stdout)
    assert document["schema"] == "pyrite.material.show"
    assert not {"ranges", "profile", "thickness", "energy", "polar", "azimuth"}.intersection(
        document["payload"]
    )
    crystal = invoke(command, ["material", "create", "crystal", "--help"])
    medium = invoke(command, ["material", "create", "amorphous", "--help"])
    assert "--cif" in crystal.stdout and "--cif" not in medium.stdout
    assert "--mass-fraction" in medium.stdout and "--mass-fraction" not in crystal.stdout


def test_remote_job_alias_preserves_json_and_warns_once(monkeypatch):
    from pyrite.remote import viewer

    monkeypatch.setattr(viewer, "jobs_raw", lambda: "j1\t12\tFalse\trun\thopg\tdone\n")
    canonical = invoke(command, ["remote", "job", "list", "-o", "json"])
    legacy = invoke(command, ["job", "list", "-o", "json"])
    assert_clean_result(canonical)
    assert legacy.exit_code == 0
    assert legacy.stdout == canonical.stdout
    assert legacy.stderr.count("is deprecated") == 1


def test_gdf_alias_keeps_machine_schema_and_stdout(tmp_path):
    from tests.cli.test_gdf import fields, write_gdf

    path = write_gdf(tmp_path / "particles.gdf", fields(), times=(0,))
    canonical = invoke(
        command, ["beam", "gdf", "inspect", str(path), "--time-s", "0", "-o", "json"]
    )
    legacy = invoke(command, ["beam", "gdf-inspect", str(path), "--time-s", "0", "-o", "json"])
    assert_clean_result(canonical)
    assert legacy.exit_code == 0
    assert legacy.stdout == canonical.stdout
    assert json.loads(canonical.stdout)["schema"] == "cxr.beam.gdf-inspect"
    assert legacy.stderr.count("is deprecated") == 1


def test_exported_catalog_reproduces_cli_configuration(tmp_path):
    from tests.helpers.user_catalog import copy_full_catalog

    selected = copy_full_catalog(tmp_path / "catalog")
    draft = CatalogDraft.open(selected)
    exported = selected.parent / "exported.toml"
    exported.write_text(draft.serialize())
    for path in (
        ["physics", "show", "quickstart"],
        ["numerics", "show", "quickstart"],
        ["profile", "ranges", "show", "quickstart", "hopg"],
    ):
        original = invoke(command, ["--catalog", str(selected), *path, "-o", "json"])
        restored = invoke(command, ["--catalog", str(exported), *path, "-o", "json"])
        assert_clean_result(original)
        assert_clean_result(restored)
        assert json.loads(original.stdout) == json.loads(restored.stdout)
