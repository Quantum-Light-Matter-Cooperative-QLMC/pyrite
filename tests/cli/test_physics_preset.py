"""Named physics presets: lifecycle, profile references and identity (#663, ADR-0021)."""

import json
from dataclasses import replace

import pytest

import pyrite
from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite.campaign.profiles import named_profile_identity
from pyrite.cli import command
from pyrite.materials import MaterialConfigError, load_material_catalog
from pyrite.materials.catalog import _refresh_default_catalog
from tests.helpers.cli import assert_clean_result, invoke


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def member(profile):
    catalog = load_material_catalog(profile=profile)
    return (catalog.profile_materials(profile) or catalog.material_keys)[0]


def ok(*args):
    result = invoke(command, list(args))
    assert_clean_result(result)
    return result


def identity(profile):
    # Identity predictors read the process-wide default catalog; reload it so
    # each call sees the edits made through the CLI.
    _refresh_default_catalog(load_material_catalog())
    material = member(profile)
    return named_profile_identity(material, catalog_profile=profile)


def profile_row(name):
    return read_raw(bundled_catalog())["profiles"][name]


def test_unit_bearing_physics_preset_resolves_before_profile_identity(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    preset_dir = _isolate_user_catalog / "physics"
    preset_dir.mkdir(exist_ok=True)
    (preset_dir / "physical.toml").write_text(
        'inelastic_model = "shell-soft-hard"\ninelastic_cutoff_eV = "2 keV"\n'
    )
    ok("profile", "set", "campaign", "--physics", "physical")
    profile_path = _isolate_user_catalog / "profiles" / "campaign.toml"
    profile_path.write_text(profile_path.read_text() + 'inelastic_cutoff_eV = "3 keV"\n')

    catalog = load_material_catalog(profile="campaign")
    physics = pyrite.load_physics("physical", catalog=load_material_catalog())
    effective = catalog.profile_numerics("campaign")
    assert physics.inelastic_cutoff_eV == 2000.0
    assert effective["inelastic_cutoff_eV"] == 3000.0


def test_bundled_reference_preset_restates_built_in_physics():
    assert pyrite.load_physics("electron-standard") == pyrite.Physics()
    refused = invoke(command, ["physics", "preset", "set", "electron-standard", "--straggling"])
    assert refused.exit_code == 1
    assert "read-only" in refused.stderr
    assert "preset create NAME --from electron-standard" in refused.stderr


def test_clone_edit_show_and_load_through_the_api(_isolate_user_catalog):
    bundled = {path: path.read_bytes() for path in bundled_catalog().rglob("*.toml")}
    ok("physics", "preset", "create", "mine", "--from", "electron-standard")
    ok("physics", "preset", "set", "mine", "--elastic-model", "mott", "--straggling")
    assert (_isolate_user_catalog / "physics" / "mine.toml").is_file()
    assert {path: path.read_bytes() for path in bundled_catalog().rglob("*.toml")} == bundled

    shown = json.loads(ok("physics", "preset", "show", "mine", "-o", "json").stdout)["payload"]
    fields = {row["key"]: row for row in shown["fields"]}
    assert shown["source"] == "user"
    assert fields["elastic_model"]["effective"] == "mott"
    assert fields["elastic_model"]["source"] == "preset mine"
    assert fields["positron_transport"]["source"] == "built-in"

    listed = json.loads(ok("physics", "preset", "list", "-o", "json").stdout)["payload"]
    sources = {row["name"]: row["source"] for row in listed["presets"]}
    assert sources == {"electron-standard": "bundled", "mine": "user"}

    physics = pyrite.load_physics("mine", catalog=load_material_catalog())
    assert physics == replace(pyrite.Physics(), elastic_model="mott", straggling=True)
    with pytest.raises(KeyError, match="unknown physics preset"):
        pyrite.load_physics("missing", catalog=load_material_catalog())


def test_profile_reference_resolution_overrides_and_provenance(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("physics", "preset", "create", "shared", "--elastic-model", "mott", "--straggling")
    ok("physics", "set", "campaign", "--straggling")
    ok("profile", "set", "campaign", "--physics", "shared")

    numerics = load_material_catalog(profile="campaign").profile_numerics("campaign")
    assert numerics is not None
    assert numerics["elastic_model"] == "mott"
    shown = json.loads(ok("physics", "show", "campaign", "-o", "json").stdout)["payload"]
    sources = {row["key"]: row["source"] for row in shown["fields"]}
    assert shown["preset"] == "shared"
    assert sources["elastic_model"] == "preset shared"
    assert sources["straggling"] == "profile"
    assert sources["inelastic_model"] == "built-in"

    refused = invoke(command, ["physics", "preset", "delete", "shared", "--yes"])
    assert refused.exit_code == 1
    assert "still referenced by profiles: campaign" in refused.stderr

    ok("physics", "preset", "rename", "shared", "common")
    assert profile_row("campaign")["physics"] == "common"
    ok("profile", "remove", "campaign", "--physics")
    assert "physics" not in profile_row("campaign")
    ok("physics", "preset", "delete", "common", "--yes")
    assert not (_isolate_user_catalog / "physics" / "common.toml").exists()


def test_adoption_and_unused_presets_keep_identity(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("physics", "set", "campaign", "--straggling", "--elastic-model", "mott")
    before = identity("campaign")

    ok("physics", "preset", "create", "unused", "--atomic-electron-deflection", "none")
    assert identity("campaign") == before

    preview = ok("physics", "preset", "adopt", "campaign", "adopted", "--dry-run")
    assert "-straggling = true" in preview.stdout
    ok("physics", "preset", "adopt", "campaign", "adopted")
    row = profile_row("campaign")
    assert row["physics"] == "adopted"
    assert "straggling" not in row and "elastic_model" not in row
    assert identity("campaign") == before

    ok("physics", "preset", "rename", "adopted", "renamed")
    assert identity("campaign") == before

    ok("physics", "preset", "set", "renamed", "--elastic-model", "elsepa")
    assert identity("campaign") != before


def test_invalid_combinations_and_references_write_nothing(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("physics", "set", "campaign", "--radiative-model", "uncoupled")
    ok("numerics", "set", "campaign", "--energy-model", "frozen")
    ok("physics", "preset", "create", "coupled", "--radiative-model", "uncoupled")
    before = snapshot(_isolate_user_catalog)

    rejected = invoke(
        command,
        [
            "physics",
            "preset",
            "create",
            "bad",
            "--inelastic-model",
            "continuous",
            "--inelastic-cutoff-ev",
            "50",
        ],
    )
    assert rejected.exit_code == 1
    # A valid preset that conflicts with a referencing profile's numerics.
    ok("profile", "set", "campaign", "--physics", "coupled")
    conflict = invoke(
        command,
        [
            "physics",
            "preset",
            "set",
            "coupled",
            "--inelastic-model",
            "shell-soft-hard",
            "--inelastic-cutoff-ev",
            "50",
        ],
    )
    assert conflict.exit_code == 1
    assert "midpoint" in conflict.stderr
    missing = invoke(command, ["profile", "set", "campaign", "--physics", "nope"])
    assert missing.exit_code == 1
    assert "unknown physics preset 'nope'" in missing.stderr
    after = snapshot(_isolate_user_catalog)
    after.pop("profiles/campaign.toml")
    before.pop("profiles/campaign.toml")
    assert after == before


def test_catalog_rejects_unknown_reference_and_preset_keys(tmp_path):
    from shutil import copytree

    catalog = tmp_path / "catalog"
    copytree(bundled_catalog(), catalog)
    (catalog / "physics" / "odd.toml").write_text("n_electrons = {values = [10]}\n")
    with pytest.raises(MaterialConfigError, match=r"physics\.odd\.n_electrons: unknown key"):
        load_material_catalog(catalog)
    (catalog / "physics" / "odd.toml").unlink()
    quickstart = catalog / "profiles" / "quickstart.toml"
    quickstart.write_text('physics = "missing"\n' + quickstart.read_text())
    with pytest.raises(MaterialConfigError, match="unknown physics preset 'missing'"):
        load_material_catalog(catalog)
