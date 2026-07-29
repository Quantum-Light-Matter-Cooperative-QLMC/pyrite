from __future__ import annotations

import json

from cxr_mc.cli import _catalog_io, profile
from tests.cli_helpers import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0, 60.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }

[profiles.sub_100keV]
materials = ["hopg"]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0, 50.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[materials.hopg]

[materials.mose2]
"""

_REFERENCED_CATALOG = (
    _CATALOG
    + """
[energy_grids.sub_100keV]
line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 100.0, num = 10 } }, source = "derived" },
]
"""
)


def _catalog(tmp_path, monkeypatch, text=_CATALOG):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(text)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def test_list_text_and_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    text = invoke(profile.command, ["list"])
    assert_clean_result(text)
    assert "standard: all materials (implicit), 1 material overrides" in text.stdout
    assert "sub_100keV: 1 materials, 0 material overrides" in text.stdout

    machine = invoke(profile.command, ["list", "--json"])
    assert_clean_result(machine)
    document = json.loads(machine.stdout)
    assert document["schema"] == "cxr.profile.list"
    (standard, sub) = document["payload"]["profiles"]
    assert standard["name"] == "standard"
    assert standard["materials"] is None
    assert standard["overrides"] == ["hopg"]
    assert sub["materials"] == ["hopg"]


def test_show_and_bare_name_alias(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(shown)
    assert "[sub_100keV]" in shown.stdout
    assert "energy: [30, 50]" in shown.stdout
    assert "materials: hopg" in shown.stdout

    aliased = invoke(profile.command, ["sub_100keV"])
    assert_clean_result(aliased)
    assert aliased.stdout == shown.stdout

    machine = invoke(profile.command, ["show", "standard", "--json"])
    assert_clean_result(machine)
    payload = json.loads(machine.stdout)["payload"]
    assert payload["materials"] is None
    assert payload["overrides"] == {"hopg": ["thickness_ang"]}


def test_show_unknown_profile_suggests_and_points_to_create(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["show", "sub_100kv"])

    assert result.exit_code == 1
    assert "unknown profile: sub_100kv" in result.stderr
    assert "Did you mean: sub_100keV" in result.stderr
    assert "cxr profile create sub_100kv" in result.stderr


def test_create_clones_source_and_applies_range_overrides(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command, ["create", "sub_200keV", "--from", "sub_100keV", "--energy", "150,200"]
    )

    assert_clean_result(result, stdout="created profile sub_200keV\n")
    text = catalog.read_text()
    assert "[profiles.sub_200keV]" in text
    assert "energy_keV = {values = [150.0, 200.0]}" in text
    # Cloned grids carry over; membership and overrides are not cloned.
    assert "tilt_deg = {values = [5.0]}" in text
    section = text.split("[profiles.sub_200keV]", 1)[1].split("\n[", 1)[0]
    assert "materials" not in section


def test_create_range_syntax_expands_stop_inclusive(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["create", "ranged", "--energy", "30,50:100:25"])

    assert_clean_result(result)
    assert "energy_keV = {values = [30.0, 50.0, 75.0, 100.0]}" in catalog.read_text()


def test_create_accepts_atomic_initial_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["create", "demo", "--materials", "hopg,mose2"],
    )

    assert_clean_result(result, stdout="created profile demo\n")
    section = catalog.read_text().split("[profiles.demo]", 1)[1].split("\n[", 1)[0]
    assert 'materials = ["hopg", "mose2"]' in section


def test_create_existing_or_invalid_name_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    existing = invoke(profile.command, ["create", "standard"])
    invalid = invoke(profile.command, ["create", "bad name"])

    assert existing.exit_code == 1
    assert "already exists" in existing.stderr
    assert "cxr profile set standard" in existing.stderr
    assert invalid.exit_code == 2
    assert "invalid profile name" in invalid.stderr


def test_set_replaces_grid_and_members_set_owns_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--polar", "10:30:10"])
    members = invoke(profile.command, ["members", "set", "sub_100keV", "hopg", "mose2"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert_clean_result(members, stdout="updated profile sub_100keV membership\n")
    text = catalog.read_text()
    assert "tilt_deg = {values = [10.0, 20.0, 30.0]}" in text
    assert 'materials = ["hopg", "mose2"]' in text


def test_set_unknown_material_in_membership_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "set", "sub_100keV", "unobtainium"])

    assert result.exit_code == 1
    assert "unknown material: unobtainium" in result.stderr


def test_set_on_standard_prompts_and_yes_skips(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(profile.command, ["set", "standard", "--energy", "40"], input="n\n")
    assert declined.exit_code == 1
    assert "profile 'standard'" in declined.stderr
    assert catalog.read_text() == original

    accepted = invoke(profile.command, ["set", "standard", "--energy", "40", "-y"])
    assert_clean_result(accepted, stdout="updated profile standard\n")
    assert "energy_keV = {values = [40.0]}" in catalog.read_text()


def test_set_nonstandard_never_prompts(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    # No input supplied: an unwanted prompt would hit EOF and abort.
    result = invoke(profile.command, ["set", "sub_100keV", "--energy", "40"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")


def test_add_unions_sorts_deduplicates(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--energy", "75,30"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "energy_keV = {values = [30.0, 50.0, 75.0]}" in catalog.read_text()


def test_hidden_add_material_option_warns_and_remains_compatible(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["add", "sub_100keV", "--energy", "75", "--materials", "mose2,hopg"],
    )

    assert result.exit_code == 0
    assert result.stdout == "updated profile sub_100keV: added mose2; already members: hopg\n"
    assert "use 'cxr profile members add sub_100keV mose2 hopg'" in result.stderr
    text = catalog.read_text()
    assert "energy_keV = {values = [30.0, 50.0, 75.0]}" in text
    assert 'materials = ["hopg", "mose2"]' in text


def test_add_accepts_electron_count_grids(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--ne-line", "100,200"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "n_electrons = {values = [100, 200]}" in catalog.read_text()


def test_remove_values_and_missing_value_error(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    removed = invoke(profile.command, ["remove", "sub_100keV", "--energy", "50"])
    assert_clean_result(removed, stdout="updated profile sub_100keV\n")
    assert "energy_keV = {values = [30.0]}" in catalog.read_text()

    missing = invoke(profile.command, ["remove", "sub_100keV", "--energy", "999"])
    assert missing.exit_code == 1
    assert "energy values not present in profile sub_100keV: 999" in missing.stderr


def test_members_remove_and_no_op_requires_range_option(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "remove", "sub_100keV", "hopg", "diamond"])
    assert_clean_result(result)
    assert "removed hopg" in result.stdout
    assert "not members: diamond" in result.stdout
    assert "materials = []" in catalog.read_text()

    bare = invoke(profile.command, ["remove", "sub_100keV"])
    assert bare.exit_code == 2
    assert "provide a range option" in bare.stderr


def test_member_group_selectors_expand_in_catalog_order_and_support_dry_run(tmp_path, monkeypatch):
    from cxr_mc import scan

    catalog = _catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(
        scan,
        "load_manifest_groups",
        lambda: {
            "no_verified_dw": ["mose2"],
            "high_energy_materials": ["hopg"],
        },
    )

    set_result = invoke(
        profile.command,
        ["members", "set", "sub_100keV", "--unverified-dw", "hopg"],
    )
    assert_clean_result(set_result, stdout="updated profile sub_100keV membership\n")
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()

    removed = invoke(
        profile.command,
        ["members", "remove", "sub_100keV", "--high-energy-only"],
    )
    assert_clean_result(removed, stdout="updated profile sub_100keV: removed hopg\n")
    assert 'materials = ["mose2"]' in catalog.read_text()

    original = catalog.read_text()
    dry_run = invoke(
        profile.command,
        ["members", "add", "sub_100keV", "--high-energy-only", "--dry-run"],
    )
    assert_clean_result(dry_run)
    assert 'materials = ["hopg", "mose2"]' in dry_run.stdout
    assert catalog.read_text() == original


def test_member_group_selectors_require_a_selector_or_material(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "set", "sub_100keV"])

    assert result.exit_code == 1
    assert "provide MATERIAL keys, --unverified-dw, or --high-energy-only" in result.stderr


def test_members_reset_restores_implicit_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    reset = invoke(profile.command, ["members", "reset", "sub_100keV"])
    shown = invoke(profile.command, ["show", "sub_100keV"])

    assert_clean_result(
        reset,
        stdout="reset profile sub_100keV membership to all in-use materials (implicit)\n",
    )
    section = catalog.read_text().split("[profiles.sub_100keV]", 1)[1].split("\n[", 1)[0]
    assert "materials" not in section
    assert_clean_result(shown)
    assert "materials: all in-use materials (implicit)" in shown.stdout


def test_add_on_standard_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(profile.command, ["add", "standard", "--energy", "40"], input="n\n")

    assert declined.exit_code == 1
    assert "add values to profile 'standard'" in declined.stderr
    assert catalog.read_text() == original


def test_empty_updates_are_usage_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    for verb in ("set", "add", "remove"):
        result = invoke(profile.command, [verb, "sub_100keV"])
        assert result.exit_code == 2
        assert "provide a range option" in result.stderr


def test_delete_requires_yes_and_removes_profile(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(profile.command, ["delete", "sub_100keV"], input="n\n")
    assert declined.exit_code == 1
    assert "delete profile 'sub_100keV'" in declined.stderr
    assert catalog.read_text() == original

    deleted = invoke(profile.command, ["delete", "sub_100keV", "-y"])
    assert_clean_result(deleted, stdout="deleted profile sub_100keV\n")
    assert "[profiles.sub_100keV]" not in catalog.read_text()


def test_delete_forbids_standard(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["delete", "standard", "-y"])

    assert result.exit_code == 1
    assert "cannot delete profile 'standard'" in result.stderr


def test_delete_blocked_by_energy_grid_store_referent(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _REFERENCED_CATALOG)
    original = catalog.read_text()

    result = invoke(profile.command, ["delete", "sub_100keV", "-y"])

    assert result.exit_code == 1
    assert "still referenced by" in result.stderr
    assert "energy_grids.sub_100keV" in result.stderr
    assert catalog.read_text() == original


def test_delete_json_requires_yes_and_reports_envelope(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    refused = invoke(profile.command, ["delete", "sub_100keV", "--json"])
    assert refused.exit_code == 2
    assert "--json requires --yes" in refused.stderr

    deleted = invoke(profile.command, ["delete", "sub_100keV", "--json", "-y"])
    assert_clean_result(deleted)
    document = json.loads(deleted.stdout)
    assert document["schema"] == "cxr.profile.delete"
    assert document["ok"] is True
    assert document["payload"] == {"deleted": "sub_100keV"}


def test_rename_moves_profile_and_energy_grid_bucket(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _REFERENCED_CATALOG)

    result = invoke(profile.command, ["rename", "sub_100keV", "sub100"])

    assert_clean_result(result, stdout="renamed profile sub_100keV to sub100\n")
    text = catalog.read_text()
    assert "[profiles.sub_100keV]" not in text
    assert "[profiles.sub100]" in text
    assert "[energy_grids.sub_100keV]" not in text
    assert "[energy_grids.sub100]" in text


def test_rename_forbids_standard(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["rename", "standard", "renamed"])

    assert result.exit_code == 1
    assert "cannot rename profile 'standard'" in result.stderr


def test_rename_rejects_existing_or_same_name(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    collision = invoke(profile.command, ["rename", "sub_100keV", "standard"])
    assert collision.exit_code == 1
    assert "already exists" in collision.stderr

    same = invoke(profile.command, ["rename", "sub_100keV", "sub_100keV"])
    assert same.exit_code == 1
    assert "already named" in same.stderr

    missing = invoke(profile.command, ["rename", "sub_100kv", "sub100"])
    assert missing.exit_code == 1
    assert "unknown profile: sub_100kv" in missing.stderr

    assert catalog.read_text() == original


def test_rename_dry_run_writes_nothing(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(profile.command, ["rename", "sub_100keV", "sub100", "--dry-run"])

    assert_clean_result(result)
    assert catalog.read_text() == original


def test_add_material_and_remove_material_roundtrip(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    added = invoke(profile.command, ["add-material", "sub_100keV", "mose2", "hopg"])
    assert added.exit_code == 0
    assert "use 'cxr profile members add sub_100keV MATERIAL...'" in added.stderr
    assert "added mose2" in added.stdout
    assert "already members: hopg" in added.stdout
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()

    removed = invoke(profile.command, ["remove-material", "sub_100keV", "hopg", "diamond"])
    assert removed.exit_code == 0
    assert "use 'cxr profile members remove sub_100keV MATERIAL...'" in removed.stderr
    assert "removed hopg" in removed.stdout
    assert "not members: diamond" in removed.stdout
    assert 'materials = ["mose2"]' in catalog.read_text()


def test_membership_verbs_require_explicit_list(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "add", "standard", "mose2"])

    assert result.exit_code == 1
    assert "implicit all-in-use-materials membership" in result.stderr
    assert "cxr profile members set standard MATERIAL" in result.stderr


def test_add_materials_explains_implicit_membership_is_already_all(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "add", "standard", "mose2"])

    assert result.exit_code == 1
    assert "already includes every material" in result.stderr
    assert "cxr profile members set standard MATERIAL" in result.stderr


def test_membership_verbs_reject_unknown_material(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "add", "sub_100keV", "unobtainium"])

    assert result.exit_code == 1
    assert "unknown material: unobtainium" in result.stderr


def test_add_material_all_seeds_implicit_membership(tmp_path, monkeypatch):
    """--all seeds an implicit all-in-use profile straight from mats_to_sim.toml's
    verified list -- the escape hatch `test_membership_verbs_require_explicit_list`
    otherwise requires (`cxr profile set NAME --materials KEY,...`, typed by hand)."""
    from cxr_mc import scan

    catalog = _catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["hopg", "mose2"])

    result = invoke(profile.command, ["add-material", "standard", "--all", "-y"])

    assert result.exit_code == 0
    assert "cxr profile members add standard MATERIAL..." in result.stderr
    assert "added hopg, mose2" in result.stdout
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()


def test_add_material_all_extends_and_skips_existing_members(tmp_path, monkeypatch):
    from cxr_mc import scan

    catalog = _catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["hopg", "mose2"])

    result = invoke(profile.command, ["add-material", "sub_100keV", "--all"])

    assert result.exit_code == 0
    assert "cxr profile members add sub_100keV MATERIAL..." in result.stderr
    assert "added mose2" in result.stdout
    assert "already members: hopg" in result.stdout
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()


def test_add_material_requires_materials_or_all(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add-material", "sub_100keV"])

    assert result.exit_code == 2
    assert "provide MATERIAL keys or --all" in result.stderr


def test_add_material_short_all_flag(tmp_path, monkeypatch):
    """-a is the short form of --all, matching the other listing options."""
    from cxr_mc import scan

    catalog = _catalog(tmp_path, monkeypatch)
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["hopg", "mose2"])

    result = invoke(profile.command, ["add-material", "sub_100keV", "-a"])

    assert result.exit_code == 0
    assert "cxr profile members add sub_100keV MATERIAL..." in result.stderr
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()


def test_dry_run_writes_nothing_and_never_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    create = invoke(profile.command, ["create", "demo", "--dry-run"])
    assert_clean_result(create)
    assert "+[profiles.demo]" in create.stdout.replace(" ", "")

    delete = invoke(profile.command, ["delete", "sub_100keV", "--dry-run"])
    assert_clean_result(delete)
    assert "-[profiles.sub_100keV]" in delete.stdout.replace(" ", "")

    standard = invoke(profile.command, ["add", "standard", "--energy", "40", "--dry-run"])
    assert_clean_result(standard)

    assert catalog.read_text() == original
