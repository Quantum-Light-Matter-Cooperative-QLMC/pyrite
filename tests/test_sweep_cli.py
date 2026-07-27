from __future__ import annotations

import json

from cxr_mc.cli import sweep
from tests.cli_helpers import assert_clean_result, invoke


def test_show_material_json_reports_effective_ranges_and_sources():
    result = invoke(sweep.command, ["show", "hopg", "--json"])

    assert_clean_result(result)
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.sweep.show"
    assert document["payload"]["material"] == "hopg"
    assert document["payload"]["profile"] == "standard"
    assert {row["name"] for row in document["payload"]["ranges"]} == {
        "thickness",
        "energy",
        "polar",
        "azimuth",
    }
    assert all(not row["overridden"] for row in document["payload"]["ranges"])


def test_show_no_material_json_reports_profiles_and_overriding_materials(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }

[profiles.standard.overrides.diamond]
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[materials.hopg]

[materials.mose2]

[materials.diamond]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)

    result = invoke(sweep.command, ["show", "--json"])

    assert_clean_result(result)
    document = json.loads(result.stdout)
    (profile,) = document["payload"]["profiles"]
    assert profile["name"] == "standard"
    assert profile["materials"] == [{"material": "hopg", "overrides": ["thickness"]}]
    assert document["payload"]["inheriting_profiles"] == {"standard": 2}


def test_show_no_material_text_lists_profile_defaults_overrides_and_inheritance(
    monkeypatch, tmp_path
):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }

[materials.hopg]

[materials.mose2]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)

    result = invoke(sweep.command, ["show"])

    assert_clean_result(result)
    assert "[standard]" in result.stdout
    assert "hopg: overrides thickness" in result.stdout
    assert "1 materials inherit standard" in result.stdout


def test_set_dry_run_preserves_catalog_and_energy_grid_siblings(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    original = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[materials.hopg]
"""
    catalog.write_text(original)
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "hopg", "--thickness", "2000", "--dry-run"])

    assert_clean_result(result)
    assert catalog.read_text() == original
    assert "thickness_ang = {values = [2000.0]}" in result.stdout
    assert "E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }" in result.stdout


def test_set_reset_removes_one_override_without_touching_energy_grid(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[materials.hopg]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "hopg", "--reset", "thickness"])

    assert_clean_result(result, stdout="updated materials.hopg\n")
    text = catalog.read_text()
    assert "thickness_ang = { values = [2000.0] }" not in text
    assert "E_grid_brem" in text


def test_set_rejects_profile_material_and_empty_updates():
    conflict = invoke(
        sweep.command, ["set", "hopg", "--profile", "standard", "--thickness", "1000"]
    )
    empty = invoke(sweep.command, ["set"])

    assert conflict.exit_code == 2
    assert "mutually exclusive" in conflict.stderr
    assert empty.exit_code == 2
    assert "provide a range option or --reset" in empty.stderr


def _catalog_with_existing_thickness_override(tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }

[materials.hopg]
"""
    )
    return catalog


def test_set_declining_overwrite_confirmation_aborts_without_writing(monkeypatch, tmp_path):
    catalog = _catalog_with_existing_thickness_override(tmp_path)
    original = catalog.read_text()
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "hopg", "--thickness", "3000"], input="n\n")

    assert result.exit_code == 1
    assert "overwrite thickness for materials.hopg?" in result.stderr
    assert "Aborted!" in result.stderr
    assert catalog.read_text() == original


def test_set_accepting_overwrite_confirmation_writes_new_value(monkeypatch, tmp_path):
    catalog = _catalog_with_existing_thickness_override(tmp_path)
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "hopg", "--thickness", "3000"], input="y\n")

    assert result.exit_code == 0
    assert result.stdout == "updated materials.hopg\n"
    assert "overwrite thickness for materials.hopg?" in result.stderr
    assert "thickness_ang = {values = [3000.0]}" in catalog.read_text()


def test_set_yes_flag_skips_overwrite_confirmation(monkeypatch, tmp_path):
    catalog = _catalog_with_existing_thickness_override(tmp_path)
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "hopg", "--thickness", "3000", "--yes"])

    assert_clean_result(result, stdout="updated materials.hopg\n")
    assert "thickness_ang = {values = [3000.0]}" in catalog.read_text()


def test_set_dry_run_never_prompts_even_when_overwriting(monkeypatch, tmp_path):
    catalog = _catalog_with_existing_thickness_override(tmp_path)
    original = catalog.read_text()
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    # No input supplied: if --dry-run prompted, this would hit EOF and abort.
    result = invoke(sweep.command, ["set", "hopg", "--thickness", "3000", "--dry-run"])

    assert_clean_result(result)
    assert catalog.read_text() == original
    assert "thickness_ang = {values = [3000.0]}" in result.stdout


def test_set_new_override_does_not_prompt(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[materials.hopg]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    # No prior thickness override and no input supplied: an unwanted prompt
    # would hit EOF and abort.
    result = invoke(sweep.command, ["set", "hopg", "--thickness", "3000"])

    assert_clean_result(result, stdout="updated materials.hopg\n")
    assert "thickness_ang = {values = [3000.0]}" in catalog.read_text()


def test_set_new_override_creates_overrides_table_when_absent(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[materials.hopg]

[materials.mose2]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(sweep, "_validate", lambda *_args: None)

    result = invoke(sweep.command, ["set", "mose2", "--energy", "40"])

    assert_clean_result(result, stdout="updated materials.mose2\n")
    text = catalog.read_text()
    assert "[profiles.standard.overrides.mose2]" in text
    assert "energy_keV = {values = [40.0]}" in text


def test_set_unknown_material_reports_error(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[materials.hopg]
"""
    )
    monkeypatch.setattr(sweep, "_MATERIALS_TOML", catalog)

    result = invoke(sweep.command, ["set", "unobtainium", "--thickness", "1000"])

    assert result.exit_code == 1
    assert "unknown material: unobtainium" in result.stderr
