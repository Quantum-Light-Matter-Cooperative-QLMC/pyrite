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
    assert {row["name"] for row in document["payload"]["ranges"]} == {
        "thickness",
        "energy",
        "polar",
        "azimuth",
    }
    assert all(not row["overridden"] for row in document["payload"]["ranges"])


def test_set_dry_run_preserves_catalog_and_energy_grid_siblings(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    original = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[materials.hopg]
profile = "standard"
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }
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

[materials.hopg]
profile = "standard"
thickness_ang = { values = [2000.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }
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
