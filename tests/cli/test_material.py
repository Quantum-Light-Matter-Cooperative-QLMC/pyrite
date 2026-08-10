from __future__ import annotations

import json

from pyrite import cli
from pyrite.cli import _catalog_io
from pyrite.cli.commands import material, sweep
from tests.helpers.cli import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[profiles.survey]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }

[profiles.survey.overrides.hopg]
energy_keV = { values = [60.0] }

[materials.hopg]

[materials.mose2]
"""


def _catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(_CATALOG)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def test_show_default_profile_reports_effective_sources(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg"])

    assert_clean_result(result)
    assert "hopg: profile standard" in result.stdout
    assert "thickness: [2000] (overridden)" in result.stdout
    assert "energy: [30] (inherited)" in result.stdout
    assert "profiles this material belongs to" not in result.stdout


def test_show_nonstandard_profile_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg", "--profile", "survey", "-o", "json"])

    assert_clean_result(result)
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.material.show"
    assert document["payload"]["profile"] == "survey"
    rows = {row["name"]: row for row in document["payload"]["ranges"]}
    assert rows["energy"]["values"] == [60.0]
    assert rows["energy"]["source"] == "overridden"
    assert rows["thickness"]["values"] == [500.0]
    assert rows["thickness"]["source"] == "inherited"


def test_set_nonstandard_profile_override(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        material.command,
        ["set", "mose2", "--profile", "survey", "--energy", "70"],
    )

    assert_clean_result(result, stdout="updated profile survey, material mose2\n")
    text = catalog.read_text()
    assert "[profiles.survey.overrides.mose2]" in text
    assert "energy_keV = {values = [70.0]}" in text


def test_material_set_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        material.command,
        [
            "set",
            "mose2",
            "--profile",
            "survey",
            "--energy",
            "40",
            "--energy",
            "50:70:20",
        ],
    )

    assert_clean_result(result, stdout="updated profile survey, material mose2\n")
    assert "energy_keV = {values = [40.0, 50.0, 70.0]}" in catalog.read_text()


def test_material_range_help_documents_repeatability() -> None:
    result = invoke(material.command, ["set", "--help"])

    assert_clean_result(result)
    assert " ".join(result.stdout.split()).count("repeat to combine") == 4


def test_reset_preserves_non_range_override_siblings(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["set", "hopg", "--reset", "thickness"])

    assert_clean_result(result, stdout="updated profile standard, material hopg\n")
    section = (
        catalog.read_text().split("[profiles.standard.overrides.hopg]", 1)[1].split("\n[", 1)[0]
    )
    assert "thickness_ang" not in section
    assert "E_grid_brem" in section


def test_overwrite_confirmation_and_dry_run(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(material.command, ["set", "hopg", "--thickness", "3000"], input="n\n")
    assert declined.exit_code == 1
    assert "profile standard, material hopg" in declined.stderr
    assert catalog.read_text() == original

    dry_run = invoke(
        material.command,
        ["set", "hopg", "--thickness", "3000", "--dry-run"],
    )
    assert_clean_result(dry_run)
    assert "+thickness_ang={values=[3000.0]}" in dry_run.stdout.replace(" ", "")
    assert catalog.read_text() == original


def test_unknown_names_report_actionable_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    unknown_material = invoke(material.command, ["show", "hpg"])
    unknown_profile = invoke(material.command, ["show", "hopg", "--profile", "missing"])

    assert unknown_material.exit_code == 1
    assert "unknown material: hpg. Did you mean: hopg?" in unknown_material.stderr
    assert unknown_profile.exit_code == 1
    assert "unknown profile: missing" in unknown_profile.stderr


def test_hidden_sweep_paths_warn_and_delegate(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    shown = invoke(sweep.command, ["show", "hopg", "-o", "json"])
    changed = invoke(
        sweep.command,
        ["set", "hopg", "--profile", "survey", "--azimuth", "100"],
    )

    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["schema"] == "cxr.sweep.show"
    assert "use 'pyrite material show hopg'" in shown.stderr
    assert changed.exit_code == 0
    assert "use 'pyrite material set hopg --profile survey'" in changed.stderr
    assert "tilt_azim_deg = {values = [100.0]}" in catalog.read_text()


def test_hidden_sweep_set_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        sweep.command,
        [
            "set",
            "mose2",
            "--profile",
            "survey",
            "--energy",
            "40",
            "--energy",
            "50:70:20",
        ],
    )

    assert result.exit_code == 0
    assert "use 'pyrite material set mose2 --profile survey'" in result.stderr
    assert "energy_keV = {values = [40.0, 50.0, 70.0]}" in catalog.read_text()


def test_hidden_sweep_show_without_material_preserves_overview(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(sweep.command, ["show", "-o", "json"])

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.sweep.show"
    assert [row["name"] for row in document["payload"]["profiles"]] == [
        "standard",
        "survey",
    ]
    assert "use 'pyrite profile list'" in result.stderr


def test_root_and_group_help_expose_new_ownership_only():
    root = invoke(cli.command, ["--help"])
    profile_help = invoke(cli.command, ["profile", "--help"])
    material_help = invoke(cli.command, ["material", "--help"])

    assert_clean_result(root)
    assert "material" in root.stdout
    assert "\n  blaze " not in root.stdout
    assert "\n  catalog " not in root.stdout
    assert "\n  sweep " not in root.stdout
    assert_clean_result(profile_help)
    assert "members" in profile_help.stdout
    assert "add-material" not in profile_help.stdout
    assert "remove-material" not in profile_help.stdout
    assert_clean_result(material_help)
    assert "show" in material_help.stdout
    assert "set" in material_help.stdout
    assert "validate" in material_help.stdout
    assert "blaze" in material_help.stdout
    assert "list" not in material_help.stdout


def test_material_group_lazily_routes_validate_and_blaze():
    validate_help = invoke(cli.command, ["material", "validate", "--help"])
    blaze_help = invoke(cli.command, ["material", "blaze", "--help"])

    assert_clean_result(validate_help)
    assert "Validate bundled material catalog" in validate_help.stdout
    assert_clean_result(blaze_help)
    assert "blazed-crystal MC sweep" in blaze_help.stdout
