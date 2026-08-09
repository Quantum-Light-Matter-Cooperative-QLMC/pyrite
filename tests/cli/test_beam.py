from __future__ import annotations

import json

from cxr_mc.cli import _catalog_io
from cxr_mc.cli.commands import beam
from tests.helpers.cli import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.attached]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }
beam = "rf_gun_200fs"

[beams.rf_gun_200fs]
label = "RF gun, 200 fs"
rep_rate_hz = 1000.0
bunch_charge_pc = 2.5

[materials.hopg]

[materials.mose2]
"""


def _catalog(tmp_path, monkeypatch, text=_CATALOG):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(text)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def test_list_text_and_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    text = invoke(beam.command, ["list"])
    assert_clean_result(text)
    assert "rf_gun_200fs (RF gun, 200 fs): 1 profiles" in text.stdout

    machine = invoke(beam.command, ["list", "-o", "json"])
    assert_clean_result(machine)
    document = json.loads(machine.stdout)
    assert document["schema"] == "cxr.beam.list"
    (row,) = document["payload"]["beams"]
    assert row["name"] == "rf_gun_200fs"
    assert row["label"] == "RF gun, 200 fs"
    assert row["referenced_by"] == ["attached"]


def test_show_text_and_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(beam.command, ["show", "rf_gun_200fs"])
    assert_clean_result(shown)
    assert "[rf_gun_200fs]" in shown.stdout
    assert "label: RF gun, 200 fs" in shown.stdout
    assert "rep_rate_hz: 1000" in shown.stdout
    assert "bunch_charge_pc: 2.5" in shown.stdout

    machine = invoke(beam.command, ["show", "rf_gun_200fs", "-o", "json"])
    assert_clean_result(machine)
    payload = json.loads(machine.stdout)["payload"]
    assert payload == {
        "name": "rf_gun_200fs",
        "label": "RF gun, 200 fs",
        "rep_rate_hz": 1000.0,
        "bunch_charge_pc": 2.5,
    }


def test_show_unknown_beam_suggests_and_points_to_create(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["show", "rf_gun_200f"])

    assert result.exit_code == 1
    assert "unknown beam: rf_gun_200f" in result.stderr
    assert "Did you mean: rf_gun_200fs" in result.stderr
    assert "cxr beam create rf_gun_200f" in result.stderr


def test_create_requires_at_least_one_beam_field(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["create", "empty"])

    assert result.exit_code == 2
    assert "provide at least one beam-field option" in result.stderr


def test_create_writes_fields_and_label(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    created = invoke(
        beam.command,
        [
            "create",
            "lab_gun",
            "--label",
            "Lab gun",
            "--rep-rate-hz",
            "5000",
            "--bunch-charge-pc",
            "1",
        ],
    )
    assert_clean_result(created, stdout="created beam lab_gun\n")

    text = catalog.read_text()
    assert "[beams.lab_gun]" in text
    assert 'label = "Lab gun"' in text
    assert "rep_rate_hz = 5000.0" in text
    assert "bunch_charge_pc = 1.0" in text


def test_create_existing_or_invalid_name_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    existing = invoke(beam.command, ["create", "rf_gun_200fs", "--rep-rate-hz", "10"])
    invalid = invoke(beam.command, ["create", "bad name", "--rep-rate-hz", "10"])

    assert existing.exit_code == 1
    assert "already exists" in existing.stderr
    assert "cxr beam set rf_gun_200fs" in existing.stderr
    assert invalid.exit_code == 2
    assert "invalid beam name" in invalid.stderr


def test_create_dry_run_writes_nothing(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(beam.command, ["create", "demo", "--rep-rate-hz", "10", "--dry-run"])

    assert_clean_result(result)
    assert "+[beams.demo]" in result.stdout.replace(" ", "")
    assert catalog.read_text() == original


def test_set_updates_field_and_prompts_on_overwrite(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(beam.command, ["set", "rf_gun_200fs", "--rep-rate-hz", "2000"], input="n\n")
    assert declined.exit_code == 1
    assert "overwrite rep_rate_hz on beam rf_gun_200fs" in declined.stderr
    assert catalog.read_text() == original

    accepted = invoke(beam.command, ["set", "rf_gun_200fs", "--rep-rate-hz", "2000", "-y"])
    assert_clean_result(accepted, stdout="updated beam rf_gun_200fs\n")
    assert "rep_rate_hz = 2000.0" in catalog.read_text()


def test_set_new_field_never_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["set", "rf_gun_200fs", "--energy-spread", "0.001"])

    assert_clean_result(result, stdout="updated beam rf_gun_200fs\n")
    assert "energy_spread_frac = 0.001" in catalog.read_text()


def test_set_requires_label_or_field(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["set", "rf_gun_200fs"])

    assert result.exit_code == 2
    assert "provide --label or a beam-field option" in result.stderr


def test_set_unknown_beam_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["set", "bogus", "--rep-rate-hz", "10"])

    assert result.exit_code == 1
    assert "unknown beam: bogus" in result.stderr


def test_rename_moves_beam_and_updates_referencing_profile(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["rename", "rf_gun_200fs", "lab_gun"])

    assert_clean_result(result, stdout="renamed beam rf_gun_200fs to lab_gun\n")
    text = catalog.read_text()
    assert "[beams.rf_gun_200fs]" not in text
    assert "[beams.lab_gun]" in text
    section = text.split("[profiles.attached]", 1)[1].split("\n[", 1)[0]
    assert 'beam = "lab_gun"' in section


def test_rename_rejects_existing_or_same_name(tmp_path, monkeypatch):
    text = _CATALOG + "\n[beams.other]\nrep_rate_hz = 1.0\n"
    _catalog(tmp_path, monkeypatch, text)

    collision = invoke(beam.command, ["rename", "rf_gun_200fs", "other"])
    assert collision.exit_code == 1
    assert "already exists" in collision.stderr

    same = invoke(beam.command, ["rename", "rf_gun_200fs", "rf_gun_200fs"])
    assert same.exit_code == 1
    assert "already named" in same.stderr

    missing = invoke(beam.command, ["rename", "bogus", "renamed"])
    assert missing.exit_code == 1
    assert "unknown beam: bogus" in missing.stderr


def test_rename_dry_run_writes_nothing(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(beam.command, ["rename", "rf_gun_200fs", "lab_gun", "--dry-run"])

    assert_clean_result(result)
    assert catalog.read_text() == original


def test_delete_blocked_by_referencing_profile(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(beam.command, ["delete", "rf_gun_200fs", "-y"])

    assert result.exit_code == 1
    assert "still referenced by profiles: attached" in result.stderr
    assert catalog.read_text() == original


def test_delete_previews_and_yes_removes_beam(tmp_path, monkeypatch):
    text = _CATALOG + "\n[beams.orphan]\nrep_rate_hz = 1.0\n"
    catalog = _catalog(tmp_path, monkeypatch, text)
    original = catalog.read_text()

    preview = invoke(beam.command, ["delete", "orphan"])
    assert preview.exit_code == 0
    assert "-[beams.orphan]" in preview.stdout.replace(" ", "")
    assert catalog.read_text() == original

    deleted = invoke(beam.command, ["delete", "orphan", "-y"])
    assert_clean_result(deleted, stdout="deleted beam orphan\n")
    assert "[beams.orphan]" not in catalog.read_text()


def test_delete_dry_run_writes_nothing(tmp_path, monkeypatch):
    text = _CATALOG + "\n[beams.orphan]\nrep_rate_hz = 1.0\n"
    catalog = _catalog(tmp_path, monkeypatch, text)
    original = catalog.read_text()

    result = invoke(beam.command, ["delete", "orphan", "--dry-run"])

    assert_clean_result(result)
    assert "-[beams.orphan]" in result.stdout.replace(" ", "")
    assert catalog.read_text() == original


def test_delete_json_requires_yes_and_reports_envelope(tmp_path, monkeypatch):
    text = _CATALOG + "\n[beams.orphan]\nrep_rate_hz = 1.0\n"
    _catalog(tmp_path, monkeypatch, text)

    refused = invoke(beam.command, ["delete", "orphan", "-o", "json"])
    assert refused.exit_code == 2
    assert "--output json requires --yes" in refused.stderr

    deleted = invoke(beam.command, ["delete", "orphan", "-o", "json", "-y"])
    assert_clean_result(deleted)
    document = json.loads(deleted.stdout)
    assert document["schema"] == "cxr.beam.delete"
    assert document["ok"] is True
    assert document["payload"] == {"deleted": "orphan"}


def test_delete_unknown_beam_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(beam.command, ["delete", "bogus", "-y"])

    assert result.exit_code == 1
    assert "unknown beam: bogus" in result.stderr


_MINIMAL_VALID_CATALOG = """schema_version = 1
[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = 0.0
E_grid_line = { arange = { start = 20.0, stop = 40.0, step = 2.0 } }
E_grid_brem = 0.0

[crystals.mos2]
cif = "cifs/mos2.cif"
validation_id = "test-fixture"
B_ang2 = 0.6
beam_uvw = [0, 0, 2]
layers_per_cell = 2
E_grid = { values = [100.0, 200.0] }

[media.sio2]
composition = { Si = 0.02205, O = 0.04410 }

[materials.mos2]
label = "mos2"
crystal = "mos2"
"""


def test_create_matches_hand_written_toml_block_value_equality(tmp_path, monkeypatch):
    """A beam built purely from ``cxr beam create`` flags must decode to the
    same resolved ``BeamSpec`` payload as an equivalent hand-written
    ``[beams.NAME]`` block, negative ``alpha_twiss`` included."""
    from cxr_mc.materials import load_material_catalog

    (tmp_path / "built").mkdir()
    built_catalog = _catalog(tmp_path / "built", monkeypatch, _MINIMAL_VALID_CATALOG)
    result = invoke(
        beam.command,
        [
            "create",
            "twiss_beam",
            "--emittance",
            "1.0",
            "--twiss-beta",
            "0.5",
            "--twiss-alpha",
            "-0.8",
            "--energy-spread",
            "0.002",
        ],
    )
    assert_clean_result(result, stdout="created beam twiss_beam\n")

    hand_written = (
        _MINIMAL_VALID_CATALOG + "\n[beams.twiss_beam]\n"
        "energy_spread_frac = 0.002\n"
        "\n[beams.twiss_beam.transverse]\n"
        "normalized_emittance_x_mm_mrad = 1.0\n"
        "beta_twiss_x_m = 0.5\n"
        "alpha_twiss_x = -0.8\n"
    )
    (tmp_path / "hand").mkdir()
    hand_catalog_path = tmp_path / "hand" / "materials.toml"
    hand_catalog_path.write_text(hand_written)

    built_document = load_material_catalog(built_catalog)
    hand_document = load_material_catalog(hand_catalog_path)
    assert built_document.beams["twiss_beam"] == hand_document.beams["twiss_beam"]
