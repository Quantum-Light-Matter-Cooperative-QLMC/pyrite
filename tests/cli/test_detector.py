import json

from pyrite.cli import _catalog_io
from pyrite.cli.commands import detector
from tests.helpers.cli import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
detector = "standard_90"
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.attached]
detector = "eds"
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }

[detectors.standard_90]
label = "Standard geometry"
observation_angle_deg = 90.0

[detectors.eds]
observation_angle_deg = 119.0
polar_acceptance_deg = 16.6
solid_angle_sr = 0.066

[materials.hopg]
"""


def _catalog(tmp_path, monkeypatch, text=_CATALOG):
    path = tmp_path / "materials.toml"
    path.write_text(text)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", path)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return path


def test_list_and_show_text_and_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    listed = invoke(detector.command, ["list"])
    shown = invoke(detector.command, ["show", "eds"])
    machine = invoke(detector.command, ["show", "eds", "-o", "json"])

    assert_clean_result(listed)
    assert "standard_90 (Standard geometry): 1 profiles" in listed.stdout
    assert "eds: 1 profiles" in listed.stdout
    assert_clean_result(shown)
    assert "observation angle: 119 deg" in shown.stdout
    assert "polar acceptance (full span): 16.6 deg" in shown.stdout
    assert "solid angle: 0.066 sr" in shown.stdout
    assert_clean_result(machine)
    assert json.loads(machine.stdout)["payload"] == {
        "name": "eds",
        "observation_angle_deg": 119.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
    }


def test_unknown_name_suggests_and_points_to_create(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(detector.command, ["show", "standard_9"])

    assert result.exit_code == 1
    assert "Did you mean: standard_90" in result.stderr
    assert "pyrite detector create standard_9" in result.stderr


def test_create_set_and_dry_run(tmp_path, monkeypatch):
    path = _catalog(tmp_path, monkeypatch)

    created = invoke(
        detector.command,
        [
            "create",
            "custom",
            "--label",
            "Custom EDS",
            "--observation-angle",
            "100",
            "--dry-run",
        ],
    )
    assert_clean_result(created)
    assert "+[detectors.custom]" in created.stdout.replace(" ", "")
    assert "[detectors.custom]" not in path.read_text()

    created = invoke(
        detector.command,
        ["create", "custom", "--observation-angle", "100"],
    )
    assert_clean_result(created, stdout="created detector custom\n")

    updated = invoke(
        detector.command,
        ["set", "custom", "--solid-angle", "0.02"],
    )
    assert_clean_result(updated, stdout="updated detector custom\n")
    assert "solid_angle_sr = 0.02" in path.read_text()


def test_scalar_options_validate_domains(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    bad_acceptance = invoke(
        detector.command,
        ["set", "eds", "--polar-acceptance", "0"],
    )
    bad_solid_angle = invoke(
        detector.command,
        ["create", "bad", "--solid-angle", "13"],
    )

    assert bad_acceptance.exit_code == 2
    assert "0<x<=180" in bad_acceptance.stderr
    assert bad_solid_angle.exit_code == 2
    assert "12.566" in bad_solid_angle.stderr


def test_create_and_set_validate_names_fields_and_overwrites(tmp_path, monkeypatch):
    path = _catalog(tmp_path, monkeypatch)
    original = path.read_text()

    empty = invoke(detector.command, ["create", "empty", "--label", "Only label"])
    invalid = invoke(detector.command, ["create", "bad name", "--observation-angle", "90"])
    declined = invoke(
        detector.command,
        ["set", "eds", "--observation-angle", "100"],
        input="n\n",
    )

    assert empty.exit_code == 2
    assert "at least one detector-geometry option" in empty.stderr
    assert invalid.exit_code == 2
    assert "invalid detector name" in invalid.stderr
    assert declined.exit_code == 1
    assert "overwrite observation_angle_deg on detector eds" in declined.stderr
    assert path.read_text() == original


def test_rename_updates_all_references_atomically(tmp_path, monkeypatch):
    path = _catalog(tmp_path, monkeypatch)

    renamed = invoke(detector.command, ["rename", "eds", "lab_eds"])

    assert_clean_result(renamed, stdout="renamed detector eds to lab_eds\n")
    text = path.read_text()
    assert "[detectors.eds]" not in text
    assert "[detectors.lab_eds]" in text
    attached = text.split("[profiles.attached]", 1)[1].split("\n[", 1)[0]
    assert 'detector = "lab_eds"' in attached


def test_delete_refuses_referenced_and_removes_orphan(tmp_path, monkeypatch):
    text = _CATALOG + "\n[detectors.orphan]\nobservation_angle_deg = 80.0\n"
    path = _catalog(tmp_path, monkeypatch, text)

    blocked = invoke(detector.command, ["delete", "eds", "-y"])
    preview = invoke(detector.command, ["delete", "orphan"])
    deleted = invoke(detector.command, ["delete", "orphan", "-o", "json", "-y"])

    assert blocked.exit_code == 1
    assert "still referenced by profiles: attached" in blocked.stderr
    assert_clean_result(preview)
    assert "-[detectors.orphan]" in preview.stdout.replace(" ", "")
    assert_clean_result(deleted)
    assert json.loads(deleted.stdout)["payload"] == {"deleted": "orphan"}
    assert "[detectors.orphan]" not in path.read_text()
