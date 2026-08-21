from __future__ import annotations

import json

from pyrite.cli import _catalog_io, _core
from pyrite.cli._deprecations import flag_message
from pyrite.cli.commands import profile
from pyrite.energy_grid import artifacts
from tests.helpers.cli import assert_clean_result, invoke

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

_BEAM_CATALOG = (
    _CATALOG
    + """
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

[beams.other_beam]
rep_rate_hz = 500.0
"""
)

_DETECTOR_CATALOG = (
    _CATALOG
    + """
[profiles.attached]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }
detector = "eds"

[detectors.standard_90]
observation_angle_deg = 90.0

[detectors.eds]
observation_angle_deg = 119.0
polar_acceptance_deg = 16.6
solid_angle_sr = 0.066
"""
)


def _catalog(tmp_path, monkeypatch, text=_CATALOG):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(text)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def _artifact_ref_catalog(digest):
    return _CATALOG.replace(
        "\n[materials.hopg]",
        f'\nenergy_grid_refs = {{ hopg = "{digest}" }}\n\n[materials.hopg]',
    )


def _artifact(tmp_path):
    return artifacts.write_artifact(
        tmp_path / "energy-grid-artifacts",
        artifacts.artifact_identity(
            "hopg",
            [{"energy_keV": 30, "start_eV": 10, "stop_eV": 100, "num": 10}],
            {"stop_eV": 1000, "step_eV": 10},
            [30],
        ),
    )


def test_list_text_and_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    text = invoke(profile.command, ["list"])
    assert_clean_result(text)
    assert "standard: all materials (implicit), 1 material overrides" in text.stdout
    assert "sub_100keV: 1 materials, 0 material overrides" in text.stdout

    machine = invoke(profile.command, ["list", "-o", "json"])
    assert_clean_result(machine)
    document = json.loads(machine.stdout)
    assert document["schema"] == "cxr.profile.list"
    (standard, sub) = document["payload"]["profiles"]
    assert standard["name"] == "standard"
    assert standard["materials"] is None
    assert standard["overrides"] == ["hopg"]
    assert sub["materials"] == ["hopg"]


def test_list_and_show_expose_energy_grid_refs(tmp_path, monkeypatch):
    stored = _artifact(tmp_path)
    _catalog(tmp_path, monkeypatch, _artifact_ref_catalog(stored.digest))

    listed = invoke(profile.command, ["list"])
    assert_clean_result(listed)
    assert "standard: all materials (implicit), 1 material overrides, 0 energy-grid refs" in (
        listed.stdout
    )
    assert "sub_100keV: 1 materials, 0 material overrides, 1 energy-grid refs" in listed.stdout

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(shown)
    assert f"    hopg -> {stored.digest}" in shown.stdout

    # A profile with no refs still resolves through the legacy catalog tables.
    legacy = invoke(profile.command, ["show", "standard"])
    assert_clean_result(legacy)
    assert "energy grids: legacy catalog tables (no artifact refs)" in legacy.stdout

    machine = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])
    assert_clean_result(machine)
    assert json.loads(machine.stdout)["payload"]["energy_grid_refs"] == {"hopg": stored.digest}

    listed_json = invoke(profile.command, ["list", "-o", "json"])
    assert_clean_result(listed_json)
    (standard, sub) = json.loads(listed_json.stdout)["payload"]["profiles"]
    assert standard["energy_grid_refs"] == {}
    assert sub["energy_grid_refs"] == {"hopg": stored.digest}


def test_show_and_bare_name_alias(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(shown)
    assert "[sub_100keV]" in shown.stdout
    assert "energy: [30, 50]" in shown.stdout
    assert "materials: hopg" in shown.stdout
    assert "observation angle: 90 deg" in shown.stdout
    assert "polar acceptance (full span): unspecified" in shown.stdout
    assert "solid angle: unspecified" in shown.stdout

    aliased = invoke(profile.command, ["sub_100keV"])
    assert_clean_result(aliased)
    assert aliased.stdout == shown.stdout

    machine = invoke(profile.command, ["show", "standard", "-o", "json"])
    assert_clean_result(machine)
    payload = json.loads(machine.stdout)["payload"]
    assert payload["materials"] is None
    assert payload["detector"] == {
        "observation_angle_deg": 90.0,
        "polar_acceptance_deg": None,
        "solid_angle_sr": None,
    }
    assert payload["overrides"] == {"hopg": ["thickness_ang"]}


def test_show_create_and_set_round_trip_longitudinal_beam(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    created = invoke(
        profile.command,
        [
            "create",
            "microtrain",
            "--material",
            "hopg",
            "--transverse-fwhm-mm",
            "0.1",
            "--rep-rate-hz",
            "5000",
            "--bunch-charge-pc",
            "1",
            "--longitudinal",
            "microtrain",
            "--envelope-rms-fs",
            "200",
        ],
    )
    assert_clean_result(
        created,
        stdout="created profile microtrain\n",
        stderr="".join(
            flag_message("profile create", flag, replacement) + "\n"
            for flag, replacement in (
                ("--transverse-fwhm-mm", "pyrite beam create/set --transverse-fwhm-mm"),
                ("--rep-rate-hz", "pyrite beam create/set --rep-rate-hz"),
                ("--bunch-charge-pc", "pyrite beam create/set --bunch-charge-pc"),
                ("--longitudinal", "pyrite beam create/set --longitudinal"),
                ("--envelope-rms-fs", "pyrite beam create/set --envelope-rms-fs"),
            )
        ),
    )

    shown = invoke(profile.command, ["show", "microtrain", "-o", "json"])
    assert_clean_result(shown)
    beam = json.loads(shown.stdout)["payload"]["beam"]
    assert beam == {
        "transverse_fwhm_mm": 0.1,
        "rep_rate_hz": 5000.0,
        "bunch_charge_pc": 1.0,
        "longitudinal": {"kind": "microtrain", "envelope_rms_fs": 200.0},
    }

    changed = invoke(
        profile.command,
        ["set", "microtrain", "--longitudinal", "compressed", "--bunch-charge-pc", "1"],
    )
    assert_clean_result(
        changed,
        stdout="updated profile microtrain\n",
        stderr="".join(
            flag_message("profile set", flag, replacement) + "\n"
            for flag, replacement in (
                ("--bunch-charge-pc", "pyrite beam create/set --bunch-charge-pc"),
                ("--longitudinal", "pyrite beam create/set --longitudinal"),
            )
        ),
    )
    text = catalog.read_text()
    section = text.split("[profiles.microtrain.beam.longitudinal]", 1)[1].split("\n[", 1)[0]
    assert 'kind = "compressed"' in section
    assert "envelope_rms_fs" not in section


def test_add_does_not_merge_longitudinal_policy(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(
        profile.command,
        ["add", "sub_100keV", "--longitudinal", "compressed"],
    )

    assert result.exit_code == 2
    assert "No such option '--longitudinal'" in result.stderr
    assert catalog.read_text() == original


def test_show_unknown_profile_suggests_and_points_to_create(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["show", "sub_100kv"])

    assert result.exit_code == 1
    assert "unknown profile: sub_100kv" in result.stderr
    assert "Did you mean: sub_100keV" in result.stderr
    assert "pyrite profile create sub_100kv" in result.stderr


def test_show_inherits_standard_detector_when_profile_block_is_absent(tmp_path, monkeypatch):
    _catalog(
        tmp_path,
        monkeypatch,
        _CATALOG
        + "\n[profiles.standard.detector]\n"
        + "observation_angle_deg = 91.0\n"
        + "polar_acceptance_deg = 16.6\n"
        + "solid_angle_sr = 0.066\n",
    )

    shown = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])

    assert_clean_result(shown)
    assert json.loads(shown.stdout)["payload"]["detector"] == {
        "observation_angle_deg": 91.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
    }


def test_create_clones_source_and_applies_range_overrides(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command, ["create", "sub_200keV", "--from", "sub_100keV", "--energy", "150,200"]
    )

    assert_clean_result(result, stdout="created profile sub_200keV\n")
    text = catalog.read_text()
    assert "[profiles.sub_200keV]" in text
    assert "energy_keV = {values = [150.0, 200.0]}" in text
    # Cloned grids and membership carry over; per-material overrides do not.
    assert "tilt_deg = {values = [5.0]}" in text
    section = text.split("[profiles.sub_200keV]", 1)[1].split("\n[", 1)[0]
    assert 'materials = ["hopg"]' in section


def test_create_clones_energy_grid_refs_without_copying_artifact_bytes(tmp_path, monkeypatch):
    stored = _artifact(tmp_path)
    catalog = _catalog(tmp_path, monkeypatch, _artifact_ref_catalog(stored.digest))
    original_bytes = stored.path.read_bytes()

    result = invoke(profile.command, ["create", "clone", "--from", "sub_100keV"])

    assert_clean_result(result, stdout="created profile clone\n")
    text = catalog.read_text()
    section = text.split("[profiles.clone]", 1)[1].split("\n[", 1)[0]
    assert f'energy_grid_refs={{hopg="{stored.digest}"}}' in section.replace(" ", "")
    assert stored.path.read_bytes() == original_bytes


def test_create_range_syntax_expands_stop_inclusive(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["create", "ranged", "--energy", "30,50:100:25"])

    assert_clean_result(result)
    assert "energy_keV = {values = [30.0, 50.0, 75.0, 100.0]}" in catalog.read_text()


def test_create_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["create", "ranged", "--energy", "30", "--energy", "50:100:25"],
    )

    assert_clean_result(result)
    assert "energy_keV = {values = [30.0, 50.0, 75.0, 100.0]}" in catalog.read_text()


def test_range_help_documents_repeatability() -> None:
    result = invoke(profile.command, ["set", "--help"])

    assert_clean_result(result)
    assert " ".join(result.stdout.split()).count("repeat to combine") == 4


def test_create_accepts_atomic_initial_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["create", "demo", "--material", "hopg,mose2"],
    )

    assert_clean_result(result, stdout="created profile demo\n")
    section = catalog.read_text().split("[profiles.demo]", 1)[1].split("\n[", 1)[0]
    assert 'materials = ["hopg", "mose2"]' in section


def test_create_and_show_round_trip_detector_scalars(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    created = invoke(
        profile.command,
        [
            "create",
            "zhai",
            "--observation-angle",
            "119",
            "--polar-acceptance",
            "16.6",
            "--solid-angle",
            "0.066",
        ],
    )
    shown = invoke(profile.command, ["show", "zhai", "-o", "json"])

    assert_clean_result(
        created,
        stdout="created profile zhai\n",
        stderr="".join(
            flag_message("profile create", flag, f"pyrite detector create/set {flag}") + "\n"
            for flag in ("--observation-angle", "--polar-acceptance", "--solid-angle")
        ),
    )
    assert_clean_result(shown)
    assert json.loads(shown.stdout)["payload"]["detector"] == {
        "observation_angle_deg": 119.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
    }
    text = catalog.read_text()
    assert "[profiles.zhai.detector]" in text
    assert "observation_angle_deg = 119.0" in text
    assert "polar_acceptance_deg = 16.6" in text
    assert "solid_angle_sr = 0.066" in text


def test_create_existing_or_invalid_name_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    existing = invoke(profile.command, ["create", "standard"])
    invalid = invoke(profile.command, ["create", "bad name"])

    assert existing.exit_code == 1
    assert "already exists" in existing.stderr
    assert "pyrite profile set standard" in existing.stderr
    assert invalid.exit_code == 2
    assert "invalid profile name" in invalid.stderr


def test_set_replaces_grid_and_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--polar", "10:30:10"])
    members = invoke(
        profile.command,
        ["set", "sub_100keV", "--material", "hopg,mose2"],
    )

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert_clean_result(members, stdout="updated profile sub_100keV\n")
    text = catalog.read_text()
    assert "tilt_deg = {values = [10.0, 20.0, 30.0]}" in text
    assert 'materials = ["hopg", "mose2"]' in text


def test_set_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["set", "sub_100keV", "--energy", "30", "--energy", "75:100:25"],
    )

    assert_clean_result(result)
    assert "energy_keV = {values = [30.0, 75.0, 100.0]}" in catalog.read_text()


def test_set_unknown_material_in_membership_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--material", "foobarium"])

    assert result.exit_code == 1
    assert "unknown material: foobarium" in result.stderr


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


def test_set_replaces_detector_scalars_and_standard_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    updated = invoke(
        profile.command,
        [
            "set",
            "sub_100keV",
            "--observation-angle",
            "119",
            "--polar-acceptance",
            "16.6",
            "--solid-angle",
            "0.066",
        ],
    )
    declined = invoke(
        profile.command,
        ["set", "standard", "--observation-angle", "91"],
        input="n\n",
    )

    assert_clean_result(
        updated,
        stdout="updated profile sub_100keV\n",
        stderr="".join(
            flag_message("profile set", flag, f"pyrite detector create/set {flag}") + "\n"
            for flag in ("--observation-angle", "--polar-acceptance", "--solid-angle")
        ),
    )
    assert declined.exit_code == 1
    assert "profile 'standard'" in declined.stderr
    text = catalog.read_text()
    assert "[profiles.sub_100keV.detector]" in text
    assert "observation_angle_deg = 119.0" in text
    assert "polar_acceptance_deg = 16.6" in text
    assert "solid_angle_sr = 0.066" in text
    assert "[profiles.standard.detector]" not in text


def test_detector_scalar_options_validate_domains_and_dry_run(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    dry_run = invoke(
        profile.command,
        ["set", "standard", "--observation-angle", "91", "--dry-run"],
    )
    bad_acceptance = invoke(
        profile.command,
        ["set", "sub_100keV", "--polar-acceptance", "0"],
    )
    bad_solid_angle = invoke(
        profile.command,
        ["create", "bad", "--solid-angle", "13"],
    )

    assert_clean_result(
        dry_run,
        stderr=flag_message(
            "profile set",
            "--observation-angle",
            "pyrite detector create/set --observation-angle",
        )
        + "\n",
    )
    assert "+observation_angle_deg = 91.0" in dry_run.stdout
    assert catalog.read_text() == original
    assert bad_acceptance.exit_code == 2
    assert "0<x<=180" in bad_acceptance.stderr
    assert bad_solid_angle.exit_code == 2
    assert "12.566" in bad_solid_angle.stderr


def test_add_unions_sorts_deduplicates(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--energy", "75,30"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "energy_keV = {values = [30.0, 50.0, 75.0]}" in catalog.read_text()


def test_add_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["add", "sub_100keV", "--energy", "75", "--energy", "100"],
    )

    assert_clean_result(result)
    assert "energy_keV = {values = [30.0, 50.0, 75.0, 100.0]}" in catalog.read_text()


def test_add_material_is_canonical_and_composes_with_ranges(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["add", "sub_100keV", "--energy", "75", "--material", "mose2,hopg"],
    )

    assert result.exit_code == 0
    assert result.stdout == "updated profile sub_100keV: added mose2; already members: hopg\n"
    assert result.stderr == ""
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


def test_remove_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["remove", "sub_100keV", "--energy", "30", "--energy", "30"],
    )

    assert_clean_result(result)
    assert "energy_keV = {values = [50.0]}" in catalog.read_text()


def test_remove_material_and_no_op_requires_option(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["remove", "sub_100keV", "--material", "hopg,diamond"],
    )
    assert_clean_result(result)
    assert "removed hopg" in result.stdout
    assert "not members: diamond" in result.stdout
    assert "materials = []" in catalog.read_text()

    bare = invoke(profile.command, ["remove", "sub_100keV"])
    assert bare.exit_code == 2
    assert "provide a range, membership, beam, or emission option" in bare.stderr


def test_show_emission_defaults_to_incoherent(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(shown)
    assert "emission: incoherent (default)" in shown.stdout

    machine = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])
    assert_clean_result(machine)
    assert json.loads(machine.stdout)["payload"]["emission"] is None


def test_set_emission_replaces_and_reports_in_show(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--emission", "coherent"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert 'emission = "coherent"' in catalog.read_text()
    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "emission: coherent" in shown.stdout

    replaced = invoke(profile.command, ["set", "sub_100keV", "--emission", "both"])
    assert_clean_result(replaced, stdout="updated profile sub_100keV\n")
    assert 'emission = "both"' in catalog.read_text()


def test_set_emission_invalid_value_rejected(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--emission", "bogus"])

    assert result.exit_code == 2
    assert "not one of" in result.stderr.lower()


def test_set_transport_numerics_round_trips_and_validates_coupling(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    invalid = invoke(profile.command, ["set", "sub_100keV", "--max-de-frac", "0.02"])
    assert invalid.exit_code == 1
    assert "requires energy_model='midpoint'" in invalid.stderr

    result = invoke(
        profile.command,
        [
            "set",
            "sub_100keV",
            "--straggling",
            "--energy-model",
            "midpoint",
            "--max-de-frac",
            "0.02",
        ],
    )
    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    text = catalog.read_text()
    assert "straggling = true" in text
    assert 'energy_model = "midpoint"' in text
    assert "max_dE_frac = 0.02" in text

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "straggling: True" in shown.stdout
    assert "energy model: midpoint" in shown.stdout
    assert "max dE fraction: 0.02" in shown.stdout


def test_xray_dispersion_is_no_longer_a_selectable_field(tmp_path, monkeypatch):
    """The in-medium dispersion is unconditional physics now: no --xray-dispersion
    flag on set/add/remove, and nothing about it in ``profile show``."""
    _catalog(tmp_path, monkeypatch)

    for verb in ("set", "add", "remove"):
        rejected = invoke(profile.command, [verb, "sub_100keV", "--xray-dispersion", "refractive"])
        assert rejected.exit_code == 2
        assert "no such option" in rejected.stderr.lower()

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(shown)
    assert "dispersion" not in shown.stdout

    machine = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])
    assert_clean_result(machine)
    assert "xray_dispersion" not in json.loads(machine.stdout)["payload"]


def test_set_emission_on_standard_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(profile.command, ["set", "standard", "--emission", "coherent"], input="n\n")
    assert declined.exit_code == 1
    assert "profile 'standard'" in declined.stderr
    assert catalog.read_text() == original

    accepted = invoke(profile.command, ["set", "standard", "--emission", "coherent", "-y"])
    assert_clean_result(accepted, stdout="updated profile standard\n")
    assert 'emission = "coherent"' in catalog.read_text()


def test_add_coherent_sets_explicit_mode(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--coherent"])

    assert result.exit_code == 0
    assert "emission: added coherent" in result.stdout
    assert "auto-switched" not in result.stdout
    assert 'emission = "coherent"' in catalog.read_text()


def test_add_coherent_then_incoherent_auto_switches_to_both(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    invoke(profile.command, ["add", "sub_100keV", "--coherent"])
    result = invoke(profile.command, ["add", "sub_100keV", "--incoherent"])

    assert result.exit_code == 0
    assert "emission: added incoherent (auto-switched to 'both')" in result.stdout
    assert 'emission = "both"' in catalog.read_text()


def test_add_both_flags_at_once_sets_both(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--coherent", "--incoherent"])

    assert result.exit_code == 0
    assert "emission: added coherent, incoherent (auto-switched to 'both')" in result.stdout
    assert 'emission = "both"' in catalog.read_text()


def test_add_emission_mode_already_present_is_a_no_op(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    invoke(profile.command, ["add", "sub_100keV", "--coherent"])
    before = catalog.read_text()
    result = invoke(profile.command, ["add", "sub_100keV", "--coherent"])

    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "emission" not in result.stdout
    assert catalog.read_text() == before


def test_remove_incoherent_from_both_leaves_coherent(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, ["add", "sub_100keV", "--coherent", "--incoherent"])

    result = invoke(profile.command, ["remove", "sub_100keV", "--incoherent"])

    assert result.exit_code == 0
    assert "emission: removed incoherent (now 'coherent')" in result.stdout
    assert 'emission = "coherent"' in catalog.read_text()


def test_remove_sole_emission_mode_drops_key_entirely(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, ["add", "sub_100keV", "--coherent"])

    result = invoke(profile.command, ["remove", "sub_100keV", "--coherent"])

    assert result.exit_code == 0
    assert "emission: removed coherent (no explicit emission left)" in result.stdout
    assert "emission" not in catalog.read_text()
    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "emission: incoherent (default)" in shown.stdout


def test_remove_emission_mode_not_present_errors(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    result = invoke(profile.command, ["remove", "sub_100keV", "--coherent"])

    assert result.exit_code == 1
    assert "profile sub_100keV emission does not include: coherent" in result.stderr
    assert catalog.read_text() == original


def test_retired_member_group_selectors_are_rejected(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    for flag in ("--unverified-dw", "--high-energy-only"):
        result = invoke(profile.command, ["members", "add", "sub_100keV", flag])
        assert result.exit_code == 2
        assert f"No such option '{flag}'" in result.stderr
        assert catalog.read_text() == original


def test_member_group_selectors_require_a_selector_or_material(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["members", "set", "sub_100keV"])

    assert result.exit_code == 1
    assert "provide MATERIAL keys" in result.stderr


def test_members_reset_restores_implicit_membership(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    reset = invoke(profile.command, ["set", "sub_100keV", "--all-materials"])
    shown = invoke(profile.command, ["show", "sub_100keV"])

    assert_clean_result(
        reset,
        stdout="updated profile sub_100keV\n",
    )
    section = catalog.read_text().split("[profiles.sub_100keV]", 1)[1].split("\n[", 1)[0]
    assert "materials" not in section
    assert_clean_result(shown)
    assert "materials: all catalog materials (implicit)" in shown.stdout


def test_set_all_materials_conflicts_with_explicit_materials(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["set", "sub_100keV", "--material", "hopg", "--all-materials"],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_members_path_warns_and_dispatches_compatibly(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        ["members", "set", "sub_100keV", "hopg", "mose2"],
    )

    assert result.exit_code == 0
    assert result.stdout == "updated profile sub_100keV membership\n"
    assert result.stderr.count("is deprecated") == 1
    assert "pyrite profile set NAME --material MATERIAL,..." in result.stderr
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()


def test_add_on_standard_prompts(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(profile.command, ["add", "standard", "--energy", "40"], input="n\n")

    assert declined.exit_code == 1
    assert "add values to profile 'standard'" in declined.stderr
    assert catalog.read_text() == original


def test_empty_updates_are_usage_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    expected_by_verb = {
        "set": "provide a range, beam, detector, transport, membership, or emission option",
        "add": "provide a range, membership, or emission option",
        "remove": "provide a range, membership, beam, or emission option",
    }
    for verb, expected in expected_by_verb.items():
        result = invoke(profile.command, [verb, "sub_100keV"])
        assert result.exit_code == 2
        assert expected in result.stderr


def test_delete_previews_non_tty_prompts_on_tty_and_yes_removes_profile(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    preview = invoke(profile.command, ["delete", "sub_100keV"])
    assert preview.exit_code == 0
    assert "-[profiles.sub_100keV]" in preview.stdout.replace(" ", "")
    assert "preview only; re-run with -y/--yes to execute" in preview.stdout
    assert preview.stderr == ""
    assert catalog.read_text() == original

    monkeypatch.setattr(_core, "_stdin_is_tty", lambda: True)
    declined = invoke(profile.command, ["delete", "sub_100keV"], input="n\n")
    assert declined.exit_code == 0
    assert "delete profile 'sub_100keV'" in declined.stderr
    assert "-[profiles.sub_100keV]" in declined.stdout.replace(" ", "")
    assert catalog.read_text() == original

    deleted = invoke(profile.command, ["delete", "sub_100keV", "-y"])
    assert_clean_result(deleted, stdout="deleted profile sub_100keV\n")
    assert "[profiles.sub_100keV]" not in catalog.read_text()


def test_delete_fails_closed_when_catalog_changes_after_preview(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    def concurrent_edit(_yes, _prompt):
        catalog.write_text(original + "\n# concurrent edit\n")
        return True

    monkeypatch.setattr(profile, "confirm_destructive", concurrent_edit)

    result = invoke(profile.command, ["delete", "sub_100keV"])

    assert result.exit_code == 1
    assert "material catalog changed after preview; rerun command" in result.stderr
    assert catalog.read_text().endswith("# concurrent edit\n")


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


def test_delete_ref_only_profile_orphans_but_does_not_delete_artifact_bytes(tmp_path, monkeypatch):
    stored = _artifact(tmp_path)
    catalog = _catalog(tmp_path, monkeypatch, _artifact_ref_catalog(stored.digest))
    original_bytes = stored.path.read_bytes()

    result = invoke(profile.command, ["delete", "sub_100keV", "-y"])

    assert_clean_result(result, stdout="deleted profile sub_100keV\n")
    assert "[profiles.sub_100keV]" not in catalog.read_text()
    assert stored.path.read_bytes() == original_bytes


def test_delete_json_requires_yes_and_reports_envelope(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    refused = invoke(profile.command, ["delete", "sub_100keV", "-o", "json"])
    assert refused.exit_code == 2
    assert "--output json requires --yes" in refused.stderr

    deleted = invoke(profile.command, ["delete", "sub_100keV", "-o", "json", "-y"])
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


def test_rename_preserves_energy_grid_ref_digest(tmp_path, monkeypatch):
    stored = _artifact(tmp_path)
    catalog = _catalog(tmp_path, monkeypatch, _artifact_ref_catalog(stored.digest))
    original_bytes = stored.path.read_bytes()

    result = invoke(profile.command, ["rename", "sub_100keV", "sub100"])

    assert_clean_result(result, stdout="renamed profile sub_100keV to sub100\n")
    text = catalog.read_text()
    assert "[profiles.sub_100keV]" not in text
    section = text.split("[profiles.sub100]", 1)[1].split("\n[", 1)[0]
    assert f'energy_grid_refs={{hopg="{stored.digest}"}}' in section.replace(" ", "")
    assert stored.path.read_bytes() == original_bytes


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
    assert "use 'pyrite profile add NAME --material MATERIAL,...'" in added.stderr
    assert "added mose2" in added.stdout
    assert "already members: hopg" in added.stdout
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()

    removed = invoke(profile.command, ["remove-material", "sub_100keV", "hopg", "diamond"])
    assert removed.exit_code == 0
    assert "use 'pyrite profile remove NAME --material MATERIAL,...'" in removed.stderr
    assert "removed hopg" in removed.stdout
    assert "not members: diamond" in removed.stdout
    assert 'materials = ["mose2"]' in catalog.read_text()


def test_membership_verbs_require_explicit_list(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "standard", "--material", "mose2"])

    assert result.exit_code == 1
    assert "implicit all-catalog-materials membership" in result.stderr
    assert "pyrite profile set standard --material MATERIAL" in result.stderr


def test_add_materials_explains_implicit_membership_is_already_all(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "standard", "--material", "mose2"])

    assert result.exit_code == 1
    assert "already includes every material" in result.stderr
    assert "pyrite profile set standard --material MATERIAL" in result.stderr


def test_membership_verbs_reject_unknown_material(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add", "sub_100keV", "--material", "foobarium"])

    assert result.exit_code == 1
    assert "unknown material: foobarium" in result.stderr


def test_add_material_all_seeds_implicit_membership(tmp_path, monkeypatch):
    """--all seeds an implicit profile from standard-profile membership -- the
    escape hatch `test_membership_verbs_require_explicit_list`
    otherwise requires (`pyrite profile set NAME --material KEY,...`, typed by hand)."""
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add-material", "standard", "--all", "-y"])

    assert result.exit_code == 0
    assert "pyrite profile add NAME --material MATERIAL,..." in result.stderr
    assert "added hopg, mose2" in result.stdout
    assert 'materials = ["hopg", "mose2"]' in catalog.read_text()


def test_add_material_all_extends_and_skips_existing_members(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add-material", "sub_100keV", "--all"])

    assert result.exit_code == 0
    assert "pyrite profile add NAME --material MATERIAL,..." in result.stderr
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
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["add-material", "sub_100keV", "-a"])

    assert result.exit_code == 0
    assert "pyrite profile add NAME --material MATERIAL,..." in result.stderr
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


def test_create_writes_transverse_twiss_block(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        [
            "create",
            "twiss",
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

    assert_clean_result(
        result,
        stdout="created profile twiss\n",
        stderr="".join(
            flag_message("profile create", flag, replacement) + "\n"
            for flag, replacement in (
                ("--emittance", "pyrite beam create/set --emittance"),
                ("--twiss-beta", "pyrite beam create/set --twiss-beta"),
                ("--twiss-alpha", "pyrite beam create/set --twiss-alpha"),
                ("--energy-spread", "pyrite beam create/set --energy-spread"),
            )
        ),
    )
    text = catalog.read_text().replace(" ", "")
    assert "[profiles.twiss.beam.transverse]" in text
    assert "normalized_emittance_x_mm_mrad=1.0" in text
    assert "beta_twiss_x_m=0.5" in text
    # A diverging beam past its waist is legitimate input, not a bad magnitude.
    assert "alpha_twiss_x=-0.8" in text
    assert "energy_spread_frac=0.002" in text


def test_create_rejects_emittance_together_with_spot_fwhm(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command,
        [
            "create",
            "clash",
            "--emittance",
            "1.0",
            "--twiss-beta",
            "0.5",
            "--transverse-fwhm-mm",
            "1.0",
        ],
    )

    assert result.exit_code == 2
    assert "--emittance replaces --transverse-fwhm-mm" in result.output


def test_create_requires_beta_alongside_emittance(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["create", "partial", "--emittance", "1.0"])

    assert result.exit_code == 2
    assert "--emittance requires --twiss-beta" in result.output


def test_create_rejects_twiss_without_emittance(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["create", "orphan", "--twiss-alpha", "-0.5"])

    assert result.exit_code == 2
    assert "--twiss-beta and --twiss-alpha require --emittance" in result.output


def test_set_transverse_retires_the_legacy_spot(tmp_path, monkeypatch):
    """The two spellings are mutually exclusive at decode, so writing one
    must remove the other or the edit leaves an unloadable profile."""
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, ["create", "spot", "--transverse-fwhm-mm", "2.0"])
    assert "transverse_fwhm_mm" in catalog.read_text()

    result = invoke(
        profile.command,
        ["set", "spot", "--emittance", "1.0", "--twiss-beta", "0.5"],
    )

    assert_clean_result(
        result,
        stderr="".join(
            flag_message("profile set", flag, replacement) + "\n"
            for flag, replacement in (
                ("--emittance", "pyrite beam create/set --emittance"),
                ("--twiss-beta", "pyrite beam create/set --twiss-beta"),
            )
        ),
    )
    section = catalog.read_text().split("[profiles.spot.beam]", 1)[1]
    assert "transverse_fwhm_mm" not in section.split("\n[profiles.spot.beam.transverse]", 1)[0]


def test_set_attaches_named_beam_by_reference(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["set", "sub_100keV", "--beam", "rf_gun_200fs"])
    assert_clean_result(result, stdout="updated profile sub_100keV\n")

    section = catalog.read_text().split("[profiles.sub_100keV]", 1)[1].split("\n[", 1)[0]
    assert 'beam = "rf_gun_200fs"' in section

    shown = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])
    assert_clean_result(shown)
    payload = json.loads(shown.stdout)["payload"]
    assert payload["beam_ref"] == "rf_gun_200fs"
    assert payload["beam"] is None

    text_shown = invoke(profile.command, ["show", "sub_100keV"])
    assert_clean_result(text_shown)
    assert "beam: rf_gun_200fs (named reference)" in text_shown.stdout


def test_create_attaches_named_beam_by_reference(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(
        profile.command,
        ["create", "newprof", "--material", "hopg", "--beam", "rf_gun_200fs"],
    )
    assert_clean_result(result, stdout="created profile newprof\n")

    shown = invoke(profile.command, ["show", "newprof", "-o", "json"])
    assert_clean_result(shown)
    assert json.loads(shown.stdout)["payload"]["beam_ref"] == "rf_gun_200fs"


def test_set_unknown_beam_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["set", "sub_100keV", "--beam", "bogus"])

    assert result.exit_code == 1
    assert "unknown beam: bogus" in result.stderr
    assert "Create it first with: pyrite beam create bogus" in result.stderr


def test_create_unknown_beam_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(
        profile.command,
        ["create", "newprof", "--material", "hopg", "--beam", "bogus"],
    )

    assert result.exit_code == 1
    assert "unknown beam: bogus" in result.stderr


def test_set_beam_and_inline_flag_are_mutually_exclusive(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(
        profile.command,
        ["set", "sub_100keV", "--beam", "rf_gun_200fs", "--rep-rate-hz", "100"],
    )

    assert result.exit_code == 2
    assert "--beam replaces the inline beam flags; pass only one" in result.stderr


def test_create_beam_and_inline_flag_are_mutually_exclusive(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(
        profile.command,
        [
            "create",
            "newprof",
            "--material",
            "hopg",
            "--beam",
            "rf_gun_200fs",
            "--rep-rate-hz",
            "100",
        ],
    )

    assert result.exit_code == 2
    assert "--beam replaces the inline beam flags; pass only one" in result.stderr


def test_set_and_create_attach_named_detector_and_show_resolved_geometry(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _DETECTOR_CATALOG)

    updated = invoke(profile.command, ["set", "sub_100keV", "--detector", "eds"])
    created = invoke(
        profile.command,
        ["create", "newprof", "--material", "hopg", "--detector", "standard_90"],
    )
    shown = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])

    assert_clean_result(updated, stdout="updated profile sub_100keV\n")
    assert_clean_result(created, stdout="created profile newprof\n")
    section = catalog.read_text().split("[profiles.sub_100keV]", 1)[1].split("\n[", 1)[0]
    assert 'detector = "eds"' in section
    payload = json.loads(shown.stdout)["payload"]
    assert payload["detector_ref"] == "eds"
    assert payload["detector"] == {
        "observation_angle_deg": 119.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
    }


def test_named_detector_unknown_conflict_and_inline_edit_compatibility(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _DETECTOR_CATALOG)

    unknown = invoke(profile.command, ["set", "sub_100keV", "--detector", "ed"])
    conflict = invoke(
        profile.command,
        ["set", "sub_100keV", "--detector", "eds", "--observation-angle", "90"],
    )
    inline = invoke(profile.command, ["set", "attached", "--observation-angle", "100"])

    assert unknown.exit_code == 1
    assert "Did you mean: eds" in unknown.stderr
    assert "pyrite detector create ed" in unknown.stderr
    assert conflict.exit_code == 2
    assert "--detector replaces inline detector flags" in conflict.stderr
    assert_clean_result(
        inline,
        stdout="updated profile attached\n",
        stderr=flag_message(
            "profile set",
            "--observation-angle",
            "pyrite detector create/set --observation-angle",
        )
        + "\n",
    )
    text = catalog.read_text()
    assert "[profiles.attached.detector]" in text
    assert "observation_angle_deg = 100.0" in text
    assert "polar_acceptance_deg = 16.6" in text


def test_set_inline_flag_onto_named_reference_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["set", "attached", "--rep-rate-hz", "100"])

    assert result.exit_code == 1
    assert "profile attached has beam = 'rf_gun_200fs' (a named reference)" in result.stderr
    assert "pyrite beam set rf_gun_200fs" in result.stderr


def test_remove_beam_detaches_named_reference(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["remove", "attached", "--beam"])

    assert_clean_result(result, stdout="updated profile attached; beam: detached rf_gun_200fs\n")
    section = catalog.read_text().split("[profiles.attached]", 1)[1].split("\n[", 1)[0]
    assert "beam" not in section

    shown = invoke(profile.command, ["show", "attached", "-o", "json"])
    assert_clean_result(shown)
    payload = json.loads(shown.stdout)["payload"]
    assert payload["beam_ref"] is None
    assert payload["beam"] is None


def test_remove_beam_when_absent_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["remove", "sub_100keV", "--beam"])

    assert result.exit_code == 1
    assert "profile sub_100keV has no beam to remove" in result.stderr


def test_remove_bare_no_op_error_mentions_beam(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)

    result = invoke(profile.command, ["remove", "sub_100keV"])

    assert result.exit_code == 2
    assert "provide a range, membership, beam, or emission option" in result.stderr
