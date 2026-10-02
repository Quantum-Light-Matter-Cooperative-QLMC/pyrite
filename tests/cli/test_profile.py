import json

import tomlkit

from pyrite import _energy_grid_artifacts as artifacts
from pyrite.cli import _catalog_io
from pyrite.cli import command as root_command
from pyrite.cli._deprecations import option_message
from pyrite.cli.commands import profile
from pyrite.console import output as _core
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

_INLINE_BEAM_CATALOG = (
    _CATALOG
    + """
[profiles.inline]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }

[profiles.inline.beam]
transverse_fwhm_mm = 0.1
rep_rate_hz = 5000.0
bunch_charge_pc = 1.0

[profiles.inline.beam.longitudinal]
kind = "microtrain"
envelope_rms_fs = 200.0
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
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", catalog)
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

    # A profile with no refs uses inline catalog grids.
    legacy = invoke(profile.command, ["show", "standard"])
    assert_clean_result(legacy)
    assert "energy grids: inline (E_grid_brem + material overrides)" in legacy.stdout

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
    assert "  detectors:\n    default: scalar\n" in shown.stdout
    assert "pixel" not in shown.stdout
    assert "emission: incoherent (default)" in shown.stdout
    assert "straggling: False (default)" in shown.stdout
    assert "energy model: midpoint (default)" in shown.stdout
    assert "max dE fraction: 0 (default)" in shown.stdout

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


_FILTER_ADD = [
    "filter",
    "add",
    "standard",
    "--name",
    "half",
    "--material",
    "silicon",
    "--thickness-mm",
    "0.1",
    "--size-mm",
    "7.04",
    "14.08",
    "--distance-mm",
    "200",
    "--offset-mm",
    "3.52",
    "0",
]


def test_filter_crud_exposes_json(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    created = invoke(
        profile.command,
        [
            "physical-detector",
            "set",
            "standard",
            "-y",
            "--distance-mm",
            "300",
            "--shape",
            "2",
            "3",
            "--pitch-mm",
            "0.1",
            "0.2",
        ],
    )
    assert_clean_result(created, stdout="updated physical detector for profile standard\n")

    added = invoke(profile.command, _FILTER_ADD)
    assert_clean_result(added, stdout="added filter to profile standard\n")
    assert "[[profiles.standard.filters]]" in catalog.read_text()

    listed = invoke(profile.command, ["filter", "list", "standard", "-o", "json"])
    assert_clean_result(listed)
    row = json.loads(listed.stdout)["payload"]["filters"][0]
    assert row["name"] == "half"
    assert row["offset_mm"] == [3.52, 0.0]

    shown = invoke(profile.command, ["show", "standard", "-o", "json"])
    assert_clean_result(shown)
    payload = json.loads(shown.stdout)["payload"]
    assert payload["filters"][0]["material"] == "silicon"
    assert payload["physical_detector"]["shape"] == [2, 3]

    removed = invoke(profile.command, ["filter", "rm", "standard", "half"])
    assert_clean_result(removed, stdout="removed filter half from profile standard\n")
    assert "filters" not in catalog.read_text()


def test_filter_add_no_longer_edits_the_physical_detector(tmp_path, monkeypatch):
    catalog = _catalog(
        tmp_path,
        monkeypatch,
        _CATALOG
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
shape = [4, 5]
""",
    )

    added = invoke(profile.command, _FILTER_ADD)
    assert_clean_result(added)
    text = catalog.read_text()
    assert "distance_mm = 400.0" in text
    assert "shape = [4, 5]" in text

    moved = invoke(profile.command, [*_FILTER_ADD, "--detector-distance-mm", "300"])
    assert moved.exit_code == 2
    assert "No such option" in moved.stderr


def test_filter_set_updates_in_place_and_keeps_order(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, _FILTER_ADD)
    second = [*_FILTER_ADD]
    second[second.index("half")] = "second"
    invoke(profile.command, second)

    updated = invoke(
        profile.command,
        ["filter", "set", "standard", "half", "--thickness-mm", "0.25", "--name", "thick"],
    )
    assert_clean_result(updated, stdout="updated filter thick on profile standard\n")
    rows = json.loads(invoke(profile.command, ["filter", "list", "standard", "-o", "json"]).stdout)[
        "payload"
    ]["filters"]
    assert [row["name"] for row in rows] == ["thick", "second"]
    assert rows[0]["thickness_mm"] == 0.25

    before = catalog.read_text()
    duplicate = invoke(profile.command, ["filter", "set", "standard", "1", "--name", "second"])
    assert duplicate.exit_code == 2
    assert "already has a filter named 'second'" in duplicate.stderr
    empty = invoke(profile.command, ["filter", "set", "standard", "1"])
    assert empty.exit_code == 2
    missing = invoke(profile.command, ["filter", "set", "standard", "nope", "--thickness-mm", "1"])
    assert missing.exit_code == 2
    assert catalog.read_text() == before


def test_show_renders_hand_authored_inline_beam_block(tmp_path, monkeypatch):
    """`profile` no longer writes `[profiles.NAME.beam]` (issue #54), but the
    block is still a valid hand-authored catalog shape, so `show` must keep
    resolving it."""
    _catalog(tmp_path, monkeypatch, _INLINE_BEAM_CATALOG)

    shown = invoke(profile.command, ["show", "inline", "-o", "json"])

    assert_clean_result(shown)
    payload = json.loads(shown.stdout)["payload"]
    assert payload["beam_ref"] is None
    assert payload["beam"] == {
        "transverse_fwhm_mm": 0.1,
        "rep_rate_hz": 5000.0,
        "bunch_charge_pc": 1.0,
        "longitudinal": {"kind": "microtrain", "envelope_rms_fs": 200.0},
    }


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


def test_show_does_not_inherit_standard_detector_when_profile_block_is_absent(
    tmp_path, monkeypatch
):
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
        "observation_angle_deg": 90.0,
        "polar_acceptance_deg": None,
        "solid_angle_sr": None,
    }


def test_create_clones_source_and_applies_range_overrides(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        profile.command, ["create", "sub_200keV", "--from", "sub_100keV", "--energy", "150,200"]
    )

    assert_clean_result(result, stdout="created profile sub_200keV from sub_100keV\n")
    text = catalog.read_text()
    assert "[profiles.sub_200keV]" in text
    assert "energy_keV = {values = [150.0, 200.0]}" in text
    # Cloned grids and membership carry over; per-material overrides do not.
    assert "tilt_deg = {values = [5.0]}" in text
    section = text.split("[profiles.sub_200keV]", 1)[1].split("\n[", 1)[0]
    assert 'materials = ["hopg"]' in section


def test_create_uses_packaged_template_and_explicit_clone_lists_inherited_sections(
    tmp_path, monkeypatch
):
    source = _CATALOG.replace(
        "[profiles.standard]\n",
        '[profiles.standard]\ndetector = "default"\nemission = "coherent"\nstraggling = true\n',
    )
    source = source.replace(
        "energy_keV = { values = [30.0, 60.0] }", "energy_keV = { values = [999.0] }", 1
    )
    source += """
[profiles.standard.physical_detector]
distance_mm = 400.0
polar_deg = 60.0
shape = [2, 3]

[[profiles.standard.filters]]
material = "silicon"
thickness_mm = 0.1
size_mm = [7.04, 14.08]
distance_mm = 200.0
polar_deg = 90.0
"""
    catalog = _catalog(tmp_path, monkeypatch, source)

    created = invoke(profile.command, ["create", "fresh"])
    assert_clean_result(created, stdout="created profile fresh\n")
    fresh = tomlkit.parse(catalog.read_text())["profiles"]["fresh"]
    assert fresh["energy_keV"]["values"][0] == 30.0
    assert fresh["E_grid_brem"]["arange"]["step"] == 25.0
    # Packaged standard carries no uniform brem override for hopg (issue #256).
    assert "overrides" not in fresh
    assert list(fresh["materials"]) == ["hopg"]
    for key in ("detector", "physical_detector", "filters", "emission", "straggling"):
        assert key not in fresh

    cloned = invoke(profile.command, ["create", "clone", "--from", "standard"])
    assert_clean_result(
        cloned,
        stdout=(
            "created profile clone from standard (inherited: detector, physical detector, "
            "filters, emission, transport numerics)\n"
        ),
    )
    clone = tomlkit.parse(catalog.read_text())["profiles"]["clone"]
    for key in ("detector", "physical_detector", "filters", "emission", "straggling"):
        assert key in clone
    assert "overrides" not in clone


def test_create_clones_energy_grid_refs_without_copying_artifact_bytes(tmp_path, monkeypatch):
    stored = _artifact(tmp_path)
    catalog = _catalog(tmp_path, monkeypatch, _artifact_ref_catalog(stored.digest))
    original_bytes = stored.path.read_bytes()

    result = invoke(profile.command, ["create", "clone", "--from", "sub_100keV"])

    assert_clean_result(result, stdout="created profile clone from sub_100keV\n")
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

    assert_clean_result(
        result,
        stdout="updated profile sub_100keV: polar changed from 5 to 10, 20, 30\n",
    )
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


def test_set_reports_a_new_profile_parameter(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--ne-line", "100,200"])

    assert_clean_result(
        result,
        stdout="updated profile sub_100keV: line electrons set to 100, 200\n",
    )


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
    assert_clean_result(
        accepted,
        stdout="updated profile standard: energy changed from 30, 60 to 40\n",
    )
    assert "energy_keV = {values = [40.0]}" in catalog.read_text()


def test_set_nonstandard_never_prompts(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    # No input supplied: an unwanted prompt would hit EOF and abort.
    result = invoke(profile.command, ["set", "sub_100keV", "--energy", "40"])

    assert_clean_result(
        result,
        stdout="updated profile sub_100keV: energy changed from 30, 50 to 40\n",
    )


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


def test_set_temporal_profile_toggles_key_and_show(tmp_path, monkeypatch):
    """#292: --temporal-profile writes the opt-in key; --no-temporal-profile drops it."""
    catalog = _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "temporal profile: off (default)" in shown.stdout
    machine = invoke(profile.command, ["show", "sub_100keV", "-o", "json"])
    assert json.loads(machine.stdout)["payload"]["temporal_profile"] is False

    result = invoke(profile.command, ["set", "sub_100keV", "--temporal-profile"])
    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "temporal_profile = true" in catalog.read_text()
    assert "temporal profile: on" in invoke(profile.command, ["show", "sub_100keV"]).stdout

    result = invoke(profile.command, ["set", "sub_100keV", "--no-temporal-profile"])
    assert_clean_result(result, stdout="updated profile sub_100keV\n")
    assert "temporal_profile" not in catalog.read_text()


def test_set_emission_invalid_value_rejected(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(profile.command, ["set", "sub_100keV", "--emission", "bogus"])

    assert result.exit_code == 2
    assert "not one of" in result.stderr.lower()


def test_set_transport_numerics_round_trips_and_validates_coupling(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    invalid = invoke(
        profile.command,
        ["set", "sub_100keV", "--energy-model", "frozen", "--max-de-frac", "0.02"],
    )
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


def test_numerics_show_reports_effective_values_and_sources(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(
        root_command, ["profile", "numerics", "show", "sub_100keV", "--fidelity", "survey"]
    )
    fidelity_warning = option_message("profile numerics show", "--fidelity") + "\n"
    assert_clean_result(shown, stderr=fidelity_warning)
    assert "sampling:" in shown.stdout
    assert "line electrons: 60 (fidelity)" in shown.stdout
    assert "reflection families: 2 (fidelity)" in shown.stdout
    assert "mosaic nodes: 5 (built-in)" in shown.stdout
    assert "energy model: midpoint (built-in)" in shown.stdout

    machine = invoke(
        root_command,
        ["profile", "numerics", "show", "sub_100keV", "--fidelity", "survey", "-o", "json"],
    )
    assert_clean_result(machine, stderr=fidelity_warning)
    envelope = json.loads(machine.stdout)
    assert envelope["schema"] == "cxr.profile.numerics.show"
    assert envelope["payload"]["profile"] == "sub_100keV"
    fields = {row["key"]: row for group in envelope["payload"]["groups"] for row in group["fields"]}
    assert fields["n_electrons"] == {
        "key": "n_electrons",
        "label": "line electrons",
        "explicit": None,
        "effective": 60,
        "source": "fidelity",
    }
    assert fields["max_reflections"]["effective"] == 4
    assert fields["max_reflections"]["source"] == "fidelity"


def test_numerics_help_exposes_scientific_controls_not_execution_tuning():
    group = invoke(profile.command, ["numerics", "--help"])
    assert_clean_result(group)
    assert "show" in group.stdout
    assert "set" in group.stdout
    assert "reset" in group.stdout

    setting = invoke(profile.command, ["numerics", "set", "--help"])
    assert_clean_result(setting)
    for option in (
        "--line-electrons",
        "--bremsstrahlung-electrons",
        "--reflection-families",
        "--maximum-reflections",
        "--mosaic-nodes",
        "--mosaic-route",
        "--maximum-fractional-energy-loss",
    ):
        assert option in setting.stdout
    for excluded in ("--workers", "--backend", "--spec-chunk", "--transport-core"):
        assert excluded not in setting.stdout


def test_numerics_set_dry_run_write_validation_and_reset(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    dry_run = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--line-electrons",
            "12",
            "--reflection-families",
            "3",
            "--mosaic-nodes",
            "7",
            "--mosaic-route",
            "mc",
            "--dry-run",
        ],
    )
    assert_clean_result(dry_run)
    assert "+n_electrons = {values = [12]}" in dry_run.stdout
    assert "+n_families = 3" in dry_run.stdout
    assert catalog.read_text() == _CATALOG

    invalid = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--energy-model",
            "frozen",
            "--maximum-fractional-energy-loss",
            "0.02",
        ],
    )
    assert invalid.exit_code == 1
    assert "requires energy_model='midpoint'" in invalid.stderr

    written = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--energy-model",
            "midpoint",
            "--maximum-fractional-energy-loss",
            "0.02",
            "--maximum-reflections",
            "6",
        ],
    )
    assert_clean_result(written, stdout="updated numerics for profile sub_100keV\n")
    assert 'energy_model = "midpoint"' in catalog.read_text()
    assert "max_dE_frac = 0.02" in catalog.read_text()
    assert "max_reflections = 6" in catalog.read_text()

    shown = invoke(profile.command, ["numerics", "show", "sub_100keV"])
    assert_clean_result(shown)
    assert "maximum reflections: 6 (profile); explicit: 6" in shown.stdout

    reset = invoke(
        profile.command,
        [
            "numerics",
            "reset",
            "sub_100keV",
            "maximum-reflections",
            "maximum-fractional-energy-loss",
        ],
    )
    assert_clean_result(reset, stdout="reset numerics for profile sub_100keV\n")
    assert "max_reflections" not in catalog.read_text()
    assert "max_dE_frac" not in catalog.read_text()


def test_numerics_standard_mutation_requires_confirmation(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(
        profile.command,
        ["numerics", "set", "standard", "--reflection-families", "3"],
        input="n\n",
    )
    assert declined.exit_code == 1
    assert catalog.read_text() == original

    accepted = invoke(
        profile.command,
        ["numerics", "set", "standard", "--reflection-families", "3", "--yes"],
    )
    assert_clean_result(accepted, stdout="updated numerics for profile standard\n")
    assert "n_families = 3" in catalog.read_text()


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
        "set": (
            "provide a range, beam, detector, transport, membership, emission, "
            "or temporal-profile option"
        ),
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


def test_inline_beam_flags_are_gone_from_create_and_set(tmp_path, monkeypatch):
    """Issue #54: beam phase space is set only through `pyrite beam`. The nine
    inline spellings are removed outright rather than deprecated, so Click
    rejects them as unknown options and nothing reaches the catalog."""
    catalog = _catalog(tmp_path, monkeypatch, _BEAM_CATALOG)
    original = catalog.read_text()
    retired = (
        ("--transverse-fwhm-mm", "0.1"),
        ("--rep-rate-hz", "5000"),
        ("--bunch-charge-pc", "1"),
        ("--longitudinal", "microtrain"),
        ("--envelope-rms-fs", "200"),
        ("--emittance", "1.0"),
        ("--twiss-beta", "0.5"),
        ("--twiss-alpha", "-0.8"),
        ("--energy-spread", "0.002"),
    )

    for flag, value in retired:
        created = invoke(profile.command, ["create", "newprof", "--material", "hopg", flag, value])
        updated = invoke(profile.command, ["set", "sub_100keV", flag, value])
        for result in (created, updated):
            assert result.exit_code == 2
            assert f"No such option '{flag}'" in result.stderr

    assert catalog.read_text() == original


def test_inline_detector_flags_are_gone_from_create_and_set(tmp_path, monkeypatch):
    """Issue #62: detector geometry is set only through `pyrite detector`. The
    three inline spellings are removed outright rather than deprecated, so
    Click rejects them as unknown options and nothing reaches the catalog."""
    catalog = _catalog(tmp_path, monkeypatch, _DETECTOR_CATALOG)
    original = catalog.read_text()
    retired = (
        ("--observation-angle", "100"),
        ("--polar-acceptance", "16.6"),
        ("--solid-angle", "0.066"),
    )

    for flag, value in retired:
        created = invoke(profile.command, ["create", "newprof", "--material", "hopg", flag, value])
        updated = invoke(profile.command, ["set", "sub_100keV", flag, value])
        for result in (created, updated):
            assert result.exit_code == 2
            assert f"No such option '{flag}'" in result.stderr

    assert catalog.read_text() == original


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


def test_named_detector_unknown_suggests_create(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch, _DETECTOR_CATALOG)

    unknown = invoke(profile.command, ["set", "sub_100keV", "--detector", "ed"])

    assert unknown.exit_code == 1
    assert "Did you mean: eds" in unknown.stderr
    assert "pyrite detector create ed" in unknown.stderr


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


def test_numerics_set_shell_inelastic_mode_validates_and_resets(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    missing_cutoff = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--energy-model",
            "midpoint",
            "--inelastic-model",
            "shell-soft-hard",
        ],
    )
    assert missing_cutoff.exit_code == 1
    assert "requires a finite positive inelastic_cutoff_eV" in missing_cutoff.stderr

    frozen = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--inelastic-model",
            "shell-soft-hard",
            "--energy-model",
            "frozen",
            "--inelastic-cutoff-ev",
            "50",
        ],
    )
    assert frozen.exit_code == 1
    assert "requires energy_model='midpoint'" in frozen.stderr

    written = invoke(
        profile.command,
        [
            "numerics",
            "set",
            "sub_100keV",
            "--energy-model",
            "midpoint",
            "--inelastic-model",
            "shell-soft-hard",
            "--inelastic-cutoff-ev",
            "50",
        ],
    )
    assert_clean_result(written, stdout="updated numerics for profile sub_100keV\n")
    assert 'inelastic_model = "shell-soft-hard"' in catalog.read_text()
    assert "inelastic_cutoff_eV = 50.0" in catalog.read_text()
    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "inelastic model: shell-soft-hard (W_c 50 eV)" in shown.stdout

    zero = invoke(profile.command, ["numerics", "set", "sub_100keV", "--inelastic-cutoff-ev", "0"])
    assert zero.exit_code == 2

    secondary = invoke(
        profile.command, ["numerics", "set", "sub_100keV", "--secondary-threshold-ev", "1000"]
    )
    assert_clean_result(secondary, stdout="updated numerics for profile sub_100keV\n")
    assert "secondary_threshold_eV = 1000.0" in catalog.read_text()
    shown = invoke(profile.command, ["show", "sub_100keV"])
    assert "secondary threshold: 1000 eV" in shown.stdout
    cleared = invoke(
        profile.command,
        ["numerics", "reset", "sub_100keV", "secondary-threshold-ev", "--yes"],
    )
    assert cleared.exit_code == 0
    assert "secondary_threshold_eV" not in catalog.read_text()

    reset = invoke(
        profile.command,
        ["numerics", "reset", "sub_100keV", "inelastic-model", "inelastic-cutoff-ev"],
    )
    assert_clean_result(reset, stdout="reset numerics for profile sub_100keV\n")
    assert "inelastic_model" not in catalog.read_text()
