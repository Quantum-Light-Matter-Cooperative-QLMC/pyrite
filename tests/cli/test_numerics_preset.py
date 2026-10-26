"""Named numerics presets, run provenance and remote/worker parity (#663, ADR-0021)."""

import json
import pickle
from argparse import Namespace
from dataclasses import replace

import pytest

import pyrite
from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite._precision import DEFAULT_PRECISION
from pyrite.cli import command
from pyrite.materials import MaterialConfigError, load_material_catalog
from pyrite.materials.catalog import _refresh_default_catalog
from tests.helpers.cli import assert_clean_result, invoke


def member(profile):
    catalog = load_material_catalog(profile=profile)
    return (catalog.profile_materials(profile) or catalog.material_keys)[0]


def ok(*args):
    result = invoke(command, list(args))
    assert_clean_result(result)
    return result


def payload(*args):
    return json.loads(ok(*args, "-o", "json").stdout)["payload"]


def profile_row(name):
    return read_raw(bundled_catalog())["profiles"][name]


def resolved(profile):
    from pyrite.runs.scan import _resolved_run

    _refresh_default_catalog(load_material_catalog())
    material = member(profile)
    args = Namespace(
        quick=False,
        fidelity="full",
        catalog_profile=profile,
        detector_id=None,
        high_energy_floor_map={},
    )
    return _resolved_run(args, material)[2]


def write_preset(layer, table, name, text):
    (layer / table).mkdir(exist_ok=True)
    (layer / table / f"{name}.toml").write_text(text)


def test_bundled_numerics_reference_restates_built_ins():
    assert pyrite.config.load_numerics("default") == pyrite.Numerics()
    refused = invoke(command, ["numerics", "preset", "set", "default", "--mosaic-nodes", "7"])
    assert refused.exit_code == 1
    assert "pyrite numerics preset create NAME --from default" in refused.stderr


def test_unit_bearing_numerics_preset_resolves_to_canonical_values(_isolate_user_catalog):
    write_preset(
        _isolate_user_catalog,
        "numerics",
        "unit-inputs",
        'channeling_dz_ang = "0.1 nm"\nchanneling_zone_max_ang = "10 nm"\n',
    )

    numerics = pyrite.config.load_numerics("unit-inputs", catalog=load_material_catalog())
    assert numerics.channeling_dz_ang == 1.0
    assert numerics.channeling_zone_max_ang == 100.0


def test_unit_bearing_numerics_preset_rejects_wrong_dimensions(_isolate_user_catalog):
    write_preset(
        _isolate_user_catalog,
        "numerics",
        "bad-units",
        'channeling_dz_ang = "2 keV"\n',
    )

    with pytest.raises(MaterialConfigError, match="numerics.channeling_dz_ang.*length"):
        load_material_catalog()


def test_numerics_preset_resolves_into_profiles_with_provenance(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("numerics", "preset", "create", "fast", "--cxr-trials", "200", "--mosaic-nodes", "7")
    ok("profile", "set", "campaign", "--numerics", "fast")

    catalog = load_material_catalog(profile="campaign")
    material = member("campaign")
    assert catalog.material(material).scan.n_electrons == (200,)
    shown = payload("numerics", "show", "campaign")
    sources = {row["key"]: row["source"] for row in shown["fields"]}
    assert shown["preset"] == "fast"
    assert sources["n_electrons"] == "preset fast"
    assert sources["mosaic_nodes"] == "preset fast"
    assert sources["n_electrons_brem"] == "fidelity"
    summary = payload("profile", "show", "campaign")
    assert summary["presets"] == {"physics": None, "numerics": "fast"}

    numerics = pyrite.config.load_numerics("fast", catalog=load_material_catalog())
    expected = replace(
        pyrite.Numerics(),
        cxr_trials=200,
        cxr_reflections=pyrite.config.CxrReflections(mosaic_nodes=7),
    )
    assert numerics == expected
    assert pickle.loads(pickle.dumps(numerics)) == numerics


def test_profile_sampling_policy_replaces_the_presets_whole_group(_isolate_user_catalog):
    write_preset(
        _isolate_user_catalog,
        "numerics",
        "adaptive",
        "mosaic_nodes = 7\n[precision]\n" + _precision_toml(),
    )
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("profile", "set", "campaign", "--numerics", "adaptive")
    summary = payload("profile", "show", "campaign")
    assert summary["precision"]["source"] == "preset adaptive"

    ok("numerics", "set", "campaign", "--cxr-trials", "300")
    catalog = load_material_catalog(profile="campaign")
    assert catalog.profile_precision("campaign") is None
    assert catalog.profile_numerics("campaign")["mosaic_nodes"] == 7
    assert payload("profile", "show", "campaign")["precision"]["source"] != "preset adaptive"


def _precision_toml():
    values = DEFAULT_PRECISION.to_dict()
    lines = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, list | tuple):
            lines.append(f"{key} = {json.dumps(list(value))}")
        else:
            lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def test_adoption_keeps_identity_and_records_preset_provenance(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    ok("physics", "set", "campaign", "--straggling")
    ok("numerics", "set", "campaign", "--mosaic-nodes", "7")
    before = resolved("campaign")
    assert "presets" not in before

    ok("physics", "preset", "adopt", "campaign", "my-physics")
    ok("numerics", "preset", "adopt", "campaign", "my-numerics")
    row = profile_row("campaign")
    assert row["physics"] == "my-physics" and row["numerics"] == "my-numerics"
    assert "straggling" not in row and "mosaic_nodes" not in row

    adopted = resolved("campaign")
    assert adopted["parameter_sha256"] == before["parameter_sha256"]
    labels = adopted["presets"]
    assert labels["physics"]["name"] == "my-physics"
    assert labels["numerics"]["name"] == "my-numerics"

    ok("numerics", "preset", "rename", "my-numerics", "shared")
    renamed = resolved("campaign")
    assert renamed["parameter_sha256"] == before["parameter_sha256"]
    assert renamed["presets"]["numerics"] == {
        "name": "shared",
        "sha256": labels["numerics"]["sha256"],
    }

    ok("numerics", "preset", "set", "shared", "--mosaic-nodes", "9")
    edited = resolved("campaign")
    assert edited["parameter_sha256"] != before["parameter_sha256"]
    assert edited["presets"]["numerics"]["sha256"] != labels["numerics"]["sha256"]


def test_adopt_keeps_multi_value_trial_grids_on_the_profile(_isolate_user_catalog):
    ok("profile", "create", "campaign", "--from", "quickstart")
    profile = _isolate_user_catalog / "profiles" / "campaign.toml"
    # Root keys must precede the profile's sub-tables.
    profile.write_text(
        "n_electrons = {values = [100, 200]}\nmosaic_nodes = 7\n" + profile.read_text()
    )
    ok("numerics", "preset", "adopt", "campaign", "mine")
    # On disk, trial counts use the canonical ``cxr_trials`` spelling.
    row = profile_row("campaign")
    assert row["cxr_trials"] == {"values": [100, 200]}
    assert "mosaic_nodes" not in row
    preset = read_raw(bundled_catalog())["numerics"]["mine"]
    assert "cxr_trials" not in preset and preset["mosaic_nodes"] == 7


def test_user_presets_ship_to_remote_jobs(_isolate_user_catalog):
    from pyrite.remote.transport import _user_catalog_entries

    write_preset(_isolate_user_catalog, "physics", "mine", "straggling = true\n")
    write_preset(_isolate_user_catalog, "numerics", "mine", "mosaic_nodes = 7\n")
    shipped = {name for name, _path in _user_catalog_entries()}
    assert any(name.endswith("physics/mine.toml") for name in shipped)
    assert any(name.endswith("numerics/mine.toml") for name in shipped)


def test_loaded_physics_survives_worker_serialization():
    physics = pyrite.config.load_physics("electron-standard")
    assert pickle.loads(pickle.dumps(physics)) == physics
