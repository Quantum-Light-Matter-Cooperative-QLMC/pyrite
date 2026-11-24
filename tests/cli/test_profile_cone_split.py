"""``pyrite profile numerics cone-split`` and the profile ``cone_split`` key (#1384)."""

import json
from dataclasses import replace

import pytest

from pyrite._cone_splitting import ConeSplitting
from pyrite.api import build_configured_cases
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import canonical_settings, dataset_identity, profile_run_settings
from pyrite.cli import _catalog_io
from pyrite.cli import command as root_command
from pyrite.materials._schema import MaterialConfigError
from pyrite.materials.catalog import load_material_catalog
from tests.helpers.cli import assert_clean_result, invoke
from tests.helpers.user_catalog import copy_full_catalog

# 5 MeV h-BN: the #203 regime where splitting pays.
_SPLIT_PROFILE = """materials = ["hbn"]
thickness_ang = {values = [10000.0]}
energy_keV = {values = [5000.0]}
tilt_deg = {values = [45.0]}
tilt_azim_deg = {values = [180.0]}
cxr_histories = {values = [20]}
brem_histories = {values = [20]}
"""


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "catalog"
    copy_full_catalog(root)
    (root / "profiles" / "split.toml").write_text(_SPLIT_PROFILE)
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", root)
    monkeypatch.setenv("PYRITE_CATALOG", str(root))
    load_material_catalog.cache_clear()
    yield root
    load_material_catalog.cache_clear()


def _cone(*args):
    return invoke(root_command, ["profile", "numerics", "cone-split", *args])


def _file(catalog, name="split"):
    return (catalog / "profiles" / f"{name}.toml").read_text()


def _stored(catalog, name="split"):
    load_material_catalog.cache_clear()
    return load_material_catalog(catalog, profile=name).profile_cone_split(name)


def _show(name="split"):
    result = _cone("show", name, "-o", "json")
    assert_clean_result(result)
    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.profile.numerics.cone-split.show"
    return envelope["payload"]


def test_help_lists_every_field_with_its_measured_default():
    result = _cone("set", "--help")
    assert_clean_result(result)
    flat = " ".join(result.stdout.split())
    for text in (
        "--cone-factor FACTOR",
        "Default 8.",
        "--min-energy-fraction FRACTION",
        "Default 0.4.",
        "--angle-scale none|U_REF",
        "Default 1.",
        "--target-weight WEIGHT",
        "Default 0.001.",
        "--max-copies-per-electron N",
        "Default 1024.",
    ):
        assert text in flat
    group = " ".join(invoke(root_command, ["profile", "numerics", "--help"]).stdout.split())
    assert "cone-split" in group


def test_enable_set_show_reset_disable_round_trip(catalog):
    assert _stored(catalog) is None
    assert _show()["enabled"] is False

    assert_clean_result(_cone("enable", "split"), stdout=None)
    assert "cone_split = true" in _file(catalog)
    assert _stored(catalog) == ConeSplitting()
    assert _cone("enable", "split").stdout.startswith("profile split: detector-cone splitting")

    written = _cone("set", "split", "--cone-factor", "6", "--angle-scale", "none")
    assert_clean_result(written)
    assert _stored(catalog) == ConeSplitting(cone_factor=6.0, angle_scale=None)
    shown = {row["key"]: row for row in _show()["fields"]}
    assert shown["cone_factor"] == {
        "key": "cone_factor",
        "explicit": 6.0,
        "effective": 6.0,
        "source": "profile",
    }
    assert shown["angle_scale"]["effective"] == "none"
    assert shown["target_weight"]["source"] == "built-in default"

    assert_clean_result(_cone("reset", "split", "cone-factor"))
    assert _stored(catalog) == ConeSplitting(angle_scale=None)
    assert_clean_result(_cone("reset", "split"))
    assert _stored(catalog) == ConeSplitting()  # reset never turns splitting off
    assert_clean_result(_cone("disable", "split"))
    assert "cone_split" not in _file(catalog)
    assert _stored(catalog) is None


def test_set_and_reset_need_an_enabled_profile(catalog):
    before = _file(catalog)
    refused = _cone("set", "split", "--cone-factor", "6")
    assert refused.exit_code == 2
    assert "pyrite profile numerics cone-split enable split" in refused.stderr
    assert _cone("reset", "split").exit_code == 2
    assert _cone("set", "split").exit_code == 2
    assert _file(catalog) == before


@pytest.mark.parametrize(
    "argv",
    [
        ("--cone-factor", "0"),
        ("--min-energy-fraction", "1"),
        ("--angle-scale", "-1"),
        ("--target-weight", "nan-ish"),
        ("--max-copies-per-electron", "0"),
    ],
)
def test_invalid_field_values_are_usage_errors(catalog, argv):
    before = _file(catalog)
    result = _cone("enable", "split", *argv)
    assert result.exit_code == 2
    assert argv[0] in result.stderr
    assert _file(catalog) == before


@pytest.mark.parametrize(
    ("profile", "argv", "message"),
    [
        ("coherent_demo", (), "does not support emission 'both'"),
        ("high_energy_demo", (), "lower cone_factor"),
    ],
)
def test_refused_profiles_exit_nonzero_without_writing(catalog, profile, argv, message):
    path = catalog / "profiles" / f"{profile}.toml"
    before = path.read_text()
    result = _cone("enable", profile, *argv)
    assert result.exit_code == 1
    assert message in result.stderr
    assert "Traceback" not in result.output
    assert path.read_text() == before


def test_low_energy_profile_warns_on_stderr(catalog):
    result = _cone("enable", "high_energy_demo", "--cone-factor", "4")
    assert result.exit_code == 0
    assert "below 3000 keV (1000)" in result.stderr
    assert _stored(catalog, "high_energy_demo") == ConeSplitting(cone_factor=4.0)


def test_hand_written_key_is_validated_at_catalog_load(catalog):
    profile = catalog / "profiles" / "split.toml"
    for value, message in (
        ("{ cone_factor = 6.0, factor = 2 }", "unknown field(s) ['factor']"),
        ('{ model = "other" }', "unknown field(s) ['model']"),
        ("false", "must be true (the measured defaults) or a table"),
        ("{ target_weight = 0.0 }", "target_weight must be positive"),
    ):
        profile.write_text(f"{_SPLIT_PROFILE}cone_split = {value}\n")
        load_material_catalog.cache_clear()
        with pytest.raises(MaterialConfigError, match="profiles.split.cone_split") as raised:
            load_material_catalog(catalog, profile="split")
        assert message in str(raised.value)
    refused = invoke(root_command, ["profile", "numerics", "validate", "split"])
    assert refused.exit_code == 1


def _cases_and_identity(catalog):
    load_material_catalog.cache_clear()
    loaded = load_material_catalog(catalog, profile="split")
    sweep = material_sweep("hbn", catalog_profile="split")
    settings = profile_run_settings(default_settings(), "split", loaded, sweep=sweep)
    cases = build_configured_cases(sweep, settings)
    identity = dataset_identity("hbn", settings, sweep, catalog_profile="split")
    return settings, cases, identity


def test_profile_key_reaches_cases_and_identity_only_when_present(catalog):
    analog_settings, analog_cases, analog_identity = _cases_and_identity(catalog)
    assert analog_settings.cone_split is None
    assert all("cone_split" not in case for case in analog_cases)
    # Absent key: nothing joins the payload, so pinned identities stay put.
    assert "cone_split" not in json.dumps(analog_identity["resolved_parameters"])

    assert_clean_result(_cone("enable", "split"))
    split_settings, split_cases, split_identity = _cases_and_identity(catalog)
    record = ConeSplitting().to_dict()
    # Same payload as Numerics(cone_split=ConeSplitting()) on the Python route.
    assert split_cases == [replace(case, cone_split=record) for case in analog_cases]
    assert split_identity["parameter_sha256"] != analog_identity["parameter_sha256"]
    numerics = split_identity["resolved_parameters"]["transport_numerics"]
    assert numerics["cone_split"] == record
    assert not canonical_settings(split_settings)
