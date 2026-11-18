"""'Off follows type' (#1242): ``none`` turns optional settings off; ``off`` warns."""

import warnings
from dataclasses import replace

import pytest

from pyrite._channeling_table import channeling_table_values
from pyrite._config_presets import normalize_field_units
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import dataset_identity, resolve_numerics
from pyrite.cli import _catalog_io, command
from pyrite.config import Physics
from tests.helpers.cli import invoke


def _run(*argv):
    return invoke(command, list(argv))


def _profile(name="campaign"):
    created = _run("profile", "create", name, "--from", "quickstart")
    assert created.exit_code == 0, created.output


def _catalog_text():
    return _catalog_io.catalog_text()[0]


@pytest.fixture
def cascades():
    """Profile ``campaign`` attached to physics preset ``cascades`` (pair production on)."""
    _profile()
    for argv in (
        [
            "physics",
            "create",
            "cascades",
            "--secondary-threshold-ev",
            "1000",
            "--pair-production-model",
            "penelope-2024",
        ],
        ["profile", "set", "campaign", "--physics", "cascades"],
    ):
        result = _run(*argv)
        assert result.exit_code == 0, result.output


#: Every optional physics setting in the #1242 inventory: (off argv, stored TOML).
_INVENTORY = {
    "straggling": (["--no-straggling"], "straggling = false"),
    "positron-transport": (["--no-positron-transport"], "positron_transport = false"),
    "atomic-electron-deflection": (
        ["--atomic-electron-deflection", "none"],
        'atomic_electron_deflection = "none"',
    ),
    "secondary-threshold-ev": (
        ["--secondary-threshold-ev", "none"],
        'secondary_threshold_eV = "none"',
    ),
    "pair-production-model": (
        ["--pair-production-model", "none"],
        'pair_production_model = "none"',
    ),
    "channeling-model": (["--channeling-model", "none"], 'model = "none"'),
    "transition-radiation": (
        ["--transition-radiation", "none"],
        'transition_radiation = "none"',
    ),
}


@pytest.mark.parametrize("option", sorted(_INVENTORY))
def test_every_inventory_setting_turns_off(option):
    _profile()
    argv, stored = _INVENTORY[option]
    result = _run("profile", "physics", "set", "campaign", *argv, "--yes")
    assert result.exit_code == 0, result.output
    assert "deprecated" not in result.stderr
    assert stored in _catalog_text()
    help_text = _run("profile", "physics", "set", "--help").stdout
    off_spelling = argv[0] if argv[0].startswith("--no-") else "none"
    assert off_spelling in help_text


@pytest.mark.parametrize(
    "option", ["channeling-model", "transition-radiation", "pair-production-model"]
)
def test_off_is_a_deprecated_alias_of_none(option):
    _profile()
    result = _run("profile", "physics", "set", "campaign", f"--{option}", "off", "--yes")
    assert result.exit_code == 0, result.output
    assert f"'--{option} off' is deprecated" in result.stderr
    assert f"use '--{option} none'" in result.stderr
    assert '"none"' in _catalog_text()


def test_none_overrides_a_preset_and_keeps_adaptive_sampling(cascades):
    # Cascades run the default policy (#1240); turning them off changes no count mode.
    shown = _run("profile", "numerics", "show", "campaign")
    assert "adaptive (default policy)" in shown.stdout
    assert "adaptive sampling:" not in shown.stdout
    off = _run(
        "profile",
        "physics",
        "set",
        "campaign",
        "--pair-production-model",
        "none",
        "--secondary-threshold-ev",
        "none",
        "--yes",
    )
    assert off.exit_code == 0, off.output
    shown = _run("profile", "numerics", "show", "campaign")
    assert "adaptive (default policy)" in shown.stdout
    assert "adaptive sampling:" not in shown.stdout


def test_threshold_off_under_preset_pair_production_names_the_fix(cascades):
    result = _run(
        "profile", "physics", "set", "campaign", "--secondary-threshold-ev", "none", "--yes"
    )
    assert result.exit_code == 2
    assert "--pair-production-model none --secondary-threshold-ev none" in result.stderr


def test_reset_threshold_under_profile_pair_production_names_the_fix():
    _profile()
    on = _run(
        "profile",
        "physics",
        "set",
        "campaign",
        "--secondary-threshold-ev",
        "1000",
        "--pair-production-model",
        "penelope-2024",
        "--yes",
    )
    assert on.exit_code == 0, on.output
    result = _run("profile", "physics", "reset", "campaign", "secondary-threshold-ev", "--yes")
    assert result.exit_code == 2
    assert (
        "pyrite profile physics set campaign --pair-production-model none "
        "--secondary-threshold-ev none"
    ) in result.stderr


def test_preset_threshold_off_names_the_preset_fix(cascades):
    result = _run("physics", "set", "cascades", "--secondary-threshold-ev", "none")
    assert result.exit_code == 2
    assert (
        "pyrite physics set cascades --pair-production-model none --secondary-threshold-ev none"
    ) in result.stderr


def test_api_accepts_none_and_warns_on_off():
    assert Physics(pair_production_model="none", secondary_threshold_eV="none") == Physics()
    assert Physics(transition_radiation="none") == Physics()
    with pytest.warns(FutureWarning, match='transition_radiation="off" is deprecated'):
        assert Physics(transition_radiation="off") == Physics()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Physics()


def test_toml_off_warns_and_maps_to_the_off_value():
    with pytest.warns(FutureWarning, match="model"):
        assert channeling_table_values({"channeling": {"model": "off"}}) == {
            "channeling_model": "none"
        }
    with pytest.warns(FutureWarning, match="transition_radiation"):
        values = normalize_field_units({"transition_radiation": "off"}, field_prefix="p")
    assert values == {"transition_radiation": "none"}
    stored = normalize_field_units(
        {"secondary_threshold_eV": "none", "pair_production_model": "none"}, field_prefix="p"
    )
    assert stored == {"secondary_threshold_eV": None, "pair_production_model": None}


def test_none_off_and_absent_share_identity():
    sweep = material_sweep("silicon")
    settings = default_settings()
    digest = dataset_identity("silicon", settings, sweep)["parameter_sha256"]
    for spelling in ("none", "off"):
        spelled = replace(settings, transition_radiation=spelling, channeling_model=spelling)
        assert spelled.transition_radiation == "off" == spelled.channeling_model
        assert dataset_identity("silicon", spelled, sweep)["parameter_sha256"] == digest
    explicit = resolve_numerics(
        {
            "secondary_threshold_eV": None,
            "pair_production_model": None,
            "transition_radiation": "none",
        }
    ).effective
    absent = resolve_numerics().effective
    for key in ("secondary_threshold_eV", "pair_production_model", "transition_radiation"):
        assert explicit[key] == absent[key]
