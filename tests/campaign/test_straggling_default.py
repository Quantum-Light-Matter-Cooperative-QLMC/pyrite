"""Model-aware straggling default (#642): shell on, continuous explicit, omissions historical."""

import pickle
from dataclasses import replace

import pytest

from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import (
    canonical_settings,
    case_content_key,
    dataset_identity,
    resolve_numerics,
)
from pyrite.campaign.sweep import build_cases
from pyrite.config import Physics
from pyrite.results import Settings

CHOICE = "requires an explicit straggling choice"


def test_omitted_straggling_resolves_with_the_inelastic_model():
    assert Physics().straggling is None
    assert Settings().straggling is True
    assert Settings(inelastic_model="auto", inelastic_cutoff_eV=None).straggling is False
    sweep = material_sweep("silicon")
    assert all(case["straggling"] is True for case in build_cases(sweep, 4, 4))
    legacy = build_cases(sweep, 4, 4, inelastic_model="auto")
    assert all("straggling" not in case for case in legacy)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Physics(inelastic_model="continuous"),
        lambda: Settings(inelastic_model="continuous"),
        lambda: build_cases(material_sweep("silicon"), 4, 4, inelastic_model="continuous"),
        lambda: resolve_numerics({"inelastic_model": "continuous"}),
        # Replacing a default Physics carries the unresolved default, not "on".
        lambda: replace(Physics(), inelastic_model="continuous", inelastic_cutoff_eV=None),
    ],
)
def test_continuous_stopping_never_inherits_a_straggling_default(make):
    with pytest.raises(ValueError, match=CHOICE):
        make()


@pytest.mark.parametrize("straggling", [True, False])
def test_explicit_continuous_choice_is_honoured(straggling):
    physics = Physics(inelastic_model="continuous", straggling=straggling)
    assert physics.straggling is straggling
    resolved = resolve_numerics({"inelastic_model": "continuous", "straggling": straggling})
    assert resolved.effective["straggling"] is straggling
    case = build_cases(
        material_sweep("sapphire"), 4, 4, inelastic_model="continuous", straggling=straggling
    )[0]
    assert case.get("straggling", False) is straggling


def test_explicit_off_restores_historical_cases_and_identity():
    sweep = material_sweep("silicon")
    off = build_cases(sweep, 4, 4, straggling=False)
    legacy = build_cases(sweep, 4, 4, inelastic_model="auto")
    assert [case_content_key(c) for c in off] == [case_content_key(c) for c in legacy]
    on = build_cases(sweep, 4, 4)
    assert case_content_key(on[0]) != case_content_key(off[0])

    settings = default_settings()
    default_id = dataset_identity("silicon", settings, sweep)
    off_id = dataset_identity("silicon", replace(settings, straggling=False), sweep)
    assert default_id["parameter_sha256"] != off_id["parameter_sha256"]
    assert default_id["resolved_parameters"]["transport_numerics"]["straggling"] is True
    assert "straggling" not in off_id["resolved_parameters"]["transport_numerics"]


def test_profile_resolution_reports_the_resolved_default():
    resolved = resolve_numerics()
    assert resolved.effective["straggling"] is True
    assert resolved.sources["straggling"] == "built-in"
    assert resolve_numerics({"inelastic_model": "auto"}).effective["straggling"] is False
    assert resolve_numerics({"straggling": False}).effective["straggling"] is False


def test_pickled_settings_without_straggling_keep_deterministic_loss():
    old = Settings(inelastic_model="auto", inelastic_cutoff_eV=None)
    del old.__dict__["straggling"]
    assert pickle.loads(pickle.dumps(old)).straggling is False


def test_bare_stem_follows_the_new_default():
    settings = default_settings()
    assert canonical_settings(settings)
    assert not canonical_settings(replace(settings, straggling=False))
