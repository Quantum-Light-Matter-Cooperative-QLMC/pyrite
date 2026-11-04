"""Specific inelastic selection preserves resolved historical checkpoint identities."""

import pickle
from dataclasses import replace

import pytest

from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.inelastic_cases import validate_explicit_shell_cases
from pyrite.campaign.profiles import case_content_key, dataset_identity, resolve_numerics
from pyrite.campaign.sweep import build_cases
from pyrite.config import Physics
from pyrite.results import Settings


@pytest.mark.parametrize("owner", [Physics, Settings])
def test_new_defaults_and_historical_pickle_omissions(owner):
    current = owner()
    assert (current.inelastic_model, current.inelastic_cutoff_eV, current.straggling) == (
        "shell-soft-hard",
        50.0,
        False,
    )
    assert pickle.loads(pickle.dumps(current)) == current
    old = owner(inelastic_model="auto")
    del old.__dict__["inelastic_model"]
    del old.__dict__["inelastic_cutoff_eV"]
    restored = pickle.loads(pickle.dumps(old))
    assert (restored.inelastic_model, restored.inelastic_cutoff_eV) == ("auto", None)
    assert owner(inelastic_model="continuous").inelastic_cutoff_eV is None


def test_resolved_legacy_cases_and_dataset_identities_are_reusable():
    sweep = material_sweep("silicon")
    current = default_settings()
    legacy = replace(current, inelastic_model="auto", inelastic_cutoff_eV=None)
    assert (
        dataset_identity("silicon", "full", current, sweep)["parameter_sha256"]
        == (dataset_identity("silicon", "full", legacy, sweep)["parameter_sha256"])
    )
    cases = build_cases(sweep, 4, 4)
    old = build_cases(sweep, 4, 4, inelastic_model="auto")
    assert cases == old
    assert case_content_key(cases[0]) == case_content_key(pickle.loads(pickle.dumps(old[0])))
    assert all(not case.get("straggling", False) for case in cases)


@pytest.mark.parametrize(
    "material,energy,reason",
    [
        ("hfs2", "midpoint", "choose inelastic_model='continuous'"),
        ("silicon", "frozen", "requires energy_model='midpoint'"),
    ],
)
def test_unsupported_default_scenes_require_explicit_continuous(material, energy, reason):
    sweep = material_sweep(material)
    with pytest.raises(ValueError, match=reason):
        build_cases(sweep, 4, 4, energy_model=energy, radiative_model="uncoupled")
    continuous = build_cases(
        sweep, 4, 4, energy_model=energy, inelastic_model="continuous", radiative_model="uncoupled"
    )
    assert all(case.get("inelastic_model") is None for case in continuous)


def test_layer_and_groove_admission():
    base = dict(build_cases(material_sweep("silicon"), 4, 4)[0])
    layers = [(0.0, 10.0, base["composition"]), (10.0, 20.0, base["composition"])]
    stacked = {
        **base,
        "abs_layers": layers,
        "layer_radiators": [{"crystal": "silicon"}, {"crystal": "mos2"}],
    }
    validate_explicit_shell_cases([stacked], 50.0, "midpoint")
    for changed, reason in [
        ({**stacked, "layer_radiators": [{"crystal": "silicon"}, None]}, "absorber layer"),
        ({**stacked, "layer_radiators": [{"crystal": "silicon"}, {"crystal": "hfs2"}]}, "hfs2"),
        ({**base, "groove_spacing_ang": 1000.0}, "grooved transport"),
    ]:
        with pytest.raises(ValueError, match=reason):
            validate_explicit_shell_cases([changed], 50.0, "midpoint")


@pytest.mark.parametrize("model", ["continuous", "auto"])
def test_profile_explicit_model_does_not_inherit_shell_only_cutoff(model):
    resolved = resolve_numerics({"inelastic_model": model})
    assert resolved.effective["inelastic_model"] == model
    assert resolved.effective["inelastic_cutoff_eV"] is None


def test_continuous_preset_survives_partial_frozen_profile_override():
    resolved = resolve_numerics(
        {"energy_model": "frozen"},
        named={"inelastic_model": "continuous", "radiative_model": "uncoupled"},
        named_sources={"inelastic_model": "physics preset", "radiative_model": "physics preset"},
    )
    assert resolved.effective["inelastic_model"] == "continuous"
    assert resolved.effective["inelastic_cutoff_eV"] is None
    with pytest.raises(ValueError, match="midpoint"):
        resolve_numerics({"energy_model": "frozen"})


def test_pixel_demo_keeps_historical_frozen_continuous_identity():
    from pyrite import api
    from pyrite.campaign.single_scene import compose_pixel_scene, resolve_pixel_scene
    from pyrite.materials import load_material_catalog

    catalog = load_material_catalog(profile="pixelated_detector_demo")
    current = compose_pixel_scene(resolve_pixel_scene(catalog, "hopg", "pixelated_detector_demo"))
    assert current.numerics.energy_model == "frozen"
    assert current.physics.inelastic_model == "continuous"
    assert current.physics.inelastic_cutoff_eV is None
    case = api.build_case(
        current.scene,
        numerics=current.numerics,
        physics=current.physics,
        observation=current.observation,
    )
    legacy = api.build_case(
        current.scene,
        numerics=current.numerics,
        physics=replace(current.physics, inelastic_model="auto"),
        observation=current.observation,
    )
    assert case_content_key(case) == case_content_key(legacy)
