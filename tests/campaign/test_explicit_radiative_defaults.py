"""Explicit BremsLib/coupled radiative defaults keep resolved legacy identities (#642, #262)."""

import pickle
from dataclasses import replace

import pytest

from pyrite._physics_fallback import PhysicsFallbackWarning
from pyrite.campaign.bremsstrahlung_cases import resolve_run_models
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.geometry import BlazedGrooves, Footprint, Layer, Slab, Stack
from pyrite.campaign.profiles import (
    canonical_settings,
    case_content_key,
    dataset_identity,
    resolve_numerics,
)
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.config import Physics
from pyrite.results import Settings
from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.bremslib import tables

LEGACY = {"bremsstrahlung_model": "auto", "radiative_model": "auto", "radiative_cutoff_eV": None}


def _selectors(owner):
    return (owner.bremsstrahlung_model, owner.radiative_model, owner.radiative_cutoff_eV)


@pytest.mark.parametrize("owner", [Physics, Settings])
def test_new_defaults_and_historical_pickle_omissions(owner):
    current = owner()
    assert _selectors(current) == ("bremslib", "bremslib-soft-hard", 1000.0)
    assert pickle.loads(pickle.dumps(current)) == current
    old = owner(**LEGACY)
    for key in LEGACY:
        del old.__dict__[key]
    assert _selectors(pickle.loads(pickle.dumps(old))) == ("auto", "auto", None)
    assert owner(radiative_model="uncoupled").radiative_cutoff_eV is None
    assert owner(radiative_model="auto").radiative_cutoff_eV is None


def _stack_sweep():
    return Sweep(
        material="hopg",
        beam=BeamSpec(energy_keV=[30.0]),
        target=Stack((Layer("hopg", 2000.0), Layer("silicon", 3000.0)), 45.0, footprint=None),
    )


@pytest.mark.parametrize("sweep_factory", [lambda: material_sweep("silicon"), _stack_sweep])
def test_covered_scene_cases_and_identities_match_legacy_auto(sweep_factory):
    sweep = sweep_factory()
    cases = build_cases(sweep, 4, 4)
    legacy = build_cases(sweep, 4, 4, **LEGACY)
    assert cases == legacy
    assert [case_content_key(c) for c in cases] == [case_content_key(c) for c in legacy]
    assert cases[0]["radiative_model"] == "bremslib-soft-hard"
    assert cases[0]["radiative_cutoff_eV"] == 1000.0
    current = default_settings()
    old = replace(current, **LEGACY)
    assert (
        dataset_identity("hopg", current, sweep)["parameter_sha256"]
        == dataset_identity("hopg", old, sweep)["parameter_sha256"]
    )
    elements = ["C", "Si"]
    resolved, events = resolve_run_models(current, elements, grooved=False)
    resolved_old, old_events = resolve_run_models(old, elements, grooved=False)
    assert events == old_events == []
    assert _selectors(resolved)[:2] == _selectors(resolved_old)[:2]
    assert (
        dataset_identity("hopg", resolved, sweep)["parameter_sha256"]
        == dataset_identity("hopg", resolved_old, sweep)["parameter_sha256"]
    )
    assert canonical_settings(resolved) and canonical_settings(resolved_old)


def _grooved_sweep():
    return Sweep(
        material="hopg",
        beam=BeamSpec(energy_keV=100.0),
        target=Slab(
            "hopg",
            2.0e5,
            tilt_deg=45.0,
            tilt_azim_deg=180.0,
            entrance_face=BlazedGrooves(2.0e4),
            footprint=Footprint(5.0, 5.0),
        ),
    )


def test_grooved_default_requires_explicit_uncoupled():
    sweep = _grooved_sweep()
    with pytest.raises(ValueError, match="radiative_model='uncoupled'.*--radiative-model"):
        build_cases(sweep, 4, 4)
    with pytest.raises(ValueError, match="--allow-physics-fallback"):
        resolve_run_models(default_settings(), ["C"], grooved=True)
    continuous = {"inelastic_model": "continuous"}  # shell transport also refuses grooves
    with pytest.warns(UserWarning, match="unavailable for grooves"):
        legacy = build_cases(sweep, 4, 4, radiative_model="auto", **continuous)
    explicit = build_cases(sweep, 4, 4, radiative_model="uncoupled", **continuous)
    assert explicit == legacy
    assert all("radiative_model" not in case for case in explicit)
    assert all(case["bremsstrahlung_model"] == "bremslib" for case in explicit)


def test_eedl_scene_requires_explicit_uncoupled():
    with pytest.raises(ValueError, match="select radiative_model='uncoupled' explicitly"):
        Physics(bremsstrahlung_model="eedl")
    sweep = material_sweep("silicon")
    with pytest.raises(ValueError, match="radiative_model='uncoupled'"):
        build_cases(sweep, 4, 4, bremsstrahlung_model="eedl")
    eedl = build_cases(sweep, 4, 4, bremsstrahlung_model="eedl", radiative_model="uncoupled")
    legacy = build_cases(sweep, 4, 4, bremsstrahlung_model="eedl", radiative_model="auto")
    assert eedl == legacy
    assert all("radiative_model" not in c and "bremsstrahlung_model" not in c for c in eedl)


def test_missing_tables_raise_in_run_preflight_with_guidance(monkeypatch):
    from argparse import Namespace

    from pyrite.runs import scan

    def missing(*_args, **_kwargs):
        raise TableNotFoundError("test catalog has no BremsLib tables")

    monkeypatch.setattr(tables, "resolve_element_table", missing)
    with pytest.raises(TableNotFoundError, match="radiative_model='uncoupled'"):
        scan._resolved_run(Namespace(catalog_profile="standard"), "hopg")


@pytest.mark.parametrize("grooved", [False, True])
def test_allow_fallback_degrades_default_and_records_it(monkeypatch, grooved):
    def missing(*_args, **_kwargs):
        raise TableNotFoundError("test catalog has no BremsLib tables")

    if not grooved:
        monkeypatch.setattr(tables, "resolve_element_table", missing)
    settings = replace(default_settings(), allow_fallback=True)
    legacy = replace(settings, **LEGACY)
    with pytest.warns(PhysicsFallbackWarning):
        resolved, events = resolve_run_models(settings, ["Si"], grooved=grooved)
    with pytest.warns(PhysicsFallbackWarning):
        resolved_old, old_events = resolve_run_models(legacy, ["Si"], grooved=grooved)
    assert _selectors(resolved) == _selectors(resolved_old)
    assert resolved.radiative_model == "uncoupled" and resolved.radiative_cutoff_eV is None
    assert resolved.bremsstrahlung_model == ("bremslib" if grooved else "eedl")
    assert [e["model_used"] for e in events] == [e["model_used"] for e in old_events]
    assert events[0]["model_requested"] == "bremslib-soft-hard"
    assert old_events[0]["model_requested"] == "auto (bremslib-soft-hard)"


def test_explicit_uncoupled_records_no_coupled_identity():
    sweep = material_sweep("silicon")
    settings = replace(default_settings(), radiative_model="uncoupled", radiative_cutoff_eV=None)
    numerics = dataset_identity("silicon", settings, sweep)["resolved_parameters"][
        "transport_numerics"
    ]
    assert "radiative_model" not in numerics
    assert all(
        "radiative_model" not in case
        for case in build_cases(sweep, 4, 4, radiative_model="uncoupled")
    )


def test_profile_defaults_fill_cutoff_only_for_coupled_model():
    effective = resolve_numerics({}).effective
    assert (
        effective["bremsstrahlung_model"],
        effective["radiative_model"],
        effective["radiative_cutoff_eV"],
    ) == ("bremslib", "bremslib-soft-hard", 1000.0)
    for model in ("uncoupled", "auto"):
        resolved = resolve_numerics({"radiative_model": model}).effective
        assert resolved["radiative_cutoff_eV"] is None


def test_typed_trajectory_lowering_inherits_explicit_defaults():
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.campaign.lowering import build_case
    from pyrite.config import FixedSampling, Numerics
    from pyrite.geometry import Slab as TypedSlab

    scene = Scene(
        Beam(30.0, transverse=PointSpot(), longitudinal=PointBunch()),
        TypedSlab("silicon", 100.0, tilt_deg=30.0, width=None, height=None),
        detector=None,
    )
    numerics = Numerics(sampling=FixedSampling(3, 2), seed=19)
    case = build_case(scene, physics=Physics(), numerics=numerics)
    legacy = build_case(scene, physics=Physics(**LEGACY), numerics=numerics)
    assert case_content_key(case) == case_content_key(legacy)
    assert case["radiative_model"] == "bremslib-soft-hard"
