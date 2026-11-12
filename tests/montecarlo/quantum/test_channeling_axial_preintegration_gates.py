"""Public API/profile and early-refusal tests for the #624 preparation.

These require a normal PyRITE checkout and are not runnable with the isolated
schema-only test harness. They do not mark #623's physical contracts ready.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyrite._numerics import Numerics, Physics, validate_profile_numerics
from pyrite.campaign.channeling_cases import resolve_channeling
from pyrite.campaign.lowering import scene_physics
from pyrite.quantum import Channeling
from pyrite.quantum._channeling import validate_channeling_run
from pyrite.results.store import Settings

UNCOUPLED = Physics(radiative_model="uncoupled")


def _spec():
    return {
        "axis_uvw": [0, 0, 2],
        "zone_max_ang": 1000.0,
        "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
    }


def _axial(**updates):
    return Channeling(**{"model": "axial", "axial": _spec(), **updates})


def test_api_and_profile_normalize_using_the_same_resolver():
    channeling = _axial()
    settings = Settings(
        channeling_model="axial",
        channeling_axial=_spec(),
        radiative_model="uncoupled",
        radiative_cutoff_eV=None,
    )
    assert channeling.axial == settings.channeling_axial
    assert channeling.axial["axis_uvw"] == [0, 0, 1]
    validate_profile_numerics(
        {"channeling_model": "axial", "channeling_axial": _spec(), "radiative_model": "uncoupled"}
    )


def test_scene_refuses_before_any_case_material_work():
    scene = SimpleNamespace(physics=UNCOUPLED, channeling=_axial())
    with pytest.raises(ValueError, match="#623"):
        scene_physics(scene, Numerics())


def test_direct_channeling_resolver_cannot_fall_through_to_planar_solver():
    with pytest.raises(ValueError, match="#623"):
        resolve_channeling([], _axial())


def test_off_and_planar_selector_compatibility():
    assert Channeling("planar", plane=(1, -1, 0)).axial is None
    with pytest.raises(ValueError, match="requires model='axial'"):
        Channeling("planar", plane=(1, -1, 0), axial=_spec())


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"axial": None}, "requires axial"),
        ({"plane": (1, 0, 0)}, "planar-only"),
    ],
)
def test_incompatible_selections_refused(updates, reason):
    with pytest.raises(ValueError, match=reason):
        _axial(**updates)


@pytest.mark.parametrize(
    "physics,reason",
    [
        (Physics(radiative_model="auto"), "radiative_model='uncoupled'"),
        (
            replace(
                UNCOUPLED,
                secondary_threshold_eV=100.0,
                inelastic_model="shell-soft-hard",
                inelastic_cutoff_eV=50.0,
            ),
            "secondary transport",
        ),
        (replace(UNCOUPLED, emission="coherent"), "incoherent"),
    ],
)
def test_incompatible_physics_refused(physics, reason):
    with pytest.raises(ValueError, match=reason):
        validate_channeling_run(_axial(), physics, "midpoint")


def test_axial_does_not_allow_non_midpoint_energy_model():
    with pytest.raises(ValueError, match="midpoint"):
        validate_channeling_run(_axial(), UNCOUPLED, "frozen")
