"""Public API/profile and early-refusal tests for the #624 preparation.

These require a normal PyRITE checkout and are not runnable with the isolated
schema-only test harness. They do not mark #623's physical contracts ready.
"""

from types import SimpleNamespace

import pytest

from pyrite._numerics import Numerics, Physics, validate_profile_numerics
from pyrite.campaign.channeling_cases import resolve_channeling
from pyrite.campaign.lowering import scene_physics
from pyrite.results.store import Settings


def _spec():
    return {
        "axis_uvw": [0, 0, 2],
        "zone_max_ang": 1000.0,
        "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
    }


def _physics(**updates):
    values = {
        "channeling_model": "axial",
        "channeling_axial": _spec(),
        "radiative_model": "uncoupled",
    }
    return Physics(**{**values, **updates})


def test_api_and_profile_normalize_using_the_same_resolver():
    physics = _physics()
    settings = Settings(
        channeling_model="axial", channeling_axial=_spec(), radiative_model="uncoupled"
    )
    assert physics.channeling_axial == settings.channeling_axial
    assert physics.channeling_axial["axis_uvw"] == [0, 0, 1]
    validate_profile_numerics(
        {"channeling_model": "axial", "channeling_axial": _spec(), "radiative_model": "uncoupled"}
    )


def test_scene_refuses_before_any_case_material_work():
    physics = _physics()
    with pytest.raises(ValueError, match="#623"):
        scene_physics(SimpleNamespace(physics=physics), Numerics())


def test_direct_channeling_resolver_cannot_fall_through_to_planar_solver():
    with pytest.raises(ValueError, match="#623"):
        resolve_channeling([], _physics(), Numerics())


def test_off_and_planar_selector_compatibility():
    assert Physics().channeling_axial is None
    assert (
        Physics(
            channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled"
        ).channeling_axial
        is None
    )
    with pytest.raises(ValueError, match="matching channeling_model"):
        Physics(channeling_axial=_spec())
    with pytest.raises(ValueError, match="requires channeling_model='axial'"):
        Physics(
            channeling_model="planar",
            channeling_plane=(1, -1, 0),
            channeling_axial=_spec(),
            radiative_model="uncoupled",
        )


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"channeling_axial": None}, "requires channeling_axial"),
        ({"channeling_plane": (1, 0, 0)}, "planar-only"),
        ({"radiative_model": "auto"}, "radiative_model='uncoupled'"),
        (
            {
                "secondary_threshold_eV": 100.0,
                "inelastic_model": "shell-soft-hard",
                "inelastic_cutoff_eV": 50.0,
            },
            "secondaries",
        ),
        ({"emission": "coherent"}, "incoherent"),
    ],
)
def test_incompatible_physics_refused(updates, reason):
    with pytest.raises(ValueError, match=reason):
        _physics(**updates)


def test_axial_does_not_allow_non_midpoint_energy_model():
    with pytest.raises(ValueError, match="midpoint"):
        _physics().validate_transport("frozen")
