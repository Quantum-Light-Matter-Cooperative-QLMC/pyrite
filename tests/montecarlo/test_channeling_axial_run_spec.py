"""Pre-integration axial schema tests; none grants physical run eligibility."""

import json

import pytest

from pyrite.campaign.axial_spec import (
    AXIAL_MODEL_MARKER,
    axial_spec_identity,
    refuse_unverified_axial_run,
    resolve_axial_spec,
)


def recipe(**overrides):
    base = {
        "axis_uvw": [0, 0, 2],
        "zone_max_ang": 1500.0,
        "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
    }
    base.update(overrides)
    return base


def test_defaults_are_canonical_json_safe_and_detached():
    raw = recipe()
    a = resolve_axial_spec(raw)
    assert a["axis_uvw"] == [0, 0, 1]
    assert a["shape"] == [16, 16]
    assert a["entry"]["impact_ang"] == [0.0, 0.0]
    assert a["boundary"]["kind"] == "pending-physical-validation"
    assert a["source"]["kind"] == "pending-physical-validation"
    json.dumps(a, allow_nan=False)
    assert resolve_axial_spec(a) == a  # profile Settings can normalize twice
    a["entry"]["impact_ang"][0] = 7.0
    assert raw["entry"] == {"version": 1, "kind": "gaussian", "waist_ang": 0.5}


def test_axis_reduction_preserves_sign_and_fingerprint():
    a = resolve_axial_spec(recipe(axis_uvw=[-2, 4, 0]))
    b = resolve_axial_spec(recipe(axis_uvw=[-1, 2, 0]))
    c = resolve_axial_spec(recipe(axis_uvw=[1, -2, 0]))
    assert a["axis_uvw"] == [-1, 2, 0]
    assert axial_spec_identity(a) == axial_spec_identity(b)
    assert axial_spec_identity(a) != axial_spec_identity(c)
    assert AXIAL_MODEL_MARKER.startswith("axial-")


def test_identity_ignores_execution_limits_but_tracks_science():
    default = resolve_axial_spec(recipe())
    larger = resolve_axial_spec(recipe(limits={"max_working_bytes": 4 * 1024**3}))
    assert axial_spec_identity(default) == axial_spec_identity(larger)
    for change in (
        {"seed": 2},
        {"ensemble_electrons": 32},
        {"shape": [32, 16]},
        {"dz_ang": 2.5},
        {"zone_max_ang": 1550},
        {"temperature_K": 300},
        {"entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.6}},
    ):
        assert axial_spec_identity(resolve_axial_spec(recipe(**change))) != axial_spec_identity(
            default
        )


def test_omitted_defaults_and_reversed_mechanisms_do_not_change_identity():
    base = resolve_axial_spec(recipe())
    explicit = resolve_axial_spec(
        recipe(
            mechanisms=["electronic", "tds"],
            cells=(1, 1),
            shape=(16, 16),
            seed=0,
            temperature_K=293.15,
        )
    )
    assert explicit == base
    assert axial_spec_identity(explicit) == axial_spec_identity(base)


@pytest.mark.parametrize(
    "kind,details",
    [
        ("hermite_gauss", {"m": 1, "n": 0}),
        ("laguerre_gauss", {"ell": -2, "p": 1}),
        ("bessel_gauss", {"ell": 1, "radial_k_inv_ang": 0.0}),
    ],
)
def test_pure_entry_families(kind, details):
    entry = {"version": 1, "kind": kind, "waist_ang": 0.5, **details}
    assert resolve_axial_spec(recipe(entry=entry))["entry"]["kind"] == kind


def test_ordered_incoherent_mixture_and_weight_normalization():
    g = {"version": 1, "kind": "gaussian", "waist_ang": 0.5}
    e = {"version": 1, "kind": "hermite_gauss", "waist_ang": 0.5, "m": 1, "n": 0}
    mixture = {
        "version": 1,
        "kind": "mixture",
        "components": [
            {"weight": 3, "entry": g},
            {"weight": 1, "entry": e},
        ],
    }
    a = resolve_axial_spec(recipe(entry=mixture))
    assert [x["weight"] for x in a["entry"]["components"]] == [0.75, 0.25]
    assert axial_spec_identity(a) != axial_spec_identity(
        resolve_axial_spec(
            recipe(entry={**mixture, "components": list(reversed(mixture["components"]))})
        )
    )


@pytest.mark.parametrize(
    "update,match",
    [
        ({"axis_uvw": [0, 0, 0]}, "axis_uvw"),
        ({"axis_uvw": [True, 0, 1]}, "axis_uvw"),
        ({"shape": [15, 16]}, "shape"),
        ({"shape": [16, 16], "cells": [3, 1]}, "shape"),
        ({"cells": [0, 1]}, "cells"),
        ({"zone_max_ang": -1}, "zone_max_ang"),
        ({"dz_ang": 0}, "dz_ang"),
        ({"temperature_K": float("nan")}, "temperature_K"),
        ({"seed": True}, "seed"),
        ({"ensemble_electrons": 0}, "ensemble_electrons"),
        ({"mechanisms": ["tds"]}, "mechanisms"),
        ({"mechanisms": [["tds"], "electronic"]}, "mechanisms"),
        ({"boundary": {"version": 1, "kind": "grid-sink"}}, "boundary"),
        ({"source": {"version": 2, "kind": "pending-physical-validation"}}, "source"),
        ({"junk": 1}, "unknown"),
        ({"limits": {"max_grid_points": 8}}, "ceilings"),
        ({"limits": {"max_steps": 5}}, "ceiling"),
        ({"limits": {"max_ensemble_electrons": 2}}, "ceilings"),
        ({"limits": {"max_working_bytes": 100}}, "memory"),
    ],
)
def test_invalid_configuration(update, match):
    with pytest.raises(ValueError, match=match):
        resolve_axial_spec(recipe(**update))


@pytest.mark.parametrize(
    "entry",
    [
        {"version": 1, "kind": "gaussian", "waist_ang": 0},
        {"version": 1, "kind": "gaussian", "waist_ang": 1.0, "dimension": 1},
        {"version": 1, "kind": "gaussian", "waist_ang": 1.0, "impact_average": "uniform"},
        {"version": 1, "kind": "gaussian", "waist_ang": 1.0, "tilt_offset_rad": [0, float("inf")]},
        {"version": 1, "kind": "laguerre_gauss", "waist_ang": 1.0, "ell": 1, "p": True},
        {"version": 1, "kind": "mixture", "components": []},
        {
            "version": 1,
            "kind": "mixture",
            "components": [
                {"weight": 0, "entry": {"version": 1, "kind": "gaussian", "waist_ang": 1}}
            ],
        },
    ],
)
def test_invalid_entry_recipes(entry):
    with pytest.raises(ValueError):
        resolve_axial_spec(recipe(entry=entry))


def test_refuse_all_unverified_production_cases():
    with pytest.raises(ValueError, match=r"#623"):
        refuse_unverified_axial_run()
