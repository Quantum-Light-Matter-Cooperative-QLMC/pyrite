"""Public policy relocation preserves source and observation content identity."""

import pickle
import warnings
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._api_deprecations import API_DEPRECATIONS
from pyrite.campaign.profiles import case_content_key


def scene(**kwargs):
    return pr.Scene(
        pr.Beam(30.0, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=1000.0, tilt_deg=30.0, footprint=None),
        **kwargs,
    )


def digest(case):
    # No transport or table resolution: both payloads use the same table markers.
    return case_content_key(case, xsgen_tables={})


def test_none_brem_source_preserves_disabled_policy_and_case_identity():
    disabled = pr.Physics(brem_source=None)
    legacy = pr.Physics(brem_source="none")
    assert disabled == legacy
    assert disabled.brem_source == "none"
    before = api.build_case(scene(physics=legacy))
    after = api.build_case(scene(physics=disabled))
    assert before == after
    assert digest(before) == digest(after)
    assert pickle.loads(pickle.dumps(disabled)) == legacy


@pytest.mark.parametrize("source", ["mc", "external", "none", None])
def test_brem_source_accepts_existing_and_none_policies(source):
    assert pr.Physics(brem_source=source).brem_source == ("none" if source is None else source)


@pytest.mark.parametrize("source", ["invalid", "None", 0, False])
def test_brem_source_rejects_invalid_policies(source):
    with pytest.raises(ValueError, match="brem_source"):
        pr.Physics(brem_source=source)


def test_models_are_frozen_and_integration_constraints_remain():
    with pytest.raises(FrozenInstanceError):
        pr.Physics().straggling = True
    with pytest.raises(ValueError, match="midpoint"):
        api.build_case(
            scene(physics=pr.Physics(inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)),
            pr.Numerics(energy_model="frozen"),
        )
    with pytest.raises(ValueError, match="pair_production_model"):
        pr.Physics(positron_transport=True)
    with pytest.raises(ValueError, match="secondary_threshold_eV"):
        pr.Physics(pair_production_model="penelope-2024")


@pytest.mark.parametrize("name", [f.name for f in fields(pr.Physics)])
def test_model_keywords_are_removed_from_numerics(name):
    with pytest.raises(TypeError, match=name):
        pr.Numerics(**{name: getattr(pr.Physics(), name)})
    changed = replace(scene(), physics=pr.Physics(straggling=True))
    assert api.build_case(changed)["straggling"] is True


def test_removed_policy_aliases_and_paths_are_rejected():
    assert not hasattr(pr, "Convergence")
    with pytest.raises(TypeError, match="convergence"):
        pr.Numerics(convergence=pr.CxrReflections())
    for name in ("emission", "brem_source"):
        with pytest.raises(TypeError, match=name):
            scene(**{name: "none"})
        with pytest.raises(TypeError, match=name):
            pr.simulate(scene().beam, scene().target, pr.Detector(), **{name: "none"})
    for path in (
        "numerics.straggling",
        "numerics.convergence.n_families",
        "detector.energy_bins.cxr",
        "emission",
        "brem_source",
    ):
        with pytest.raises(ValueError, match="has no field"):
            pr.Sweep(scene(), {path: [1]})
    sweep = pr.Sweep(
        scene(), {"physics.straggling": [False, True], "numerics.cxr_trials": [10, 20]}
    )
    cases = sweep.cases()
    assert [c["Ne"] for c in cases] == [10, 20, 10, 20]
    assert [c.get("straggling", False) for c in cases] == [False, False, True, True]
    for factory in (
        pr.Detector,
        lambda **kw: pr.PlanarDetector(pr.PlanarPose((0, 0, 100)), size_mm=(1, 1), **kw),
    ):
        with pytest.raises(TypeError, match="energy_bins"):
            factory(energy_bins=pr.EnergyBins())
    bins = pr.EnergyBins(cxr=np.array([1000.0, 1050.0]), brem=np.array([1000.0, 2000.0]))
    case = api.build_case(scene(numerics=pr.Numerics(energy_bins=bins)))
    from pyrite._energy_grid_encoding import decode_energy_grid

    np.testing.assert_array_equal(decode_energy_grid(case.E_grid_line), bins.cxr)
    assert pickle.loads(pickle.dumps(bins)) == bins
    assert pr.EnergyBins.__module__ == "pyrite.detectors.spec"


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("incoherent", ("incoherent",)),
        ("coherent", ("coherent",)),
        (("coherent",), ("coherent",)),
        (["incoherent"], ("incoherent",)),
        ({"coherent"}, ("coherent",)),
        (frozenset({"incoherent"}), ("incoherent",)),
        (("coherent", "incoherent", "coherent"), ("incoherent", "coherent")),
        (["coherent", "incoherent"], ("incoherent", "coherent")),
        ({"incoherent", "coherent"}, ("incoherent", "coherent")),
        (frozenset({"incoherent", "coherent"}), ("incoherent", "coherent")),
    ],
)
def test_emission_selection_normalizes_and_preserves_identity(selection, expected):
    physics = pr.Physics(emission=selection)
    canonical = pr.Physics(emission=expected)
    assert physics.emission == expected
    assert physics == canonical
    assert hash(physics) == hash(canonical)
    assert pickle.loads(pickle.dumps(physics)) == canonical
    first = api.build_case(scene(physics=physics))
    second = api.build_case(scene(physics=canonical))
    assert first == second
    assert digest(first) == digest(second)
    assert first.get("coherent_emission", False) == ("coherent" in expected)


@pytest.mark.parametrize("selection", [(), [], set(), frozenset(), "unknown", ("both",), [1], [[]]])
def test_emission_rejects_empty_and_invalid_modes(selection):
    with pytest.raises(ValueError, match="emission.*one or both"):
        pr.Physics(emission=selection)


@pytest.mark.parametrize("selection", [None, 1, {"coherent": True}, iter(["coherent"])])
def test_emission_rejects_unsupported_collections(selection):
    with pytest.raises(TypeError, match="emission.*mode string"):
        pr.Physics(emission=selection)


def test_emission_does_not_retain_mutable_input():
    selection = ["coherent", "incoherent"]
    physics = pr.Physics(emission=selection)
    before = hash(physics)
    selection.clear()
    assert physics.emission == ("incoherent", "coherent")
    assert hash(physics) == before
    assert replace(physics, straggling=True).emission == physics.emission


def test_both_alias_warns_once_and_matches_collection():
    with pytest.warns(DeprecationWarning, match='Physics\\(emission="both"\\)') as caught:
        old = pr.Physics(emission="both")
    assert len(caught) == 1
    assert "0.15.0" in str(caught[0].message)
    new = pr.Physics(emission=("incoherent", "coherent"))
    assert old == new
    assert hash(old) == hash(new)
    assert api.build_case(scene(physics=old)) == api.build_case(scene(physics=new))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        replace(old, straggling=True)
    assert not caught
    entry = API_DEPRECATIONS['Physics(emission="both")']
    assert entry.deprecated_in == "0.13.1"
    assert entry.remove_in == "0.15.0"


def test_emission_collection_sweep_paths_normalize_each_selection():
    selections = ["incoherent", ("coherent",), ["coherent", "incoherent", "coherent"]]
    sweep = pr.Sweep(scene(), {"physics.emission": selections})
    assert [s.physics.emission for _, s in sweep.expand()] == [
        ("incoherent",),
        ("coherent",),
        ("incoherent", "coherent"),
    ]
    assert [case.get("coherent_emission", False) for case in sweep.cases()] == [False, True, True]
