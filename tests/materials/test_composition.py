"""Substance lookup preserves namespace and persisted-composition contracts."""

from dataclasses import replace
from types import MappingProxyType

import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign import geometry
from pyrite.instrument import model, observation
from pyrite.materials import CATALOG, MediumSpec, attenuation
from pyrite.materials._composition import resolve_composition


@pytest.fixture
def selected_catalog():
    # Deliberate namespace collision, ordered repeated elements, and a medium
    # absent from the bundled catalog. These must not be sorted or coalesced.
    medium = MediumSpec("silicon", (("O", 0.04), ("Si", 0.02), ("O", 0.01)))
    return replace(
        CATALOG,
        media=MappingProxyType({"silicon": medium, "external-foil": medium}),
        crystals=MappingProxyType({"silicon": CATALOG.crystals["silicon"]}),
        materials=MappingProxyType({"recipe-only": CATALOG.materials["hopg"]}),
    )


def test_explicit_catalog_controls_namespace_precedence_and_composition_order(selected_catalog):
    expected = (("O", 0.04), ("Si", 0.02), ("O", 0.01))
    assert resolve_composition("silicon", catalog=selected_catalog) == expected
    assert resolve_composition("external-foil", catalog=selected_catalog) == expected
    assert (
        resolve_composition("silicon", catalog=selected_catalog, namespaces=("crystal", "medium"))
        == CATALOG.crystals["silicon"].composition
    )


@pytest.mark.parametrize("key", ["recipe-only", "hopg", "SILICON"])
def test_resolver_never_falls_back_to_recipe_or_bundled_namespaces(selected_catalog, key):
    with pytest.raises(KeyError, match=key):
        resolve_composition(key, catalog=selected_catalog)


def test_inline_composition_keeps_order_without_catalog_lookup(selected_catalog):
    medium = MediumSpec("inline", (("Si", 0.02), ("O", 0.04), ("Si", 0.01)))
    assert (
        resolve_composition(medium, catalog=selected_catalog, namespaces=()) is medium.composition
    )


def test_legacy_callers_pass_their_selected_catalog_and_keep_case_rules(
    selected_catalog, monkeypatch
):
    for owner in (api, geometry, model, observation, attenuation):
        monkeypatch.setattr(owner, "CATALOG", selected_catalog)
    expected = selected_catalog.media["external-foil"].composition
    assert geometry.substrate_composition("EXTERNAL-FOIL") == list(expected)
    assert api._resolved_composition("external-foil") == expected
    assert observation._resolved_composition("external-foil") == expected
    assert attenuation._resolve_composition("external-foil") == expected
    assert pr.FilterPlate("external-foil", 0.1, (1, 1), pr.PlanarPose((0, 0, 10))).material == (
        "external-foil"
    )
    with pytest.raises(ValueError, match="unknown substrate"):
        geometry.substrate_composition("HOPG")
    with pytest.raises(ValueError, match="unknown FilterPlate material"):
        pr.FilterPlate("EXTERNAL-FOIL", 0.1, (1, 1), pr.PlanarPose((0, 0, 10)))
    with pytest.raises(ValueError, match="unknown filter material"):
        attenuation._resolve_composition("EXTERNAL-FOIL")


def test_crystal_substrate_keys_remain_case_sensitive(selected_catalog, monkeypatch):
    monkeypatch.setattr(geometry, "CATALOG", replace(selected_catalog, media=MappingProxyType({})))
    assert geometry.substrate_composition("silicon") == list(
        CATALOG.crystals["silicon"].composition
    )
    with pytest.raises(ValueError, match="unknown substrate"):
        geometry.substrate_composition("SILICON")


def test_recipe_only_keys_keep_contextual_rejections(selected_catalog, monkeypatch):
    for owner in (geometry, model, attenuation):
        monkeypatch.setattr(owner, "CATALOG", selected_catalog)
    with pytest.raises(ValueError, match="unknown substrate"):
        geometry.substrate_composition("recipe-only")
    with pytest.raises(ValueError, match="unknown FilterPlate material"):
        pr.FilterPlate("recipe-only", 0.1, (1, 1), pr.PlanarPose((0, 0, 10)))
    with pytest.raises(ValueError, match="expected a catalog crystal or medium key"):
        attenuation._resolve_composition("recipe-only")


@pytest.mark.parametrize("composition", [(), (("Si", 0.0),), (("Si", float("nan")),), (("", 0.1),)])
def test_inline_validation_remains_at_filter_and_attenuation_boundaries(composition):
    medium = MediumSpec("invalid", composition)
    with pytest.raises(ValueError, match="FilterPlate MediumSpec composition"):
        pr.FilterPlate(medium, 0.1, (1, 1), pr.PlanarPose((0, 0, 10)))
    with pytest.raises(ValueError, match="positive element number densities"):
        attenuation._resolve_composition(medium)
