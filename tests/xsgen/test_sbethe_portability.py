"""Published SBETHE input pins survive host roundoff and reject changed physics."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.xsgen import TableNotFoundError
from pyrite.xsgen.sbethe import catalog
from pyrite.xsgen.sbethe.deck import SbetheDeck
from pyrite.xsgen.sbethe.generate import material_request
from pyrite.xsgen.sbethe.release import SbetheReleaseEntry, SbetheReleaseIndex, load_release_index
from pyrite.xsgen.store import manifest_digest, store


def _request(material, projectile="electron"):
    return material_request(
        material.key,
        material.composition,
        density_g_cm3=material.density_g_cm3,
        mean_excitation_eV=material.mean_excitation_eV,
        band_gap_eV=material.band_gap_eV,
        projectile=projectile,
    )


def _deck(material, **kwargs):
    return SbetheDeck(
        material.key,
        material.composition,
        material.density_g_cm3,
        material.mean_excitation_eV,
        material.band_gap_eV,
        **kwargs,
    )


@pytest.fixture
def published(isolated_dirs, packaged_dir, monkeypatch, request):
    projectile = getattr(request, "param", "electron")
    material = catalog.catalog_material("silicon")
    table_request = _request(material, projectile)
    table = store(
        table_request,
        {
            "stopping_energy_eV": np.array([1e3, 1e9]),
            "stopping_eV_per_angstrom": np.array([0.01, 0.001]),
        },
    )
    entry = SbetheReleaseEntry("silicon", table.key, table.digest)
    index = SbetheReleaseIndex(
        "fixture",
        "0" * 64,
        0,
        (),
        (entry,),
        input_decks={table.key: _deck(material, projectile=projectile).input_sha256},
    )
    monkeypatch.setattr("pyrite.xsgen.sbethe.release.load_release_index", lambda **_: index)
    real = catalog.material_inputs_from_composition

    def drift(*args, **kwargs):
        value = real(*args, **kwargs)
        return replace(value, density_g_cm3=math.nextafter(value.density_g_cm3, math.inf))

    monkeypatch.setattr(catalog, "material_inputs_from_composition", drift)
    changed = drift("silicon", CATALOG.crystal("silicon").composition)
    assert _request(changed, projectile).key != table.key
    assert _deck(changed).input_sha256 == _deck(material).input_sha256
    return SimpleNamespace(
        material=material,
        changed=changed,
        table=table,
        index=index,
        monkeypatch=monkeypatch,
        projectile=projectile,
    )


@pytest.mark.parametrize("published", ["electron", "positron"], indirect=True)
def test_roundoff_uses_the_published_table_without_rekeying_it(published):
    actual = catalog.resolve_catalog_table("silicon", projectile=published.projectile)
    assert actual.key == published.table.key
    assert actual.digest == published.table.digest
    np.testing.assert_array_equal(actual.arrays()["stopping_eV_per_angstrom"], [0.01, 0.001])


@pytest.mark.parametrize("projectile", ["electron", "positron"])
def test_shipped_input_pins_cover_each_published_table(projectile):
    index = load_release_index(projectile=projectile)
    assert index is not None
    assert set(index.input_decks) == {entry.key for entry in index.tables}
    for entry in index.tables:
        for label in entry.label.split(","):
            material = catalog.catalog_material(label)
            assert (
                index.input_decks[entry.key] == _deck(material, projectile=projectile).input_sha256
            )


def test_display_name_does_not_change_physical_input_equivalence(published):
    actual = catalog.resolve_composition_table(
        "anonymous-layer", CATALOG.crystal("silicon").composition
    )
    assert actual.key == published.table.key


def test_direct_request_still_precedes_release_compatibility(published, monkeypatch):
    exact = store(
        _request(published.changed),
        {
            "stopping_energy_eV": np.array([1e3, 1e9]),
            "stopping_eV_per_angstrom": np.array([0.02, 0.002]),
        },
    )

    def forbidden(**_):
        pytest.fail("an exact request must not consult release compatibility")

    monkeypatch.setattr("pyrite.xsgen.sbethe.release.load_release_index", forbidden)
    assert catalog.resolve_catalog_table("silicon").key == exact.key


@pytest.mark.parametrize(
    "field, value",
    [
        ("density_g_cm3", 3.0),
        ("mean_excitation_eV", 200.0),
        ("composition", {14: 1.0, 8: 1.0}),
        ("band_gap_eV", 1.1),
    ],
)
def test_changed_physical_input_is_not_accepted(published, monkeypatch, field, value):
    changed = replace(published.changed, **{field: value})
    monkeypatch.setattr(catalog, "material_inputs_from_composition", lambda *_a, **_k: changed)
    with pytest.raises(TableNotFoundError):
        catalog.resolve_catalog_table("silicon")


@pytest.mark.parametrize(
    "field, value",
    [
        ("code", "elsepa"),
        ("code_version", "different-source"),
        ("quantity", "different-output"),
        ("model", {"projectile": "positron"}),
        ("target", {"kind": "element", "Z": 14}),
        ("key", "0" * 64),
        ("schema", "unsupported-schema"),
    ],
)
def test_a_pinned_but_incompatible_manifest_is_not_accepted(published, monkeypatch, field, value):
    manifest = dict(published.table.manifest)
    manifest[field] = value
    manifest["manifest_sha256"] = manifest_digest(manifest)
    published.table.path.with_suffix(".json").write_text(json.dumps(manifest))
    entry = replace(published.index.tables[0], manifest_sha256=manifest["manifest_sha256"])
    index = replace(published.index, tables=(entry,))
    monkeypatch.setattr("pyrite.xsgen.sbethe.release.load_release_index", lambda **_: index)
    with pytest.raises(TableNotFoundError):
        catalog.resolve_catalog_table("silicon")


def test_modified_manifest_cannot_keep_its_claimed_release_digest(published):
    manifest = dict(published.table.manifest)
    manifest["created_utc"] = "changed-without-updating-digest"
    published.table.path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(TableNotFoundError):
        catalog.resolve_catalog_table("silicon")


def test_projectile_uses_its_own_input_pin(published):
    with pytest.raises(TableNotFoundError):
        catalog.resolve_catalog_table("silicon", projectile="positron")


def test_legacy_index_does_not_guess_equivalent_keys(published, monkeypatch):
    record = published.index.record()
    record.pop("input_decks")
    legacy = SbetheReleaseIndex.from_record(record)
    assert not legacy.input_decks
    assert "input_decks" not in legacy.record()
    monkeypatch.setattr("pyrite.xsgen.sbethe.release.load_release_index", lambda **_: legacy)
    with pytest.raises(TableNotFoundError):
        catalog.resolve_catalog_table("silicon")
