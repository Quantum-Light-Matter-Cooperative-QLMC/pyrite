"""Species-specific requests preserve the released electron identities (#276)."""

import shutil

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.xsgen.elsepa import ElsepaDeck
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
from pyrite.xsgen.elsepa.generate import element_request, muffin_tin_request
from pyrite.xsgen.sbethe.catalog import catalog_material, resolve_catalog_table
from pyrite.xsgen.sbethe.generate import material_request
from pyrite.xsgen.store import store


@pytest.mark.parametrize("code", ["elsepa", "sbethe"])
def test_species_release_round_trip_keeps_electron_archive_intact(
    code, isolated_dirs, packaged_dir, tmp_path, monkeypatch
):
    from pyrite.xsgen.elsepa import release as elastic
    from pyrite.xsgen.elsepa.catalog import (
        PRODUCTION_ENERGIES_EV,
        _muffin_tin_energies,
        elemental_solid,
    )
    from pyrite.xsgen.fetch import fetch_elsepa, fetch_sbethe_tables
    from pyrite.xsgen.sbethe import release as stopping
    from pyrite.xsgen.store import resolve, user_table_dir

    release = elastic if code == "elsepa" else stopping
    fetch = fetch_elsepa if code == "elsepa" else fetch_sbethe_tables
    if code == "elsepa":
        monkeypatch.setattr(elastic, "TRANSPORT_ELEMENTS", {"Si": {"Z": 14}})
        monkeypatch.setattr(elastic, "elementary_crystals", lambda: ("silicon",))
    else:
        monkeypatch.setattr(stopping, "catalog_keys", lambda: ("silicon",))
    entries = []
    for projectile, value in (("electron", 1.0), ("positron", 2.0)):
        if code == "elsepa":
            solid = elemental_solid("silicon")
            assert solid is not None
            requests = [
                element_request(14, PRODUCTION_ENERGIES_EV, projectile=projectile)[0],
                muffin_tin_request(
                    solid.key,
                    solid.z,
                    _muffin_tin_energies(),
                    radius_cm=solid.radius_cm,
                    density_g_cm3=solid.density_g_cm3,
                    projectile=projectile,
                )[0],
            ]
        else:
            material = catalog_material("silicon")
            requests = [
                material_request(
                    material.key,
                    material.composition,
                    density_g_cm3=material.density_g_cm3,
                    mean_excitation_eV=material.mean_excitation_eV,
                    projectile=projectile,
                )
            ]
        for request in requests:
            store(request, {"species_reference": np.array([value])})
        archive, index = release.build_release(tmp_path / "out", projectile=projectile)
        entries.append((archive, archive.read_bytes(), index))
    assert entries[0][0].name == f"{code}-tables.zip"
    assert entries[1][0].name == f"{code}-positron-tables.zip"
    assert entries[0][0].read_bytes() == entries[0][1]
    assert {e.key for e in entries[0][2].tables}.isdisjoint(e.key for e in entries[1][2].tables)
    shutil.rmtree(user_table_dir())
    for (archive, _, index), value in zip(entries, (1.0, 2.0), strict=True):
        species = "electron" if value == 1.0 else "positron"
        monkeypatch.setattr(
            f"pyrite.xsgen.fetch.load_{code}_release_index",
            lambda *, projectile, expected=species, pinned=index: (
                pinned if projectile == expected else None
            ),
        )
        fetch(archive, projectile=species)
        for entry in index.tables:
            table = resolve(entry.key)
            assert table is not None
            assert table.arrays()["species_reference"].item() == value
    rebuilt, _ = release.build_release(tmp_path / "again", projectile="positron")
    assert rebuilt.read_bytes() == entries[1][1]
    loaded = release.load_release_index(entries[1][0].with_suffix(".json"))
    assert loaded == entries[1][2]


def test_shipped_positron_stopping_pin_covers_catalog_without_electron_keys():
    from pyrite.xsgen.sbethe.release import catalog_keys, load_release_index

    electron = load_release_index()
    positron = load_release_index(projectile="positron")
    assert electron is not None and positron is not None
    assert {entry.key for entry in electron.tables}.isdisjoint(
        entry.key for entry in positron.tables
    )
    assert {name for entry in positron.tables for name in entry.label.split(",")} == set(
        catalog_keys()
    )
    assert positron.archive_bytes > 0
    assert len(positron.archive_sha256) == 64


def test_shipped_positron_elastic_pin_mirrors_electron_labels_with_disjoint_keys():
    from pyrite.xsgen.elsepa.release import load_release_index

    electron = load_release_index()
    positron = load_release_index(projectile="positron")
    assert electron is not None and positron is not None
    assert {entry.key for entry in electron.tables}.isdisjoint(
        entry.key for entry in positron.tables
    )
    assert sorted(entry.label for entry in positron.tables) == sorted(
        entry.label for entry in electron.tables
    )
    assert positron.archive_bytes > 0
    assert len(positron.archive_sha256) == 64


@pytest.mark.parametrize("muffin", [False, True])
def test_projectile_changes_deck_and_identity_without_changing_electron_defaults(muffin):
    def request(**options):
        if muffin:
            return muffin_tin_request(
                "silicon", 14, [1e3], radius_cm=1.175e-8, density_g_cm3=2.33, **options
            )
        return element_request(14, [1e3], **options)

    historical, electron = request()
    explicit, _ = request(projectile="electron")
    alternate, positron = request(projectile="positron")
    assert historical.key == explicit.key
    assert alternate.key != historical.key
    assert "projectile" not in historical.model
    assert alternate.model["projectile"] == "positron"
    assert "IELEC  -1\n" in electron.render()
    assert "IELEC  1\n" in positron.render()
    assert "MEXCH  0\n" in positron.render()
    assert "MCPOL  2\n" in positron.render()
    assert positron.output_names == electron.output_names


@pytest.mark.parametrize("projectile", ["proton", "", "POSITRON"])
def test_invalid_projectile_is_rejected_before_elsepa_runs(projectile):
    with pytest.raises(ValueError, match="projectile"):
        ElsepaDeck.free_atom(14, [1e3], projectile=projectile)


def test_positron_cannot_request_an_exchange_potential():
    with pytest.raises(ValueError, match="exchange_model=0"):
        ElsepaDeck.free_atom(14, [1e3], projectile="positron", exchange_model=1)


def test_catalog_lookup_selects_both_free_and_muffin_tin_positron_tables(monkeypatch):
    from pyrite.xsgen.elsepa import catalog

    seen = []

    class Table:
        def arrays(self):
            return {
                "mu": np.array([0.0, 1.0]),
                "energy_eV": np.array([1e3, 1e7]),
                "total_elastic_cm2": np.array([1e-16, 1e-18]),
                "transport1_cm2": np.array([1e-17, 1e-19]),
                "dcs_cm2_sr": np.ones((2, 2)),
            }

    def capture(request, deck):
        seen.append(request.model)
        return request, deck

    monkeypatch.setattr(
        catalog, "element_request", lambda *a, **k: capture(*element_request(*a, **k))
    )
    monkeypatch.setattr(
        catalog, "muffin_tin_request", lambda *a, **k: capture(*muffin_tin_request(*a, **k))
    )
    monkeypatch.setattr(catalog, "resolve", lambda key: Table())
    result = resolve_layer_tables(
        CATALOG.crystal("silicon").info.composition, projectile="positron"
    )
    assert len(result[0].tables) == 2
    assert [model["mode"] for model in seen] == ["free_atom", "muffin_tin"]
    assert all(model["projectile"] == "positron" for model in seen)


def test_sbethe_catalog_resolves_species_independently(isolated_dirs, packaged_dir):
    material = catalog_material("silicon")
    for projectile, stopping in (("electron", 1.0), ("positron", 2.0)):
        request = material_request(
            material.key,
            material.composition,
            density_g_cm3=material.density_g_cm3,
            mean_excitation_eV=material.mean_excitation_eV,
            projectile=projectile,
        )
        store(request, {"stopping_eV_per_angstrom": np.array([stopping])})
    assert resolve_catalog_table("silicon").arrays()["stopping_eV_per_angstrom"].item() == 1.0
    assert (
        resolve_catalog_table("silicon", projectile="positron")
        .arrays()["stopping_eV_per_angstrom"]
        .item()
        == 2.0
    )


@pytest.mark.parametrize("code", ["elsepa", "sbethe"])
def test_missing_positron_table_recovery_requests_positron_species(monkeypatch, code):
    from pyrite.xsgen._errors import TableNotFoundError

    monkeypatch.setattr(f"pyrite.xsgen.{code}.catalog.resolve", lambda key: None)
    with pytest.raises(TableNotFoundError, match="--projectile positron"):
        if code == "elsepa":
            resolve_layer_tables(CATALOG.crystal("silicon").composition, projectile="positron")
        else:
            resolve_catalog_table("silicon", projectile="positron")
