"""Released SBETHE catalogue stopping tables: the maintainer build and the user fetch.

Stand-in tables are stored under the real catalogue request keys, so the
round trip -- build, pin, install, resolve -- runs without compiling or
running SBETHE and without the network.
"""

import shutil
from dataclasses import replace

import numpy as np
import pytest

from pyrite.xsgen import DataFetchError, TableNotFoundError
from pyrite.xsgen.fetch import fetch_sbethe_tables
from pyrite.xsgen.sbethe.catalog import catalog_material, resolve_catalog_table
from pyrite.xsgen.sbethe.generate import material_request
from pyrite.xsgen.sbethe.release import (
    build_release,
    catalog_keys,
    load_release_index,
    release_tables,
)
from pyrite.xsgen.store import resolve, store, user_table_dir


def _request(key):
    material = catalog_material(key)
    return material_request(
        material.key,
        material.composition,
        density_g_cm3=material.density_g_cm3,
        mean_excitation_eV=material.mean_excitation_eV,
        band_gap_eV=material.band_gap_eV,
    )


def _store_all(scale=1.0):
    energy = np.geomspace(1.0e3, 1.0e9, 4)
    for key in catalog_keys():
        request = _request(key)
        if resolve(request.key) is None:
            store(
                request,
                {"stopping_energy_eV": energy, "stopping_eV_per_angstrom": scale * energy**0.1},
            )


@pytest.fixture
def release(tmp_path, isolated_dirs, packaged_dir, monkeypatch):
    """Build a release from maintainer-side tables, then empty the user store."""
    _store_all()
    archive, index = build_release(tmp_path / "out")
    shutil.rmtree(user_table_dir())
    index_path = tmp_path / "out" / "sbethe-tables.json"
    monkeypatch.setattr("pyrite.xsgen.sbethe.release.release_index_path", lambda: index_path)
    return archive, index


def test_release_has_one_entry_per_table_and_labels_every_catalogue_key(release):
    _, index = release

    labelled = [name for entry in index.tables for name in entry.label.split(",")]
    assert sorted(labelled) == list(catalog_keys())
    assert len({entry.key for entry in index.tables}) == len(index.tables)
    shared = next(entry for entry in index.tables if "mos2-on-sapphire" in entry.label)
    assert shared.key == _request("mos2").key


def test_fetch_installs_tables_that_catalogue_resolution_then_finds(release):
    archive, index = release

    result = fetch_sbethe_tables(archive)

    assert result.installed is True
    assert result.code == "sbethe-tables"
    assert result.file_count == len(index.tables)
    assert resolve_catalog_table("pdse2").tier == "user"
    assert fetch_sbethe_tables().installed is False


def test_rebuilding_from_fetched_tables_gives_the_same_archive(release, tmp_path):
    archive, index = release
    fetch_sbethe_tables(archive)

    again, rebuilt = build_release(tmp_path / "again")

    assert rebuilt.archive_sha256 == index.archive_sha256
    assert again.read_bytes() == archive.read_bytes()


def test_an_unpublished_release_names_the_local_archive_route(release, monkeypatch):
    _, index = release
    unpublished = replace(index, urls=())

    with pytest.raises(DataFetchError, match="fetch sbethe-tables --archive PATH"):
        fetch_sbethe_tables(index=unpublished)


def test_building_without_every_table_names_the_generation_route(isolated_dirs, packaged_dir):
    with pytest.raises(TableNotFoundError, match="pyrite tables generate --code sbethe"):
        release_tables()


def test_the_shipped_index_pins_every_catalogue_table_under_its_current_key():
    index = load_release_index()

    assert index is not None
    by_label = {name: entry for entry in index.tables for name in entry.label.split(",")}
    assert sorted(by_label) == list(catalog_keys())
    for key, entry in by_label.items():
        assert entry.key == _request(key).key, key
