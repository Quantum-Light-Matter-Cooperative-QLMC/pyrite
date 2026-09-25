"""Released ELSEPA elastic tables: the maintainer build and the user-side fetch.

Stand-in tables are stored under the real production request keys, so the
round trip -- build, pin, install, resolve -- runs without compiling or
running ELSEPA and without the network.
"""

import shutil
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.xsgen import DataFetchError, TableNotFoundError
from pyrite.xsgen.elsepa.catalog import (
    PRODUCTION_ENERGIES_EV,
    _muffin_tin_energies,
    elemental_solid,
    resolve_layer_tables,
)
from pyrite.xsgen.elsepa.generate import element_request, muffin_tin_request
from pyrite.xsgen.elsepa.release import build_release, elementary_crystals, release_tables
from pyrite.xsgen.fetch import fetch_elsepa
from pyrite.xsgen.store import resolve, store, user_table_dir


def _arrays(energies, scale=1.0):
    energies = np.asarray(energies, dtype=float)
    mu = np.linspace(0.0, 1.0, 5)
    total = scale * np.geomspace(1e-16, 1e-18, energies.size)
    return {
        "energy_eV": energies,
        "mu": mu,
        "dcs_cm2_sr": np.ones((energies.size, mu.size)),
        "total_elastic_cm2": total,
        "transport1_cm2": total / 10.0,
    }


def _store_all(scale=1.0):
    for z in sorted(int(row["Z"]) for row in TRANSPORT_ELEMENTS.values()):
        request, _ = element_request(z, PRODUCTION_ENERGIES_EV)
        store(request, _arrays(PRODUCTION_ENERGIES_EV, scale))
    for key in elementary_crystals():
        solid = elemental_solid(key)
        request, _ = muffin_tin_request(
            key,
            solid.z,
            _muffin_tin_energies(),
            radius_cm=solid.radius_cm,
            density_g_cm3=solid.density_g_cm3,
        )
        store(request, _arrays(_muffin_tin_energies(), scale))


@pytest.fixture
def release(tmp_path, isolated_dirs, packaged_dir, monkeypatch):
    """Build a release from maintainer-side tables, then empty the user store."""
    _store_all()
    archive, index = build_release(tmp_path / "out")
    shutil.rmtree(user_table_dir())
    index_path = tmp_path / "out" / "elsepa-tables.json"
    monkeypatch.setattr("pyrite.xsgen.elsepa.release.release_index_path", lambda: index_path)
    return archive, index


def test_release_covers_every_transport_element_and_elementary_crystal(release):
    _, index = release
    labels = [entry.label for entry in index.tables]

    zs = sorted(int(row["Z"]) for row in TRANSPORT_ELEMENTS.values())
    assert labels[: len(zs)] == [f"Z={z}" for z in zs]
    assert labels[len(zs) :] == sorted({"silicon", "diamond", "hopg", "black_phosphorus"})
    assert all(len(CATALOG.crystal(key).info.composition) == 1 for key in labels[len(zs) :])


def test_rebuilding_from_fetched_tables_gives_the_same_archive(release, tmp_path):
    archive, index = release
    fetch_elsepa(archive)

    again, rebuilt = build_release(tmp_path / "again")

    assert rebuilt.archive_sha256 == index.archive_sha256
    assert again.read_bytes() == archive.read_bytes()


def test_fetch_installs_tables_that_the_default_model_then_resolves(release):
    archive, index = release

    result = fetch_elsepa(archive)

    assert result.installed is True
    assert result.file_count == len(index.tables)
    silicon = resolve_layer_tables(CATALOG.crystal("silicon").info.composition)
    assert [table.key for table in silicon[0].tables] == [
        entry.key for entry in index.tables if entry.label in {"Z=14", "silicon"}
    ]
    assert fetch_elsepa().installed is False


def test_a_locally_generated_table_under_a_released_key_counts_as_installed(release):
    archive, index = release
    _store_all(scale=2.0)  # same keys, different manifests (a local generation)

    result = fetch_elsepa(archive)

    assert result.installed is False
    kept = resolve(index.tables[0].key)
    assert kept.digest != index.tables[0].manifest_sha256


def test_an_unpublished_release_names_the_local_archive_route(release):
    with pytest.raises(DataFetchError, match="fetch elsepa --archive PATH"):
        fetch_elsepa()


def test_an_index_disagreeing_with_its_archive_installs_nothing(release):
    archive, index = release
    first = index.tables[0]
    tampered = replace(index, tables=(replace(first, manifest_sha256="0" * 64), *index.tables[1:]))

    with pytest.raises(DataFetchError, match="ELSEPA table manifest for Z=5"):
        fetch_elsepa(archive, index=tampered)
    assert not any(user_table_dir().glob("*.npz"))


def test_a_wrong_archive_installs_nothing(release, tmp_path):
    impostor = tmp_path / "impostor.zip"
    with zipfile.ZipFile(impostor, "w") as bundle:
        bundle.writestr("tables/x.json", "{}")

    with pytest.raises(DataFetchError, match="SHA-256 mismatch"):
        fetch_elsepa(impostor)
    assert not any(user_table_dir().glob("*.npz"))


def test_missing_tables_name_the_fetch_command(release):
    with pytest.raises(TableNotFoundError, match="pyrite tables fetch elsepa"):
        resolve_layer_tables([("Si", 0.03)])


def test_building_without_every_table_names_the_generation_route(isolated_dirs, tmp_path):
    with pytest.raises(TableNotFoundError, match="generation enabled"):
        release_tables()


def test_the_shipped_index_pins_every_production_table():
    from pyrite.xsgen.elsepa.release import load_release_index

    index = load_release_index()

    assert index is not None
    expected = [f"Z={z}" for z in sorted(int(r["Z"]) for r in TRANSPORT_ELEMENTS.values())]
    assert [e.label for e in index.tables] == [*expected, *elementary_crystals()]
    for entry in index.tables[: len(expected)]:
        z = int(entry.label.removeprefix("Z="))
        assert entry.key == element_request(z, PRODUCTION_ENERGIES_EV)[0].key
