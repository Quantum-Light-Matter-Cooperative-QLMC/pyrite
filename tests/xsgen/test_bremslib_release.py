"""Released BremsLib tables: the maintainer build and the user-side fetch.

The release is built from the same miniature library the converter tests use,
so the whole round trip -- build, pin, install, resolve -- runs without the
810 MB upstream library and without the network.
"""

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.xsgen import DataFetchError, TableNotFoundError
from pyrite.xsgen import fetch as fetch_module
from pyrite.xsgen.bremslib import generate_element
from pyrite.xsgen.bremslib.release import (
    DROPPED_ARRAYS,
    VARIANT,
    build_release,
    catalogue_elements,
    catalogue_table,
    load_release_index,
    release_arrays,
)
from pyrite.xsgen.fetch import fetch_bremslib
from pyrite.xsgen.store import resolve, user_table_dir

from .test_bremslib import _write_library as _write_element

ELEMENTS = (6, 79)


@pytest.fixture
def library(tmp_path) -> Path:
    """A miniature library holding every element in :data:`ELEMENTS`."""
    merged = tmp_path / "deposit" / "BremsLib_v2.0.8"
    for z in ELEMENTS:
        single = _write_element(tmp_path / f"z{z}", z)
        shutil.copytree(single, merged, dirs_exist_ok=True)
    return merged


@pytest.fixture
def release(library, tmp_path, isolated_dirs, packaged_dir, monkeypatch):
    """Build a release and pin its index as the one this build ships."""
    archive, index = build_release(tmp_path / "out", source_path=library, elements=ELEMENTS)
    index_path = tmp_path / "out" / "bremslib-tables.json"
    monkeypatch.setattr("pyrite.xsgen.bremslib.release.release_index_path", lambda: index_path)
    return archive, index


# --- the maintainer build ------------------------------------------------


def test_catalogue_elements_cover_every_transport_element():
    assert catalogue_elements() == tuple(sorted(row["Z"] for row in TRANSPORT_ELEMENTS.values()))


def test_release_transform_drops_uncertainties_and_narrows_the_ddcs(library):
    from pyrite.xsgen.bremslib import build_table

    full = build_table(library, 79)
    released = release_arrays(full)

    assert not set(DROPPED_ARRAYS) & set(released)
    assert released["ddcs_mb_sr"].dtype == np.float32
    # Normalization-bearing arrays keep their precision.
    assert released["sdcs_mb"].dtype == np.float64
    assert released["node_angular_integral_mb"].dtype == np.float64
    np.testing.assert_allclose(released["ddcs_mb_sr"], full["ddcs_mb_sr"], rtol=1e-7)


def test_a_released_panel_reports_its_missing_uncertainties_as_unknown(library):
    from pyrite.xsgen.bremslib import build_table, panel_of

    full = build_table(library, 79)
    panel = panel_of(release_arrays(full), 0)

    np.testing.assert_array_equal(panel.theta_deg, panel_of(full, 0).theta_deg)
    assert np.isnan(panel.rel_err).all()


def test_release_transform_refuses_an_unconverted_table():
    with pytest.raises(ValueError, match="not a converted BremsLib table"):
        release_arrays({"t1_MeV": np.ones(2)})


def test_release_index_pins_the_archive_and_every_table(release):
    archive, index = release

    assert [entry.z for entry in index.tables] == list(ELEMENTS)
    assert index.variant == VARIANT
    assert "10.17632/6zfsc9xsz8.9" in index.upstream
    assert index.archive_bytes == archive.stat().st_size
    assert load_release_index() == index
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(bundle.namelist())
    assert names == sorted(
        f"tables/{entry.key}{suffix}" for entry in index.tables for suffix in (".json", ".npz")
    )


def test_released_manifests_state_the_release_modifications(release, tmp_path):
    archive, index = release
    entry = index.tables[0]
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(f"tables/{entry.key}.json"))

    assert "float32" in manifest["modifications"]
    assert "uncertainties" in manifest["modifications"]
    assert "10.17632/6zfsc9xsz8.9" in manifest["upstream"]


def test_released_tables_key_apart_from_locally_generated_ones(release, library):
    _, index = release

    local = generate_element(79, source_path=library).table

    assert local.key != index.entry(79).key


def test_rebuilding_from_the_same_library_keeps_the_same_keys(
    library, tmp_path, isolated_dirs, packaged_dir
):
    _, first = build_release(tmp_path / "a", source_path=library, elements=(79,))
    _, second = build_release(tmp_path / "b", source_path=library, elements=(79,))

    # Keys depend on the library files and the variant, not on when or where
    # the release was built, so a refresh against an unchanged deposit keeps
    # every key a user's run identity already recorded.
    assert first.tables[0].key == second.tables[0].key


# --- the user-side fetch -------------------------------------------------


def test_fetch_installs_from_a_local_archive_then_resolves(release):
    archive, index = release

    result = fetch_bremslib(archive)

    assert result.installed is True
    assert result.file_count == len(ELEMENTS)
    assert result.path == user_table_dir()
    for entry in index.tables:
        table = catalogue_table(entry.z)
        assert table.key == entry.key
        assert table.tier == "user"
        assert table.arrays()["ddcs_mb_sr"].dtype == np.float32


def test_a_complete_install_needs_neither_archive_nor_network(release, monkeypatch):
    archive, _ = release
    fetch_bremslib(archive)
    monkeypatch.setattr(fetch_module, "urlopen", lambda *a, **k: pytest.fail("network used"))

    result = fetch_bremslib()

    assert result.installed is False


def test_fetch_downloads_the_pinned_url(release, monkeypatch):
    import io

    archive, index = release
    pinned = replace(index, urls=("https://example.invalid/tables.zip",))
    seen = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_url(request, **kwargs):
        seen.append(request.full_url)
        return _Response(archive.read_bytes())

    monkeypatch.setattr(fetch_module, "urlopen", open_url)

    assert fetch_bremslib(index=pinned).installed is True
    assert seen == ["https://example.invalid/tables.zip"]


def test_an_unpublished_release_names_the_local_archive_route(release):
    with pytest.raises(DataFetchError, match="--archive PATH"):
        fetch_bremslib()


def test_a_build_pinning_no_release_says_how_to_generate(isolated_dirs, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pyrite.xsgen.bremslib.release.release_index_path", lambda: tmp_path / "absent.json"
    )

    with pytest.raises(DataFetchError, match="pyrite tables generate --code bremslib"):
        fetch_bremslib()


def test_a_wrong_archive_installs_nothing(release, tmp_path):
    impostor = tmp_path / "impostor.zip"
    with zipfile.ZipFile(impostor, "w") as bundle:
        bundle.writestr("tables/x.json", "{}")

    with pytest.raises(DataFetchError, match="SHA-256 mismatch"):
        fetch_bremslib(impostor)
    assert not any(user_table_dir().glob("*.npz"))


def test_an_index_disagreeing_with_its_archive_installs_nothing(release):
    archive, index = release
    first = index.tables[0]
    tampered = replace(index, tables=(replace(first, manifest_sha256="0" * 64), *index.tables[1:]))

    with pytest.raises(DataFetchError, match="does not match the pinned release"):
        fetch_bremslib(archive, index=tampered)
    assert not any(user_table_dir().glob("*.npz"))


def test_a_foreign_table_under_a_released_key_is_not_overwritten(release):
    archive, index = release
    entry = index.tables[0]
    user_table_dir().mkdir(parents=True, exist_ok=True)
    manifest = user_table_dir() / f"{entry.key}.json"
    manifest.write_text(json.dumps({"manifest_sha256": "f" * 64}), encoding="utf-8")
    (user_table_dir() / f"{entry.key}.npz").write_bytes(b"theirs")

    with pytest.raises(DataFetchError, match="move or remove it"):
        fetch_bremslib(archive)
    assert (user_table_dir() / f"{entry.key}.npz").read_bytes() == b"theirs"


# --- resolution ----------------------------------------------------------


def test_an_unfetched_release_names_the_fetch_command(release):
    with pytest.raises(TableNotFoundError, match="pyrite tables fetch bremslib"):
        catalogue_table(79)


def test_an_element_outside_the_release_names_the_generate_command(release):
    with pytest.raises(TableNotFoundError, match="--element 26"):
        catalogue_table(26)


def test_a_table_resolving_to_other_numbers_is_refused(release):
    archive, index = release
    fetch_bremslib(archive)
    entry = index.entry(79)
    table = resolve(entry.key)
    body = dict(table.manifest)
    body["manifest_sha256"] = "e" * 64
    table.path.with_suffix(".json").write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(TableNotFoundError, match="does not match the pinned release"):
        catalogue_table(79)


def test_a_missing_checkout_points_catalogue_elements_at_the_fetch(isolated_dirs, tmp_path):
    from pyrite.xsgen import SourceUnavailableError

    with pytest.raises(SourceUnavailableError, match="pyrite tables fetch bremslib"):
        generate_element(74, source_path=tmp_path / "no-checkout")


def test_a_missing_checkout_does_not_offer_a_fetch_for_other_elements(isolated_dirs, tmp_path):
    from pyrite.xsgen import SourceUnavailableError

    with pytest.raises(SourceUnavailableError) as raised:
        generate_element(79, source_path=tmp_path / "no-checkout")
    assert "fetch" not in str(raised.value)
