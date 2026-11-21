"""Zenodo table bundles: deterministic packaging and URL pinning."""

import hashlib
import json
import zipfile

import pytest

from pyrite import datasets
from pyrite.xsgen import bundles


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    return path


def _index(path, archive, keys):
    path.write_text(
        json.dumps(
            {
                "schema": "x",
                "archive": {
                    "sha256": _sha(archive),
                    "bytes": archive.stat().st_size,
                    "urls": [f"https://example.invalid/{archive.name}"],
                },
                "tables": [{"label": key, "key": key, "manifest_sha256": "m"} for key in keys],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def codes(tmp_path, monkeypatch):
    """Two archives (electron, positron) and one dataset, under a fake data tree."""
    (tmp_path / "xsgen").mkdir()
    local = tmp_path / "archives"
    local.mkdir()
    shared = {"tables/s.json": b"{}", "tables/s.npz": b"S"}
    electron = _archive(local / "e.zip", {**shared, "tables/a.json": b"{}", "tables/a.npz": b"A"})
    positron = _archive(local / "p.zip", {**shared, "tables/b.json": b"{}", "tables/b.npz": b"B"})
    _index(tmp_path / "xsgen" / "e.json", electron, ["a", "s"])
    _index(tmp_path / "xsgen" / "p.json", positron, ["b", "s"])
    eadl = tmp_path / "EADL2025.ALL"
    eadl.write_bytes(b"relaxation")
    specs = (
        bundles.BundleSpec("code", "test code", indexes=("e.json", "p.json")),
        bundles.BundleSpec("epics", "test data", datasets=("eadl",)),
    )
    monkeypatch.setattr(bundles, "BUNDLES", specs)
    monkeypatch.setattr(bundles, "_index_path", lambda name: tmp_path / "xsgen" / name)
    monkeypatch.setattr(bundles, "require_verified", lambda name: eadl)
    return tmp_path, local


def test_one_deterministic_archive_and_checksum_per_code(codes, tmp_path):
    root, local = codes
    first = bundles.build_bundles(tmp_path / "out1", archives=local)
    second = bundles.build_bundles(tmp_path / "out2", archives=local)

    assert first == second
    with zipfile.ZipFile(tmp_path / "out1" / "code.zip") as zipped:
        assert zipped.namelist() == [
            f"tables/{key}{suffix}" for key in "abs" for suffix in (".json", ".npz")
        ]
    with zipfile.ZipFile(tmp_path / "out1" / "epics.zip") as zipped:
        assert zipped.read("eadl/EADL2025.ALL") == b"relaxation"
    for name, entry in first["bundles"].items():
        archive = tmp_path / "out1" / entry["file"]
        assert entry["sha256"] == _sha(archive)
        checksum = (tmp_path / "out1" / f"{entry['file']}.sha256").read_text()
        assert checksum == f"{entry['sha256']}  {name}.zip\n"
    assert first["concept_doi"] == bundles.CONCEPT_DOI and first["record"] is None


def test_conflicting_members_across_archives_are_refused(codes, tmp_path):
    root, local = codes
    positron = _archive(local / "p.zip", {"tables/s.json": b"{}", "tables/s.npz": b"other"})
    _index(root / "xsgen" / "p.json", positron, ["s"])

    with pytest.raises(ValueError, match="differs"):
        bundles.build_bundles(tmp_path / "out", archives=local)


def test_pinning_records_zenodo_urls_and_repins_table_indexes(codes, tmp_path):
    root, local = codes
    index = bundles.build_bundles(tmp_path / "out", archives=local)
    old = json.loads((root / "xsgen" / "e.json").read_text())

    pinned = bundles.pin_bundles(index, 42, root=root)

    url = "https://zenodo.org/records/42/files/code.zip?download=1"
    assert pinned["record"] == 42 and pinned["bundles"]["code"]["urls"] == [url]
    assert json.loads((root / "xsgen" / "table-bundles.json").read_text()) == pinned
    for name in ("e.json", "p.json"):
        archive = json.loads((root / "xsgen" / name).read_text())["archive"]
        # Different bytes: the old GitHub URL cannot serve the new digest.
        assert archive == {
            "sha256": index["bundles"]["code"]["sha256"],
            "bytes": index["bundles"]["code"]["bytes"],
            "urls": [url],
        }
    assert old["archive"]["sha256"] != index["bundles"]["code"]["sha256"]


def test_pinning_keeps_an_old_url_that_serves_the_same_bytes(codes, tmp_path):
    root, local = codes
    index = bundles.build_bundles(tmp_path / "out", archives=local)
    record = json.loads((root / "xsgen" / "e.json").read_text())
    record["archive"]["sha256"] = index["bundles"]["code"]["sha256"]
    (root / "xsgen" / "e.json").write_text(json.dumps(record))

    bundles.pin_bundles(index, 7, root=root)

    urls = json.loads((root / "xsgen" / "e.json").read_text())["archive"]["urls"]
    assert urls == [
        "https://zenodo.org/records/7/files/code.zip?download=1",
        "https://example.invalid/e.zip",
    ]


def test_the_shipped_bundle_index_covers_every_bundle_and_table_pin():
    record = json.loads(datasets.bundle_index_path().read_text(encoding="utf-8"))
    assert record["schema"] == datasets.BUNDLE_SCHEMA
    assert record["concept_doi"] == bundles.CONCEPT_DOI
    assert list(record["bundles"]) == [spec.name for spec in bundles.BUNDLES]
    for spec in bundles.BUNDLES:
        entry = record["bundles"][spec.name]
        assert entry["file"] == spec.filename
        for name in spec.indexes:
            archive = json.loads(bundles._index_path(name).read_text())["archive"]
            if entry["urls"]:
                # Once pinned, every table index fetches its bundle.
                assert archive["sha256"] == entry["sha256"]
                assert archive["urls"][0] == entry["urls"][0]
    # The BremsLib bundle reproduces the archive already published on GitHub.
    bremslib = json.loads(bundles._index_path("bremslib-tables.json").read_text())
    assert record["bundles"]["bremslib"]["sha256"] == bremslib["archive"]["sha256"]


#: GitHub table/mirror tags published before tables moved to the Zenodo
#: dataset record. Every GitHub release is archived into the *software*
#: Zenodo family, so a new table pin must not add a tag here.
FROZEN_GITHUB_TABLE_TAGS = {
    "mirror-eedl2025-1",
    "mirror-eadl2025-1",
    "mirror-sbethe2-1",
    "tables-epdl-1",
    "tables-epdl-differential-1",
    "tables-brem-lattice-share-1",
    "tables-bremslib-4",
    "tables-elsepa-5",
    "tables-elsepa-positron-5",
    "tables-sbethe-4",
    "tables-sbethe-positron-4",
}


def test_no_new_table_pin_points_at_a_github_release():
    from pyrite.xsgen._urls import archive_urls
    from pyrite.xsgen.fetch import SBETHE_ARCHIVE_URLS

    urls = [*SBETHE_ARCHIVE_URLS]
    for dataset in (*datasets.DATASETS.values(), *datasets.OPTIONAL_DATASETS.values()):
        urls.extend(dataset.urls)
    for path in sorted(datasets.bundle_index_path().parent.glob("*-tables.json")):
        urls.extend(archive_urls(json.loads(path.read_text(encoding="utf-8"))["archive"]))
    marker = "/pyrite/releases/download/"
    tags = {url.split(marker, 1)[1].split("/", 1)[0] for url in urls if marker in url}
    assert tags <= FROZEN_GITHUB_TABLE_TAGS
