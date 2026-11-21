"""Package the pinned tables and datasets as one Zenodo dataset record.

PyRITE's table products are archived together in one Zenodo dataset family
(concept DOI :data:`CONCEPT_DOI`), apart from the software family that the
GitHub integration fills from code releases. Each version of that record
holds one deterministic zip and one ``sha256sum`` file per upstream code:

- ``elsepa.zip`` -- ELSEPA elastic tables, electron and positron;
- ``sbethe.zip`` -- SBETHE stopping tables, electron and positron;
- ``bremslib.zip`` -- BremsLib-derived bremsstrahlung tables;
- ``epics.zip`` -- EEDL and EADL as pinned, and the PyRITE-derived EPDL files;
- ``pyrite.zip`` -- data computed by PyRITE itself.

A bundle repackages bytes that are already pinned: table members are copied
unchanged from the verified release archives, dataset members from verified
installs, so every table key, manifest digest and dataset SHA-256 is
unchanged. Only the archive digests are new. :func:`build_bundles` writes the
staging directory a maintainer uploads; :func:`pin_bundles` then records the
record's file URLs in :data:`BUNDLE_INDEX` and re-pins the table indexes to
their bundle. SBETHE's upstream ``sdbase`` mirror and NIST SRD 64 are never
bundled (see ``THIRD-PARTY-NOTICES.md``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._util.hashing import file_sha256
from ..datasets import BUNDLE_SCHEMA, bundle_index_path, bundle_member, require_verified
from ..paths import data_dir
from ._urls import archive_urls
from .bremslib.release import archive_member

#: Concept DOI of the table dataset family (all versions).
CONCEPT_DOI = "10.5281/zenodo.23220627"
#: Zenodo download URL of one file of a published version record.
ZENODO_FILE_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"

#: Fixed member timestamp, so identical members give identical bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleSpec:
    """What goes into one bundle.

    ``indexes`` names the shipped table indexes (files under
    ``data/xsgen``) whose tables the bundle carries; ``datasets`` names
    :mod:`pyrite.datasets` entries. A bundle has one kind or the other.
    """

    name: str
    description: str
    indexes: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        """Archive name in the record."""
        return f"{self.name}.zip"


#: The record's bundles, in listing order.
BUNDLES: tuple[BundleSpec, ...] = (
    BundleSpec(
        "elsepa",
        "ELSEPA elastic-scattering tables, electron and positron (CC BY-NC 3.0 adaptations)",
        indexes=("elsepa-tables.json", "elsepa-positron-tables.json"),
    ),
    BundleSpec(
        "sbethe",
        "SBETHE stopping-power tables, electron and positron (CC BY-NC 3.0 adaptations)",
        indexes=("sbethe-tables.json", "sbethe-positron-tables.json"),
    ),
    BundleSpec(
        "bremslib",
        "BremsLib 2.0.8 bremsstrahlung tables, release float32 variant (CC BY 4.0 adaptations)",
        indexes=("bremslib-tables.json",),
    ),
    BundleSpec(
        "epics",
        "EPICS2025 EEDL and EADL as pinned, and PyRITE-derived EPDL tables (CC BY 4.0)",
        datasets=("eedl", "eadl", "epdl", "epdl-differential"),
    ),
    BundleSpec(
        "pyrite",
        "Tables computed by PyRITE: the lattice-coherent bremsstrahlung share kernel",
        datasets=("brem-lattice-share",),
    ),
)


def _index_path(name: str) -> Path:
    return data_dir() / "xsgen" / name


def _write_zip(archive: Path, members: Mapping[str, Callable[[], bytes]]) -> None:
    """Write ``members`` (name -> bytes reader), sorted and timestamp-free."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            bundle.writestr(info, members[name]())


def _source_archive(record: Mapping[str, Any], label: str, archives: Path | None, work: Path):
    """Return a verified copy of an index's published archive."""
    from ._urls import archive_urls
    from .fetch import _obtain

    archive = record["archive"]
    local = None
    if archives is not None:
        urls = archive_urls(archive)
        names = [url.rsplit("/", 1)[-1].split("?", 1)[0] for url in urls]
        # A stale local copy (other digest) is skipped in favour of the URLs.
        local = next(
            (
                archives / n
                for n in names
                if (archives / n).is_file() and file_sha256(archives / n) == archive["sha256"]
            ),
            None,
        )
    work.mkdir(parents=True, exist_ok=True)
    path, _ = _obtain(local, archive_urls(archive), work, label, str(archive["sha256"]))
    return path


def _table_members(
    spec: BundleSpec, archives: Path | None, work: Path
) -> dict[str, Callable[[], bytes]]:
    """Collect every table member of ``spec``'s indexes from their archives."""
    members: dict[str, bytes] = {}
    for index_name in spec.indexes:
        record = json.loads(_index_path(index_name).read_text(encoding="utf-8"))
        source = _source_archive(record, index_name, archives, work / index_name)
        with zipfile.ZipFile(source) as zipped:
            for row in record["tables"]:
                for suffix in (".json", ".npz"):
                    name = archive_member(str(row["key"]), suffix)
                    data = zipped.read(name)
                    if members.setdefault(name, data) != data:
                        raise ValueError(f"{name} differs between the {spec.name} archives")
    return {name: (lambda data=data: data) for name, data in members.items()}


def _dataset_members(spec: BundleSpec) -> dict[str, Callable[[], bytes]]:
    """Collect verified installed datasets; run ``pyrite tables fetch`` first."""
    from ..datasets import get

    members: dict[str, Callable[[], bytes]] = {}
    for name in spec.datasets:
        path = require_verified(name)
        members[bundle_member(get(name))] = path.read_bytes
    return members


def build_bundles(
    out_dir: str | Path,
    *,
    archives: str | Path | None = None,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write the record's bundles, checksum files and manifest into ``out_dir``.

    Parameters
    ----------
    out_dir
        Staging directory, created if absent; the upload set.
    archives
        Directory holding local copies of the pinned table release archives,
        under their published names. Missing ones are downloaded from the
        index URLs. Either way each is checked against its pinned digest.
    names
        Bundles to build; defaults to all of :data:`BUNDLES`.

    Returns
    -------
    dict
        The bundle index (``table-bundles.json``) with empty URLs, also
        written into ``out_dir``.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {spec.name for spec in BUNDLES} if names is None else set(names)
    unknown = wanted - {spec.name for spec in BUNDLES}
    if unknown:
        raise KeyError(f"unknown bundle(s): {', '.join(sorted(unknown))}")
    local = None if archives is None else Path(archives)
    bundles: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=".bundles-", dir=destination) as temp:
        work = Path(temp)
        for spec in BUNDLES:
            if spec.name not in wanted:
                continue
            members = _table_members(spec, local, work) if spec.indexes else _dataset_members(spec)
            partial = work / spec.filename
            _write_zip(partial, members)
            archive = destination / spec.filename
            shutil.move(partial, archive)
            digest = file_sha256(archive)
            (destination / f"{spec.filename}.sha256").write_text(
                f"{digest}  {spec.filename}\n", encoding="utf-8"
            )
            bundles[spec.name] = {
                "file": spec.filename,
                "sha256": digest,
                "bytes": archive.stat().st_size,
                "urls": [],
                "members": len(members),
            }
    index = {
        "schema": BUNDLE_SCHEMA,
        "concept_doi": CONCEPT_DOI,
        "record": None,
        "bundles": bundles,
    }
    (destination / "table-bundles.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def pin_bundles(index: Mapping[str, Any], record: int, *, root: Path | None = None) -> dict:
    """Pin ``index`` at Zenodo version ``record`` and re-pin the table indexes.

    Writes :func:`pyrite.datasets.bundle_index_path` and rewrites the
    ``archive`` of every table index a bundle carries, so ``pyrite tables
    fetch`` downloads the bundle. ``root`` redirects both to another
    ``data/`` tree (tests).
    """
    pinned = json.loads(json.dumps(index))
    pinned["record"] = int(record)
    for entry in pinned["bundles"].values():
        entry["urls"] = [ZENODO_FILE_URL.format(record=int(record), name=entry["file"])]
    base = data_dir() if root is None else Path(root)
    target = bundle_index_path() if root is None else base / "xsgen" / "table-bundles.json"
    target.write_text(json.dumps(pinned, indent=2) + "\n", encoding="utf-8")
    for spec in BUNDLES:
        entry = pinned["bundles"].get(spec.name)
        if entry is None:
            continue
        for index_name in spec.indexes:
            path = base / "xsgen" / index_name
            record_json = json.loads(path.read_text(encoding="utf-8"))
            old = record_json["archive"]
            # Same bytes already published elsewhere stay as a fallback.
            kept = archive_urls(old) if old["sha256"] == entry["sha256"] else ()
            urls = list(entry["urls"]) + [url for url in kept if url not in entry["urls"]]
            record_json["archive"] = {
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "urls": urls,
            }
            path.write_text(json.dumps(record_json, indent=2) + "\n", encoding="utf-8")
    return pinned


#: Upstream deposits each bundle derives from, for the record's relations.
UPSTREAM_DOIS = {
    "elsepa": "10.17632/w4hm5vymym.1",
    "sbethe": "10.17632/7zw25f428t.2",
    "bremslib": "10.17632/6zfsc9xsz8.9",
}


def render_readme(index: Mapping[str, Any], version: str) -> str:
    """Return the record README listing every bundle and how to install it."""
    rows = "\n".join(
        f"| `{spec.filename}` | `{spec.filename}.sha256` | {spec.description} |"
        for spec in BUNDLES
        if spec.name in index["bundles"]
    )
    return f"""# PyRITE runtime tables

Physics data and precomputed tables read by PyRITE, a Python toolkit for
electron transport and X-ray emission in crystalline and amorphous targets.
Table set version {version}; all versions share concept DOI {CONCEPT_DOI}.
PyRITE software releases are archived in a separate Zenodo record family.

## Contents

One zip and one `sha256sum` file per upstream code.

| Archive | Checksum | Contents |
|---|---|---|
{rows}

Members are byte-identical copies of what PyRITE pins: table members are
`tables/<key>.json` manifests with their `tables/<key>.npz` arrays; dataset
members are `<dataset>/<file>`. Each table manifest records its upstream
source, version and modifications.

## Install

A PyRITE build that pins this version downloads it with
`pyrite tables fetch`. Offline, download the archives and run, for example,
`pyrite tables fetch elsepa --archive elsepa.zip` (likewise
`sbethe-tables`, `bremslib`, and each EPICS dataset with `epics.zip`).
PyRITE checks every archive against its pinned SHA-256 before installing.

## Verify

    sha256sum -c *.sha256

## Licences and attribution

Each component keeps its upstream licence; see `THIRD-PARTY-NOTICES.md`.
ELSEPA and SBETHE tables are CC BY-NC 3.0 adaptations; BremsLib tables and
EPICS2025 data are CC BY 4.0. PyRITE's software licence does not cover these
data. Cite the PyRITE software version you used, this table set version,
and the upstream references listed in the notices.
"""


def deposit_metadata(version: str, creators: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return Zenodo deposit metadata for a new version of the table record."""
    related = [
        {"relation": "isDerivedFrom", "identifier": doi, "scheme": "doi"}
        for doi in UPSTREAM_DOIS.values()
    ]
    related.append(
        {
            "relation": "isSupplementTo",
            "identifier": "https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite",
            "scheme": "url",
        }
    )
    return {
        "metadata": {
            "upload_type": "dataset",
            "title": "PyRITE runtime tables",
            "version": version,
            "creators": list(creators),
            "description": (
                "Physics data and precomputed tables read by PyRITE: ELSEPA elastic, "
                "SBETHE stopping and BremsLib bremsstrahlung tables, EPICS2025 "
                "EEDL/EADL and PyRITE-derived EPDL data, and tables computed by PyRITE. "
                "One archive and one SHA-256 file per upstream code; upstream licences "
                "and attribution are in THIRD-PARTY-NOTICES.md."
            ),
            "access_right": "open",
            "license": "other-nc",
            "keywords": ["PyRITE", "cross sections", "ELSEPA", "SBETHE", "BremsLib", "EPICS2025"],
            "related_identifiers": related,
        }
    }


def verify_published(index: Mapping[str, Any], work: str | Path) -> list[str]:
    """Download every pinned bundle URL and return mismatch messages (empty: OK)."""
    from ._errors import DataFetchError
    from .fetch import _download

    problems = []
    for name, entry in index["bundles"].items():
        for url in entry["urls"]:
            staged = Path(work) / entry["file"]
            try:
                actual = _download(url, staged, f"{name} bundle ({url})")
            except DataFetchError as exc:
                problems.append(str(exc))
                continue
            if actual != entry["sha256"]:
                problems.append(f"{url}: expected {entry['sha256']}, got {actual}")
            staged.unlink(missing_ok=True)
    return problems


__all__ = [
    "BUNDLES",
    "CONCEPT_DOI",
    "ZENODO_FILE_URL",
    "BundleSpec",
    "UPSTREAM_DOIS",
    "build_bundles",
    "deposit_metadata",
    "pin_bundles",
    "render_readme",
    "verify_published",
]
