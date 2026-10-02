"""Build, pin, and look up released BremsLib-derived tables.

The released tables cover every element a runnable catalogue material may
contain, so bremsstrahlung from BremsLib works without a local checkout. They
are too large for the wheel -- 10 to 18 MB even at reduced precision, against
a whole-subsystem budget of about 5 MB -- so they are fetched on demand, the
way SBETHE's ``sdbase/`` is (``pyrite tables fetch bremslib``).

Two halves, split by who holds a BremsLib checkout:

- **Maintainer.** :func:`build_release` reads the library, converts each
  element, applies the release transform, and writes one deterministic zip
  plus an index recording the archive digest and each table's key and
  manifest digest. ``scripts/release_bremslib_tables.py`` drives it. Run it
  when the catalogue gains an element or the upstream deposit versions.
- **User.** The index ships in the wheel. It is the only way a user without
  a checkout can know a table's key, because the key covers a digest of the
  library files the table was read from. :func:`catalogue_table` resolves
  through it; :func:`pyrite.xsgen.fetch.fetch_bremslib` installs from it.

The release transform stores the DDCS as float32 and drops the two
uncertainty arrays. That is a released-table decision, not a change to
:func:`~pyrite.xsgen.bremslib.convert.build_table`: a locally generated table
stays exact. Because the transform changes the numbers, the variant is part
of the key, so a released table and a locally generated one never share a
key while holding different arrays.
"""

import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...materials._transport_data import TRANSPORT_ELEMENTS
from ...paths import data_dir
from .._errors import TableNotFoundError
from .._urls import archive_urls, as_urls
from ..sources import resolve_source
from ..store import MODIFICATIONS_NOTE, StoredTable, resolve, store
from .convert import build_table
from .generate import element_request
from .read import COMPLETE_T1_MAX_MEV, library_root

#: Schema of the release index.
RELEASE_SCHEMA = "pyrite.xsgen.bremslib-release.v1"

#: Key parameter naming the release transform. Bump it when the transform
#: changes, which re-keys every released table.
VARIANT = "release-float32-v1"

#: Arrays the release drops. Both are per-point upstream uncertainties, which
#: no consumer samples from, and together they are about a third of a table.
DROPPED_ARRAYS = ("sdcs_rel_err", "ddcs_rel_err")

#: Arrays the release stores as float32. The DDCS dominates the payload, and
#: its upstream precision (``%.8e``) exceeds what sampling an angle needs. The
#: grids, the SDCS and both angular integrals stay float64: they set the
#: normalization, and they are small.
FLOAT32_ARRAYS = ("ddcs_mb_sr",)

#: The release's modifications, recorded in every released manifest on top
#: of what any generated table already says, for the CC BY 4.0 term.
RELEASE_MODIFICATIONS = (
    f"{MODIFICATIONS_NOTE} Released variant {VARIANT}: the DDCS is stored as "
    "float32, and the per-point relative uncertainties of the SDCS and the "
    "DDCS are removed."
)

#: Directory holding the tables inside the release archive.
ARCHIVE_TABLE_DIR = "tables"

#: Fixed member timestamp, so rebuilding identical tables gives identical bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_DIGEST_CHUNK = 1 << 20


def release_index_path() -> Path:
    """Return the release index shipped inside the wheel."""
    return data_dir() / "xsgen" / "bremslib-tables.json"


def catalogue_elements() -> tuple[int, ...]:
    """Return the atomic numbers a released table set must cover.

    Every element a runnable catalogue material is allowed to contain, not
    only those the catalogue happens to use today: the material parser
    rejects any other element, so this set bounds every material a user can
    run without their own checkout.
    """
    return tuple(sorted(int(row["Z"]) for row in TRANSPORT_ELEMENTS.values()))


def release_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the release transform to one converted table."""
    missing = [name for name in (*DROPPED_ARRAYS, *FLOAT32_ARRAYS) if name not in arrays]
    if missing:
        raise ValueError(f"not a converted BremsLib table; missing {', '.join(missing)}")
    return {
        name: np.asarray(value, dtype=np.float32) if name in FLOAT32_ARRAYS else np.asarray(value)
        for name, value in arrays.items()
        if name not in DROPPED_ARRAYS
    }


@dataclass(frozen=True)
class ReleaseEntry:
    """One released table: its element, key, and manifest digest."""

    z: int
    key: str
    manifest_sha256: str

    @property
    def label(self) -> str:
        """Name of the table in fetch messages."""
        return f"Z={self.z}"


@dataclass(frozen=True)
class ReleaseIndex:
    """The pinned description of one released table archive.

    Parameters
    ----------
    upstream
        Upstream deposit and library directory the tables were read from, as
        recorded in each table's manifest.
    variant
        Release transform, from :data:`VARIANT`.
    t1_max_MeV
        Energy bound every table was built with.
    archive_sha256, archive_bytes
        Digest and size of the release zip.
    urls
        Where the archive is published, tried in order (see
        :mod:`pyrite.xsgen._urls`); empty if it is not yet. A missing URL
        still permits installing from a local copy.
    tables
        One entry per element, sorted by atomic number.
    """

    upstream: str
    variant: str
    t1_max_MeV: float
    archive_sha256: str
    archive_bytes: int
    urls: tuple[str, ...]
    tables: tuple[ReleaseEntry, ...]

    def record(self) -> dict[str, Any]:
        """Return the JSON-serializable index body."""
        return {
            "schema": RELEASE_SCHEMA,
            "upstream": self.upstream,
            "variant": self.variant,
            "t1_max_MeV": self.t1_max_MeV,
            "archive": {
                "sha256": self.archive_sha256,
                "bytes": self.archive_bytes,
                "urls": list(self.urls),
            },
            "tables": [
                {"z": entry.z, "key": entry.key, "manifest_sha256": entry.manifest_sha256}
                for entry in self.tables
            ],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReleaseIndex:
        """Parse an index body, rejecting any other schema."""
        if record.get("schema") != RELEASE_SCHEMA:
            raise ValueError(
                f"unsupported BremsLib release index schema {record.get('schema')!r}; "
                f"expected {RELEASE_SCHEMA!r}"
            )
        archive = record["archive"]
        return cls(
            upstream=str(record["upstream"]),
            variant=str(record["variant"]),
            t1_max_MeV=float(record["t1_max_MeV"]),
            archive_sha256=str(archive["sha256"]),
            archive_bytes=int(archive["bytes"]),
            urls=archive_urls(archive),
            tables=tuple(
                ReleaseEntry(
                    z=int(row["z"]),
                    key=str(row["key"]),
                    manifest_sha256=str(row["manifest_sha256"]),
                )
                for row in record["tables"]
            ),
        )

    def entry(self, z: int) -> ReleaseEntry | None:
        """Return the entry for element ``z``, or ``None`` if it was not released."""
        return next((entry for entry in self.tables if entry.z == int(z)), None)


def load_release_index(path: Path | None = None) -> ReleaseIndex | None:
    """Return the shipped release index, or ``None`` if this build pins none."""
    source = release_index_path() if path is None else Path(path)
    if not source.is_file():
        return None
    return ReleaseIndex.from_record(json.loads(source.read_text(encoding="utf-8")))


def archive_member(key: str, suffix: str) -> str:
    """Return the archive member name for one file of table ``key``."""
    return f"{ARCHIVE_TABLE_DIR}/{key}{suffix}"


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_DIGEST_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def write_archive(archive: Path, tables: Iterable[StoredTable]) -> None:
    """Write ``tables`` into a zip whose bytes depend on their contents only."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        # The payloads are already deflated ``.npz`` files, so storing them
        # recompresses nothing and keeps the build fast.
        for table in sorted(tables, key=lambda table: table.key):
            manifest = table.path.with_suffix(".json")
            for source, suffix in ((manifest, ".json"), (table.path, ".npz")):
                info = zipfile.ZipInfo(archive_member(table.key, suffix), date_time=_ZIP_EPOCH)
                info.external_attr = 0o644 << 16
                bundle.writestr(info, source.read_bytes())


def build_release(
    out_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    elements: Iterable[int] | None = None,
    t1_max_MeV: float = COMPLETE_T1_MAX_MEV,
    urls: Iterable[str] = (),
) -> tuple[Path, ReleaseIndex]:
    """Build the release archive and its index from a BremsLib checkout.

    Maintainer-only: it needs the 810 MB library. Writes
    ``bremslib-tables.zip`` and ``bremslib-tables.json`` into ``out_dir``;
    the JSON is what gets committed to the wheel, the zip is what gets
    published at ``urls``.

    Parameters
    ----------
    out_dir
        Output directory. Created if absent; existing release files in it
        are replaced.
    source_path
        BremsLib checkout, resolved as for table generation.
    elements
        Atomic numbers to release. Defaults to :func:`catalogue_elements`.
    t1_max_MeV
        Energy bound for every table.
    urls
        Where the archive will be published, in fetch order, recorded in the
        index.

    Returns
    -------
    tuple of Path and ReleaseIndex
        The archive and its index.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source = resolve_source("bremslib", source_path)
    library = library_root(source.root)
    upstream = f"{source.spec.upstream}; library data {library.name}"
    zs = sorted({int(z) for z in (catalogue_elements() if elements is None else elements)})
    if not zs:
        raise ValueError("a BremsLib release needs at least one element")

    archive = destination / "bremslib-tables.zip"
    with tempfile.TemporaryDirectory(prefix=".bremslib-release-", dir=destination) as temp:
        staging = Path(temp)
        tables: list[StoredTable] = []
        for z in zs:
            request = element_request(library, z, t1_max_MeV=t1_max_MeV, model={"variant": VARIANT})
            arrays = release_arrays(build_table(library, z, t1_max_MeV=t1_max_MeV))
            tables.append(
                store(
                    request,
                    arrays,
                    compiler=None,
                    source_origin="release",
                    upstream=upstream,
                    modifications=RELEASE_MODIFICATIONS,
                    root=staging,
                )
            )
        partial = staging / archive.name
        write_archive(partial, tables)
        shutil.move(partial, archive)

    index = ReleaseIndex(
        upstream=upstream,
        variant=VARIANT,
        t1_max_MeV=float(t1_max_MeV),
        archive_sha256=file_sha256(archive),
        archive_bytes=archive.stat().st_size,
        urls=as_urls(urls),
        tables=tuple(
            ReleaseEntry(z=z, key=table.key, manifest_sha256=table.digest)
            for z, table in zip(zs, tables, strict=True)
        ),
    )
    (destination / "bremslib-tables.json").write_text(
        json.dumps(index.record(), indent=2) + "\n", encoding="utf-8"
    )
    return archive, index


def catalogue_table(z: int) -> StoredTable:
    """Return the released table for element ``z``.

    Raises
    ------
    TableNotFoundError
        If this build pins no release, the release does not cover ``z``, or
        the table has not been fetched -- each with the command that fixes it.
        Also if the installed table's manifest is not the pinned one, which
        would otherwise serve numbers the release did not ship.
    """
    index = load_release_index()
    entry = None if index is None else index.entry(z)
    if index is None or entry is None:
        raise TableNotFoundError(
            f"no released BremsLib table for Z={int(z)}; generate one from a BremsLib "
            f"checkout with `pyrite tables generate --code bremslib --element {int(z)}`"
        )
    table = resolve(entry.key)
    if table is None:
        raise TableNotFoundError(
            f"the released BremsLib table for Z={int(z)} is not installed; "
            "run `pyrite tables fetch bremslib`"
        )
    if table.digest != entry.manifest_sha256:
        raise TableNotFoundError(
            f"the installed BremsLib table for Z={int(z)} ({table.path}) does not match "
            "the pinned release; remove it and run `pyrite tables fetch bremslib`"
        )
    return table


__all__ = [
    "ARCHIVE_TABLE_DIR",
    "DROPPED_ARRAYS",
    "FLOAT32_ARRAYS",
    "RELEASE_MODIFICATIONS",
    "RELEASE_SCHEMA",
    "VARIANT",
    "ReleaseEntry",
    "ReleaseIndex",
    "archive_member",
    "build_release",
    "catalogue_elements",
    "catalogue_table",
    "file_sha256",
    "load_release_index",
    "release_arrays",
    "release_index_path",
    "write_archive",
]
