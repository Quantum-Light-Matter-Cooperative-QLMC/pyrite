"""Build, pin, and look up the released ELSEPA elastic tables.

The production elastic model needs a free-atom table for every element a
runnable catalogue material may contain, plus a muffin-tin table for every
elementary catalogue crystal (:mod:`pyrite.xsgen.elsepa.catalog`). Generating
them compiles and runs ELSEPA for minutes per element, and the set is about
25 MB -- too large for the wheel -- so it is released like the BremsLib
tables and installed with ``pyrite tables fetch elsepa``.

Two halves:

- **Maintainer.** :func:`build_release` resolves (or, with ``generate``,
  generates) every production table and writes one deterministic zip plus an
  index recording the archive digest and each table's key and manifest
  digest. ``scripts/release_elsepa_tables.py`` drives it. Run it when the
  catalogue gains an element or elementary crystal, or the vendored ELSEPA
  source, deck, or energy grid changes -- each re-keys tables.
- **User.** The index ships in the wheel; :func:`pyrite.xsgen.fetch.fetch_elsepa`
  installs from it into the user table directory.

Unlike BremsLib there is no release transform: a released table is exactly
the table generation stores, under the same key. The key covers the vendored
source digest and the deck, not the machine, so a table generated locally
from the same deck is an equally valid install of that key, and runtime
resolution (:func:`pyrite.xsgen.elsepa.catalog.resolve_layer_tables`) needs
no index.
"""

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...materials import CATALOG
from ...materials._transport_data import TRANSPORT_ELEMENTS
from ...paths import data_dir
from .._errors import TableNotFoundError
from .._urls import archive_urls, as_urls
from ..bremslib.release import file_sha256, write_archive
from ..store import StoredTable, resolve
from .catalog import (
    PRODUCTION_ENERGIES_EV,
    _muffin_tin_energies,
    elemental_solid,
)
from .generate import (
    element_request,
    generate_element,
    generate_muffin_tin,
    muffin_tin_request,
)

#: Schema of the release index.
RELEASE_SCHEMA = "pyrite.xsgen.elsepa-release.v1"

#: Upstream deposit, as recorded in every generated manifest.
UPSTREAM = "ELSEPA 2020, Mendeley Data doi:10.17632/w4hm5vymym.1"


@dataclass(frozen=True)
class ElsepaReleaseEntry:
    """One released table: what it describes, its key, and its manifest digest.

    ``label`` is ``"Z=<z>"`` for a free atom and the crystal key for a
    muffin-tin table.
    """

    label: str
    key: str
    manifest_sha256: str


@dataclass(frozen=True)
class ElsepaReleaseIndex:
    """The pinned description of one released ELSEPA table archive.

    ``urls`` is empty until the archive is published and is tried in order
    (see :mod:`pyrite.xsgen._urls`); installing from a local copy works
    either way.
    """

    upstream: str
    archive_sha256: str
    archive_bytes: int
    urls: tuple[str, ...]
    tables: tuple[ElsepaReleaseEntry, ...]

    def record(self) -> dict[str, Any]:
        """Return the JSON-serializable index body."""
        return {
            "schema": RELEASE_SCHEMA,
            "upstream": self.upstream,
            "archive": {
                "sha256": self.archive_sha256,
                "bytes": self.archive_bytes,
                "urls": list(self.urls),
            },
            "tables": [
                {"label": e.label, "key": e.key, "manifest_sha256": e.manifest_sha256}
                for e in self.tables
            ],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ElsepaReleaseIndex:
        """Parse an index body, rejecting any other schema."""
        if record.get("schema") != RELEASE_SCHEMA:
            raise ValueError(
                f"unsupported ELSEPA release index schema {record.get('schema')!r}; "
                f"expected {RELEASE_SCHEMA!r}"
            )
        archive = record["archive"]
        return cls(
            upstream=str(record["upstream"]),
            archive_sha256=str(archive["sha256"]),
            archive_bytes=int(archive["bytes"]),
            urls=archive_urls(archive),
            tables=tuple(
                ElsepaReleaseEntry(
                    label=str(row["label"]),
                    key=str(row["key"]),
                    manifest_sha256=str(row["manifest_sha256"]),
                )
                for row in record["tables"]
            ),
        )


def release_index_path() -> Path:
    """Return the release index shipped inside the wheel."""
    return data_dir() / "xsgen" / "elsepa-tables.json"


def load_release_index(path: Path | None = None) -> ElsepaReleaseIndex | None:
    """Return the shipped release index, or ``None`` if this build pins none."""
    source = release_index_path() if path is None else Path(path)
    if not source.is_file():
        return None
    return ElsepaReleaseIndex.from_record(json.loads(source.read_text(encoding="utf-8")))


def elementary_crystals() -> tuple[str, ...]:
    """Return every catalogue crystal that takes a muffin-tin table, sorted."""
    return tuple(sorted(key for key in CATALOG.crystals if elemental_solid(key) is not None))


def _required(table: StoredTable | None, label: str) -> StoredTable:
    if table is None:
        raise TableNotFoundError(
            f"no ELSEPA table for {label}; build the release with generation enabled"
        )
    return table


def release_tables(*, generate: bool = False) -> list[tuple[str, StoredTable]]:
    """Return ``(label, table)`` for every production table, in index order.

    Free atoms for every transport element by atomic number, then muffin-tin
    tables by crystal key. With ``generate``, a missing table is generated;
    otherwise it is an error.
    """
    out = []
    for z in sorted(int(row["Z"]) for row in TRANSPORT_ELEMENTS.values()):
        request, _ = element_request(z, PRODUCTION_ENERGIES_EV)
        table = resolve(request.key)
        if table is None and generate:
            table = generate_element(z, PRODUCTION_ENERGIES_EV).table
        out.append((f"Z={z}", _required(table, f"Z={z}")))
    for key in elementary_crystals():
        solid = elemental_solid(key)
        assert solid is not None
        energies = _muffin_tin_energies()
        radius, density = solid.radius_cm, solid.density_g_cm3
        request, _ = muffin_tin_request(
            key, solid.z, energies, radius_cm=radius, density_g_cm3=density
        )
        table = resolve(request.key)
        if table is None and generate:
            table = generate_muffin_tin(
                key, solid.z, energies, radius_cm=radius, density_g_cm3=density
            ).table
        out.append((key, _required(table, key)))
    return out


def build_release(
    out_dir: str | Path, *, urls: Iterable[str] = (), generate: bool = False
) -> tuple[Path, ElsepaReleaseIndex]:
    """Build the release archive and its index from stored production tables.

    Writes ``elsepa-tables.zip`` and ``elsepa-tables.json`` into ``out_dir``;
    the JSON is committed to the wheel and the zip is published at ``urls``.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labelled = release_tables(generate=generate)
    archive = destination / "elsepa-tables.zip"
    with tempfile.TemporaryDirectory(prefix=".elsepa-release-", dir=destination) as temp:
        partial = Path(temp) / archive.name
        write_archive(partial, [table for _, table in labelled])
        shutil.move(partial, archive)
    index = ElsepaReleaseIndex(
        upstream=UPSTREAM,
        archive_sha256=file_sha256(archive),
        archive_bytes=archive.stat().st_size,
        urls=as_urls(urls),
        tables=tuple(
            ElsepaReleaseEntry(label=label, key=table.key, manifest_sha256=table.digest)
            for label, table in labelled
        ),
    )
    (destination / "elsepa-tables.json").write_text(
        json.dumps(index.record(), indent=2) + "\n", encoding="utf-8"
    )
    return archive, index


__all__ = [
    "RELEASE_SCHEMA",
    "UPSTREAM",
    "ElsepaReleaseEntry",
    "ElsepaReleaseIndex",
    "build_release",
    "elementary_crystals",
    "load_release_index",
    "release_index_path",
    "release_tables",
]
