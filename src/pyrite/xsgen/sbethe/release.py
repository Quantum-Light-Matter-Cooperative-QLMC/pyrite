"""Build, pin, and look up the released SBETHE stopping tables.

Every runnable catalogue material and every amorphous medium used in target
stacks resolves one SBETHE table by material identity
(:func:`pyrite.xsgen.sbethe.catalog.resolve_catalog_table`). Generating them
needs gfortran and the fetched ``sdbase/``, so the set is released as one
archive and installed with ``pyrite tables fetch sbethe-tables``, like the
ELSEPA tables.

Two halves:

- **Maintainer.** :func:`build_release` resolves (or, with ``generate``,
  generates) every catalogue table and writes one deterministic zip plus an
  index recording the archive digest and each table's key and manifest
  digest. ``scripts/release_sbethe_tables.py`` drives it. Run it when the
  catalogue gains a material or medium, or a composition, the vendored
  SBETHE source, or the deck changes -- each re-keys tables.
- **User.** The index ships in the wheel; :func:`pyrite.xsgen.fetch.fetch_sbethe_tables`
  installs from it into the user table directory.

As for ELSEPA, a released table is exactly the table generation stores, under
the same key, and the key fixes the material inputs, vendored source, and
deck, so a locally generated table under it is an equally valid install.
"""

import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from ...materials import CATALOG
from ...paths import data_dir
from .._errors import TableNotFoundError
from .._urls import as_urls
from ..bremslib.release import file_sha256, write_archive
from ..elsepa.release import ElsepaReleaseEntry, ElsepaReleaseIndex
from ..store import StoredTable
from .catalog import catalog_material, resolve_catalog_table
from .generate import generate_material

#: Schema of the release index.
RELEASE_SCHEMA = "pyrite.xsgen.sbethe-release.v1"

#: Upstream deposit, as recorded in every generated manifest.
UPSTREAM = "SBETHE, Mendeley Data doi:10.17632/7zw25f428t.2"

#: One released table: ``label`` lists the catalogue keys that resolve to it.
SbetheReleaseEntry = ElsepaReleaseEntry


class SbetheReleaseIndex(ElsepaReleaseIndex):
    """The pinned description of one released SBETHE table archive."""

    SCHEMA: ClassVar[str] = RELEASE_SCHEMA
    NAME: ClassVar[str] = "SBETHE"


def release_index_path(*, projectile: str = "electron") -> Path:
    """Return the release index shipped inside the wheel."""
    if projectile not in ("electron", "positron"):
        raise ValueError("projectile must be electron or positron")
    suffix = "" if projectile == "electron" else "-positron"
    return data_dir() / "xsgen" / f"sbethe{suffix}-tables.json"


def load_release_index(
    path: Path | None = None, *, projectile: str = "electron"
) -> SbetheReleaseIndex | None:
    """Return the shipped release index, or ``None`` if this build pins none."""
    source = release_index_path(projectile=projectile) if path is None else Path(path)
    if not source.is_file():
        return None
    return SbetheReleaseIndex.from_record(json.loads(source.read_text(encoding="utf-8")))


def catalog_keys() -> tuple[str, ...]:
    """Return runnable materials plus amorphous media used in target stacks, sorted."""
    return tuple(sorted(set(CATALOG.materials) | set(CATALOG.media)))


def release_tables(
    keys: Iterable[str] | None = None, *, generate: bool = False, projectile: str = "electron"
) -> list[tuple[str, StoredTable]]:
    """Return ``(label, table)`` for every distinct catalogue table, by key.

    Several catalogue keys may share a table (a material and its crystal
    resolve the same composition); the label lists them all. With
    ``generate``, a missing table is generated; otherwise it is an error.
    """
    by_key: dict[str, tuple[list[str], StoredTable]] = {}
    for name in catalog_keys() if keys is None else keys:
        if generate:
            try:
                resolve_catalog_table(name, projectile=projectile)
            except TableNotFoundError:
                material = catalog_material(name)
                generate_material(
                    material.key,
                    material.composition,
                    density_g_cm3=material.density_g_cm3,
                    mean_excitation_eV=material.mean_excitation_eV,
                    band_gap_eV=material.band_gap_eV,
                    projectile=projectile,
                )
        table = resolve_catalog_table(name, projectile=projectile)
        by_key.setdefault(table.key, ([], table))[0].append(name)
    return [(",".join(sorted(names)), table) for names, table in by_key.values()]


def build_release(
    out_dir: str | Path,
    *,
    urls: Iterable[str] = (),
    generate: bool = False,
    keys: Iterable[str] | None = None,
    projectile: str = "electron",
) -> tuple[Path, SbetheReleaseIndex]:
    """Build the release archive and its index from stored catalogue tables.

    Writes ``sbethe-tables.zip`` and ``sbethe-tables.json`` into ``out_dir``;
    the JSON is committed to the wheel and the zip is published at ``urls``.
    ``projectile="positron"`` uses ``sbethe-positron-tables`` for both files,
    preserving the electron release. Both species use the existing table
    schema; their request keys identify the projectile.
    """
    if projectile not in ("electron", "positron"):
        raise ValueError("projectile must be electron or positron")
    suffix = "" if projectile == "electron" else "-positron"
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labelled = sorted(
        release_tables(keys, generate=generate, projectile=projectile), key=lambda row: row[0]
    )
    archive = destination / f"sbethe{suffix}-tables.zip"
    with tempfile.TemporaryDirectory(prefix=".sbethe-release-", dir=destination) as temp:
        partial = Path(temp) / archive.name
        write_archive(partial, [table for _, table in labelled])
        shutil.move(partial, archive)
    index = SbetheReleaseIndex(
        upstream=UPSTREAM,
        archive_sha256=file_sha256(archive),
        archive_bytes=archive.stat().st_size,
        urls=as_urls(urls),
        tables=tuple(
            SbetheReleaseEntry(label=label, key=table.key, manifest_sha256=table.digest)
            for label, table in labelled
        ),
    )
    (destination / f"sbethe{suffix}-tables.json").write_text(
        json.dumps(index.record(), indent=2) + "\n", encoding="utf-8"
    )
    return archive, index


__all__ = [
    "RELEASE_SCHEMA",
    "UPSTREAM",
    "SbetheReleaseEntry",
    "SbetheReleaseIndex",
    "build_release",
    "catalog_keys",
    "load_release_index",
    "release_index_path",
    "release_tables",
]
