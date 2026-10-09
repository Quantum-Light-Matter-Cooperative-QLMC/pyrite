"""Key, store, and resolve generated cross-section tables.

One namespace, one key (D2). A table is identified by the SHA-256 of a
normalized request record -- code, code version, target, model parameters --
so "is this the table I asked for" and "has anything it depends on changed"
are the same question. Change the deck, patch the Fortran, or edit the
material, and the key changes; a stale table is then not found rather than
silently served.

Resolution is two-tier (D3): the user table directory first, then the packaged
one. Consumers call :func:`resolve` and never learn which tier answered, so a
shipped table and a user-generated one are indistinguishable to
:mod:`pyrite.montecarlo.transport.lut` and its peers. With an explicit
workspace the pre-workspace ``<user data dir>/xsgen/tables`` is no longer
searched (removed in 0.5.0, ADR-0014); ``pyrite tables migrate`` copies tables
from it into the selected workspace.

Each table carries a sidecar JSON manifest recording how it was made (D5):
source digest, compiler version, deck hash, model parameters, PyRITE version,
timestamp, and the CC BY modifications note that redistributing an adaptation
of ELSEPA, SBETHE, or BremsLib output requires. The manifest's own digest is
what feeds run identity, so regenerating a table with different deck
parameters cannot resume a checkpoint computed from the old one.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..paths import data_dir, user_data_dir
from ._errors import TableNotFoundError

#: Schema of the request record hashed into a table key. Bumping this
#: re-keys every table, so it changes only when the record's *meaning*
#: changes, never when a field is added to a model payload.
KEY_SCHEMA = "pyrite.xsgen.key.v1"

#: Schema of the sidecar manifest.
MANIFEST_SCHEMA = "pyrite.xsgen.manifest.v1"

#: Attribution text stamped into every manifest. ELSEPA, SBETHE, and BremsLib
#: all carry a Creative Commons BY clause, which requires derived material to
#: be marked as derived and its modifications indicated. A table parsed out of
#: a code's output and restructured is an adaptation, so this is a licensing
#: obligation, not a courtesy. It describes what every generator does; a table
#: modified further -- a reduced-precision release -- says so through
#: ``store(modifications=...)``.
MODIFICATIONS_NOTE = (
    "Derived from the named external code's output: parsed from its output "
    "files and stored on the code's own grids, with derived quantities such as "
    "integrals or sampling CDFs added. Not the upstream data. See "
    "THIRD-PARTY-NOTICES.md for the upstream author, citation, and licence."
)


def _canonical(payload: Any) -> str:
    """Render ``payload`` as the one JSON spelling that is hashed.

    Sorted keys and no whitespace, matching how
    :func:`pyrite.campaign.profiles.dataset_identity` encodes run identity --
    the same discipline for the same reason: two records that mean the same
    thing must produce the same bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not part of a table key")


def arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Return a SHA-256 over the stored array payload itself.

    The rest of the manifest describes how a table was *made*; this describes
    what it *contains*. Without it two tables can share a manifest digest
    despite holding different numbers -- same code, same deck, same shapes and
    dtypes, written within the same second, which is exactly what regenerating
    a table looks like. Since that digest is what invalidates checkpoints, the
    collision would serve results computed from the superseded table.

    Byte order is normalized to little-endian so a manifest shipped with a
    table verifies on a big-endian host rather than appearing corrupt.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name])
        little = values.astype(values.dtype.newbyteorder("<"), copy=False)
        for part in (
            name.encode("utf-8"),
            str(values.dtype).encode("utf-8"),
            repr(values.shape).encode("utf-8"),
        ):
            digest.update(part)
            digest.update(b"\0")
        digest.update(little.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ElementTarget:
    """A free atom, identified by atomic number.

    The right target for free-atom ELSEPA (``MUFFIN 0``) and for BremsLib,
    both of which are per-element.
    """

    z: int

    def record(self) -> dict[str, Any]:
        """Return the hashable spelling of this target."""
        return {"kind": "element", "z": int(self.z)}

    def label(self) -> str:
        """Return a short human-readable target name."""
        return f"Z={self.z}"


@dataclass(frozen=True)
class MaterialTarget:
    """A material, identified by a digest of the properties a deck consumes.

    The catalogue *key* is carried for display only and is deliberately not
    what identifies the target. Keying on the name would not change when the
    catalogue's composition or density for that name changed, which is
    exactly the stale-table case the key exists to catch.

    Parameters
    ----------
    key
        Catalogue key, for display.
    identity
        Digest from :func:`material_identity`.
    """

    key: str
    identity: str

    def record(self) -> dict[str, Any]:
        """Return the hashable spelling of this target."""
        return {"kind": "material", "identity": self.identity}

    def label(self) -> str:
        """Return a short human-readable target name."""
        return self.key


Target = ElementTarget | MaterialTarget


def material_identity(
    *,
    composition: Mapping[int, float],
    density_g_cm3: float,
    mean_excitation_eV: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Return the identity digest of a material, for :class:`MaterialTarget`.

    Hashes what the external decks actually consume, not what the material is
    called. ``composition`` is atomic number to atom fraction; fractions are
    renormalized so that two spellings of the same stoichiometry -- ``{42: 1,
    16: 2}`` and ``{42: 1/3, 16: 2/3}`` -- agree.

    ``extra`` carries code-specific material scalars that are still properties
    of the material rather than of the deck: ELSEPA's muffin-tin radius from
    the nearest-neighbour distance, and the band gap feeding its absorption
    potential. They belong here because changing them changes the material
    the table describes.

    Notes
    -----
    Spec D2 states this digest comes from ``pyrite.materials._identity``. It
    does not: that module derives a *display* identity -- formula, phase,
    slab cut -- and hashes nothing. The digest is defined here instead. See
    the task record for issue #161.

    Parameters
    ----------
    composition
        Atomic number to atom fraction. Must be non-empty with a positive
        total.
    density_g_cm3
        Mass density.
    mean_excitation_eV
        Mean excitation energy, where the code takes one (SBETHE). ``None``
        where it does not.
    extra
        Further material scalars, hashed as given.

    Returns
    -------
    str
        Hex SHA-256 digest.

    Raises
    ------
    ValueError
        If the composition is empty, carries a non-positive fraction, or the
        density is not positive.
    """
    if not composition:
        raise ValueError("material composition is empty")
    if any(fraction <= 0.0 for fraction in composition.values()):
        raise ValueError("material composition has a non-positive atom fraction")
    if not density_g_cm3 > 0.0:
        raise ValueError(f"material density must be positive, got {density_g_cm3!r}")
    total = float(sum(composition.values()))
    record = {
        "schema": "pyrite.xsgen.material.v1",
        "composition": {str(int(z)): float(n) / total for z, n in sorted(composition.items())},
        "density_g_cm3": float(density_g_cm3),
        "mean_excitation_eV": None if mean_excitation_eV is None else float(mean_excitation_eV),
        "extra": dict(extra or {}),
    }
    return _digest(record)


@dataclass(frozen=True)
class TableRequest:
    """The normalized record a table key is computed from.

    Parameters
    ----------
    code
        Which external code produces the table.
    code_version
        Digest of that code's sources, from
        :func:`pyrite.xsgen.sources.source_digest`.
    target
        Element or material the table describes.
    quantity
        Which of the code's outputs this table holds, e.g. ``"elastic_dcs"``.
        One program emits several tables, so the key must separate them.
    model
        Deck parameters. Everything that changes the numbers belongs here;
        anything that does not -- output verbosity, file names -- must not,
        or identical physics will key twice.
    """

    code: str
    code_version: str
    target: Target
    quantity: str
    model: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """Return the exact record that :attr:`key` hashes."""
        return {
            "schema": KEY_SCHEMA,
            "code": self.code,
            "code_version": self.code_version,
            "quantity": self.quantity,
            "target": self.target.record(),
            "model": dict(self.model),
        }

    @property
    def key(self) -> str:
        """Hex SHA-256 identifying this table."""
        return _digest(self.record())

    @property
    def deck_hash(self) -> str:
        """Hex SHA-256 of the model parameters alone.

        Recorded separately in the manifest so two tables differing only in
        deck can be told apart without recomputing the whole key.
        """
        return _digest(dict(self.model))


@dataclass(frozen=True)
class TableManifest:
    """Provenance for one stored table.

    Every field answers "would this table be different if that changed?".
    :attr:`digest` over the whole record is what feeds run identity, so any
    of them moving invalidates checkpoints computed from the old table.
    """

    key: str
    code: str
    code_version: str
    quantity: str
    target: Mapping[str, Any]
    target_label: str
    model: Mapping[str, Any]
    deck_hash: str
    compiler: str | None
    pyrite_version: str
    created_utc: str
    arrays: Mapping[str, Any]
    arrays_sha256: str
    source_origin: str | None = None
    upstream: str | None = None
    modifications: str = MODIFICATIONS_NOTE
    schema: str = MANIFEST_SCHEMA

    def record(self) -> dict[str, Any]:
        """Return the JSON-serializable manifest body."""
        return {
            "schema": self.schema,
            "key": self.key,
            "code": self.code,
            "code_version": self.code_version,
            "quantity": self.quantity,
            "target": dict(self.target),
            "target_label": self.target_label,
            "model": dict(self.model),
            "deck_hash": self.deck_hash,
            "compiler": self.compiler,
            "pyrite_version": self.pyrite_version,
            "created_utc": self.created_utc,
            "arrays": dict(self.arrays),
            "arrays_sha256": self.arrays_sha256,
            "source_origin": self.source_origin,
            "upstream": self.upstream,
            "modifications": self.modifications,
        }

    @property
    def digest(self) -> str:
        """Hex SHA-256 of the manifest record.

        Covers both how the table was made and, through
        :attr:`arrays_sha256`, what it holds. A regeneration that produced
        identical numbers at a different timestamp still re-keys, which costs
        recomputation; the converse -- serving numbers from a table that no
        longer exists -- is the failure this exists to prevent.
        """
        return _digest(self.record())


def user_table_dir() -> Path:
    """Return the writable table directory in the selected workspace."""
    from ..console.config import xsgen_data_root

    return xsgen_data_root() / "tables"


def packaged_table_dir() -> Path:
    """Return the packaged table directory shipped inside the wheel."""
    return data_dir() / "xsgen" / "tables"


def legacy_table_dir() -> Path:
    """Return the pre-workspace table directory, the source of ``tables migrate``.

    ``<user data dir>/xsgen/tables``. When no workspace is selected this *is*
    the selected tier; with an explicit workspace it is no longer searched.
    """
    return user_data_dir() / "xsgen" / "tables"


def _legacy_tier_active() -> bool:
    return legacy_table_dir() != user_table_dir()


def search_dirs() -> tuple[Path, ...]:
    """Return the resolution tiers, most-preferred first."""
    return tuple(dict.fromkeys((user_table_dir(), packaged_table_dir())))


def _tier(root: Path) -> str:
    if root == packaged_table_dir():
        return "packaged"
    return "user"


@dataclass(frozen=True)
class StoredTable:
    """A table found on disk, with its provenance and its tier.

    Parameters
    ----------
    key
        Table key.
    path
        The ``.npz`` payload.
    manifest
        Parsed sidecar manifest.
    tier
        ``"user"`` or ``"packaged"``. Display and
        diagnostics only; no consumer branches on it.
    """

    key: str
    path: Path
    manifest: Mapping[str, Any]
    tier: str

    @property
    def digest(self) -> str:
        """Manifest digest, as recorded when the table was written."""
        return str(self.manifest["manifest_sha256"])

    def arrays(self) -> dict[str, np.ndarray]:
        """Load the stored arrays."""
        with np.load(self.path) as loaded:
            return {name: loaded[name] for name in loaded.files}


def _paths_for(root: Path, key: str) -> tuple[Path, Path]:
    return root / f"{key}.npz", root / f"{key}.json"


def resolve(key: str) -> StoredTable | None:
    """Return the stored table for ``key``, or ``None`` if there is none.

    Checks the user directory before the packaged one (D3). A table present
    in both is served from the user directory, so a locally regenerated table
    overrides a shipped one without having to delete it.
    """
    for root in search_dirs():
        tier = _tier(root)
        payload, manifest_path = _paths_for(root, key)
        if not (payload.is_file() and manifest_path.is_file()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return StoredTable(key=key, path=payload, manifest=manifest, tier=tier)
    return None


def require(key: str, *, hint: str = "") -> StoredTable:
    """Return the stored table for ``key``, raising when absent.

    Raises
    ------
    TableNotFoundError
        Naming both tiers searched, so the user can see whether the table was
        expected to be shipped or generated.
    """
    found = resolve(key)
    if found is not None:
        return found
    tiers = "\n".join(f"  - {root}" for root in search_dirs())
    suffix = f"\n{hint}" if hint else ""
    raise TableNotFoundError(
        f"no cross-section table {key} in:\n{tiers}\n"
        f"generate it with `pyrite tables generate`{suffix}"
    )


def store(
    request: TableRequest,
    arrays: Mapping[str, np.ndarray],
    *,
    compiler: str | None = None,
    source_origin: str | None = None,
    upstream: str | None = None,
    modifications: str = MODIFICATIONS_NOTE,
    root: Path | None = None,
    overwrite: bool = False,
) -> StoredTable:
    """Write a table and its manifest into the user table directory.

    Parameters
    ----------
    request
        The request that produced ``arrays``; supplies the key.
    arrays
        Named arrays on the code's native grid. Not resampled onto PyRITE's
        electron energy grid: that grid is configurable, so baking it in here
        would make every table stale on a grid change. Resampling happens at
        load.
    compiler, source_origin, upstream
        Provenance recorded in the manifest.
    modifications
        How the table differs from the upstream output, recorded for the
        CC BY modifications term. Defaults to :data:`MODIFICATIONS_NOTE`.
    root
        Destination directory. Defaults to :func:`user_table_dir`; the
        packaged directory is written only by the maintainer-side release
        step.
    overwrite
        Replace an existing table with this key.

    Returns
    -------
    StoredTable
        The table as written.

    Raises
    ------
    FileExistsError
        If the key is already stored and ``overwrite`` is false.
    ValueError
        If ``arrays`` is empty.
    """
    if not arrays:
        raise ValueError("refusing to store a table with no arrays")
    destination = user_table_dir() if root is None else Path(root)
    payload, manifest_path = _paths_for(destination, request.key)
    if payload.exists() and not overwrite:
        raise FileExistsError(f"table {request.key} already stored at {payload}")

    # ``Any`` rather than ``ndarray``: ``savez_compressed`` takes its arrays as
    # ``**kwds``, which the stub cannot distinguish from its ``allow_pickle``
    # keyword, so a precisely typed mapping is rejected at the call below.
    prepared: dict[str, Any] = {name: np.asarray(value) for name, value in arrays.items()}
    manifest = TableManifest(
        key=request.key,
        code=request.code,
        code_version=request.code_version,
        quantity=request.quantity,
        target=request.target.record(),
        target_label=request.target.label(),
        model=dict(request.model),
        deck_hash=request.deck_hash,
        compiler=compiler,
        pyrite_version=__version__,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        arrays={
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(prepared.items())
        },
        arrays_sha256=arrays_digest(prepared),
        source_origin=source_origin,
        upstream=upstream,
        modifications=modifications,
    )
    body = manifest.record()
    # The digest is over the record *without* itself, then carried inside the
    # stored document so a reader need not recompute it to compare identities.
    body["manifest_sha256"] = manifest.digest

    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(payload, **prepared)
    manifest_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return StoredTable(key=request.key, path=payload, manifest=body, tier="user")


def manifest_digest(body: Mapping[str, Any]) -> str:
    """Recompute the digest of a stored manifest document.

    The stored document carries its own ``manifest_sha256``, computed over
    everything else; this recomputes it, so a manifest arriving from outside
    -- a fetched release -- can be checked rather than trusted.
    """
    return _digest({name: value for name, value in body.items() if name != "manifest_sha256"})


def identity_markers(tables: Iterable[StoredTable]) -> dict[str, str]:
    """Map each table's key to its provenance-manifest digest, for run identity.

    What a run records so that changing a table it read invalidates the
    results computed from it. The manifest digest covers the Fortran source,
    the compiler, the deck, the model parameters and the PyRITE version, so
    every way a table can become a different table moves this value.

    Both identity surfaces take it: :func:`pyrite.campaign.profiles.dataset_identity`,
    which gates the checkpoint stem, and
    :func:`pyrite.campaign.profiles.case_content_key`, which gates the
    content-addressable blob store. Passing it to only one leaves the other
    serving results computed from the superseded table.

    An empty result means the run read no generated table, and both surfaces
    then leave their digests exactly as they were.
    """
    return {table.key: table.digest for table in tables}


def iter_stored() -> Iterator[StoredTable]:
    """Yield every stored table, user tier first, without duplicate keys."""
    seen: set[str] = set()
    for root in search_dirs():
        tier = _tier(root)
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.glob("*.json")):
            key = manifest_path.stem
            payload = root / f"{key}.npz"
            if key in seen or not payload.is_file():
                continue
            seen.add(key)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            yield StoredTable(key=key, path=payload, manifest=manifest, tier=tier)


@dataclass(frozen=True)
class MigrationReport:
    """Outcome of :func:`migrate_legacy_tables`.

    ``copied`` lists keys copied (or, in a dry run, that would be);
    ``present`` keys already in the destination, which are never replaced.
    """

    source: Path
    destination: Path
    copied: tuple[str, ...]
    present: tuple[str, ...]
    dry_run: bool
    active: bool


def migrate_legacy_tables(*, dry_run: bool = False) -> MigrationReport:
    """Copy legacy-tier tables into the selected table directory.

    Non-destructive: legacy files are never modified or deleted, and a key the
    destination already holds is left alone (it already wins resolution).
    Each pair is copied payload first, each file through a temporary name and
    an atomic rename, so an interrupted run leaves at most a payload without a
    manifest, which :func:`resolve` ignores. Inactive (nothing to do) when no
    workspace is selected, because the legacy directory is then the selected
    one.
    """
    source = legacy_table_dir()
    destination = user_table_dir()
    active = _legacy_tier_active()
    copied: list[str] = []
    present: list[str] = []
    if active and source.is_dir():
        for manifest_path in sorted(source.glob("*.json")):
            key = manifest_path.stem
            payload = source / f"{key}.npz"
            if not payload.is_file():
                continue
            target_payload, target_manifest = _paths_for(destination, key)
            if target_payload.is_file() and target_manifest.is_file():
                present.append(key)
                continue
            copied.append(key)
            if dry_run:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for src, dst in ((payload, target_payload), (manifest_path, target_manifest)):
                fd, temporary = tempfile.mkstemp(dir=destination, prefix=f".{dst.name}.")
                os.close(fd)
                try:
                    shutil.copyfile(src, temporary)
                    os.replace(temporary, dst)
                except BaseException:
                    Path(temporary).unlink(missing_ok=True)
                    raise
    return MigrationReport(
        source=source,
        destination=destination,
        copied=tuple(copied),
        present=tuple(present),
        dry_run=dry_run,
        active=active,
    )


__all__ = [
    "KEY_SCHEMA",
    "MANIFEST_SCHEMA",
    "MODIFICATIONS_NOTE",
    "ElementTarget",
    "MaterialTarget",
    "MigrationReport",
    "StoredTable",
    "TableManifest",
    "TableRequest",
    "Target",
    "arrays_digest",
    "identity_markers",
    "iter_stored",
    "legacy_table_dir",
    "manifest_digest",
    "material_identity",
    "migrate_legacy_tables",
    "packaged_table_dir",
    "require",
    "resolve",
    "search_dirs",
    "store",
    "user_table_dir",
]
