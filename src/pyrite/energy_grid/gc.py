"""Reachability-safe garbage collection for immutable energy-grid artifacts.

Planning records orphan age outside immutable artifact bytes.  Deletion is a
separate explicit operation and fails closed whenever any reachability input
changed after the plan was made.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from pyrite import _energy_grid_artifacts as artifacts
from pyrite.checkpoints import campaign_lock

DEFAULT_GRACE = timedelta(days=14)
"""Default retention period for newly unreachable artifacts."""

METADATA_SCHEMA = "cxr.energy-grid-gc-metadata.v1"
METADATA_NAME = ".gc-metadata.json"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactGCError(RuntimeError):
    """Fail-closed reachability, metadata, or concurrent-change error."""


@dataclass(frozen=True)
class FileSnapshot:
    """Path and content signature captured in a GC plan."""

    path: Path
    digest: str


@dataclass(frozen=True)
class GcCandidate:
    """One orphan approved for deletion by a particular immutable plan."""

    digest: str
    path: Path
    orphaned_at: float


@dataclass(frozen=True)
class GcPlan:
    """Immutable snapshot required by :func:`execute_gc`."""

    catalog_path: Path
    checkpoint_dir: Path
    store_root: Path
    roots: tuple[str, ...]
    inventory: tuple[FileSnapshot, ...]
    catalog: FileSnapshot
    locks: tuple[FileSnapshot, ...]
    metadata: FileSnapshot | None
    candidates: tuple[GcCandidate, ...]
    retained: tuple[str, ...]
    prune_all: bool


@dataclass(frozen=True)
class VerificationIssue:
    """Missing or corrupt artifact found by :func:`verify_artifacts`."""

    source: str
    digest: str
    message: str


@dataclass(frozen=True)
class VerificationReport:
    """Artifact integrity report; malformed reachability inputs raise instead."""

    roots: tuple[str, ...]
    inventory: tuple[str, ...]
    issues: tuple[VerificationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def metadata_path(store_root: Path | str) -> Path:
    """Return the out-of-band orphan metadata location for one store."""
    return Path(store_root) / METADATA_NAME


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        path.lstat()
    except FileNotFoundError:
        raise ArtifactGCError(f"{label} is missing: {path}") from None
    if path.is_symlink() or not path.is_file():
        raise ArtifactGCError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArtifactGCError(f"cannot read {label} {path}: {exc}") from exc


def _snapshot(path: Path, *, label: str) -> FileSnapshot:
    return FileSnapshot(
        path=path, digest=hashlib.sha256(_regular_bytes(path, label=label)).hexdigest()
    )


def _optional_snapshot(path: Path, *, label: str) -> FileSnapshot | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _snapshot(path, label=label)


def _validate_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ArtifactGCError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _catalog_roots(catalog_path: Path) -> tuple[set[str], FileSnapshot]:
    data = _regular_bytes(catalog_path, label="catalog")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ArtifactGCError(f"invalid catalog {catalog_path}: {exc}") from None
    profiles = raw.get("profiles") if isinstance(raw, Mapping) else None
    if not isinstance(profiles, Mapping):
        raise ArtifactGCError(f"invalid catalog {catalog_path}: profiles must be a table")
    roots: set[str] = set()
    for profile, row in profiles.items():
        if not isinstance(profile, str) or not isinstance(row, Mapping):
            raise ArtifactGCError(
                f"invalid catalog {catalog_path}: profiles entries must be tables"
            )
        if "energy_grid_refs" not in row:
            continue
        refs = row["energy_grid_refs"]
        if not isinstance(refs, Mapping):
            raise ArtifactGCError(
                f"invalid catalog {catalog_path}: profiles.{profile}.energy_grid_refs must be a table"
            )
        for material, digest in refs.items():
            if not isinstance(material, str) or not material:
                raise ArtifactGCError(
                    f"invalid catalog {catalog_path}: empty energy-grid material ref"
                )
            roots.add(
                _validate_digest(
                    digest,
                    label=f"catalog profiles.{profile}.energy_grid_refs.{material}",
                )
            )
    return roots, FileSnapshot(catalog_path, hashlib.sha256(data).hexdigest())


def _lock_paths(checkpoint_dir: Path) -> tuple[Path, ...]:
    """Find exact campaign-lock names without traversing symlinked directories."""
    if not checkpoint_dir.exists() and not checkpoint_dir.is_symlink():
        return ()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ArtifactGCError(
            f"checkpoint directory must be a non-symlink directory: {checkpoint_dir}"
        )
    found: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        walk(path)
                    elif entry.is_file(follow_symlinks=False) and (
                        entry.name == "cxr.lock.json" or entry.name.endswith(".lock.json")
                    ):
                        found.append(path)
        except OSError as exc:
            raise ArtifactGCError(f"cannot scan checkpoint directory {directory}: {exc}") from exc

    walk(checkpoint_dir)
    return tuple(sorted(found))


def _lock_roots(
    checkpoint_dir: Path,
) -> tuple[set[str], tuple[FileSnapshot, ...], dict[str, set[str]]]:
    roots: set[str] = set()
    sources: dict[str, set[str]] = {}
    snapshots: list[FileSnapshot] = []
    for path in _lock_paths(checkpoint_dir):
        _regular_bytes(path, label="campaign lock")
        try:
            payload = campaign_lock.read_lock(path)
        except ValueError as exc:
            raise ArtifactGCError(str(exc)) from None
        grid_refs = payload["artifacts"]["energy_grid"]
        if not isinstance(grid_refs, Mapping):  # Defensive: read_lock validates this today.
            raise ArtifactGCError(
                f"invalid campaign lock {path}: artifacts.energy_grid must be a table"
            )
        for material, digest in grid_refs.items():
            if not isinstance(material, str) or not material:
                raise ArtifactGCError(
                    f"invalid campaign lock {path}: empty energy-grid material key"
                )
            value = _validate_digest(
                digest, label=f"campaign lock {path} artifacts.energy_grid.{material}"
            )
            roots.add(value)
            sources.setdefault(value, set()).add(str(path))
        snapshots.append(_snapshot(path, label="campaign lock"))
    return roots, tuple(snapshots), sources


def reachability_roots(catalog_path: Path | str, checkpoint_dir: Path | str) -> tuple[str, ...]:
    """Return all profile and active/archive campaign-lock artifact roots."""
    catalog_roots, _ = _catalog_roots(Path(catalog_path))
    lock_roots, _, _ = _lock_roots(Path(checkpoint_dir))
    return tuple(sorted(catalog_roots | lock_roots))


def _read_metadata(store_root: Path) -> dict[str, float]:
    path = metadata_path(store_root)
    if not path.exists() and not path.is_symlink():
        return {}
    data = _regular_bytes(path, label="GC metadata")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactGCError(f"invalid GC metadata {path}: {exc}") from None
    if not isinstance(payload, Mapping) or payload.get("schema") != METADATA_SCHEMA:
        raise ArtifactGCError(f"invalid GC metadata {path}: expected schema {METADATA_SCHEMA}")
    orphans = payload.get("orphans")
    if not isinstance(orphans, Mapping):
        raise ArtifactGCError(f"invalid GC metadata {path}: orphans must be a table")
    output: dict[str, float] = {}
    for digest, timestamp in orphans.items():
        key = _validate_digest(digest, label=f"GC metadata {path} orphan key")
        if isinstance(timestamp, bool):
            raise ArtifactGCError(f"invalid GC metadata {path}: orphan timestamp must be finite")
        try:
            number = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ArtifactGCError(
                f"invalid GC metadata {path}: orphan timestamp must be finite"
            ) from exc
        if not math.isfinite(number) or number < 0:
            raise ArtifactGCError(f"invalid GC metadata {path}: orphan timestamp must be finite")
        output[key] = number
    return output


def _write_metadata(store_root: Path, orphans: Mapping[str, float]) -> None:
    if store_root.is_symlink() or (store_root.exists() and not store_root.is_dir()):
        raise ArtifactGCError(f"artifact store must be a non-symlink directory: {store_root}")
    store_root.mkdir(parents=True, exist_ok=True)
    destination = metadata_path(store_root)
    payload = {
        "schema": METADATA_SCHEMA,
        "orphans": {digest: orphans[digest] for digest in sorted(orphans)},
    }
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(dir=store_root, prefix=".gc-metadata.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _inventory(store_root: Path) -> tuple[FileSnapshot, ...]:
    if not store_root.exists() and not store_root.is_symlink():
        return ()
    if store_root.is_symlink() or not store_root.is_dir():
        raise ArtifactGCError(f"artifact store must be a non-symlink directory: {store_root}")
    snapshots: list[FileSnapshot] = []
    for digest in artifacts.inventory_artifacts(store_root):
        path = artifacts.artifact_path(store_root, digest)
        snapshots.append(_snapshot(path, label="artifact"))
    return tuple(snapshots)


def _coerce_grace(grace: timedelta | float | int) -> float:
    seconds = grace.total_seconds() if isinstance(grace, timedelta) else float(grace)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("grace must be a finite nonnegative duration")
    return seconds


def plan_gc(
    catalog_path: Path | str,
    checkpoint_dir: Path | str,
    *,
    store_root: Path | str | None = None,
    grace: timedelta | float = DEFAULT_GRACE,
    prune_all: bool = False,
    clock: Callable[[], float] = time.time,
) -> GcPlan:
    """Inventory roots, record orphan age, and return a deletion-free GC plan."""
    catalog = Path(catalog_path)
    checkpoints = Path(checkpoint_dir)
    store = Path(store_root) if store_root is not None else catalog.parent / "energy-grid-artifacts"
    now = float(clock())
    if not math.isfinite(now) or now < 0:
        raise ValueError("clock must return a finite nonnegative timestamp")
    grace_seconds = _coerce_grace(grace)
    catalog_roots, catalog_snapshot = _catalog_roots(catalog)
    lock_roots, locks, _ = _lock_roots(checkpoints)
    roots = catalog_roots | lock_roots
    inventory = _inventory(store)
    present = {snapshot.path.stem for snapshot in inventory}
    old_orphans = _read_metadata(store)
    new_orphans = {
        digest: timestamp for digest, timestamp in old_orphans.items() if digest in present
    }
    candidates: list[GcCandidate] = []
    retained: list[str] = []
    for snapshot in inventory:
        digest = snapshot.path.stem
        if digest in roots:
            new_orphans.pop(digest, None)
            retained.append(digest)
            continue
        orphaned_at = new_orphans.setdefault(digest, now)
        if prune_all or now - orphaned_at >= grace_seconds:
            candidates.append(GcCandidate(digest, snapshot.path, orphaned_at))
        else:
            retained.append(digest)
    if new_orphans != old_orphans:
        _write_metadata(store, new_orphans)
    return GcPlan(
        catalog_path=catalog,
        checkpoint_dir=checkpoints,
        store_root=store,
        roots=tuple(sorted(roots)),
        inventory=inventory,
        catalog=catalog_snapshot,
        locks=locks,
        metadata=_optional_snapshot(metadata_path(store), label="GC metadata"),
        candidates=tuple(candidates),
        retained=tuple(sorted(retained)),
        prune_all=prune_all,
    )


def _revalidate(plan: GcPlan) -> None:
    catalog_roots, catalog = _catalog_roots(plan.catalog_path)
    lock_roots, locks, _ = _lock_roots(plan.checkpoint_dir)
    if catalog != plan.catalog or locks != plan.locks:
        raise ArtifactGCError("catalog or campaign locks changed after GC preview; rerun command")
    if tuple(sorted(catalog_roots | lock_roots)) != plan.roots:
        raise ArtifactGCError("artifact reachability changed after GC preview; rerun command")
    if _inventory(plan.store_root) != plan.inventory:
        raise ArtifactGCError("artifact inventory changed after GC preview; rerun command")
    if _optional_snapshot(metadata_path(plan.store_root), label="GC metadata") != plan.metadata:
        raise ArtifactGCError("GC metadata changed after GC preview; rerun command")


def execute_gc(plan: GcPlan) -> tuple[Path, ...]:
    """Revalidate and delete exactly the candidates from ``plan``.

    This is the sole deletion function in this module.
    """
    _revalidate(plan)
    candidate_paths = {candidate.path: candidate for candidate in plan.candidates}
    for path, candidate in candidate_paths.items():
        expected = artifacts.artifact_path(plan.store_root, candidate.digest)
        if path != expected or _snapshot(path, label="artifact") not in plan.inventory:
            raise ArtifactGCError("artifact candidate changed after GC preview; rerun command")
    for path in candidate_paths:
        path.unlink()
    orphans = _read_metadata(plan.store_root)
    for candidate in plan.candidates:
        orphans.pop(candidate.digest, None)
    if plan.candidates or orphans:
        _write_metadata(plan.store_root, orphans)
    return tuple(candidate.path for candidate in plan.candidates)


def verify_artifacts(
    catalog_path: Path | str,
    checkpoint_dir: Path | str,
    *,
    store_root: Path | str | None = None,
) -> VerificationReport:
    """Detect missing/corrupt referenced artifacts and corrupt stored objects."""
    catalog = Path(catalog_path)
    checkpoints = Path(checkpoint_dir)
    store = Path(store_root) if store_root is not None else catalog.parent / "energy-grid-artifacts"
    catalog_roots, _ = _catalog_roots(catalog)
    lock_roots, _, lock_sources = _lock_roots(checkpoints)
    roots = catalog_roots | lock_roots
    inventory = _inventory(store)
    sources: dict[str, set[str]] = {digest: {"catalog"} for digest in catalog_roots}
    for digest, paths in lock_sources.items():
        sources.setdefault(digest, set()).update(paths)
    issues: list[VerificationIssue] = []
    checked: set[str] = set()
    for digest in sorted(roots | {snapshot.path.stem for snapshot in inventory}):
        try:
            artifacts.verify_artifact(store, digest)
        except artifacts.ArtifactError as exc:
            for source in sorted(sources.get(digest, {"store"})):
                issues.append(VerificationIssue(source, digest, str(exc)))
        checked.add(digest)
    return VerificationReport(
        roots=tuple(sorted(roots)),
        inventory=tuple(snapshot.path.stem for snapshot in inventory),
        issues=tuple(issues),
    )


__all__ = [
    "ArtifactGCError",
    "DEFAULT_GRACE",
    "FileSnapshot",
    "GcCandidate",
    "GcPlan",
    "METADATA_NAME",
    "METADATA_SCHEMA",
    "VerificationIssue",
    "VerificationReport",
    "execute_gc",
    "metadata_path",
    "plan_gc",
    "reachability_roots",
    "verify_artifacts",
]
