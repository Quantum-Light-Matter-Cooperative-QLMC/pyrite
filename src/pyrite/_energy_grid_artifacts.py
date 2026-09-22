"""Immutable, content-addressed energy-grid artifacts.

Artifact bytes are the public compatibility boundary: they are canonical UTF-8
JSON containing only the frozen identity fields below.  Provenance and other
mutable annotations must live outside this store so they cannot change a
campaign's artifact digest.

This is a store, not a stage of the derivation pipeline: it imports nothing
first-party, the bytes live beside ``materials.toml`` in
``data/energy-grid-artifacts/``, and it has two readers on opposite sides of the
package graph -- ``materials.catalog`` resolves ``[profiles.*.energy_grid_refs]``
at catalog-load time, and ``energy_grid.apply`` / ``energy_grid.gc`` write and
collect. It therefore sits at the package root, below both, rather than inside
``energy_grid/`` where it was the one leaf in an otherwise driver-level package.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SCHEMA = "cxr.energy-grid-artifact.v1"
"""Schema written into every first-generation energy-grid artifact."""

IDENTITY_FIELDS = (
    "schema",
    "material",
    "line_rows",
    "brem_grid",
    "beam_energies_keV",
)
"""Frozen ordered field tuple defining an artifact digest."""

EXCLUDED_ANNOTATION_FIELDS = (
    "provenance",
    "notes",
    "timestamp",
    "ref_name",
    "orphaned_at",
)
"""Frozen annotation names excluded from artifact bytes and identity."""

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MATERIAL_RE = re.compile(r"[a-z][a-z0-9_-]*\Z")


class ArtifactError(RuntimeError):
    """Base error for artifact-store failures."""


class ArtifactMissingError(ArtifactError):
    """Raised when an expected immutable artifact is absent."""


class ArtifactCorruptError(ArtifactError):
    """Raised when stored bytes do not match their name or canonical form."""


@dataclass(frozen=True)
class EnergyGridArtifact:
    """A verified artifact identity, digest, and immutable storage path."""

    digest: str
    identity: dict[str, Any]
    path: Path


def _finite_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "finite positive number" if positive else "finite number"
        raise ValueError(f"{field} must be a {qualifier}")
    return number


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        number = int(cast(Any, value))
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric != number or number <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return number


def normalize_material(material: object) -> str:
    """Return canonical validated material key used by the catalog/store."""
    if not isinstance(material, str):
        raise ValueError("material must be text")
    normalized = material.strip().lower()
    if not _MATERIAL_RE.fullmatch(normalized):
        raise ValueError("material must be a lowercase catalog-style key")
    return normalized


def _normalize_line_rows(line_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if isinstance(line_rows, (str, bytes)):
        raise ValueError("line_rows must be a sequence of mappings")
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(line_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"line_rows[{index}] must be a mapping")
        prefix = f"line_rows[{index}]"
        energy = _finite_float(row.get("energy_keV"), f"{prefix}.energy_keV", positive=True)
        start = _finite_float(row.get("start_eV"), f"{prefix}.start_eV", positive=True)
        stop = _finite_float(row.get("stop_eV"), f"{prefix}.stop_eV", positive=True)
        if stop <= start:
            raise ValueError(f"{prefix}.stop_eV must be greater than start_eV")
        normalized.append(
            {
                "energy_keV": energy,
                "start_eV": start,
                "stop_eV": stop,
                "num": _positive_int(row.get("num"), f"{prefix}.num"),
            }
        )
    normalized.sort(key=lambda row: cast(float, row["energy_keV"]))
    if any(
        left["energy_keV"] == right["energy_keV"]
        for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("line_rows must not contain duplicate energy_keV values")
    return normalized


def _normalize_brem_grid(brem_grid: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(brem_grid, Mapping):
        raise ValueError("brem_grid must be a mapping")
    start = _finite_float(brem_grid.get("start_eV", 0.0), "brem_grid.start_eV")
    if start < 0:
        raise ValueError("brem_grid.start_eV must be nonnegative")
    stop = _finite_float(brem_grid.get("stop_eV"), "brem_grid.stop_eV", positive=True)
    if stop <= start:
        raise ValueError("brem_grid.stop_eV must be greater than start_eV")
    return {
        "start_eV": start,
        "stop_eV": stop,
        "step_eV": _finite_float(brem_grid.get("step_eV"), "brem_grid.step_eV", positive=True),
    }


def _normalize_beam_energies(beam_energies_keV: Sequence[object]) -> list[float]:
    if isinstance(beam_energies_keV, (str, bytes)):
        raise ValueError("beam_energies_keV must be a sequence")
    return sorted(
        {_finite_float(value, "beam_energies_keV", positive=True) for value in beam_energies_keV}
    )


def normalize_annotations(annotations: Mapping[str, object] | None) -> dict[str, object]:
    """Validate out-of-band annotations without allowing them into identity bytes."""
    if annotations is None:
        return {}
    if not isinstance(annotations, Mapping):
        raise ValueError("annotations must be a mapping")
    if any(not isinstance(key, str) for key in annotations):
        raise ValueError("annotation field names must be text")
    unexpected = sorted(set(annotations) - set(EXCLUDED_ANNOTATION_FIELDS))
    if unexpected:
        raise ValueError(f"unsupported annotation fields: {', '.join(unexpected)}")
    return dict(annotations)


def artifact_identity(
    material: object,
    line_rows: Sequence[Mapping[str, object]],
    brem_grid: Mapping[str, object],
    beam_energies_keV: Sequence[object],
    *,
    annotations: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Normalize artifact identity; annotations are validated but never retained.

    Callers that need annotations should persist :func:`normalize_annotations`
    output separately, keyed by the returned digest.
    """
    normalize_annotations(annotations)
    return {
        "schema": SCHEMA,
        "material": normalize_material(material),
        "line_rows": _normalize_line_rows(line_rows),
        "brem_grid": _normalize_brem_grid(brem_grid),
        "beam_energies_keV": _normalize_beam_energies(beam_energies_keV),
    }


def canonical_bytes(identity: Mapping[str, object]) -> bytes:
    """Return canonical UTF-8 bytes after validating frozen identity fields."""
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a mapping")
    if set(identity) != set(IDENTITY_FIELDS) or len(identity) != len(IDENTITY_FIELDS):
        raise ValueError(f"identity fields must be exactly {IDENTITY_FIELDS!r}")
    if identity.get("schema") != SCHEMA:
        raise ValueError(f"identity.schema must be {SCHEMA}")
    normalized = artifact_identity(
        identity["material"],
        cast("Sequence[Mapping[str, object]]", identity["line_rows"]),
        cast("Mapping[str, object]", identity["brem_grid"]),
        cast("Sequence[object]", identity["beam_energies_keV"]),
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def artifact_digest(identity: Mapping[str, object]) -> str:
    """Return SHA-256 digest of canonical artifact bytes."""
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def artifact_path(root: Path | str, digest: str) -> Path:
    """Return the sharded storage path for a validated digest."""
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError("digest must be 64 lowercase hexadecimal characters")
    return Path(root) / digest[:2] / f"{digest}.json"


def _decode_canonical(data: bytes, digest: str) -> dict[str, Any]:
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != digest:
        raise ArtifactCorruptError(
            f"artifact {digest} bytes hash to {actual_digest}, not its filename digest"
        )
    try:
        decoded = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactCorruptError(f"artifact {digest} is not valid canonical JSON") from exc
    try:
        canonical = canonical_bytes(decoded)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCorruptError(f"artifact {digest} has invalid identity fields") from exc
    if data != canonical:
        raise ArtifactCorruptError(f"artifact {digest} is not canonical JSON")
    return decoded


def _no_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def write_artifact(root: Path | str, identity: Mapping[str, object]) -> EnergyGridArtifact:
    """Atomically create an artifact, or return an existing byte-identical copy."""
    data = canonical_bytes(identity)
    digest = hashlib.sha256(data).hexdigest()
    path = artifact_path(root, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{digest}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            existing_identity = _decode_canonical(existing, digest)
            if existing != data:
                raise ArtifactCorruptError(
                    f"artifact {digest} differs from canonical identity bytes"
                ) from None
            return EnergyGridArtifact(digest, existing_identity, path)
        return EnergyGridArtifact(digest, json.loads(data), path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_artifact(root: Path | str, digest: str) -> EnergyGridArtifact:
    """Load and fully verify a single immutable artifact."""
    path = artifact_path(root, digest)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ArtifactMissingError(f"artifact {digest} is missing") from exc
    return EnergyGridArtifact(digest, _decode_canonical(data, digest), path)


def verify_artifact(root: Path | str, digest: str) -> EnergyGridArtifact:
    """Verify and return artifact identified by DIGEST."""
    return load_artifact(root, digest)


def inventory_artifacts(root: Path | str) -> tuple[str, ...]:
    """Safely enumerate well-formed shard entries without trusting their bytes."""
    root_path = Path(root)
    if not root_path.exists():
        return ()
    found: list[str] = []
    for shard in root_path.iterdir():
        if shard.is_symlink() or not shard.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", shard.name):
            continue
        for candidate in shard.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                continue
            match = re.fullmatch(r"([0-9a-f]{64})\.json", candidate.name)
            if match and match.group(1).startswith(shard.name):
                found.append(match.group(1))
    return tuple(sorted(found))
