"""Deterministic per-dataset campaign lockfiles.

Lockfiles capture resolved mutable-profile state at successful run completion.
They are provenance records and artifact-GC roots; paused or failed runs must
not claim a completed lock.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "cxr.campaign-lock.v1"


def lock_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    """Return lockfile beside directory or legacy-pickle checkpoint metadata."""
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return path.with_suffix(".lock.json")
    return path / "cxr.lock.json"


def lock_payload(
    *,
    profile: str,
    material: str,
    dataset_identity: Mapping[str, Any],
    energy_grid_digest: str | None,
) -> dict[str, Any]:
    """Build versioned JSON payload from already-resolved run inputs."""
    if not isinstance(profile, str) or not profile:
        raise ValueError("profile must be a nonempty string")
    if not isinstance(material, str) or not material:
        raise ValueError("material must be a nonempty string")
    if energy_grid_digest is not None and (
        len(energy_grid_digest) != 64
        or any(character not in "0123456789abcdef" for character in energy_grid_digest)
    ):
        raise ValueError("energy_grid_digest must be a lowercase SHA-256 digest")
    return {
        "schema": SCHEMA,
        "profile": profile,
        "material": material,
        "artifacts": {
            "energy_grid": ({material: energy_grid_digest} if energy_grid_digest else {}),
        },
        "legacy_energy_grid": energy_grid_digest is None,
        "dataset_identity": dict(dataset_identity),
    }


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one lock deterministically without platform-specific bytes."""
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_lock(
    checkpoint_path: str | os.PathLike[str],
    *,
    profile: str,
    material: str,
    dataset_identity: Mapping[str, Any],
    energy_grid_digest: str | None,
) -> Path:
    """Atomically replace completed run's deterministic campaign lockfile."""
    destination = lock_path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = lock_payload(
        profile=profile,
        material=material,
        dataset_identity=dataset_identity,
        energy_grid_digest=energy_grid_digest,
    )
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def read_lock(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate one campaign lock's schema and artifact mapping."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid campaign lock {source}: {exc}") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"invalid campaign lock {source}: expected schema {SCHEMA}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("energy_grid"), dict):
        raise ValueError(f"invalid campaign lock {source}: missing artifacts.energy_grid")
    return payload
