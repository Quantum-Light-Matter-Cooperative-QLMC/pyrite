"""Immutable worker-produced case artifacts for distributed campaign execution.

This is an inbox, NOT an authoritative sweep checkpoint or CAS. Workers may
only write independent, content-keyed blobs. A single coordinator validates
those blobs and applies them via the existing ``run_sweep`` callback/persistence
lifecycle. Workers must never write ``meta.json``, ``parts/``, or component HDF5.

Publication assumes a POSIX-compatible shared filesystem supporting atomic
hard-link creation (e.g. the usual cluster filesystem). Cross-machine object
stores and unsupported filesystems require a separate publication backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FORMAT = "pyrite.execution.case.v1"
_BLOCK_BYTES = 1024 * 1024
_KEY_PATTERN = re.compile(r"[a-fA-F0-9]{64}\Z")


class ArtifactIntegrityError(ValueError):
    """An expected published result is corrupt, mismatched, or unreadable."""


def _key(value: str) -> str:
    if not isinstance(value, str) or _KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("case content key must be a 64-character SHA-256 hex digest")
    return value.lower()


def _material(value: str) -> str:
    if (
        not isinstance(value, str)
        or value in ("", ".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("material must be a single nonempty path component")
    return value


def _digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_BLOCK_BYTES):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


@dataclass(frozen=True, slots=True)
class CaseArtifactRef:
    """Small, serializable reference; never contains trajectory/spectral arrays.

    ``sha256`` authenticates artifact *bytes*, not the physics content key.
    The latter is produced by PyRITE's existing resolved-case identity logic.
    """

    material: str
    content_key: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _material(self.material)
        _key(self.content_key)
        _key(self.sha256)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("size_bytes must be a positive integer")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")


class CaseArtifactStore:
    """Portable shared-filesystem inbox with atomic, first-writer-wins publish.

    Publish's ``os.link`` is exclusive and never replaces a completed artifact.
    Retries and competing workers accept the first published result for the
    same physics key. A failed HDF5 write leaves only an ignored ``.tmp`` file,
    never a visible completed artifact. Callers are responsible for keeping
    source work units deterministic (including RNG seeds and table identities).

    An invalid completed artifact causes an explicit error, not a cache hit or
    silent overwrite. Operators can quarantine it before retrying.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path_for(self, material: str, content_key: str) -> Path:
        name = _material(material)
        key = _key(content_key)
        return self.root / name / ".execution-artifacts" / key[:2] / f"{key}.h5"

    def reference(self, material: str, content_key: str) -> CaseArtifactRef | None:
        """Return a committed artifact; uncommitted HDF5 and temp files are ignored.

        A sibling JSON receipt records the digest at publication time; without
        it, recovery could not distinguish a valid file from one corrupted
        *after* the writer returned (HDF5 need not detect appended bytes).
        """
        key = _key(content_key)
        name = _material(material)
        receipt = self.path_for(name, key).with_suffix(".json")
        if not receipt.exists():
            return None
        try:
            metadata = json.loads(receipt.read_text(encoding="utf-8"))
            ref = CaseArtifactRef(
                material=metadata["material"],
                content_key=metadata["content_key"],
                sha256=metadata["sha256"],
                size_bytes=metadata["size_bytes"],
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ArtifactIntegrityError(f"invalid artifact receipt: {receipt}") from exc
        if metadata.get("format") != _FORMAT or ref.material != name or ref.content_key != key:
            raise ArtifactIntegrityError(f"artifact receipt identity mismatch: {receipt}")
        return ref

    def publish(
        self, material: str, content_key: str, output: dict[str, Any]
    ) -> CaseArtifactRef:
        """Atomically publish a complete case output without modifying checkpoints.

        ``output`` is the raw ``run_case`` result, prior to ``store_result``.
        A duplicate publish returns a reference to the first completed result.
        """
        name = _material(material)
        key = _key(content_key)
        if not isinstance(output, dict):
            raise TypeError("case artifact output must be a mapping dict")
        target = self.path_for(name, key)
        prior = self.reference(name, key)
        if prior is not None:
            self.load(prior)
            return prior

        # Import here: planning and executor inspection must not import HDF5.
        from ..checkpoints import _checkpoint_io

        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            descriptor, tmp_name = tempfile.mkstemp(
                prefix=f".{key}.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            tmp = Path(tmp_name)
            try:
                _checkpoint_io.dump(
                    {"format": _FORMAT, "material": name, "content_key": key, "output": output},
                    str(tmp),
                )
                with tmp.open("rb") as stream:
                    os.fsync(stream.fileno())
                try:
                    # Atomic create-if-absent, unlike os.replace (last writer wins).
                    os.link(tmp, target)
                except FileExistsError:
                    pass  # Another worker won. Validate its file before committing.
            finally:
                tmp.unlink(missing_ok=True)

        # A crash between linking the data and committing its receipt leaves a
        # recoverable *candidate*, not a published result. Any later retry can
        # validate and finish the receipt, without replacing the immutable HDF5.
        try:
            document = _checkpoint_io.load(str(target))
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            raise ArtifactIntegrityError(f"unreadable uncommitted case artifact: {target}") from exc
        if (
            not isinstance(document, dict)
            or document.get("format") != _FORMAT
            or document.get("material") != name
            or document.get("content_key") != key
            or not isinstance(document.get("output"), dict)
        ):
            raise ArtifactIntegrityError(f"uncommitted artifact identity mismatch: {target}")
        digest, size = _digest(target)
        receipt_path = target.with_suffix(".json")
        descriptor, tmp_name = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=target.parent
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {"format": _FORMAT, "material": name, "content_key": key,
                     "sha256": digest, "size_bytes": size},
                    stream,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(tmp, receipt_path)
            except FileExistsError:
                pass  # First committed receipt wins, regardless of retry order.
        finally:
            tmp.unlink(missing_ok=True)
        ref = self.reference(name, key)
        if ref is None:
            raise ArtifactIntegrityError(f"artifact receipt publication failed: {receipt_path}")
        self.load(ref)
        return ref

    def load(
        self,
        ref: CaseArtifactRef,
        *,
        expected_material: str | None = None,
        expected_content_key: str | None = None,
    ) -> dict[str, Any]:
        """Verify digest and resolved-case identity before decoding worker data.

        Never trust a worker-supplied reference as sufficient proof of physics
        identity: the coordinator must pass the expected case key separately.
        """
        if expected_material is not None and ref.material != _material(expected_material):
            raise ArtifactIntegrityError("case artifact material does not match requested case")
        if expected_content_key is not None and ref.content_key != _key(expected_content_key):
            raise ArtifactIntegrityError("case artifact content key does not match requested case")
        path = self.path_for(ref.material, ref.content_key)
        try:
            digest, size = _digest(path)
        except OSError as exc:
            raise ArtifactIntegrityError(f"missing/unreadable case artifact: {path}") from exc
        if digest != ref.sha256.lower() or size != ref.size_bytes:
            raise ArtifactIntegrityError(f"case artifact digest or length mismatch: {path}")

        from ..checkpoints import _checkpoint_io

        try:
            document = _checkpoint_io.load(str(path))
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            raise ArtifactIntegrityError(f"unreadable case artifact: {path}") from exc
        if (
            not isinstance(document, dict)
            or document.get("format") != _FORMAT
            or document.get("material") != ref.material
            or document.get("content_key") != ref.content_key
            or not isinstance(document.get("output"), dict)
        ):
            raise ArtifactIntegrityError(f"case artifact identity/schema mismatch: {path}")
        return document["output"]

    def recover(
        self, material: str, content_key: str
    ) -> tuple[CaseArtifactRef, dict[str, Any]] | None:
        """Resolve one crash-surviving artifact, or None if no complete publish.

        Corruption raises rather than being incorrectly classified as complete.
        Designed for coordinator restart before assigning pending work.
        """
        ref = self.reference(material, content_key)
        if ref is None:
            return None
        return ref, self.load(ref, expected_material=material, expected_content_key=content_key)
