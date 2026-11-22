"""Immutable history-shard artifacts: one case's electron ranges on shared disk (#1387).

A shard is the raw, unfinalized transport of electrons ``[start, stop)`` of one
fixed-count case (:mod:`pyrite.montecarlo.runner.history_shards`). Workers
publish shards here; one join task per case reads them back in electron order
and publishes the ordinary case artifact (:class:`.artifacts.CaseArtifactStore`).

Publication mirrors the case inbox: an exclusive hard link makes the first
complete writer win, and a JSON receipt records the payload digest, so a
retried or resumed run reuses a published shard and never recomputes it.

Payloads are pickles. They carry nested metadata (tuples, NumPy scalars,
aliased arrays) that must round-trip exactly, and they are read only by the
same campaign's join task after the receipt digest matches. The trust boundary
is the Dask cluster's own, which already exchanges pickled task payloads; never
point a store at a directory other users can write.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactIntegrityError, CaseArtifactRef, _digest, _key, _material

_FORMAT = "pyrite.execution.history-shard.v1"


@dataclass(frozen=True, slots=True)
class ShardPolicy:
    """Opt-in: split each case's electron histories into shards (experimental, #1387).

    ``shard_electrons`` electrons per shard (the last is short when it does not
    divide the case). ``block_electrons`` bounds the rows a shard holds at once
    (default: the whole shard); it never changes results. ``transport_core``
    must resolve to ``"per-electron"``: the lockstep core cannot shard and GPU
    shards are not supported. ``keep_shards`` keeps shard files after their
    case artifact is published (default: delete them).
    """

    shard_electrons: int
    block_electrons: int | None = None
    transport_core: str = "per-electron"
    keep_shards: bool = False

    def __post_init__(self) -> None:
        for field in ("shard_electrons", "block_electrons"):
            value = getattr(self, field)
            if value is None and field == "block_electrons":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.transport_core != "per-electron":
            raise ValueError("history shards run on transport_core='per-electron' only")


def shard_content_key(case_key: str, start: int, stop: int, transport_core: str) -> str:
    """Key of electrons ``[start, stop)`` of the case ``case_key`` on ``transport_core``.

    The block size inside a shard is excluded: it never changes the rows.
    """
    if not 0 <= int(start) < int(stop):
        raise ValueError("a shard must cover a nonempty [start, stop) range")
    digest = hashlib.sha256(b"pyrite.history-shard.v1\x00")
    digest.update(bytes.fromhex(_key(case_key)))
    digest.update(int(start).to_bytes(8, "big") + int(stop).to_bytes(8, "big"))
    digest.update(transport_core.encode())
    return digest.hexdigest()


class HistoryShardStore:
    """Shared-filesystem shard inbox with atomic, first-writer-wins publish."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path_for(self, material: str, shard_key: str) -> Path:
        key = _key(shard_key)
        return self.root / _material(material) / ".history-shards" / key[:2] / f"{key}.pkl"

    def reference(self, material: str, shard_key: str) -> CaseArtifactRef | None:
        """The committed shard, or None; temp files and receipt-less payloads are ignored."""
        receipt = self.path_for(material, shard_key).with_suffix(".json")
        if not receipt.exists():
            return None
        try:
            meta = json.loads(receipt.read_text(encoding="utf-8"))
            ref = CaseArtifactRef(
                meta["material"], meta["content_key"], meta["sha256"], meta["size_bytes"]
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ArtifactIntegrityError(f"invalid shard receipt: {receipt}") from exc
        if meta.get("format") != _FORMAT or (ref.material, ref.content_key) != (
            _material(material),
            _key(shard_key),
        ):
            raise ArtifactIntegrityError(f"shard receipt identity mismatch: {receipt}")
        return ref

    def verify(self, ref: CaseArtifactRef) -> Path:
        """Check the payload's digest and length against its receipt; return its path."""
        path = self.path_for(ref.material, ref.content_key)
        try:
            digest, size = _digest(path)
        except OSError as exc:
            raise ArtifactIntegrityError(f"missing/unreadable shard: {path}") from exc
        if digest != ref.sha256.lower() or size != ref.size_bytes:
            raise ArtifactIntegrityError(f"shard digest or length mismatch: {path}")
        return path

    def publish(self, material: str, shard_key: str, segments: dict[str, Any]) -> CaseArtifactRef:
        """Publish one shard's rows; a shard already committed is returned unchanged."""
        prior = self.reference(material, shard_key)
        if prior is not None:
            self.verify(prior)
            return prior
        target = self.path_for(material, shard_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": _FORMAT, "content_key": _key(shard_key), "segments": segments}
        if not target.exists():
            with _temporary(target) as tmp:
                with tmp.open("wb") as stream:
                    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    stream.flush()
                    os.fsync(stream.fileno())
                _link_once(tmp, target)
        digest, size = _digest(target)
        receipt = {
            "format": _FORMAT,
            "material": _material(material),
            "content_key": _key(shard_key),
            "sha256": digest,
            "size_bytes": size,
        }
        with _temporary(target) as tmp:
            tmp.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            with tmp.open("rb") as stream:
                os.fsync(stream.fileno())
            _link_once(tmp, target.with_suffix(".json"))
        ref = self.reference(material, shard_key)
        if ref is None:
            raise ArtifactIntegrityError(f"shard receipt publication failed: {target}")
        self.verify(ref)
        return ref

    def load(self, ref: CaseArtifactRef, *, expected_content_key: str) -> dict[str, Any]:
        """The shard's rows, after digest and identity checks."""
        if ref.content_key != _key(expected_content_key):
            raise ArtifactIntegrityError("shard key does not match the requested range")
        path = self.verify(ref)
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - digest-checked campaign inbox
        if (
            not isinstance(payload, dict)
            or payload.get("format") != _FORMAT
            or payload.get("content_key") != ref.content_key
            or not isinstance(payload.get("segments"), dict)
        ):
            raise ArtifactIntegrityError(f"shard identity/schema mismatch: {path}")
        return payload["segments"]

    def remove(self, ref: CaseArtifactRef) -> None:
        """Delete a consumed shard (receipt first, so it never reads as committed)."""
        path = self.path_for(ref.material, ref.content_key)
        path.with_suffix(".json").unlink(missing_ok=True)
        path.unlink(missing_ok=True)


class _temporary:
    """A temp file beside ``target``, removed on exit."""

    def __init__(self, target: Path) -> None:
        self.target = target

    def __enter__(self) -> Path:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.target.stem}.", suffix=".tmp", dir=self.target.parent
        )
        os.close(descriptor)
        self.path = Path(name)
        return self.path

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _link_once(source: Path, target: Path) -> None:
    try:
        os.link(source, target)  # atomic create-if-absent
    except FileExistsError:
        pass  # another worker won; its file is validated by the caller
