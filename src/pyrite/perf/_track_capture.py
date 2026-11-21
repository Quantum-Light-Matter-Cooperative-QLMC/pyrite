"""Bounded, checksum-verified source-cone replay inputs for benchmark studies.

Private engineering records, not general trajectory archives or numerical gates.
Only deterministic scoring of the same retained primaries is paired by these
records; no common-random-number transport claim follows from their seed.
"""

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .._util.hashing import canonical_json_digest, file_sha256

SCHEMA = "pyrite.geant4-performance.track-capture.v1"
ARRAY_FIELDS = ("elec_id", "L_ang", "E_keV", "r_mid", "v_hat")
SCALAR_FIELDS = ("thickness_ang", "crystal_width_ang", "crystal_height_ang")


def source_fingerprint() -> str:
    """Hash actual package Python sources, independently of Git availability."""
    root = Path(__file__).resolve().parents[1]
    return canonical_json_digest(
        {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*.py"))}
    )


def _json(value: Any) -> Any:
    """Normalize numeric metadata, refusing non-JSON values and nonfinite floats."""
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    # canonical JSON also rejects NaN/Infinity, including nested entries.
    canonical_json_digest(value)
    return value


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _arrays(segments: dict, count: int) -> dict[str, np.ndarray]:
    if segments.get("Ne") != count or segments.get("channeling_flux") is not None:
        raise ValueError("source capture needs the exact unweighted incident population")
    if segments.get("_stream_escape_recovery") is not None:
        raise ValueError("callback-based tracks cannot be serialized for replay")
    fields = ARRAY_FIELDS + (("E_repr_keV",) if segments.get("E_repr_keV") is not None else ())
    arrays = {name: np.asarray(segments[name]) for name in fields}
    ids = arrays["elec_id"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("invalid local primary IDs")
    for name, array in arrays.items():
        shape = (ids.size, 3) if name in ("r_mid", "v_hat") else ids.shape
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid replay field: {name}")
    if np.any(arrays["L_ang"] < 0) or np.any(arrays["E_keV"] < 0):
        raise ValueError("negative track length or energy")
    return arrays


def _scalars(segments: dict) -> dict:
    values = {name: segments[name] for name in SCALAR_FIELDS}
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
        raise ValueError("capture requires finite positive slab dimensions")
    radiative = segments.get("radiative")
    if not radiative or radiative.get("model") != "bremslib-soft-hard":
        raise ValueError("capture requires coupled BremsLib transport identity")
    return _json(values | dict(radiative=radiative))


def _check_scalar_context(scalars: dict, metadata: dict) -> None:
    context = metadata.get("context", {})
    if "thickness_ang" in context and scalars["thickness_ang"] != context["thickness_ang"]:
        raise ValueError("captured thickness differs from scoring context")
    if "cutoff_eV" in context and scalars["radiative"]["cutoff_eV"] != context["cutoff_eV"]:
        raise ValueError("captured radiative cutoff differs from scoring context")


class TrackCapture:
    """Write contiguous whole-primary shards; publish completion only on finish."""

    def __init__(self, root: Path, histories: int, metadata: dict, max_bytes: int):
        if not _positive_integer(histories) or histories < 2 or not _positive_integer(max_bytes):
            raise ValueError("capture needs >=2 primaries and a positive byte ceiling")
        self.root = Path(root)
        self.histories = histories
        self.max_bytes = max_bytes
        self.metadata = _json(metadata)
        self.shards: list[dict] = []
        self.offset = 0
        self.bytes = 0
        self.finished = False
        self.root.mkdir(parents=True, exist_ok=False)

    def append(self, segments: dict, offset: int, count: int) -> None:
        if self.finished or offset != self.offset or not _positive_integer(count):
            raise ValueError("capture shards must cover contiguous, unrepeated whole primaries")
        if offset + count > self.histories:
            raise ValueError("capture exceeds declared incident population")
        arrays = _arrays(segments, count)
        scalars = _scalars(segments)
        _check_scalar_context(scalars, self.metadata)
        arrays["primary_ids"] = np.arange(offset, offset + count, dtype=np.int64)
        size = sum(a.nbytes for a in arrays.values())
        # Include archive headers in the on-disk ceiling; no compressed-array inflation.
        overhead = 4096 * len(arrays)
        if self.bytes + size + overhead > self.max_bytes:
            raise ValueError("capture exceeds byte ceiling; reduce population/rows, never truncate")
        name = f"tracks-{len(self.shards):06d}.npz"
        path = self.root / name
        payload: dict[str, Any] = dict(arrays)
        with path.open("xb") as stream:
            np.savez(stream, **payload)
        actual = path.stat().st_size
        self.bytes += actual
        if self.bytes > self.max_bytes:
            raise ValueError("capture exceeded byte ceiling; incomplete capture retained")
        self.shards.append(
            dict(
                file=name,
                sha256=file_sha256(path),
                bytes=actual,
                array_bytes=size,
                offset=offset,
                count=count,
                rows=int(arrays["elec_id"].size),
                scalars=scalars,
            )
        )
        self.offset += count

    def finish(self) -> dict:
        if self.finished or self.offset != self.histories:
            raise ValueError("cannot complete a partial or already completed capture")
        record = dict(
            schema=SCHEMA,
            purpose="accuracy-only",
            workload="source-cone",
            complete=True,
            histories=self.histories,
            metadata=self.metadata,
            shards=self.shards,
            bytes=self.bytes,
            numerical_gate="unmeasured",
            timing_comparison_eligible=False,
        )
        record["manifest_sha256"] = canonical_json_digest(record)
        # A failed/partial request has no completion marker and is never replayable.
        with (self.root / "capture.json").open("x") as stream:
            json.dump(record, stream, allow_nan=False, sort_keys=True, indent=1)
            stream.write("\n")
        self.finished = True
        return record


def _check_shard_file(root: Path, shard: dict, max_bytes: int) -> None:
    """Check stored/expanded size and content identity without loading track arrays."""
    path = root / shard["file"]
    if path.is_symlink() or path.stat().st_size != shard["bytes"]:
        raise ValueError("capture shard size/path mismatch")
    if shard["array_bytes"] > max_bytes:
        raise ValueError("capture exceeds replay byte ceiling")
    if file_sha256(path) != shard["sha256"]:
        raise ValueError("capture shard checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        if sum(m.file_size for m in archive.infolist()) > max_bytes:
            raise ValueError("expanded capture exceeds replay byte ceiling")


def load_index(root: Path, max_bytes: int) -> dict:
    """Check the full shard layout and all payload hashes before any scoring."""
    if not _positive_integer(max_bytes):
        raise ValueError("positive replay byte ceiling required")
    root = Path(root)
    record = json.loads((root / "capture.json").read_text())
    signature = record.pop("manifest_sha256")
    if signature != canonical_json_digest(record):
        raise ValueError("capture manifest checksum mismatch")
    record["manifest_sha256"] = signature
    if (
        record.get("schema") != SCHEMA
        or record.get("complete") is not True
        or record.get("workload") != "source-cone"
        or record.get("purpose") != "accuracy-only"
        or record.get("timing_comparison_eligible") is not False
    ):
        raise ValueError("not a complete accuracy-only source-cone capture")
    if not _positive_integer(record["histories"]) or record["histories"] < 2:
        raise ValueError("invalid captured incident count")
    offset, size, names = 0, 0, set()
    for shard in record["shards"]:
        name = shard["file"]
        if (
            not re.fullmatch(r"tracks-\d{6}\.npz", name)
            or name in names
            or shard["offset"] != offset
            or not _positive_integer(shard["count"])
        ):
            raise ValueError("invalid, repeated or noncontiguous capture shard")
        names.add(name)
        size += shard["bytes"]
        if size > max_bytes:
            raise ValueError("capture exceeds replay byte ceiling")
        _check_shard_file(root, shard, max_bytes)
        _check_scalar_context(shard["scalars"], record["metadata"])
        offset += shard["count"]
    if (
        offset != record["histories"]
        or size != record["bytes"]
        or names != {p.name for p in root.glob("tracks-*.npz")}
    ):
        raise ValueError("incomplete capture population or shard inventory")
    return record


def read_shard(root: Path, shard: dict) -> tuple[dict, np.ndarray]:
    """Restore numeric rows and exact IDs, including primaries with no rows."""
    path = Path(root) / shard["file"]
    if file_sha256(path) != shard["sha256"]:
        raise ValueError("capture shard changed after preflight")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    ids = arrays.pop("primary_ids")
    if ids.dtype.kind not in "iu" or not np.array_equal(
        ids, np.arange(shard["offset"], shard["offset"] + shard["count"])
    ):
        raise ValueError("captured incident IDs are missing, reordered or repeated")
    segments = arrays | shard["scalars"] | dict(Ne=shard["count"])
    checked = _arrays(segments, shard["count"])
    if set(arrays) != set(checked) or checked["elec_id"].size != shard["rows"]:
        raise ValueError("unexpected capture fields or row count")
    _scalars(segments)
    # Production partition audit compares tuples; JSON deliberately stores arrays.
    radiative = dict(segments["radiative"])
    radiative["bremslib_tables"] = tuple(tuple(v) for v in radiative["bremslib_tables"])
    segments["radiative"] = radiative
    return segments, ids
