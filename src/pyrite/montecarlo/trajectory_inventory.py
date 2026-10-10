"""Discover and describe trajectory artifacts without loading segment arrays.

Backs ``pyrite trajectories ls|info`` and the artifact discovery of
``export``/``score`` (#397). Metadata comes from root attributes, the small
JSON groups, and ``/transport`` node shapes and dtypes. History, track, and
generation counts scan only their integer ID columns in bounded row blocks, so
host memory never scales with the segment table.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .trajectories import (
    PARTIAL_SUFFIX,
    TrajectoryArtifactError,
    read_trajectory_header,
    row_fields,
)

_BLOCK = 1 << 16
_PER_ELECTRON_HINTS = ("initial_E_keV", "E_cut_by_electrons")


@dataclass(frozen=True)
class FoundArtifact:
    """A complete artifact, its path relative to the searched root, and header."""

    path: Path
    relative: Path
    header: dict[str, Any]


@dataclass(frozen=True)
class SkippedFile:
    """A file under a searched directory that is not a complete artifact."""

    path: Path
    status: str  # "partial", "incomplete", or "foreign"
    reason: str


@dataclass
class Discovery:
    """Outcome of :func:`discover_artifacts`."""

    artifacts: list[FoundArtifact] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)


def _classify(path: Path) -> tuple[dict[str, Any] | None, SkippedFile | None]:
    try:
        return read_trajectory_header(path), None
    except TrajectoryArtifactError as error:
        status = "incomplete" if str(error).startswith("incomplete") else "foreign"
        return None, SkippedFile(path, status, str(error))


def discover_artifacts(paths: Iterable[str | os.PathLike[str]]) -> Discovery:
    """Expand directories recursively to complete artifacts; keep explicit files.

    A directory's ``*.h5`` files that are not complete PyRITE artifacts, and
    stray ``*.h5.partial`` files of interrupted writes, are reported in
    ``skipped`` instead of aborting. An explicit file is always returned as an
    artifact (its header is then empty when unreadable) so the caller reports
    its own error. A directory's artifacts keep their layout below it (every
    case has an ``E0_<E>keV.h5`` file, so names alone collide); an explicit
    file is its own name.
    """
    found = Discovery()
    for path in map(Path, paths):
        if not path.is_dir():
            header, _ = _classify(path)
            found.artifacts.append(FoundArtifact(path, Path(path.name), header or {}))
            continue
        candidates = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and (p.suffix == ".h5" or p.name.endswith(".h5" + PARTIAL_SUFFIX))
        )
        for candidate in candidates:
            if candidate.name.endswith(PARTIAL_SUFFIX):
                found.skipped.append(
                    SkippedFile(candidate, "partial", "interrupted write (stray .partial file)")
                )
                continue
            header, skipped = _classify(candidate)
            if skipped is not None:
                found.skipped.append(skipped)
            else:
                found.artifacts.append(
                    FoundArtifact(candidate, candidate.relative_to(path), header or {})
                )
    return found


def _json_attr(handle, group: str) -> Any:
    return json.loads(handle[group].attrs["json"]) if group in handle else None


def _distinct_count(dataset, *, sorted_rows: bool) -> tuple[int, int | None, int | None]:
    """Distinct values, minimum, and maximum of an integer column in blocks."""
    n_rows = int(dataset.shape[0])
    if not n_rows:
        return 0, None, None
    if sorted_rows:
        count, previous = 0, None
        for start in range(0, n_rows, _BLOCK):
            block = np.asarray(dataset[start : start + _BLOCK])
            changes = int(np.count_nonzero(block[1:] != block[:-1])) + 1
            count += changes - (previous is not None and block[0] == previous)
            previous = block[-1]
        return count, int(dataset[0]), int(dataset[n_rows - 1])
    seen: set[int] = set()
    low = high = None
    for start in range(0, n_rows, _BLOCK):
        block = np.unique(np.asarray(dataset[start : start + _BLOCK]))
        seen.update(block.tolist())
        low = int(block[0]) if low is None else min(low, int(block[0]))
        high = int(block[-1]) if high is None else max(high, int(block[-1]))
    return len(seen), low, high


def _generation_counts(dataset) -> dict[int, int]:
    counts = np.zeros(0, dtype=np.int64)
    for start in range(0, int(dataset.shape[0]), _BLOCK):
        block = np.asarray(dataset[start : start + _BLOCK]).astype(np.int64)
        block = np.bincount(block[block >= 0])
        if block.size > counts.size:
            block[: counts.size] += counts
            counts = block
        else:
            counts[: block.size] += block
    return {generation: int(n) for generation, n in enumerate(counts) if n}


def _field_record(name: str, node, n_rows: int, n_electrons: int | None) -> dict[str, Any]:
    kind = str(node.attrs.get("pyrite_kind", "array"))
    record: dict[str, Any] = {"name": name, "kind": kind}
    if hasattr(node, "dtype") and kind != "none":
        record["dtype"] = str(node.dtype)
        record["shape"] = list(node.shape)
    if "units" in node.attrs:
        record["units"] = str(node.attrs["units"])
    shape = getattr(node, "shape", None)
    if kind == "array" and shape and shape[0] == n_rows and name in row_fields():
        record["role"] = "segment"
    elif kind == "array" and shape and n_electrons is not None and shape[0] == n_electrons:
        record["role"] = "electron"
    elif kind in {"dict", "list", "tuple"}:
        record["role"] = "nested"
        record["children"] = list(node)
    else:
        record["role"] = "scalar" if kind != "array" else "other"
    return record


def _electron_count(group) -> int | None:
    for name in _PER_ELECTRON_HINTS:
        if name in group and getattr(group[name], "shape", ()):
            return int(group[name].shape[0])
    return None


def describe_artifact(path: str | os.PathLike[str], *, counts: bool = True) -> dict[str, Any]:
    """JSON-safe description of one complete artifact.

    Reads root attributes, ``/case``/``/provenance``/``/settings`` JSON, and
    ``/transport`` node metadata. With ``counts``, scans the integer
    ``electron_id``/``track_id``/``generation`` columns in bounded blocks;
    no other segment array is read.

    Raises
    ------
    TrajectoryArtifactError
        Unreadable, foreign, incomplete, or unsupported artifact.
    """
    import h5py

    path = Path(path)
    header = read_trajectory_header(path)
    with h5py.File(path, "r") as handle:
        group = handle["transport"]
        ids_name = "electron_id" if "electron_id" in group else "elec_id"
        ids = group[ids_name] if ids_name in group else None
        n_rows = int(ids.shape[0]) if ids is not None else int(header.get("segment_count", 0))
        n_electrons = _electron_count(group)
        fields = [
            _field_record(name, group[name], n_rows, n_electrons)
            for name in group
            if name not in {"E_keV", "t_ang", "elec_id"} or name == ids_name
        ]
        case = _json_attr(handle, "case") or {}
        description: dict[str, Any] = {
            "path": os.fspath(path),
            "bytes": path.stat().st_size,
            "schema_version": int(header["schema_version"]),
            "complete": True,
            "created_utc": header.get("created_utc"),
            "pyrite_version": header.get("pyrite_version"),
            "case": {
                "name": header.get("case_name"),
                "E0_keV": float(header["E0_keV"]),
                "crystal": case.get("crystal"),
                "thickness_ang": case.get("thickness_ang"),
                "tilt_deg": case.get("tilt_deg"),
                "Ne": case.get("Ne"),
                "sha256": header.get("case_sha256"),
            },
            "seed": header.get("seed"),
            "provenance": _json_attr(handle, "provenance") or {},
            "scene": "scene" in handle,
            "spectrum_inputs": "spectrum_inputs" in handle,
            "row_order": header.get("row_order", "transport"),
            "segments": n_rows,
            "sampled_histories": n_electrons,
            "fields": fields,
        }
        if counts and ids is not None:
            sorted_rows = header.get("row_order") == "electron"
            histories, low, high = _distinct_count(ids, sorted_rows=sorted_rows)
            description["histories"] = histories
            description["history_id_range"] = None if low is None else [low, high]
            if "track_id" in group:
                description["tracks"] = _distinct_count(group["track_id"], sorted_rows=False)[0]
            if "generation" in group:
                generations = _generation_counts(group["generation"])
                description["segments_by_generation"] = {str(k): v for k, v in generations.items()}
                description["max_generation"] = max(generations, default=None)
    return description


def segment_field_names(description: dict[str, Any]) -> list[str]:
    """Per-segment field names of a :func:`describe_artifact` result."""
    return [record["name"] for record in description["fields"] if record["role"] == "segment"]
