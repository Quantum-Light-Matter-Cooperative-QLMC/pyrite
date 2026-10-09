"""Bounded whole-track HDF5 selection, independent of rendering libraries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..montecarlo.trajectories import TrajectoryArtifactError, read_trajectory_header, row_fields

_BLOCK = 65536
_REQUIRED = ("r_mid", "v_hat", "L_ang", "electron_id")
_IDENTITY = ("track_id", "parent_id", "generation")
_CONTROLS = (
    "E_start_keV",
    "t_start_ang",
    "t0_ang",
    "event_kind",
    "hard_channel",
    "flight_id",
    "substep_id",
)


@dataclass(frozen=True)
class SelectionPlan:
    """Counts and costs before geometry allocation; ranges are stored-row runs."""

    path: Path
    signature: tuple[int, int]
    ranges: tuple[tuple[int, int], ...]
    fields: tuple[str, ...]
    segments: int
    array_bytes: int
    estimated_peak_bytes: int
    case: dict
    scene: dict | None
    provenance: dict
    header: dict


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def plan_selection(path, *, histories=(), tracks=(), attributes=(), max_segments=1_000_000):
    """Intersect complete history/track IDs; fail rather than truncate a track.

    Only bounded identity blocks are scanned. Costs include selected arrays,
    original row IDs, sorting, two-point lines, one filtered copy and renderer
    copies (an admission estimate, not a hard process-memory guarantee).
    """
    import h5py

    path = Path(path)
    signature = _signature(path)
    header = read_trajectory_header(path)
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    ranges = []
    count = 0
    with h5py.File(path, "r") as handle:
        group = handle["transport"]
        for name in _REQUIRED:
            if name not in group:
                raise TrajectoryArtifactError(f"missing required segment field {name}")
        if group["L_ang"].ndim != 1:
            raise TrajectoryArtifactError("L_ang must be a one-dimensional segment field")
        n = len(group["L_ang"])
        if tracks and "track_id" not in group:
            raise TrajectoryArtifactError("track selection unavailable: capture has no track_id")
        fields = tuple(
            dict.fromkeys(
                (*_REQUIRED, *[f for f in (*_IDENTITY, *_CONTROLS) if f in group], *attributes)
            )
        )
        allowed_fields = row_fields()
        for name in fields:
            if name not in allowed_fields or name not in group or group[name].shape[:1] != (n,):
                raise TrajectoryArtifactError(f"{name!r} is not an available segment attribute")
            if not np.issubdtype(group[name].dtype, np.number):
                raise TrajectoryArtifactError(f"{name!r} is not a numeric segment attribute")
        for start in range(0, n, _BLOCK):
            stop = min(n, start + _BLOCK)
            keep = np.ones(stop - start, dtype=bool)
            if histories:
                keep &= np.isin(group["electron_id"][start:stop], histories)
            if tracks:
                keep &= np.isin(group["track_id"][start:stop], tracks)
            changes = np.diff(np.r_[False, keep, False].astype(np.int8))
            for a, b in zip(
                np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
            ):
                a, b = start + int(a), start + int(b)
                if ranges and ranges[-1][1] == a:
                    ranges[-1] = (ranges[-1][0], b)
                else:
                    ranges.append((a, b))
                count += b - a
            if count > max_segments:
                raise TrajectoryArtifactError(
                    f"selection exceeds {max_segments:,} segments; select fewer complete histories/tracks or increase --max-segments"
                )
        if not count:
            raise TrajectoryArtifactError("selection contains no segments")
        row_bytes = sum(
            node.dtype.itemsize * int(np.prod(node.shape[1:]))
            for name in fields
            for node in [group[name]]
        )
        case = json.loads(handle["case"].attrs["json"])
        scene = json.loads(handle["scene"].attrs["json"]) if "scene" in handle else None
        provenance = json.loads(handle["provenance"].attrs["json"])
    if _signature(path) != signature:
        raise TrajectoryArtifactError("capture changed during selection; retry")
    array_bytes = count * (row_bytes + 8)
    return SelectionPlan(
        path,
        signature,
        tuple(ranges),
        fields,
        count,
        array_bytes,
        1024**3 + 3 * array_bytes + count * 512,
        case,
        scene,
        provenance,
        header,
    )


def load_selection(plan: SelectionPlan, *, memory_budget_bytes=2 * 1024**3):
    """Read only planned row runs/fields, restoring captured transport order."""
    import h5py

    if plan.estimated_peak_bytes > memory_budget_bytes:
        raise TrajectoryArtifactError(
            f"estimated selection memory {plan.estimated_peak_bytes / 1024**2:.1f} MiB exceeds budget {memory_budget_bytes / 1024**2:.1f} MiB; select fewer histories/tracks or increase --memory-mib"
        )
    if _signature(plan.path) != plan.signature:
        raise TrajectoryArtifactError("capture changed after selection; retry")
    with h5py.File(plan.path, "r") as handle:
        group = handle["transport"]
        data = {
            name: np.empty((plan.segments, *group[name].shape[1:]), dtype=group[name].dtype)
            for name in plan.fields
        }
        rows = np.empty(plan.segments, dtype=np.int64)
        offset = 0
        for a, b in plan.ranges:
            size = b - a
            for name in plan.fields:
                data[name][offset : offset + size] = group[name][a:b]
            rows[offset : offset + size] = (
                handle["transport_order"][a:b] if "transport_order" in handle else np.arange(a, b)
            )
            offset += size
        order = np.argsort(rows, kind="stable")
        data = {name: values[order] for name, values in data.items()}
        data["segment_id"] = rows[order]
    if _signature(plan.path) != plan.signature:
        raise TrajectoryArtifactError("capture changed during loading; retry")
    # Legacy captures retain history identity; never invent secondary ancestry.
    for name in _IDENTITY:
        if name not in data:
            data[name] = np.full(plan.segments, -1, dtype=np.int64)
    return data
