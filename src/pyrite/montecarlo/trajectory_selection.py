"""Select whole histories or tracks of a trajectory artifact without loading it.

A selection is a list of stored-row runs. Schema-2 artifacts store rows
electron-major, so one primary history (``electron_id``, which every secondary
of its shower keeps) is one contiguous run, found by bisection. ``track_id``
is not sorted and is matched in bounded blocks. Neither path reads segment
coordinates; :func:`read_selection` and the VTK export read only the selected
runs.

Grooved vacuum legs carry only ``vacuum_elec_id``. They join a selection made
of whole histories and are left out of any track selection, whose ancestry
they cannot express.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .trajectories import TrajectoryArtifactError, read_trajectory_header, row_fields

_BLOCK = 1 << 16
# Compatibility aliases are hard links to canonical row fields.
_ALIASES = frozenset({"E_keV", "t_ang", "elec_id"})
# Resolved history IDs recorded with a selection; larger sets record the
# request (``first``/``sample`` and ``seed``) that reproduces them.
_MAX_RECORDED_IDS = 10_000

Ranges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TrajectorySelection:
    """Stored-row runs of complete histories/tracks in one artifact.

    ``ranges`` and ``vacuum_ranges`` are ascending, disjoint ``[start, stop)``
    runs of stored material rows and of vacuum-leg rows. ``request`` is the
    JSON-safe record of what was asked for and what it resolved to.
    """

    path: Path
    ranges: Ranges
    vacuum_ranges: Ranges
    segments: int
    histories: int
    request: dict[str, Any]


@dataclass(frozen=True)
class SelectedSegments:
    """Selected row fields in transported order, with their units.

    ``transport["segment_id"]`` is each row's index in the full transported
    result, so rows keep their identity outside the artifact.
    """

    selection: TrajectorySelection
    transport: dict[str, np.ndarray]
    units: dict[str, str]


def _runs(rows: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of a sorted row-index array."""
    if not rows.size:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = rows[np.r_[0, cuts]]
    stops = rows[np.r_[cuts - 1, rows.size - 1]] + 1
    return [(offset + int(a), offset + int(b)) for a, b in zip(starts, stops, strict=True)]


def _merge(ranges: Iterable[tuple[int, int]]) -> Ranges:
    merged: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if merged and merged[-1][1] >= a:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        elif b > a:
            merged.append((a, b))
    return tuple(merged)


def _bisect(ids, value: int, side: str) -> int:
    """``np.searchsorted`` on a sorted on-disk dataset, one element per probe."""
    low, high = 0, int(ids.shape[0])
    while low < high:
        mid = (low + high) // 2
        probe = int(ids[mid])
        if probe < value or (side == "right" and probe == value):
            low = mid + 1
        else:
            high = mid
    return low


def _match_runs(dataset, values, within: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Runs of rows inside ``within`` whose ``dataset`` value is in ``values``."""
    values = np.asarray(sorted(values))
    found = []
    for a, b in within:
        for start in range(a, b, _BLOCK):
            stop = min(b, start + _BLOCK)
            found.extend(_runs(np.flatnonzero(np.isin(dataset[start:stop], values)), start))
    return found


def _history_ids(ids, *, limit: int | None = None) -> np.ndarray:
    """Distinct history IDs in stored order, stopping after ``limit`` of them."""
    seen: dict[int, None] = {}
    for start in range(0, int(ids.shape[0]), _BLOCK):
        block = ids[start : start + _BLOCK]
        _, first = np.unique(block, return_index=True)
        for value in block[np.sort(first)]:
            seen.setdefault(int(value), None)
            if limit is not None and len(seen) >= limit:
                return np.fromiter(seen, dtype=np.int64)
    return np.fromiter(seen, dtype=np.int64)


def _empty_message(path: Path, handle) -> str:
    ids = handle["transport/electron_id"]
    present = _history_ids(ids)
    sampled = (
        handle["transport/initial_E_keV"].shape[0]
        if "initial_E_keV" in handle["transport"]
        else None
    )
    span = f"IDs {present.min()}..{present.max()}" if present.size else "none"
    total = f" of {sampled} sampled" if sampled is not None else ""
    return (
        f"selection contains no segments in {path}; {present.size} histories{total} "
        f"have segments ({span})"
    )


def select_trajectories(
    path: str | os.PathLike[str],
    *,
    histories: Iterable[int] = (),
    tracks: Iterable[int] = (),
    first: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> TrajectorySelection:
    """Select complete histories and/or tracks of a trajectory artifact.

    Parameters
    ----------
    path
        Complete trajectory artifact.
    histories
        Primary history IDs (``electron_id``); a history includes every
        secondary track of its shower.
    tracks
        Track IDs (``track_id``); intersected with any history selection.
    first
        The first ``first`` histories that have segments, in stored order.
    sample
        ``sample`` histories drawn without replacement from those that have
        segments, reproducibly from ``seed``; fewer if fewer exist.
    seed
        Seed of ``sample``.

    ``histories``, ``first`` and ``sample`` are mutually exclusive.

    Raises
    ------
    ValueError
        Conflicting or non-positive options.
    TrajectoryArtifactError
        Incomplete or foreign artifact, a capture without ``track_id`` asked
        for tracks, or an empty selection (the message lists the history IDs
        that have segments).
    """
    import h5py

    path = Path(path)
    histories = sorted({int(value) for value in histories})
    tracks = sorted({int(value) for value in tracks})
    if sum(bool(option) for option in (histories, first is not None, sample is not None)) > 1:
        raise ValueError("histories, first and sample are mutually exclusive")
    for name, value in (("first", first), ("sample", sample)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if any(value < 0 for value in (*histories, *tracks)):
        raise ValueError("history and track IDs are non-negative")
    header = read_trajectory_header(path)
    with h5py.File(path, "r") as handle:
        group = handle["transport"]
        if "electron_id" not in group:
            raise TrajectoryArtifactError(f"{path} has no electron_id segment field")
        if tracks and "track_id" not in group:
            raise TrajectoryArtifactError(
                f"track selection unavailable: {path} records no track_id (no transported secondaries)"
            )
        ids = group["electron_id"]
        n_rows = int(ids.shape[0])
        sorted_rows = header.get("row_order") == "electron"
        if first is not None:
            histories = sorted(_history_ids(ids, limit=first).tolist())
        elif sample is not None:
            present = _history_ids(ids)
            rng = np.random.default_rng(seed)
            histories = sorted(
                rng.choice(present, size=min(sample, present.size), replace=False).tolist()
            )
        by_history = bool(histories) or first is not None or sample is not None
        if not by_history:
            ranges: list[tuple[int, int]] = [(0, n_rows)]
        elif sorted_rows:
            ranges = [
                (_bisect(ids, value, "left"), _bisect(ids, value, "right")) for value in histories
            ]
        else:  # schema 1: rows in transported order
            ranges = _match_runs(ids, histories, [(0, n_rows)])
        if tracks:
            ranges = _match_runs(group["track_id"], tracks, _merge(ranges))
        merged = _merge(ranges)
        segments = sum(b - a for a, b in merged)
        if not segments:
            raise TrajectoryArtifactError(_empty_message(path, handle))
        vacuum: Ranges = ()
        if by_history and not tracks and "vacuum_elec_id" in group:
            legs = group["vacuum_elec_id"]
            vacuum = _merge(_match_runs(legs, histories, [(0, int(legs.shape[0]))]))
        selected = {int(value) for a, b in merged for value in np.unique(ids[a:b])}
    request: dict[str, Any] = {
        "histories": None if first is not None or sample is not None else histories or None,
        "tracks": tracks or None,
        "first": first,
        "sample": sample,
        "seed": seed if sample is not None else None,
        "selected_histories": len(selected),
        "segments": segments,
        "vacuum_legs": sum(b - a for a, b in vacuum),
    }
    if len(selected) <= _MAX_RECORDED_IDS:
        request["history_ids"] = sorted(selected)
    return TrajectorySelection(path, merged, vacuum, segments, len(selected), request)


def read_selection(
    selection: TrajectorySelection,
    fields: Iterable[str] | None = None,
    *,
    max_segments: int | None = None,
) -> SelectedSegments:
    """Read the selected rows of ``fields`` (default: every row field).

    Rows return in transported order, with ``segment_id`` giving each row's
    transported index. Only the selected runs are read.

    Raises
    ------
    TrajectoryArtifactError
        A requested field is not a row field of the artifact, or the selection
        exceeds ``max_segments``.
    """
    import h5py

    if max_segments is not None and selection.segments > max_segments:
        raise TrajectoryArtifactError(
            f"selection has {selection.segments:,} segments, more than max_segments="
            f"{max_segments:,}; select fewer histories or tracks"
        )
    with h5py.File(selection.path, "r") as handle:
        group = handle["transport"]
        n_rows = int(group["electron_id"].shape[0])
        available = [
            key
            for key in group
            if key in row_fields() and key not in _ALIASES and group[key].attrs.get("pyrite_row")
        ]
        if fields is None:
            fields = available
        else:
            fields = list(dict.fromkeys(fields))
            unknown = [
                key
                for key in fields
                if key not in group or group[key].shape[:1] != (n_rows,) or key not in row_fields()
            ]
            if unknown:
                raise TrajectoryArtifactError(
                    f"not segment fields of {selection.path}: {', '.join(unknown)}; "
                    f"available: {', '.join(available)}"
                )
        data = {
            key: np.empty((selection.segments, *group[key].shape[1:]), dtype=group[key].dtype)
            for key in fields
        }
        rows = np.empty(selection.segments, dtype=np.int64)
        order = handle["transport_order"] if "transport_order" in handle else None
        offset = 0
        for a, b in selection.ranges:
            for key in fields:
                data[key][offset : offset + b - a] = group[key][a:b]
            rows[offset : offset + b - a] = order[a:b] if order is not None else np.arange(a, b)
            offset += b - a
        units = {
            key: str(group[key].attrs["units"]) for key in fields if "units" in group[key].attrs
        }
    restore = np.argsort(rows, kind="stable")
    transport = {key: values[restore] for key, values in data.items()}
    transport["segment_id"] = rows[restore]
    units["segment_id"] = "1"
    return SelectedSegments(selection, transport, units)
