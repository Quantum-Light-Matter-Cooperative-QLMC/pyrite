"""Streaming reuse of the canonical all-generation secondary energy ledger."""

from __future__ import annotations

from collections import defaultdict
from os import PathLike

import numpy as np
import pandas as pd

from .trajectories import TrajectoryArtifactError, _read_node
from .trajectory_summary import _blocks
from .transport.secondaries import secondary_energy_balance


def _select_columns(group, key, wanted, columns, max_rows):
    parts = {name: [] for name in columns}
    indices = []
    size = len(group[key])
    for a in range(0, size, max_rows):
        values = group[key][a : a + max_rows]
        keep = np.flatnonzero(np.isin(values, wanted))
        if not keep.size:
            continue
        indices.append(keep + a)
        for name in columns:
            parts[name].append(group[name][a : a + max_rows][keep])
    result = {
        name: np.concatenate(values)
        if values
        else np.empty((0, *group[name].shape[1:]), dtype=group[name].dtype)
        for name, values in parts.items()
    }
    return result, np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)


def _pair_block(group, original_rows, max_rows):
    on = bool(_read_node(group["positrons_transported"]))
    columns = ("row", "electron_id", "electron_launched", "electron_keV", "positron_keV")
    if on:
        columns += ("positron_launched", "positron_fate")
    events, event_ids = _select_columns(group["events"], "row", original_rows, columns, max_rows)
    local = {int(row): i for i, row in enumerate(original_rows)}
    events["row"] = np.array([local[int(row)] for row in events["row"]], dtype=np.int64)
    pair = {"events": events, "positrons_transported": on}
    if on:
        photons, _ = _select_columns(
            group["annihilation_photons"],
            "event",
            event_ids,
            ("event", "absorbed", "k_keV"),
            max_rows,
        )
        event_local = {int(event): i for i, event in enumerate(event_ids)}
        photons["event"] = np.array(
            [event_local[int(event)] for event in photons["event"]], dtype=np.int64
        )
        pair["annihilation_photons"] = photons
    return pair


def capture_secondary_energy_balance(
    path: str | PathLike[str], *, max_rows: int = 65536
) -> pd.DataFrame:
    """Sum canonical ``secondary_energy_balance`` terms over whole-history blocks.

    Requires a secondary-transport capture. Pair metadata and annihilation photons
    are filtered in bounded scans and their original row/event indices remapped
    to each local block. Returns ``term, energy_keV``; no energy formula is
    redefined here. Scans may repeat for large pair-event metadata tables.
    """
    import h5py

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    totals: dict[str, float] = defaultdict(float)
    with h5py.File(path, "r") as handle:
        group = handle["transport"]
        if "secondaries" not in group or "secondary_tracks" not in group:
            raise TrajectoryArtifactError("energy ledger requires a secondary-transport capture")
        meta = {key: _read_node(group[key]) for key in ("inelastic", "secondaries")}
        offset = 0
        for block in _blocks(path, max_rows):
            block.update(meta)
            track = block["track_id"]
            order = np.lexsort((block["substep_id"], block["flight_id"], track))
            first = order[np.r_[True, np.diff(track[order]) != 0]] if len(order) else order
            unique, dense = np.unique(track, return_inverse=True)
            block["track_id"] = dense
            block["secondary_tracks"] = {
                "track_id": np.arange(len(unique)),
                "generation": block["generation"][first],
                "electron_id": block["electron_id"][first],
                "launch_E_keV": block["E_start_keV"][first],
            }
            if "pair_production" in group:
                stop = offset + len(track)
                original = (
                    handle["transport_order"][offset:stop]
                    if "transport_order" in handle
                    else np.arange(offset, stop)
                )
                block["pair_production"] = _pair_block(group["pair_production"], original, max_rows)
            for key, value in secondary_energy_balance(block).items():
                totals[key] += float(value)
            offset += len(track)
    return pd.DataFrame(sorted(totals.items()), columns=["term", "energy_keV"])
