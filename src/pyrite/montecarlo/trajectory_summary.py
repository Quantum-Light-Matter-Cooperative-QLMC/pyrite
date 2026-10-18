"""Renderer-free, streaming reductions of saved trajectory captures.

Histograms have explicit edges and units in column names. Memory scales with
``max_rows`` (or one oversized primary history), histogram bins and the number
of distinct layer/event/generation codes, never the total segment count.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from .trajectories import TrajectoryArtifactError, iter_electron_row_blocks
from .transport.events import (
    EVENT_CUTOFF,
    EVENT_DELTA,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
)

_FATES = {
    int(EVENT_EXIT_TOP): "backscattered",
    int(EVENT_EXIT_BOTTOM): "transmitted",
    int(EVENT_EXIT_SIDE): "side_exited",
    int(EVENT_CUTOFF): "cutoff_stopped",
}
_FIELDS = (
    "electron_id",
    "r_mid",
    "L_ang",
    "v_hat",
    "E_start_keV",
    "E_end_keV",
    "event_kind",
    "flight_id",
    "substep_id",
    "layer",
)
_OPTIONAL = ("track_id", "generation", "hard_channel", "hard_W_keV", "hard_radiative_k_eV")


def _edges(values: object, name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} must contain at least two finite edges")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return edges


def _blocks(path, max_rows):
    import h5py

    from .trajectories import read_trajectory_header, transport_view

    read_trajectory_header(path)
    with h5py.File(path, "r") as handle:
        names = transport_view(handle).row_fields()
    missing = set(_FIELDS) - set(names)
    if missing:
        raise TrajectoryArtifactError(f"summary requires stored row fields: {sorted(missing)}")
    fields = _FIELDS + tuple(key for key in _OPTIONAL if key in names)
    yield from iter_electron_row_blocks(path, max_rows, fields=fields, include_shared=False)


def depth_loss_profile(block: dict[str, Any], edges: object) -> pd.DataFrame:
    """Attribute continuous segment loss uniformly along its straight depth span.

    Source: the saved-segment analysis convention in
    ``docs/physics/beam-transport/transport-outputs.md``. For nonzero depth span,
    each bin receives ``(E_start-E_end) * overlap / abs(L*v_z)``. A transverse
    segment puts its whole loss in its midpoint bin. Bins are left-closed,
    right-open, except the final right edge is included. Group by stored layer;
    use edges containing layer interfaces to avoid bins straddling layers.
    Complete coverage conserves continuous loss; reversing direction preserves
    attribution. Partial coverage retains only the covered fraction. This is
    keV per bin, without mass, volume, or primary normalization; cutoff residuals
    and hard-transfer deposits are excluded.

    Validation: trajectory-depth-loss-profile
    """
    edges = _edges(edges, "depth_edges_ang")
    z = np.asarray(block["r_mid"])[:, 2]
    dz = np.asarray(block["L_ang"]) * np.asarray(block["v_hat"])[:, 2]
    low, high = z - abs(dz) / 2, z + abs(dz) / 2
    loss = np.asarray(block["E_start_keV"]) - np.asarray(block["E_end_keV"])
    layers = np.asarray(block["layer"])
    transverse = dz == 0
    point_bin = np.searchsorted(edges, z, side="right") - 1
    point_bin[z == edges[-1]] = len(edges) - 2
    rows = []
    for layer in np.unique(layers):
        where = layers == layer
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            overlap = np.maximum(0, np.minimum(high, b) - np.maximum(low, a))
            share = np.divide(overlap, abs(dz), out=np.zeros_like(z), where=~transverse)
            share[transverse] = point_bin[transverse] == j
            rows.append((int(layer), a, b, float(np.sum(loss[where] * share[where]))))
    return pd.DataFrame(
        rows, columns=["layer", "depth_low_ang", "depth_high_ang", "continuous_loss_keV"]
    )


def _history_rows(block):
    """Primary-only endpoints/ranges; descendants never determine primary fate."""
    n = len(block["electron_id"])
    primary = np.asarray(block.get("generation", np.zeros(n))) == 0
    ids = np.asarray(block["electron_id"])[primary]
    if not ids.size:
        return []
    order = np.lexsort((block["substep_id"][primary], block["flight_id"][primary], ids))
    r = block["r_mid"][primary][order]
    v = block["v_hat"][primary][order]
    length = block["L_ang"][primary][order]
    ids = ids[order]
    start, end = r - length[:, None] * v / 2, r + length[:, None] * v / 2
    cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    at = cuts[:-1]
    terminal = cuts[1:] - 1
    kinds = block["event_kind"][primary][order][terminal]
    energies = block["E_end_keV"][primary][order][terminal]
    angles = np.arccos(np.clip(v[terminal, 2], -1, 1))
    penetration = np.maximum.reduceat(np.maximum(start[:, 2], end[:, 2]), at)
    ranges = np.add.reduceat(length, at)
    return list(
        zip(
            ids[at].tolist(),
            [_FATES.get(int(kind), "unfinished") for kind in kinds],
            np.maximum(0, penetration).tolist(),
            ranges.tolist(),
            energies.tolist(),
            angles.tolist(),
            strict=True,
        )
    )


def iter_primary_summaries(
    path: str | os.PathLike[str],
    *,
    max_rows: int = 65536,
) -> Iterator[pd.DataFrame]:
    """Yield bounded tidy tables, one row per primary history with material rows.

    Columns: history ID, terminal primary fate, maximum primary depth and
    material path length (Angstrom), terminal pre-event energy (keV), and polar
    direction angle to sample +z (rad). Missed entries have no material rows;
    unfinished histories are explicit. Secondary rows do not alter these values.
    """
    for block in _blocks(path, max_rows):
        yield pd.DataFrame(
            _history_rows(block),
            columns=[
                "electron_id",
                "fate",
                "penetration_ang",
                "range_ang",
                "exit_energy_keV",
                "exit_angle_rad",
            ],
        )


def _histogram(values, edges):
    counts = np.histogram(values, edges)[0]
    # Keep out-of-range observations visible rather than silently losing them.
    return np.r_[np.count_nonzero(values < edges[0]), counts, np.count_nonzero(values > edges[-1])]


def _hist_frame(counts, edges, unit):
    return pd.DataFrame(
        {
            f"low_{unit}": np.r_[-np.inf, edges[:-1], edges[-1]],
            f"high_{unit}": np.r_[edges[0], edges[1:], np.inf],
            "count": counts,
        }
    )


def summarize_trajectories(
    path: str | os.PathLike[str],
    *,
    depth_edges_ang: object,
    energy_edges_keV: object,
    angle_edges_rad: object = np.linspace(0, np.pi, 37),
    range_edges_ang: object | None = None,
    max_rows: int = 65536,
) -> dict[str, pd.DataFrame]:
    """Reduce a complete capture to fixed-bin tidy tables without transport.

    Returns ``depth_loss``, ``fates``, ``penetration``, ``range``, ``exit_energy``,
    ``exit_angle``, ``hard_events`` and ``generations``. Fate fractions use the stored sampled primary count ``Ne`` as
    denominator, with missed and otherwise rowless histories explicit. Energy/angle distributions include escaped primaries only.
    Hard spectra use recorded transfer W or emitted photon energy, by event
    kind/channel. Generations count transported tracks once at first row and
    sum their launch energies; subthreshold particles without tracks are excluded.
    Explicit under/overflow bins preserve histogram counts. Initial sampled-beam
    metrics and the full secondary energy ledger are separate artifact analyses.
    """
    depth = _edges(depth_edges_ang, "depth_edges_ang")
    energy = _edges(energy_edges_keV, "energy_edges_keV")
    angle = _edges(angle_edges_rad, "angle_edges_rad")
    ranges = depth if range_edges_ang is None else _edges(range_edges_ang, "range_edges_ang")
    fates: dict[str, int] = dict.fromkeys([*_FATES.values(), "unfinished"], 0)
    hist = {
        "penetration": np.zeros(len(depth) + 1, dtype=np.int64),
        "range": np.zeros(len(ranges) + 1, dtype=np.int64),
        "exit_energy": np.zeros(len(energy) + 1, dtype=np.int64),
        "exit_angle": np.zeros(len(angle) + 1, dtype=np.int64),
    }
    dose: dict[tuple[int, float, float], float] = defaultdict(float)
    hard: dict[tuple[int, int], np.ndarray] = {}
    generations: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])
    for block in _blocks(path, max_rows):
        for layer, a, b, loss in depth_loss_profile(block, depth).itertuples(
            index=False, name=None
        ):
            dose[layer, a, b] += loss
        rows = _history_rows(block)
        if rows:
            values = np.asarray([row[2:] for row in rows])
            for row in rows:
                fates[row[1]] += 1
            escaped = np.array(
                [row[1] in ("backscattered", "transmitted", "side_exited") for row in rows]
            )
            hist["penetration"] += _histogram(values[:, 0], depth)
            hist["range"] += _histogram(values[:, 1], ranges)
            hist["exit_energy"] += _histogram(values[escaped, 2], energy)
            hist["exit_angle"] += _histogram(values[escaped, 3], angle)
        _reduce_events(block, energy, hard, generations)
    total = _fate_population(path, fates)
    tables = {
        key: _hist_frame(hist[key], edges, unit)
        for key, edges, unit in (
            ("penetration", depth, "ang"),
            ("range", ranges, "ang"),
            ("exit_energy", energy, "keV"),
            ("exit_angle", angle, "rad"),
        )
    }
    tables["depth_loss"] = pd.DataFrame(
        [(*key, value) for key, value in sorted(dose.items())],
        columns=["layer", "depth_low_ang", "depth_high_ang", "continuous_loss_keV"],
    )
    tables["fates"] = pd.DataFrame(
        [(key, value, value / total if total else 0.0) for key, value in fates.items()],
        columns=["fate", "count", "fraction"],
    )
    frames = [
        _hist_frame(counts, energy, "keV").assign(event_kind=kind, hard_channel=channel)
        for (kind, channel), counts in sorted(hard.items())
    ]
    tables["hard_events"] = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["low_keV", "high_keV", "count", "event_kind", "hard_channel"])
    )
    tables["generations"] = pd.DataFrame(
        [(gen, int(v[0]), v[1]) for gen, v in sorted(generations.items())],
        columns=["generation", "tracks", "launch_energy_keV"],
    )
    return tables


def _fate_population(path, fates):
    """Use the sampled primary population, preserving missed/rowless histories."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle["transport"]
        represented = sum(fates.values())
        total = int(group["Ne"][()]) if "Ne" in group else represented
        missed = int(group["n_missed"][()]) if "n_missed" in group else 0
    if missed:
        fates["missed"] = missed
    rowless = total - represented - missed
    if rowless > 0:
        fates["without_rows"] = rowless
    return total


def _reduce_events(block, edges, hard, generations):
    kind = block["event_kind"]
    channel = np.asarray(block.get("hard_channel", np.zeros(len(kind), dtype=int)))
    transfer = np.asarray(block.get("hard_W_keV", np.zeros(len(kind))))
    photon = np.asarray(block.get("hard_radiative_k_eV", np.zeros(len(kind)))) * 1e-3
    is_hard = (
        np.isin(kind, [EVENT_HARD_INELASTIC, EVENT_HARD_RADIATIVE, EVENT_DELTA])
        | (transfer > 0)
        | (photon > 0)
    )
    energy = np.where(photon > 0, photon, transfer)
    for event, ch in np.unique(np.column_stack([kind[is_hard], channel[is_hard]]), axis=0):
        key = (int(event), int(ch))
        hard.setdefault(key, np.zeros(len(edges) + 1, dtype=np.int64))
        hard[key] += _histogram(energy[is_hard & (kind == event) & (channel == ch)], edges)
    track = block.get("track_id", block["electron_id"])
    order = np.lexsort((block["substep_id"], block["flight_id"], track))
    first = order[np.r_[True, np.diff(track[order]) != 0]] if len(order) else order
    generation = np.asarray(block.get("generation", np.zeros(len(kind), dtype=int)))
    for gen in np.unique(generation[first]):
        selected = first[generation[first] == gen]
        generations[int(gen)][0] += len(selected)
        generations[int(gen)][1] += float(np.sum(block["E_start_keV"][selected]))
