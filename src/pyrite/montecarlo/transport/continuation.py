"""Opt-in bounded continuation for the exact CUDA transport core (#789).

The batch driver in :mod:`.batching` gives every electron of a launch a dense
row capacity and runs it to completion, so live scratch scales with the
longest track and a launch's lanes idle while its longest histories finish.
This driver instead keeps a fixed population of *active* histories, launches
them for at most ``rows_per_round`` rows each, and between launches retires
finished histories and refills their lanes from the pending primaries. Live
scratch is ``active x rows_per_round`` rows whatever the track length.

Results do not depend on the schedule. The kernel yields only between steps,
where its loop-carried state is the per-history record held here, and every
draw is addressed by (history stream, per-history counter); a resumed history
continues exactly as one uninterrupted run. Rows of each round are retained
in launch order and finally scattered to the canonical electron-major,
step-minor positions the full-history driver produces, so the active
population, the row quantum and the refill order change only performance.

Supported: the exact CUDA core under continuous losses, with or without
coupled radiative transport. The shell soft/hard mode, straggling, the
transport LUT and every CPU core are rejected; their state is resumable in
principle but has no continuation regression yet.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cores import (
    EXIT_BACKSCATTERED,
    EXIT_CUTOFF_STOPPED,
    EXIT_SIDE,
    EXIT_STEP_LIMITED,
    EXIT_TRANSMITTED,
)

# Kernel exit code of a history that reached its launch's row quantum.
EXIT_SUSPENDED = np.int8(6)
# Per-history kernel state, three of each: int32 (step, flight, substep),
# uint64 stream counters and float64 open optical depths (elastic, hard,
# radiative). A zero step marks a history that has not yet launched.
_STATE_WORDS = 3
# Exit tallies in the batch driver's return order.
_TALLIED = (EXIT_BACKSCATTERED, EXIT_TRANSMITTED, EXIT_SIDE, EXIT_CUTOFF_STOPPED, EXIT_STEP_LIMITED)
# Compaction temporaries per retained row: int64 row index, slot and source.
_GATHER_INDEX_BYTES = 3 * 8
# Per-lane round arrays: int32 slot, int64 row count, int8 exit code.
_LANE_BYTES = 4 + 8 + 1


@dataclass(frozen=True)
class TransportContinuationConfig:
    """Bounded-continuation schedule; no field changes results.

    ``rows_per_round`` is the row quantum each active history may emit per
    launch. ``active_histories`` caps the concurrent population; ``None``
    takes as many as ``PerElectronTransportConfig.scratch_budget_bytes`` and
    ``max_batch`` allow. The budget covers the round's row scratch, its
    compaction copy and gather indices, and the lane arrays, sized from the
    enabled row schema. Ne-sized per-history state and retained output are
    outside it: bounded scratch does not bound total trajectory storage.

    ``device_staging`` keeps each round's retained rows on the device and
    downloads the reassembled columns once, trading up to twice the retained
    output in device memory for fewer, contiguous host copies. Off, rows move
    to the host every round and are reassembled there.

    ``on_round`` (diagnostics only) receives one ``phase="round"`` dict per
    launch with the launched population, rows, finished/refilled counts and
    scratch bytes, then one ``phase="assembly"`` dict timing the reassembly.
    """

    rows_per_round: int = 256
    active_histories: int | None = None
    device_staging: bool = False
    on_round: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self):
        if int(self.rows_per_round) < 1:
            raise ValueError("rows_per_round must be a positive integer")
        if self.active_histories is not None and int(self.active_histories) < 1:
            raise ValueError("active_histories must be a positive integer or None")


def continuation_row_bytes(scratch_row):
    """Exact bytes of one row across every enabled scratch column."""
    return sum(int(a.nbytes) for a in scratch_row)


def active_population(Ne, row_bytes, config, cont):
    """Concurrent histories that fit the scratch budget, or a clear error."""
    quantum = int(cont.rows_per_round)
    per_lane = quantum * (2 * row_bytes + _GATHER_INDEX_BYTES) + _LANE_BYTES
    fit = int(config.scratch_budget_bytes) // per_lane
    if fit < 1:
        raise ValueError(
            f"one continuation lane needs {per_lane} bytes ({quantum} rows of "
            f"{row_bytes} bytes plus staging), above scratch_budget_bytes="
            f"{int(config.scratch_budget_bytes)}; lower rows_per_round"
        )
    limit = int(config.max_batch)
    if cont.active_histories is not None:
        limit = min(limit, int(cont.active_histories))
    return max(1, min(fit, limit, int(Ne))), per_lane


def _gather_index(xp, counts, quantum):
    """Flat scratch slot of each retained row, slot-major and step-minor."""
    ends = xp.cumsum(counts)
    total = int(ends[-1]) if counts.size else 0
    rows = xp.arange(total, dtype=xp.int64)
    lane = xp.searchsorted(ends, rows, side="right")
    rows -= ends[lane] - counts[lane]
    rows += lane * quantum
    return rows


def _canonical_destinations(chunks, rows_total):
    """Destination rows of every retained chunk in canonical order.

    Chunks are in launch order and each holds, lane by lane, a contiguous run
    of one history's next rows, so a per-history cursor starting at that
    history's exclusive prefix places every row exactly once.
    """
    cursor = np.cumsum(rows_total) - rows_total
    for histories, counts in chunks:
        live = counts > 0
        histories, counts = histories[live], counts[live]
        starts = np.cumsum(counts) - counts
        base = cursor[histories]
        cursor[histories] += counts
        yield np.repeat(base - starts, counts) + np.arange(int(counts.sum()), dtype=np.int64)


def _pending_order(Ne, history_order):
    pending = np.arange(Ne, dtype=np.int32) if history_order is None else history_order
    pending = np.asarray(pending, dtype=np.int32)
    if not np.array_equal(np.sort(pending), np.arange(Ne)):
        raise ValueError("history_order must be a permutation of range(Ne)")
    return pending


def _assemble(xp, chunks, rows_total, retained, columns, out_bufs, nseg, keep_on_device):
    """Scatter retained rounds into canonical rows; the joined device rows or None."""
    joined = None
    targets = out_bufs
    if keep_on_device:
        joined = tuple(
            xp.empty((nseg,) + a.shape[1:], dtype=a.dtype) if a.shape[0] else a for a in columns
        )
        targets = joined
    for dest, chunk in zip(_canonical_destinations(chunks, rows_total), retained, strict=True):
        dest = xp.asarray(dest) if keep_on_device else dest
        for target, column in zip(targets, chunk, strict=True):
            if column.shape[0]:
                target[dest] = column
    return joined


def _drive_continuation_rounds(
    core,
    core_args,
    xp,
    Ne,
    max_segments,
    d_keys,
    d_stragg,
    straggling_args,
    extra_args,
    alloc_scratch,
    out_bufs,
    to_host,
    config,
    keep_on_device,
    on_batch=None,
    history_order=None,
):
    """Host-orchestrated rounds of bounded continuation; see the module doc.

    Arguments follow :func:`.batching._drive_per_electron_batches`, with the
    row-column allocator ``alloc_scratch(rows)`` instead of the schema flags.
    ``history_order`` (tests) permutes the refill order. The return value is
    that driver's ``(nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited,
    joined, stragg_dE)``.
    """
    cont = config.continuation
    quantum = int(cont.rows_per_round)
    row_bytes = continuation_row_bytes(alloc_scratch(1))
    n_active, per_lane = active_population(Ne, row_bytes, config, cont)
    scratch = alloc_scratch(n_active * quantum)
    # Frozen rows keep seven columns; the kernel still takes the empty end fields.
    columns = scratch[: len(out_bufs)]
    on_device = keep_on_device or cont.device_staging
    seg_count = xp.zeros(n_active, dtype=xp.int64)
    exit_code = xp.zeros(n_active, dtype=xp.int8)
    state = tuple(xp.zeros((Ne, _STATE_WORDS), dtype=t) for t in (xp.int32, xp.uint64, xp.float64))
    pending = _pending_order(Ne, history_order)
    rows_total = np.zeros(Ne, dtype=np.int64)
    chunks, retained = [], []
    tallies = dict.fromkeys(_TALLIED, 0)
    active = pending[:n_active]
    next_pending = active.size
    nseg = 0
    from ..runner import _nsys_pop, _nsys_push

    while active.size:
        m = int(active.size)
        _nsys_push("pyrite.transport.continuation.launch")
        core(
            (0, m, quantum, d_keys),
            *core_args,
            scratch,
            (seg_count[:m], exit_code[:m]),
            straggling_args,
            *extra_args,
            continuation=(xp.asarray(active), *state),
        )
        _nsys_pop()
        # The first device read in the stage range is also the launch's sync.
        _nsys_push("pyrite.transport.continuation.stage")
        index = _gather_index(xp, seg_count[:m], quantum)
        gathered = tuple(a[index] if a.shape[0] else a for a in columns)
        nseg += int(index.size)
        del index
        retained.append(gathered if on_device else tuple(to_host(a) for a in gathered))
        del gathered
        counts = to_host(seg_count[:m]).astype(np.int64)
        codes = to_host(exit_code[:m])
        _nsys_pop()
        if nseg > max_segments:
            raise RuntimeError("segment buffer exhausted")
        chunks.append((active.copy(), counts))
        rows_total[active] += counts
        for code in tallies:
            tallies[code] += int((codes == code).sum())
        finished = codes != EXIT_SUSPENDED
        refill = pending[next_pending : next_pending + int(finished.sum())]
        next_pending += refill.size
        if cont.on_round is not None:
            cont.on_round(
                {
                    "phase": "round",
                    "round": len(chunks) - 1,
                    "launched": m,
                    "rows": int(counts.sum()),
                    "max_rows": int(counts.max()),
                    "finished": int(finished.sum()),
                    "refilled": int(refill.size),
                    "pending": int(Ne - next_pending),
                    "rows_per_round": quantum,
                    "row_bytes": row_bytes,
                    "scratch_bytes": int(sum(a.nbytes for a in scratch)),
                    "budgeted_lane_bytes": per_lane,
                }
            )
        active = np.concatenate((active[~finished], refill))
        if finished.any() and on_batch is not None:
            on_batch(int(next_pending - active.size), Ne)

    assembly = time.perf_counter()
    _nsys_push("pyrite.transport.continuation.assemble")
    joined = _assemble(xp, chunks, rows_total, retained, columns, out_bufs, nseg, on_device)
    if on_device and not keep_on_device:
        for buf, column in zip(out_bufs, joined, strict=True):
            if column.shape[0]:
                buf[:nseg] = to_host(column)
        joined = None
    _nsys_pop()
    if cont.on_round is not None:
        cont.on_round({"phase": "assembly", "rows": nseg, "wall_s": time.perf_counter() - assembly})
    return (nseg, *(tallies[code] for code in _TALLIED), joined, to_host(d_stragg))
