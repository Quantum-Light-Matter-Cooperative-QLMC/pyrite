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

Supported: the exact CUDA core under continuous losses, the shell soft/hard
mode (secondary directions on or off) and straggling, each with or without
coupled radiative transport (#797). Shell hard events are sampled and stored
within their step and the hard stream counter and optical depth are in the
record; straggling draws are keyed by (flight, substep) and its per-history
accumulator is reset only on a fresh history. The transport LUT and every CPU
core are rejected.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numba import njit, prange

from ..._env import env_value
from .cores import (
    EXIT_BACKSCATTERED,
    EXIT_CUTOFF_STOPPED,
    EXIT_SIDE,
    EXIT_STEP_LIMITED,
    EXIT_TRANSMITTED,
)
from .host_memory import admit_host_allocation

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
# Fraction of free device memory the derived staging budget leaves unused.
STAGING_HEADROOM = 0.1

logger = logging.getLogger(__name__)

# ``PerElectronTransportConfig.schedule`` values (#799). None of them changes
# results: continuation output is bitwise the full-history output.
TRANSPORT_SCHEDULES = ("full-history", "continuation", "auto")
# Below this many histories ``auto`` keeps full-history. #789 set 5000 when
# host-staged rounds made Si 12-27% slower at 5000. With device staging
# (#842), continuation is faster at every measured size (#825 jobs 2982/2983,
# 1000-8000 histories, Si 300 keV and W 1 MeV: 1.4-5.3x request time). A
# full-history batch is sized by the longest track, so long-track scenes
# launch about 256 histories at a time. 1000 matches the CUDA crossover
# (``batching.CUDA_TRANSPORT_MIN_ELECTRONS``); smaller passes are unmeasured.
CONTINUATION_AUTO_MIN_HISTORIES = 1000


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

    ``device_staging`` (default off since #825) keeps each round's retained
    rows on the device and downloads the reassembled columns once, for up to
    twice the retained output in device memory. Staging is bounded: while
    twice the staged bytes fit ``staging_budget_bytes`` rounds stay on the
    device; past it the staged rounds spill to the host and the run finishes
    with host reassembly. ``None`` derives the budget from free device memory
    (driver free plus the memory pool's cached blocks) after the round
    scratch is allocated, less ``STAGING_HEADROOM``. Off, rows move to the
    host every round and are reassembled there with the parallel run copy,
    which since #825 is as fast or faster on every measured scene (job 3000)
    and needs no device staging memory. Results are bitwise the same in
    every mode; runs that keep segments on the device never spill.

    ``on_round`` (diagnostics only) receives one ``phase="round"`` dict per
    launch with the launched population, rows, finished/refilled counts and
    scratch bytes, then one ``phase="assembly"`` dict timing the reassembly.
    """

    rows_per_round: int = 256
    active_histories: int | None = None
    device_staging: bool = False
    staging_budget_bytes: int | None = None
    on_round: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self):
        if int(self.rows_per_round) < 1:
            raise ValueError("rows_per_round must be a positive integer")
        if self.active_histories is not None and int(self.active_histories) < 1:
            raise ValueError("active_histories must be a positive integer or None")
        if self.staging_budget_bytes is not None and int(self.staging_budget_bytes) < 0:
            raise ValueError("staging_budget_bytes must be non-negative or None")


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


def check_schedule(schedule):
    """Validate a schedule name; ``None`` defers to the config's continuation."""
    if schedule is not None and schedule not in TRANSPORT_SCHEDULES:
        raise ValueError(
            f"transport schedule must be one of {TRANSPORT_SCHEDULES}, got {schedule!r}"
        )
    return schedule


def schedule_from_env():
    """Process default schedule: ``auto`` unless explicitly overridden."""
    raw = env_value("PYRITE_MC_TRANSPORT_SCHEDULE", "").strip()
    if not raw:
        return "auto"
    if raw not in TRANSPORT_SCHEDULES:
        raise ValueError(f"PYRITE_MC_TRANSPORT_SCHEDULE must be one of {TRANSPORT_SCHEDULES}")
    return raw


def effective_schedule(config):
    """The schedule a config asks for; an explicit ``continuation`` implies it."""
    if config.schedule is not None:
        return config.schedule
    return "full-history" if config.continuation is None else "continuation"


def _auto_fallback_reason(Ne, *, cuda, lut, launched, gdf):
    """Why ``auto`` keeps full-history for this pass, or None to continue."""
    if not cuda:
        return "the transport core is not 'cuda'"
    if lut:
        return "the CUDA transport LUT core has no continuation"
    if launched:
        # Cascade generations are covered (#797); the channeling handoff is not.
        return "launched passes (cascade generations, channeling handoff) stay full-history"
    if gdf:
        return "GDF beams have no continuation regression"
    if Ne < CONTINUATION_AUTO_MIN_HISTORIES:
        return f"Ne={Ne} is below {CONTINUATION_AUTO_MIN_HISTORIES} histories"
    return None


def resolve_schedule(config, Ne, **support):
    """Lower ``config.schedule`` to the driver's ``continuation`` field.

    ``full-history`` clears continuation, ``continuation`` keeps or creates it
    (unsupported modes then raise in the driver, as before), and ``auto``
    continues only on a supported pass of at least
    :data:`CONTINUATION_AUTO_MIN_HISTORIES` histories, logging why otherwise.
    ``support`` holds the keyword flags of :func:`_auto_fallback_reason`.
    Results are identical under every choice.
    """
    schedule = effective_schedule(config)
    if schedule == "continuation":
        cont = config.continuation or TransportContinuationConfig()
        return replace(config, schedule=schedule, continuation=cont)
    if schedule == "auto":
        reason = _auto_fallback_reason(int(Ne), **support)
        if reason is None:
            cont = config.continuation or TransportContinuationConfig()
            logger.info("transport schedule auto: continuation for %d histories", Ne)
            return replace(config, schedule="continuation", continuation=cont)
        logger.info("transport schedule auto: full-history because %s", reason)
    return replace(config, schedule="full-history", continuation=None)


def _free_device_bytes(xp):
    """Free device bytes including the memory pool's cache; None off-CUDA."""
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return None
    free, _total = cuda.runtime.memGetInfo()
    return int(free) + int(xp.get_default_memory_pool().free_bytes())


def staging_budget(xp, cont):
    """Device bytes staged rounds plus their joined copy may hold; None = unbounded."""
    if cont.staging_budget_bytes is not None:
        return int(cont.staging_budget_bytes)
    free = _free_device_bytes(xp)
    return None if free is None else int(free * (1.0 - STAGING_HEADROOM))


def _chunk_bytes(chunk):
    return sum(int(a.nbytes) for a in chunk)


class _Retained:
    """Retained rounds, staged on the device within a budget, else on the host."""

    def __init__(self, xp, cont, keep_on_device, to_host):
        self.on_device = keep_on_device or cont.device_staging
        self.budget = None if keep_on_device or not self.on_device else staging_budget(xp, cont)
        self.xp = xp
        self.to_host = to_host
        self.arena = None
        self.chunks = []
        self.staged_bytes = 0
        self.spilled_round = None

    def append(self, gathered):
        if self.on_device and self.budget is not None:
            self.staged_bytes += _chunk_bytes(gathered)
            # Assembly holds the staged rounds and their joined copy at once.
            if 2 * self.staged_bytes > self.budget:
                # Spilling copies every staged round plus this one to the host.
                admit_host_allocation(self.staged_bytes, "continuation staged-round spill")
                self.chunks = [self._host(c) for c in self.chunks]
                self.on_device, self.spilled_round = False, len(self.chunks)
                # Hand the staged rounds' cached blocks back to the driver:
                # pageable D2H runs up to ~2x slower while the pool holds most
                # of the card, which slowed every later round's download
                # (#825, jobs 2990-2995).
                pool = getattr(self.xp, "get_default_memory_pool", None)
                if pool is not None:
                    pool().free_all_blocks()
        if not self.on_device:
            # The gather is still device-resident; its host copy must fit
            # alongside every previously retained round.
            admit_host_allocation(_chunk_bytes(gathered), "continuation retained round download")
        self.chunks.append(gathered if self.on_device else self._host(gathered))

    def _host(self, chunk):
        return tuple(self._download(c, a) for c, a in enumerate(chunk))

    def _download(self, column, array):
        if not hasattr(array, "get"):
            return self.to_host(array)
        if self.arena is None:
            self.arena = _ColumnArena()
        out = self.arena.empty(column, array.shape, array.dtype)
        array.get(out=out)
        return out

    def close(self):
        """Drop the arena's open blocks; each frees once its views are released."""
        self.arena = None


class _ColumnArena:
    """Per-column host blocks that round downloads are packed into.

    Assembly copies one column at a time and releases it (``_scatter_host``),
    so a column's rounds must free their memory together. Small per-round
    arrays from the C heap stayed resident after release (#825 job 3001), and
    small private mappings fault in slowly (job 3008). Blocks of at least
    ``BLOCK_BYTES`` take the allocator's mmap path, are hugepage-advised by
    NumPy, and return to the OS once every view into them is gone.
    """

    BLOCK_BYTES = 64 << 20
    ALIGN = 64

    def __init__(self):
        self.blocks = {}

    def empty(self, column, shape, dtype):
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        block, offset = self.blocks.get(column, (None, 0))
        if block is None or offset + nbytes > block.size:
            block, offset = np.empty(max(self.BLOCK_BYTES, nbytes), dtype=np.uint8), 0
        view = block[offset : offset + nbytes].view(dtype).reshape(shape)
        self.blocks[column] = (block, offset + -(-nbytes // self.ALIGN) * self.ALIGN)
        return view


def _gather_index(xp, counts, quantum):
    """Flat scratch slot of each retained row, slot-major and step-minor."""
    ends = xp.cumsum(counts)
    total = int(ends[-1]) if counts.size else 0
    rows = xp.arange(total, dtype=xp.int64)
    lane = xp.searchsorted(ends, rows, side="right")
    rows -= ends[lane] - counts[lane]
    rows += lane * quantum
    return rows


def _canonical_runs(chunks, rows_total):
    """``(destination, source, length)`` of every lane run, chunk by chunk.

    Chunks are in launch order and each holds, lane by lane, a contiguous run
    of one history's next rows, so a per-history cursor starting at that
    history's exclusive prefix places every run exactly once.
    """
    cursor = np.cumsum(rows_total) - rows_total
    for histories, counts in chunks:
        live = counts > 0
        histories, counts = histories[live], counts[live]
        starts = np.cumsum(counts) - counts
        base = cursor[histories]
        cursor[histories] += counts
        yield base, starts, counts


def _canonical_destinations(chunks, rows_total):
    """Destination row of every retained row, chunk by chunk (device join)."""
    for base, starts, counts in _canonical_runs(chunks, rows_total):
        yield np.repeat(base - starts, counts) + np.arange(int(counts.sum()), dtype=np.int64)


@njit(parallel=True, cache=True)
def _copy_runs(dst, src, dst_start, src_start, length):  # pragma: no cover - compiled
    """``dst[d:d+n] = src[s:s+n]`` for every run; runs never overlap in ``dst``."""
    for r in prange(dst_start.size):
        d, s, n = dst_start[r], src_start[r], length[r]
        dst[d : d + n] = src[s : s + n]


def _host_rows(target, nseg):
    """The reserved host ndarray behind ``target``, as ``(nseg, width)``."""
    rows = target[:nseg]
    if not rows.flags.c_contiguous:
        # reshape would copy, and the run copies would miss the output.
        raise ValueError("continuation output columns must be C-contiguous")
    return rows.reshape(nseg, -1)


def _scatter_host(chunks, rows_total, retained, targets, nseg):
    """Copy host-retained rounds into canonical rows, one parallel copy per run.

    Pure data movement: the same bytes land in the same rows as an indexed
    scatter, without materializing a per-row destination index. Every round
    holds rows of every active history, so copying round by round faults in
    the whole output while all rounds are still resident (twice the payload,
    #825 job 3006). Copy column by column instead and release each round's
    column once copied: resident memory stays near one payload plus a column.
    """
    dst = [_host_rows(t, nseg) if t.shape[0] else None for t in targets]
    if len(retained) != len(chunks):
        raise ValueError("retained rounds and chunk records disagree")
    runs = list(_canonical_runs(chunks, rows_total))
    for i, chunk in enumerate(retained):
        if len(chunk) != len(dst):
            raise ValueError("retained round columns and output columns disagree")
        retained[i] = list(chunk)
    for c, rows in enumerate(dst):
        for i, (base, starts, counts) in enumerate(runs):
            column, retained[i][c] = retained[i][c], None
            if rows is not None and column.shape[0]:
                flat = np.ascontiguousarray(column).reshape(column.shape[0], -1)
                _copy_runs(rows, flat, base, starts, counts)
                del flat
            del column
    retained.clear()


def _pending_order(Ne, history_order):
    pending = np.arange(Ne, dtype=np.int32) if history_order is None else history_order
    pending = np.asarray(pending, dtype=np.int32)
    if not np.array_equal(np.sort(pending), np.arange(Ne)):
        raise ValueError("history_order must be a permutation of range(Ne)")
    return pending


def _assemble(xp, chunks, rows_total, retained, columns, out_bufs, nseg, keep_on_device):
    """Scatter retained rounds into canonical rows; the joined device rows or None.

    ``keep_on_device`` means the rounds are device-resident and are joined there;
    otherwise they are host arrays copied into ``out_bufs``.
    """
    if not keep_on_device:
        if out_bufs and hasattr(out_bufs[0], "reserve"):
            # Assembly must hold all staged rounds *and* their final canonical
            # columns; use the actual row count rather than Ne * max_steps.
            out_bufs[0].reserve(nseg)
        _scatter_host(chunks, rows_total, retained, out_bufs, nseg)
        return None
    joined = tuple(
        xp.empty((nseg,) + a.shape[1:], dtype=a.dtype) if a.shape[0] else a for a in columns
    )
    for dest, chunk in zip(_canonical_destinations(chunks, rows_total), retained, strict=True):
        dest = xp.asarray(dest)
        for target, column in zip(joined, chunk, strict=True):
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
    retained = _Retained(xp, cont, keep_on_device, to_host)
    seg_count = xp.zeros(n_active, dtype=xp.int64)
    exit_code = xp.zeros(n_active, dtype=xp.int8)
    state = tuple(xp.zeros((Ne, _STATE_WORDS), dtype=t) for t in (xp.int32, xp.uint64, xp.float64))
    pending = _pending_order(Ne, history_order)
    rows_total = np.zeros(Ne, dtype=np.int64)
    chunks = []
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
        retained.append(gathered)
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
    on_device = retained.on_device
    retained.close()
    joined = _assemble(xp, chunks, rows_total, retained.chunks, columns, out_bufs, nseg, on_device)
    if on_device and not keep_on_device:
        for buf, column in zip(out_bufs, joined, strict=True):
            if column.shape[0]:
                admit_host_allocation(int(column.nbytes), "continuation final host download")
                buf[:nseg] = to_host(column)
        joined = None
    _nsys_pop()
    if cont.on_round is not None:
        cont.on_round(
            {
                "phase": "assembly",
                "rows": nseg,
                "wall_s": time.perf_counter() - assembly,
                "staged": on_device,
                "staging_budget_bytes": retained.budget,
                "spilled_round": retained.spilled_round,
            }
        )
    return (nseg, *(tallies[code] for code in _TALLIED), joined, to_host(d_stragg))
