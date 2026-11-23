"""Opt-in bounded continuation for the exact CUDA transport core (#789).

The batch driver in :mod:`.batching` gives every electron of a launch a dense
row capacity and runs it to completion, so live scratch scales with the
longest track and a launch's lanes idle while its longest histories finish.
This driver instead keeps a fixed population of *active* histories, launches
them for at most ``rows_per_round`` rows each, and between launches retires
finished histories and refills their lanes from the pending primaries. Live
scratch is ``active x rows_per_round`` rows whatever the track length. An
unset ``rows_per_round`` is sized from the first round's rows per history
(#1396), so short-track scenes run few wide rounds instead of many narrow ones.

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

from ..._util.env import env_value
from ._continuation_join import RUN_BYTES, _copy_runs, join_rounds
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
#: Automatic row quantum range (#1396). The first round of an automatic pass
#: launches at the upper end, so long-track scenes (W 1 MeV, ~1,100 rows per
#: electron) keep 256 rows; short-track scenes drop toward the lower end.
AUTO_ROWS_PER_ROUND = (16, 256)
#: Quantile of first-round rows per history the automatic quantum covers.
AUTO_ROWS_QUANTILE = 0.95
#: Process-wide sizing record of automatic and explicit continuation passes,
#: read and cleared by the runner's timing collector (see
#: :func:`pop_continuation_stats`).
_STATS: dict[str, int] = {}

logger = logging.getLogger(__name__)

# ``PerElectronTransportConfig.schedule`` values (#799). None of them changes
# results: continuation output is bitwise the full-history output.
TRANSPORT_SCHEDULES = ("full-history", "continuation", "auto")
# ``auto`` has no history-count floor (#889). The 1000-history floor of #825
# sent small passes, including benchmark and streamed tail blocks, to
# full-history, whose batch is sized by the longest track: at W 1 MeV that is
# ~108 histories per launch and 334 vs 13 ns of kernel per row (#875 job 3579).


@dataclass(frozen=True)
class TransportContinuationConfig:
    """Bounded-continuation schedule; no field changes results.

    ``rows_per_round`` is the row quantum each active history may emit per
    launch. ``None`` (the default, #1396) sizes it automatically: the first
    round launches at ``AUTO_ROWS_PER_ROUND[1]`` rows, and later rounds use the
    ``AUTO_ROWS_QUANTILE`` quantile of that round's rows per history, rounded
    up to a power of two and clamped to ``AUTO_ROWS_PER_ROUND``, with the
    population resized to the smaller quantum. ``active_histories`` caps the
    concurrent population; ``None``
    takes as many as ``PerElectronTransportConfig.scratch_budget_bytes`` and
    ``max_batch`` allow (an unset ``max_batch`` leaves only the budget). The budget covers the round's row scratch, its
    compaction copy and gather indices, and the lane arrays, sized from the
    enabled row schema. Ne-sized per-history state and retained output are
    outside it: bounded scratch does not bound total trajectory storage.

    ``device_staging`` (default on again since #866; off in #825) keeps each round's retained
    rows on the device, joins them there column by column with a run-copy
    kernel and downloads each joined column through pinned chunks (#866).
    Device memory peaks at the staged rows plus their largest joined column
    and the run table. Staging is bounded: while that peak fits
    ``staging_budget_bytes`` rounds stay on the device; past it the staged
    rounds spill to the host and the run finishes with host reassembly.
    ``None`` derives the budget from free device memory (driver free plus the
    memory pool's cached blocks) after the round scratch is allocated, less
    ``STAGING_HEADROOM``. Staged transport ran at 0.47-0.80x host
    reassembly where the rows fit and tied it after a spill (#866, job
    3018). Off, rows move to the host every round and are reassembled there
    with the parallel run copy; that needs no device staging memory. Results
    are bitwise the same in every mode; runs that keep segments on the device
    never spill.

    ``on_round`` (diagnostics only) receives one ``phase="round"`` dict per
    launch with the launched population, rows, finished/refilled counts and
    scratch bytes, then one ``phase="assembly"`` dict timing the reassembly.
    """

    rows_per_round: int | None = None
    active_histories: int | None = None
    device_staging: bool = True
    staging_budget_bytes: int | None = None
    on_round: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self):
        if self.rows_per_round is not None and int(self.rows_per_round) < 1:
            raise ValueError("rows_per_round must be a positive integer or None")
        if self.active_histories is not None and int(self.active_histories) < 1:
            raise ValueError("active_histories must be a positive integer or None")
        if self.staging_budget_bytes is not None and int(self.staging_budget_bytes) < 0:
            raise ValueError("staging_budget_bytes must be non-negative or None")


def continuation_row_bytes(scratch_row):
    """Exact bytes of one row across every enabled scratch column."""
    return sum(int(a.nbytes) for a in scratch_row)


def initial_quantum(cont):
    """The row quantum of a pass's first round: the explicit one, else the automatic ceiling."""
    rows = cont.rows_per_round
    return AUTO_ROWS_PER_ROUND[1] if rows is None else int(rows)


def automatic_quantum(counts):
    """Row quantum for the rounds after a first round that emitted ``counts`` rows per lane.

    The ``AUTO_ROWS_QUANTILE`` quantile of the per-lane rows, rounded up to a
    power of two and clamped to ``AUTO_ROWS_PER_ROUND``. Lanes suspended at
    the first round's quantum count as that quantum, so a scene whose
    histories outlast it keeps the ceiling. Performance only (#1396).
    """
    low, high = AUTO_ROWS_PER_ROUND
    counts = np.asarray(counts)
    if not counts.size:
        return high
    rows = int(np.quantile(counts, AUTO_ROWS_QUANTILE, method="higher"))
    return min(max(low, 1 << max(rows - 1, 0).bit_length()), high)


def active_population(Ne, row_bytes, config, cont, quantum=None):
    """Concurrent histories that fit the scratch budget, or a clear error.

    ``quantum`` defaults to :func:`initial_quantum`. An unset
    ``config.max_batch`` caps the population only by the budget and ``Ne``.
    """
    quantum = initial_quantum(cont) if quantum is None else int(quantum)
    per_lane = quantum * (2 * row_bytes + _GATHER_INDEX_BYTES) + _LANE_BYTES
    fit = int(config.scratch_budget_bytes) // per_lane
    if fit < 1:
        raise ValueError(
            f"one continuation lane needs {per_lane} bytes ({quantum} rows of "
            f"{row_bytes} bytes plus staging), above scratch_budget_bytes="
            f"{int(config.scratch_budget_bytes)}; lower rows_per_round"
        )
    limit = int(Ne)
    if config.max_batch is not None:
        limit = min(limit, int(config.max_batch))
    if cont.active_histories is not None:
        limit = min(limit, int(cont.active_histories))
    return max(1, min(fit, limit)), per_lane


def pop_continuation_stats():
    """Continuation sizing since the last call, as timing metrics; clears the record.

    ``continuation_rows_per_round`` and ``continuation_lanes`` are the largest
    quantum and population any pass ran its later rounds with,
    ``continuation_rounds`` the launches and ``continuation_passes`` the
    driver calls. Empty when no continuation pass ran.
    """
    stats = {f"continuation_{key}": value for key, value in _STATS.items()}
    _STATS.clear()
    return stats


def _record_pass(quantum, lanes, rounds):
    _STATS["rows_per_round"] = max(_STATS.get("rows_per_round", 0), int(quantum))
    _STATS["lanes"] = max(_STATS.get("lanes", 0), int(lanes))
    _STATS["rounds"] = _STATS.get("rounds", 0) + int(rounds)
    _STATS["passes"] = _STATS.get("passes", 0) + 1


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


def _auto_fallback_reason(*, cuda, lut, launched, gdf):
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
    return None


def resolve_schedule(config, Ne, **support):
    """Lower ``config.schedule`` to the driver's ``continuation`` field.

    ``full-history`` clears continuation, ``continuation`` keeps or creates it
    (unsupported modes then raise in the driver, as before), and ``auto``
    continues on every supported pass, whatever its size, logging why
    otherwise.
    ``support`` holds the keyword flags of :func:`_auto_fallback_reason`.
    Results are identical under every choice.
    """
    schedule = effective_schedule(config)
    if schedule == "continuation":
        cont = config.continuation or TransportContinuationConfig()
        return replace(config, schedule=schedule, continuation=cont)
    if schedule == "auto":
        reason = _auto_fallback_reason(**support)
        if reason is None:
            cont = config.continuation or TransportContinuationConfig()
            logger.info("transport schedule auto: continuation for %d histories", Ne)
            return replace(config, schedule="continuation", continuation=cont)
        logger.info("transport schedule auto: full-history because %s", reason)
    return replace(config, schedule="full-history", continuation=None)


def _free_device_bytes(xp):
    """Free device bytes including the pool's cache, within its limit; None off-CUDA."""
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return None
    free, _total = cuda.runtime.memGetInfo()
    pool = xp.get_default_memory_pool()
    free = int(free) + int(pool.free_bytes())
    limit = int(pool.get_limit())
    # A pool limit (0 = none) caps allocations below what the driver offers.
    return free if not limit else min(free, limit - int(pool.used_bytes()))


def staging_budget(xp, cont):
    """Device bytes staged rounds, a joined column and run table may hold; None = unbounded."""
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
        self.column_bytes = None
        self.run_bytes = 0
        self.spilled_round = None

    def peak_bytes(self):
        """Device peak of the column-by-column join of the staged rounds (#866)."""
        largest = max(self.column_bytes) if self.column_bytes else 0
        return self.staged_bytes + largest + self.run_bytes

    def append(self, gathered, lanes=0):
        if self.on_device and self.budget is not None:
            self.staged_bytes += _chunk_bytes(gathered)
            sizes = [int(a.nbytes) for a in gathered]
            if self.column_bytes is None:
                self.column_bytes = sizes
            else:
                self.column_bytes = [a + b for a, b in zip(self.column_bytes, sizes, strict=True)]
            self.run_bytes += RUN_BYTES * int(lanes)
            # The join holds the staged rounds, one joined column (each column
            # frees its round slices once joined) and the run table.
            if self.peak_bytes() > self.budget:
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


def _reserve(out_bufs, nseg):
    if out_bufs and hasattr(out_bufs[0], "reserve"):
        # Assembly must hold all staged rounds *and* their final canonical
        # columns; use the actual row count rather than Ne * max_steps.
        out_bufs[0].reserve(nseg)


def _assemble(
    xp, chunks, rows_total, retained, columns, out_bufs, nseg, staged, keep_on_device, stats=None
):
    """Place retained rounds in canonical rows; the joined device rows or None.

    Host-retained rounds are run-copied into ``out_bufs``. Staged rounds are
    joined on the device column by column; unless ``keep_on_device`` each
    joined column is downloaded straight into ``out_bufs``.
    """
    runs = _canonical_runs(chunks, rows_total)
    if staged and keep_on_device:
        return join_rounds(xp, runs, retained, columns, nseg, None, stats)
    _reserve(out_bufs, nseg)
    if not staged:
        _scatter_host(chunks, rows_total, retained, out_bufs, nseg)
        return None
    if not nseg:  # no rows to place; an empty output has no (0, width) view
        retained.clear()
        return None
    targets = [
        _host_rows(t, nseg) if a.shape[0] else None for t, a in zip(out_bufs, columns, strict=True)
    ]
    join_rounds(xp, runs, retained, columns, nseg, targets, stats)
    return None


def _round_buffers(xp, alloc_scratch, n_active, quantum):
    """One round's row scratch and per-lane row counts and exit codes."""
    scratch = alloc_scratch(n_active * quantum)
    return scratch, xp.zeros(n_active, dtype=xp.int64), xp.zeros(n_active, dtype=xp.int8)


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
    quantum = initial_quantum(cont)
    # An automatic quantum is chosen once, after the first round (#1396).
    automatic = sizing = cont.rows_per_round is None
    row_bytes = continuation_row_bytes(alloc_scratch(1))
    n_active, per_lane = active_population(Ne, row_bytes, config, cont, quantum)
    scratch, seg_count, exit_code = _round_buffers(xp, alloc_scratch, n_active, quantum)
    retained = _Retained(xp, cont, keep_on_device, to_host)
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
        cap = quantum
        if automatic and not sizing and next_pending == Ne:
            # Draining: spread the fixed scratch over the remaining lanes.
            cap = min(AUTO_ROWS_PER_ROUND[1], max(quantum, n_active * quantum // m))
        _nsys_push("pyrite.transport.continuation.launch")
        core(
            (0, m, cap, d_keys),
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
        index = _gather_index(xp, seg_count[:m], cap)
        # Frozen rows keep seven columns; the kernel still takes the empty end fields.
        gathered = tuple(a[index] if a.shape[0] else a for a in scratch[: len(out_bufs)])
        nseg += int(index.size)
        del index
        retained.append(gathered, lanes=m)
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
        record = {
            "round": len(chunks) - 1,
            "launched": m,
            "rows_per_round": cap,
            "scratch_bytes": int(sum(a.nbytes for a in scratch)),
            "budgeted_lane_bytes": per_lane,
        }
        if sizing:
            sizing = False
            resized = automatic_quantum(counts)
            if resized != quantum:
                quantum = resized
                n_active, per_lane = active_population(Ne, row_bytes, config, cont, quantum)
                del scratch, seg_count, exit_code
                scratch, seg_count, exit_code = _round_buffers(xp, alloc_scratch, n_active, quantum)
        # Fill every free lane; equal to the finished count unless just resized.
        room = n_active - int(m - finished.sum())
        refill = pending[next_pending : next_pending + room]
        next_pending += refill.size
        if cont.on_round is not None:
            cont.on_round(
                {
                    "phase": "round",
                    **record,
                    "rows": int(counts.sum()),
                    "max_rows": int(counts.max()),
                    "finished": int(finished.sum()),
                    "refilled": int(refill.size),
                    "pending": int(Ne - next_pending),
                    "row_bytes": row_bytes,
                }
            )
        active = np.concatenate((active[~finished], refill))
        if finished.any() and on_batch is not None:
            on_batch(int(next_pending - active.size), Ne)

    _record_pass(quantum, n_active, len(chunks))
    assembly = time.perf_counter()
    _nsys_push("pyrite.transport.continuation.assemble")
    on_device = retained.on_device
    retained.close()
    join_stats = {}
    joined = _assemble(
        xp,
        chunks,
        rows_total,
        retained.chunks,
        scratch[: len(out_bufs)],
        out_bufs,
        nseg,
        on_device,
        keep_on_device,
        join_stats,
    )
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
                "staged_peak_bytes": retained.peak_bytes() if on_device else None,
                "pool_peak_bytes": join_stats.get("pool_peak_bytes"),
            }
        )
    return (nseg, *(tallies[code] for code in _TALLIED), joined, to_host(d_stragg))
