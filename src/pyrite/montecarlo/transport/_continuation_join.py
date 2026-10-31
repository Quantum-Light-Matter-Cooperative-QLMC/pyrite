"""Device-side join of staged continuation rounds (#866).

Staged rounds hold, lane by lane, contiguous runs of one history's rows.
:func:`join_rounds` copies every run to its canonical position with a CUDA
run-copy kernel driven by the ``(round, destination, source, length)`` run
table, one column at a time: the column's slices are released from every
round as soon as it is joined, so the device holds the staged payload plus
one joined column instead of twice the payload. When the rows return to the
host each joined column is downloaded in chunks through a double-buffered
pinned ring on its own stream, overlapping the next column's join, and the
pinned chunks are copied in parallel into the reserved host output.

Pure data movement: the same bytes land in the same rows as the host run
copy, so results are bitwise unchanged.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Device bytes of one run-table row: int32 round, int64 dest/source/length.
RUN_BYTES = 4 + 3 * 8
# Pinned ring: two chunks, so one D2H overlaps the previous chunk's host copy.
CHUNK_BYTES = 32 << 20
_WARP = 32
_THREADS = 256

_SOURCE = r"""
typedef {word} word_t;
extern "C" __global__ void copy_runs(
    word_t* __restrict__ dst,
    const unsigned long long* __restrict__ round_ptr,
    const int* __restrict__ run_round,
    const long long* __restrict__ run_dst,
    const long long* __restrict__ run_src,
    const long long* __restrict__ run_len,
    const long long n_runs,
    const long long row_words)
{{
    // One warp per run: runs are at most rows_per_round rows long.
    const long long run = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (run >= n_runs) return;
    const word_t* src = (const word_t*)round_ptr[run_round[run]] + run_src[run] * row_words;
    word_t* out = dst + run_dst[run] * row_words;
    const long long n = run_len[run] * row_words;
    for (long long i = threadIdx.x % 32; i < n; i += 32) out[i] = src[i];
}}
"""
_WORDS = {8: "unsigned long long", 4: "unsigned int", 2: "unsigned short", 1: "unsigned char"}
_KERNELS: dict[int, object] = {}


def _kernel(cp, word_bytes):
    kernel = _KERNELS.get(word_bytes)
    if kernel is None:
        source = _SOURCE.format(word=_WORDS[word_bytes])
        kernel = _KERNELS[word_bytes] = cp.RawKernel(source, "copy_runs")
    return kernel


def _row_bytes(column):
    return column.dtype.itemsize * int(np.prod(column.shape[1:], dtype=np.int64))


def _word_bytes(row_bytes):
    return next(w for w in (8, 4, 2, 1) if row_bytes % w == 0)


@njit(parallel=True, cache=True)
def _copy_runs(dst, src, dst_start, src_start, length):  # pragma: no cover - compiled
    """``dst[d:d+n] = src[s:s+n]`` for every run; runs never overlap in ``dst``."""
    for r in prange(dst_start.size):
        d, s, n = dst_start[r], src_start[r], length[r]
        dst[d : d + n] = src[s : s + n]


@njit(parallel=True, cache=True)
def _copy_bytes(dst, src):  # pragma: no cover - compiled
    """Parallel ``dst[:] = src`` for flat byte arrays of equal size."""
    block = 1 << 20
    for b in prange((src.size + block - 1) // block):
        lo = b * block
        hi = min(lo + block, src.size)
        dst[lo:hi] = src[lo:hi]


def _cat(parts):
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)


class RunTable:
    """Every retained run in round order: round, destination, source, length."""

    def __init__(self, runs):
        runs = list(runs)
        self.rounds = len(runs)
        sizes = [counts.size for _base, _starts, counts in runs]
        self.round = np.repeat(np.arange(self.rounds, dtype=np.int32), sizes)
        self.dst = _cat([base for base, _s, _c in runs])
        self.src = _cat([starts for _b, starts, _c in runs])
        self.length = _cat([counts for _b, _s, counts in runs])
        self._split = np.cumsum(sizes)[:-1]

    def per_round(self):
        if not self.rounds:
            return []
        parts = (np.split(a, self._split) for a in (self.dst, self.src, self.length))
        return list(zip(*parts, strict=True))

    def upload(self, cp):
        return tuple(cp.asarray(a) for a in (self.round, self.dst, self.src, self.length))


def _flat2d(column, rows):
    return column.reshape(rows, -1)


def _join_host(table, retained, c, target):
    """NumPy rounds (device-free tests): run copy straight into ``target``."""
    rows = _flat2d(target, target.shape[0])
    for (base, starts, counts), chunk in zip(table.per_round(), retained, strict=True):
        column, chunk[c] = chunk[c], None
        if column.shape[0] and counts.size:
            src = np.ascontiguousarray(column).reshape(column.shape[0], -1)
            _copy_runs(rows, src, base, starts, counts)


def _join_device(cp, table, device_table, retained, c, joined):
    """Launch the run copy of column ``c`` of every round into ``joined``."""
    pointers, sources = [], []
    for chunk in retained:
        column, chunk[c] = chunk[c], None
        if column.shape[0] and not column.flags.c_contiguous:
            column = cp.ascontiguousarray(column)
        sources.append(column)  # alive until the launch is enqueued
        pointers.append(column.data.ptr if column.shape[0] else 0)
    n_runs = int(table.length.size)
    if n_runs and joined.shape[0]:
        row_bytes = _row_bytes(joined)
        word = _word_bytes(row_bytes)
        blocks = -(-n_runs // (_THREADS // _WARP))
        _kernel(cp, word)(
            (blocks,),
            (_THREADS,),
            (
                joined,
                cp.asarray(np.asarray(pointers, dtype=np.uint64)),
                *device_table,
                np.int64(n_runs),
                np.int64(row_bytes // word),
            ),
        )
    # Dropping the round slices returns them to the current stream's pool;
    # later allocations on that stream are ordered after the launch.
    del sources


class _PinnedRing:
    """Two pinned chunks; D2H of one overlaps the host copy of the other."""

    def __init__(self, cp, stream):
        self.cp, self.stream = cp, stream
        self.buffers = [
            np.frombuffer(cp.cuda.alloc_pinned_memory(CHUNK_BYTES), np.uint8, CHUNK_BYTES)
            for _ in range(2)
        ]
        self.events = [None, None]
        self.pending = [None, None]

    def _drain(self, k):
        if self.events[k] is not None:
            self.events[k].synchronize()
            dst, n = self.pending[k]
            _copy_bytes(dst, self.buffers[k][:n])
            self.events[k] = self.pending[k] = None

    def download(self, column, target, ready):
        """Copy device ``column`` into host ``target`` once ``ready`` fires."""
        cp = self.cp
        dst = target.reshape(-1).view(np.uint8)
        nbytes = int(column.nbytes)
        if dst.size != nbytes:
            raise ValueError("joined column and host target sizes disagree")
        self.stream.wait_event(ready)
        kind = cp.cuda.runtime.memcpyDeviceToHost
        for k, offset in enumerate(range(0, nbytes, CHUNK_BYTES)):
            slot = k % 2
            self._drain(slot)
            n = min(CHUNK_BYTES, nbytes - offset)
            cp.cuda.runtime.memcpyAsync(
                self.buffers[slot].ctypes.data,
                column.data.ptr + offset,
                n,
                kind,
                self.stream.ptr,
            )
            self.events[slot] = self.stream.record()
            self.pending[slot] = (dst[offset : offset + n], n)
        self._drain(0)
        self._drain(1)


def join_rounds(xp, runs, retained, columns, nseg, targets=None, stats=None):
    """Join staged ``retained`` rounds into canonical rows, column by column.

    ``targets`` None keeps the joined columns on the device and returns them
    (disabled columns pass through as ``columns`` entries); otherwise each
    joined column is written into the matching C-contiguous host ``targets``
    array of ``nseg`` rows and None is returned. ``retained`` is consumed.
    ``stats`` (diagnostics) receives ``pool_peak_bytes`` on CUDA: the largest
    memory-pool footprint sampled after each column's join is enqueued.
    """
    table = RunTable(runs)
    if len(retained) != table.rounds:
        raise ValueError("retained rounds and chunk records disagree")
    chunks = [list(chunk) for chunk in retained]
    retained.clear()
    for chunk in chunks:
        if len(chunk) != len(columns):
            raise ValueError("retained round columns and output columns disagree")
    enabled = [c for c, a in enumerate(columns) if a.shape[0]]
    if not hasattr(xp, "cuda"):
        out = list(columns)
        for c in enabled:
            shape = (nseg,) + columns[c].shape[1:]
            target = xp.empty(shape, columns[c].dtype) if targets is None else targets[c]
            _join_host(table, chunks, c, target)
            out[c] = target
        return tuple(out) if targets is None else None
    return _join_cuda(xp, table, chunks, columns, enabled, nseg, targets, stats)


def _join_cuda(cp, table, chunks, columns, enabled, nseg, targets, stats):
    device_table = table.upload(cp)
    compute = cp.cuda.get_current_stream()
    pool = cp.get_default_memory_pool()
    peak = [int(pool.total_bytes())]

    def sample():
        peak[0] = max(peak[0], int(pool.total_bytes()))
        if stats is not None:
            stats["pool_peak_bytes"] = peak[0]

    out = list(columns)
    if targets is None:
        for c in enabled:
            out[c] = cp.empty((nseg,) + columns[c].shape[1:], columns[c].dtype)
            _join_device(cp, table, device_table, chunks, c, out[c])
            sample()
        return tuple(out)
    ring = _PinnedRing(cp, cp.cuda.Stream(non_blocking=True))

    def launch(c):
        joined = cp.empty((nseg,) + columns[c].shape[1:], columns[c].dtype)
        _join_device(cp, table, device_table, chunks, c, joined)
        sample()
        return joined, compute.record()

    # Join column i+1 while column i downloads; peak device memory stays the
    # payload plus one column, as each join frees its rounds' slices.
    current = launch(enabled[0]) if enabled else None
    for i, c in enumerate(enabled):
        following = launch(enabled[i + 1]) if i + 1 < len(enabled) else None
        joined, ready = current
        ring.download(joined, targets[c], ready)
        del joined
        current = following
    return None
