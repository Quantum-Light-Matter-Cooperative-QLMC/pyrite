"""Disk spill of streamed line-grid evidence (#1141).

The live streamed grid pass (:mod:`.streamed_line_grid`) needs every valid
sinc ``(width, t_L^2)`` sample and, for the measured bandwidth, every
production line. Both grow linearly with the electron count, so above a size
threshold they are written to a per-run scratch directory as blocks arrive and
host memory stays bounded independent of ``N``; disk use still scales with it.

Exactness: :func:`sinc_spacing_from_samples` takes a weighted quantile over
the stable width order. :func:`external_stable_sort` reproduces that order out
of core: rows are distributed into value buckets by pivots, each bucket keeps
its rows in their original order, and every bucket that fits is sorted with
the same stable argsort. Equal widths share a bucket, so the concatenated
buckets are the global stable order whatever the pivots. :func:`sorted_sinc_spacing`
then reads the sorted rows chunkwise; carrying the running sum into each
chunk's first element keeps the sequential float cumulative sum unchanged, so
the spacing is bitwise the in-memory one. Production lines are appended to
one raw file (:class:`LineFile`) and served as views of a single read-only
memory map, holding the same bits in the same block layout, so the population
consumers read identical arrays.

Files are raw native-endian arrays. The scratch
directory lives under the output ``cache/`` (always safe to delete) and is
removed on success, error and interrupt; directories of dead processes on the
same host are pruned when a new one starts.
"""

import contextlib
import itertools
import os
import shutil
import socket
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Evidence bytes above which the grid pass spills to disk (projected over the
#: run, or held). ``None`` never spills; ``0`` always does.
DEFAULT_SPILL_THRESHOLD_BYTES = 2 * 1024**3
#: Rows one in-memory stable sort takes; larger runs split into value buckets.
SORT_ROWS = 1 << 24
#: Rows read or reduced per chunk.
CHUNK_ROWS = 1 << 22
#: Widest bucket fan-out of one partition pass (open file pairs).
MAX_BUCKETS = 128
#: Pivot sample size per partition pass.
SAMPLE_ROWS = 1 << 16
SPILL_DIRNAME = "line-grid-spill"


class LineGridSpillError(RuntimeError):
    """The scratch disk cannot hold the projected grid-pass evidence."""


@dataclass(frozen=True)
class GridSpillPolicy:
    """When and where the streamed grid pass spills its evidence.

    ``threshold_bytes``: spill once the evidence held or projected over the
    run exceeds this; ``None`` never spills. ``directory``: parent of the
    per-run scratch directory, default ``<output>/cache/line-grid-spill``.
    """

    threshold_bytes: int | None = DEFAULT_SPILL_THRESHOLD_BYTES
    directory: str | os.PathLike[str] | None = None

    def parent(self) -> Path:
        """The directory holding per-run scratch directories."""
        if self.directory is not None:
            return Path(self.directory)
        from ...paths import cache_dir

        return cache_dir() / SPILL_DIRNAME

    def check_writable(self) -> None:
        """Create :meth:`parent` and refuse it with ``ValueError`` if unwritable."""
        parent = self.parent()
        try:
            parent.mkdir(parents=True, exist_ok=True)
            Path(tempfile.mkdtemp(prefix=".probe-", dir=parent)).rmdir()
        except OSError as error:
            raise ValueError(
                f"streamed line-grid spill directory {parent} is not writable "
                f"(KernelTuning.stream_spill_dir): {error}"
            ) from error


@dataclass(frozen=True)
class SpilledRun:
    """Aligned ``width``/``weight`` rows in a pair of raw files, in row order."""

    width: Path
    weight: Path
    rows: int
    dtypes: tuple[np.dtype, np.dtype]


def _host_tag() -> str:
    return socket.gethostname().replace("-", "_") or "host"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prune_stale_spill_dirs(parent: Path) -> list[Path]:
    """Remove this host's spill directories whose process no longer runs."""
    removed = []
    if not parent.is_dir():
        return removed
    prefix = f"{_host_tag()}-"
    for child in parent.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        pid = child.name[len(prefix) :].split("-", 1)[0]
        if pid.isdigit() and int(pid) != os.getpid() and not _alive(int(pid)):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed


class SpillStore:
    """One run's scratch directory with byte accounting and cleanup."""

    def __init__(self, parent: Path):
        parent.mkdir(parents=True, exist_ok=True)
        prune_stale_spill_dirs(parent)
        name = f"{_host_tag()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.root = parent / name
        self.root.mkdir()
        self.bytes = 0
        self.peak_bytes = 0
        self._names = itertools.count()

    def path(self, stem: str) -> Path:
        """A fresh file path in the store."""
        return self.root / f"{next(self._names):06d}-{stem}"

    def account(self, path: Path) -> None:
        """Count a written file."""
        self.bytes += path.stat().st_size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def unlink(self, path: Path) -> None:
        """Remove a counted file."""
        size = path.stat().st_size
        path.unlink()
        self.bytes -= size

    def free_bytes(self) -> int:
        """Free bytes on the store's file system."""
        return int(shutil.disk_usage(self.root).free)

    def close(self) -> None:
        """Remove the directory and everything in it."""
        shutil.rmtree(self.root, ignore_errors=True)
        self.bytes = 0


def write_run(store: SpillStore, width: np.ndarray, weight: np.ndarray) -> SpilledRun:
    """Write one block's samples; the files exist only if both writes succeeded."""
    paths = (store.path("width.raw"), store.path("weight.raw"))
    try:
        for path, array in zip(paths, (width, weight), strict=True):
            with path.open("wb") as stream:
                np.ascontiguousarray(array).tofile(stream)
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    for path in paths:
        store.account(path)
    return SpilledRun(*paths, int(width.size), (width.dtype, weight.dtype))


def _read(path: Path, dtype, start: int, stop: int) -> np.ndarray:
    if stop <= start:
        return np.empty(0, dtype=dtype)
    with path.open("rb") as stream:
        stream.seek(start * np.dtype(dtype).itemsize)
        return np.fromfile(stream, dtype=dtype, count=stop - start)


def iter_run_chunks(runs, chunk_rows: int | None = None):
    """Yield ``(width, weight)`` chunks of ``runs`` in row order."""
    size = CHUNK_ROWS if chunk_rows is None else chunk_rows
    for run in runs:
        for start in range(0, run.rows, size):
            stop = min(start + size, run.rows)
            yield (
                _read(run.width, run.dtypes[0], start, stop),
                _read(run.weight, run.dtypes[1], start, stop),
            )


def external_stable_sort(store: SpillStore, runs, dtypes) -> SpilledRun:
    """All rows of ``runs`` in ``np.argsort(width, kind="stable")`` order.

    Consumes ``runs`` (their files are deleted once partitioned). Peak host
    memory is about ``SORT_ROWS`` rows plus one chunk, independent of the
    total; peak disk is about twice the input.
    """
    out = SpilledRun(store.path("sorted-width.raw"), store.path("sorted-weight.raw"), 0, dtypes)
    rows = sum(run.rows for run in runs)
    with out.width.open("wb") as width_out, out.weight.open("wb") as weight_out:
        _sort_into(store, list(runs), rows, dtypes, (width_out, weight_out))
    store.account(out.width)
    store.account(out.weight)
    return SpilledRun(out.width, out.weight, rows, dtypes)


def _drop(store, runs) -> None:
    for run in runs:
        store.unlink(run.width)
        store.unlink(run.weight)


def _sort_into(store, runs, rows, dtypes, outputs) -> None:
    """Append the stable width order of ``runs`` to ``outputs``; drop ``runs``."""
    from ..spectrum.diagnostics import _stable_argsort

    width_out, weight_out = outputs
    if rows <= SORT_ROWS:
        chunks = list(iter_run_chunks(runs))
        _drop(store, runs)
        if not chunks:
            return
        width = np.concatenate([chunk[0] for chunk in chunks])
        weight = np.concatenate([chunk[1] for chunk in chunks])
        del chunks
        order = _stable_argsort(width)
        width[order].tofile(width_out)
        weight[order].tofile(weight_out)
        return
    lowest, highest, sample = _scan(runs, rows)
    if lowest == highest:
        # One width: the stable order is the row order.
        for width, weight in iter_run_chunks(runs):
            width.tofile(width_out)
            weight.tofile(weight_out)
        _drop(store, runs)
        return
    buckets = _partition(store, runs, rows, dtypes, _pivots(sample, lowest, highest, rows))
    _drop(store, runs)
    for bucket in buckets:
        if bucket.rows:
            _sort_into(store, [bucket], bucket.rows, dtypes, outputs)
        else:
            _drop(store, [bucket])


def _scan(runs, rows):
    """``(min, max, sample)`` of the widths of ``runs``."""
    stride = max(1, rows // SAMPLE_ROWS)
    lowest, highest, sample = np.inf, -np.inf, []
    for width, _ in iter_run_chunks(runs):
        if width.size:
            lowest = min(lowest, width.min())
            highest = max(highest, width.max())
            sample.append(width[::stride].copy())  # a view would pin the chunk
    return lowest, highest, np.sort(np.concatenate(sample))


def _pivots(sample, lowest, highest, rows):
    """Bucket pivots in ``(lowest, highest]``: every bucket is a strict subset.

    Rows go to bucket ``#(pivots <= width)``, so the lowest width lands below
    every pivot and the highest at or above one, in different buckets.
    """
    count = min(MAX_BUCKETS, max(2, -(-2 * rows // SORT_ROWS)))
    picks = sample[(np.arange(1, count) * sample.size) // count]
    pivots = np.unique(picks[(picks > lowest) & (picks <= highest)])
    return pivots if pivots.size else np.asarray([highest], dtype=sample.dtype)


def _partition(store, runs, rows, dtypes, pivots):
    """Distribute rows of ``runs`` into value buckets, each in row order."""
    del rows
    count = pivots.size + 1
    paths = [
        (store.path("bucket-width.raw"), store.path("bucket-weight.raw")) for _ in range(count)
    ]
    sizes = np.zeros(count, dtype=np.int64)
    with contextlib.ExitStack() as stack:
        streams = [
            (stack.enter_context(w.open("wb")), stack.enter_context(t.open("wb"))) for w, t in paths
        ]
        for width, weight in iter_run_chunks(runs):
            bucket = np.searchsorted(pivots, width, side="right")
            grouped = np.argsort(bucket, kind="stable")
            bounds = np.concatenate(([0], np.cumsum(np.bincount(bucket, minlength=count))))
            for index in np.flatnonzero(bounds[1:] > bounds[:-1]):
                rows_here = grouped[bounds[index] : bounds[index + 1]]
                width[rows_here].tofile(streams[index][0])
                weight[rows_here].tofile(streams[index][1])
            sizes += bounds[1:] - bounds[:-1]
    buckets = []
    for (width_path, weight_path), size in zip(paths, sizes, strict=True):
        store.account(width_path)
        store.account(weight_path)
        buckets.append(SpilledRun(width_path, weight_path, int(size), dtypes))
    return buckets


def sorted_sinc_spacing(run: SpilledRun, *, aliased_weight_limit=0.01):
    """:func:`.diagnostics.sinc_spacing_from_samples` of already stably sorted rows.

    Reads ``run`` chunkwise; the result is bitwise the in-memory reduction's.
    """
    if not 0.0 <= float(aliased_weight_limit) < 1.0:
        raise ValueError("aliased_weight_limit must be in [0, 1)")
    rows = run.rows
    if rows == 0:
        raise ValueError("cannot derive sinc spacing: diagnostic transport has no valid segments")
    width = np.memmap(run.width, dtype=run.dtypes[0], mode="r", shape=(rows,))
    weight = np.memmap(run.weight, dtype=run.dtypes[1], mode="r", shape=(rows,))
    try:
        total = float(_cumulative_last(weight))
        target = float(aliased_weight_limit) * total
        index = _first_reaching(weight, target)
        step = float(width[min(index, rows - 1)])
        below = int(np.searchsorted(width, step, side="left"))
        aliased = float(weight[:below].sum() / total)
    finally:
        del width, weight
    return step, aliased, int(rows)


def _cumulative_chunks(weight):
    """``(offset, np.cumsum(weight)[offset:offset + chunk])`` chunk by chunk.

    The running sum enters each chunk's first element, so every value is the
    sequential cumulative sum's own.
    """
    carry = None
    for start in range(0, weight.size, CHUNK_ROWS):
        chunk = np.array(weight[start : start + CHUNK_ROWS])
        if carry is not None:
            chunk[0] += carry
        cumulative = np.cumsum(chunk)
        carry = cumulative[-1]
        yield start, cumulative


def _cumulative_last(weight):
    last = None
    for _, cumulative in _cumulative_chunks(weight):
        last = cumulative[-1]
    return last


def _first_reaching(weight, target) -> int:
    """``np.searchsorted(np.cumsum(weight), target, side="left")``."""
    for start, cumulative in _cumulative_chunks(weight):
        index = int(np.searchsorted(cumulative, target, side="left"))
        if index < cumulative.size:
            return start + index
    return int(weight.size)


#: Byte alignment of each spilled line array, as NumPy aligns its own buffers.
_LINE_ALIGN = 64


class LineFile:
    """Production lines of every spilled block, appended to one raw file.

    One file, mapped once after transport, keeps a single descriptor however
    many blocks and reflections spill (per-array maps exhausted the process's
    descriptors at 64 h-BN blocks, job 13610). Each population is kept as
    ``(label, ((offset, dtype, size), ...))`` and served as views of that map,
    so consumers read the same arrays in the same block layout.
    """

    def __init__(self, store: SpillStore):
        self.store = store
        self.path = store.path("lines.raw")
        self.path.touch()
        self.size = 0
        self._map: np.memmap | None = None

    def append(self, populations) -> list:
        """Append one block's populations; on failure the file is as before."""
        if self._map is not None:
            raise RuntimeError("spilled lines were already mapped")
        records, offset = [], self.size
        try:
            with self.path.open("r+b") as stream:
                stream.truncate(self.size)  # drop a failed attempt's bytes
                stream.seek(self.size)
                for population in populations:
                    fields = []
                    for array in (population.energy_eV, population.weight, population.width_eV):
                        array = np.ascontiguousarray(array)
                        pad = -offset % _LINE_ALIGN
                        stream.write(b"\0" * pad)
                        offset += pad
                        array.tofile(stream)
                        fields.append((offset, array.dtype.str, int(array.size)))
                        offset += array.nbytes
                    records.append((population.label, tuple(fields)))
        except BaseException:
            with self.path.open("r+b") as stream:
                stream.truncate(self.size)
            raise
        self.store.bytes += offset - self.size
        self.store.peak_bytes = max(self.store.peak_bytes, self.store.bytes)
        self.size = offset
        return records

    def populations(self, records) -> list:
        """The recorded populations as read-only views of the file."""
        from ..spectrum.cxr_seeds import ResonancePopulation

        if self._map is None and self.size:
            self._map = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.size,))
        out = []
        for label, fields in records:
            arrays = []
            for offset, dtype, size in fields:
                dtype = np.dtype(dtype)
                if size == 0 or self._map is None:
                    arrays.append(np.empty(size, dtype=dtype))
                    continue
                stop = offset + size * dtype.itemsize
                arrays.append(self._map[offset:stop].view(dtype))
            out.append(ResonancePopulation(label, *arrays))
        return out

    def close(self) -> None:
        """Release the map."""
        self._map = None


def projected_disk_bytes(sinc_bytes: float, line_bytes: float) -> float:
    """Peak spill bytes for this much evidence: raw plus partitioned sinc rows."""
    return 2.0 * sinc_bytes + line_bytes + 2 * CHUNK_ROWS * 16
