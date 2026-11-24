"""Size live streamed blocks from a device-memory target (#868).

``KernelTuning.stream_memory_mib`` caps the CuPy pool under the target, which is
what bounds peak device memory: scoring adapts its chunks to the pool, so its
peak does not follow row count (#907). The block size only decides how much of
that budget retained rows may take. A small first block measures rows per
electron and bytes per row for this scene, and later blocks follow the largest
rate seen, so a denser stretch of the population shrinks them. Electron streams
are counter-addressed, so block sizes do not change transport.
"""

from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from numbers import Integral
from typing import Any

from ..._backend import BACKEND
from .transport_options import override_transport_tuning

#: Electrons in the measuring first block.
PROBE_ELECTRONS = 1_024
#: Share of the target retained block rows may take; the rest is scoring scratch.
RETAINED_FRACTION = 0.25
#: Continuation scratch per block electron: holds the whole block as one
#: population (#907: 8 GiB for 65,536 electrons on the RTX PRO 6000).
SCRATCH_BYTES_PER_ELECTRON = 8 * 1024**3 / 65_536
#: Share of the target continuation scratch may take.
SCRATCH_FRACTION = 0.125
#: Share of the target the pool may reserve at most.
POOL_FRACTION = 0.95
#: Target kept for allocations made outside the pool after the run starts: 0.10
#: to 0.43 GiB on the RTX 5080 and RTX PRO 6000 (#868). Memory already in use
#: outside the pool at the start (CUDA context, other processes) is read per run.
POOL_HEADROOM_BYTES = 1024**3 // 2
#: Smallest share of the target left to the pool, so small targets still run.
#: Those targets are best effort: the reserve may not fit under them.
MIN_POOL_FRACTION = 0.5
_MIB = 1024**2


class MemoryBlockSizer:
    """Choose whole-electron blocks for one streamed run from ``target_mib``."""

    def __init__(self, target_mib: int, transport: Any):
        self.target_bytes = int(target_mib) * _MIB
        self.transport = transport
        self.bytes_per_electron = 0.0
        self.spans: list[tuple[int, int]] = []

    def next_stop(self, start: int, n_electrons: int) -> int:
        """End of the block starting at ``start``."""
        if not self.spans:
            size = PROBE_ELECTRONS
        elif self.bytes_per_electron > 0:
            budget = RETAINED_FRACTION * self.target_bytes
            size = max(1, int(budget / self.bytes_per_electron))
        else:
            size = n_electrons  # no rows so far: nothing to retain
        return min(start + size, n_electrons)

    def blocks(self, n_electrons: int) -> Iterator[tuple[int, int]]:
        """Half-open electron ranges; each is sized after the previous was observed."""
        start = 0
        while start < n_electrons:
            stop = self.next_stop(start, n_electrons)
            self.spans.append((start, stop))
            yield start, stop
            start = stop

    def observe(self, start: int, stop: int, block: Mapping[str, Any]) -> None:
        """Keep the largest retained bytes per electron seen."""
        size = sum(
            int(value.nbytes)
            for value in block.values()
            if hasattr(value, "nbytes") and getattr(value, "ndim", 0) >= 1
        )
        self.bytes_per_electron = max(self.bytes_per_electron, size / max(stop - start, 1))

    @contextmanager
    def block_tuning(self, electrons: int) -> Generator[None]:
        """Run one block as one population, keeping any explicit user control."""
        scratch = min(SCRATCH_BYTES_PER_ELECTRON * electrons, SCRATCH_FRACTION * self.target_bytes)
        tuning = replace(
            self.transport,
            scratch_mib=self.transport.scratch_mib or max(512, int(scratch / _MIB)),
            electron_batch=self.transport.electron_batch or electrons,
        )
        with override_transport_tuning(tuning):
            yield

    @contextmanager
    def pool_limit(self) -> Generator[None]:
        """Cap the device pool under the target for this run, never above an existing cap."""
        previous = BACKEND.memory_limit_bytes()
        outside = BACKEND.device_bytes_outside_pool()
        if previous is None or outside is None:  # no device pool
            yield
            return
        pool_bytes = int(
            max(
                min(
                    POOL_FRACTION * self.target_bytes,
                    self.target_bytes - outside - POOL_HEADROOM_BYTES,
                ),
                MIN_POOL_FRACTION * self.target_bytes,
            )
        )
        limit = pool_bytes if previous == 0 else min(previous, pool_bytes)
        BACKEND.set_memory_limit_bytes(limit)
        try:
            yield
        finally:
            BACKEND.set_memory_limit_bytes(previous)


#: Histories one CUDA transport launch holds (#1252, RTX 5080: scratch-limited
#: fill of 7 131, #889); smaller adaptive chunks add launches without adding fill.
CUDA_FILL_HISTORIES = 7_168
#: Smallest host chunk for adaptive transport: occupies the CPU workers while
#: keeping work past a stop small.
HOST_MIN_CHUNK = 1_024
#: Device or host bytes per transported row while an adaptive chunk is measured.
#: #1253 job 13569 (RTX 5080, per-chunk pool window): 0.8--1.6 KiB per row on
#: 160k--410k-row chunks; 4 KiB keeps a 2.5x margin and caps the densest
#: cleo_sub scene (37.7 rows/electron) near 1M rows, half the 1.8M-row #1252
#: block that filled the card. (#1252's 7--12 KiB included retained phases.)
MEASURE_BYTES_PER_ROW = 4 * 1024
#: Memory one adaptive chunk's rows may take while measured (#1253).
ADAPTIVE_CHUNK_BUDGET_BYTES = 4 * 1024**3


class AdaptiveChunkPlan:
    """Transport chunks for an adaptive run: fill, memory and decision points (#1253).

    A chunk reaches the next decision point when that is at least ``floor``
    electrons away, else it takes ``floor`` electrons, which may span several
    points. Above the floor it never exceeds ``budget / (rows per electron x
    bytes per row)``, with rows per electron the largest rate seen so far (the
    first chunk is ``floor``); the floor wins when the cap is smaller.
    An optional streamed ``block_limit`` overrides that floor and memory cap.
    Chunk boundaries never change decisions, counts or output;
    the stopping monitor folds values at decision points and the driver keeps
    exactly the realized prefix.
    """

    def __init__(
        self,
        floor: int,
        budget_bytes: int = ADAPTIVE_CHUNK_BUDGET_BYTES,
        *,
        block_limit: int | MemoryBlockSizer | None = None,
    ):
        if type(floor) is not int or floor < 1:
            raise ValueError("chunk floor must be a positive integer")
        self.floor = floor
        self.budget_bytes = int(budget_bytes)
        self.rows_per_electron = 0.0
        self.block_limit = block_limit

    @classmethod
    def for_core(
        cls, core: str, hint: int | None = None, *, block_limit=None
    ) -> "AdaptiveChunkPlan":
        """The plan for a transport core; ``hint`` (deprecated ``block_histories``) sets the floor."""
        if hint is not None:
            return cls(hint, block_limit=block_limit)
        return cls(
            CUDA_FILL_HISTORIES if core == "cuda" else HOST_MIN_CHUNK, block_limit=block_limit
        )

    def next_stop(self, start: int, next_point: int | None, limit: int) -> int:
        """End of the chunk starting at ``start``, within ``(start, limit]``."""
        size = self.floor
        if self.rows_per_electron > 0.0:
            fits = int(self.budget_bytes / (self.rows_per_electron * MEASURE_BYTES_PER_ROW))
            size = max(self.floor, fits)
        target = start + self.floor
        if next_point is not None and next_point - start >= self.floor:
            target = next_point
        stop = max(start + 1, min(target, start + size, limit))
        # A streamed route's bound wins even over a legacy chunk-floor hint.
        if isinstance(self.block_limit, Integral):
            stop = min(stop, start + max(1, int(self.block_limit)))
        elif isinstance(self.block_limit, MemoryBlockSizer):
            stop = self.block_limit.next_stop(start, stop)
            self.block_limit.spans.append((start, stop))
        return stop

    def observe(self, start: int, stop: int, block: Mapping[str, Any]) -> None:
        """Keep the largest rows per electron seen."""
        ids = block.get("electron_id")
        rows = 0 if ids is None else int(ids.shape[0])
        self.rows_per_electron = max(self.rows_per_electron, rows / max(stop - start, 1))
        if isinstance(self.block_limit, MemoryBlockSizer):
            self.block_limit.observe(start, stop, block)
