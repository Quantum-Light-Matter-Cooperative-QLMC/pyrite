"""Size live streamed blocks from a device-memory target (#868).

``KernelTuning.stream_memory_mib`` caps the CuPy pool under the target, which is
what bounds peak device memory: scoring adapts its chunks to the pool, so its
peak does not follow row count (#907). The block size only decides how much of
that budget retained rows may take. A small first block measures rows per
electron and bytes per row for this scene, and later blocks follow the largest
rate seen, so a denser stretch of the population shrinks them. Electron streams
are counter-addressed, so block sizes do not change transport.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
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
#: Target kept outside the pool: the process held about 1 GiB beyond the pool
#: (CUDA context plus driver allocations) at 16-64 GiB targets on the RTX PRO
#: 6000 (#868). The context is not separable per process through CuPy, so this
#: is a measured constant rather than a per-run reading.
POOL_RESERVE_BYTES = 5 * 1024**3 // 4
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
    def block_tuning(self, electrons: int) -> Iterator[None]:
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
    def pool_limit(self) -> Iterator[None]:
        """Cap the device pool under the target for this run, never above an existing cap."""
        previous = BACKEND.memory_limit_bytes()
        if previous is None:  # no device pool
            yield
            return
        pool_bytes = int(
            max(
                min(POOL_FRACTION * self.target_bytes, self.target_bytes - POOL_RESERVE_BYTES),
                MIN_POOL_FRACTION * self.target_bytes,
            )
        )
        limit = pool_bytes if previous == 0 else min(previous, pool_bytes)
        BACKEND.set_memory_limit_bytes(limit)
        try:
            yield
        finally:
            BACKEND.set_memory_limit_bytes(previous)
