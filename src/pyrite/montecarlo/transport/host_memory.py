"""Admission and demand-sized host output buffers for charged-particle transport.

The per-electron transport drivers know the number of *emitted* rows only after
running a batch/round. ``max_steps`` is a termination ceiling, not a reasonable
allocation size. Grow physical host output from emitted rows instead of reserving
``Ne * max_steps`` in a potentially overcommitted Linux process.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from ..._backend import BackendResourceError
from ..._env import env_value


class HostMemoryResourceError(BackendResourceError):
    """A retained transport allocation cannot fit the effective host budget."""

    def __init__(self, phase: str, required_bytes: int, available_bytes: int):
        self.phase = phase
        self.required_bytes = int(required_bytes)
        self.available_bytes = int(available_bytes)
        super().__init__(
            f"host memory admission: phase={phase}; required={required_bytes} bytes "
            f"({required_bytes / 2**30:.2f} GiB), available={available_bytes} bytes "
            f"({available_bytes / 2**30:.2f} GiB). Reduce histories or retained "
            "trajectory output, use bounded consumption when supported (#611), "
            "or request a larger enforced memory limit. No histories were truncated."
        )


def _cgroup_headroom_bytes() -> int | None:
    """Free space under the tightest visible cgroup v2 ancestor (if any)."""
    root = Path("/sys/fs/cgroup")
    if not root.exists():
        return None
    # Most containerized/HPC processes see their current cgroup mounted at
    # /sys/fs/cgroup; on other hosts the /proc/self/cgroup suffix is visible.
    bases = [root]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                child = root / line.split("::", 1)[1].lstrip("/")
                if child.is_dir():
                    bases.insert(0, child)
    except OSError:
        pass
    candidate_dirs = set()
    for base in bases:
        current = base
        while current == root or root in current.parents:
            candidate_dirs.add(current)
            if current == root:
                break
            current = current.parent
    remaining = []
    for folder in candidate_dirs:
        try:
            raw = (folder / "memory.max").read_text().strip()
            if raw == "max":
                continue
            maximum = int(raw)
            used = int((folder / "memory.current").read_text().strip())
        except (OSError, ValueError):
            continue
        remaining.append(max(0, maximum - used))
    return min(remaining) if remaining else None


def host_headroom_bytes() -> int:
    """Conservative allocatable RAM, considering MemAvailable and cgroup v2.

    ``PYRITE_MC_HOST_LIMIT_BYTES`` is an optional process-RSS ceiling for
    reproducible small-budget checks. It can only *reduce* system headroom.
    """
    available = int(psutil.virtual_memory().available)
    cgroup = _cgroup_headroom_bytes()
    if cgroup is not None:
        available = min(available, cgroup)
    raw = env_value("PYRITE_MC_HOST_LIMIT_BYTES", "").strip()
    if raw:
        try:
            limit = int(raw)
            if limit < 0:
                raise ValueError
        except ValueError as error:
            raise HostMemoryResourceError("invalid PYRITE_MC_HOST_LIMIT_BYTES", 1, 0) from error
        available = min(available, max(0, limit - int(psutil.Process().memory_info().rss)))
    # A reserve leaves room for Python/CUDA housekeeping and unrelated workers.
    reserve = min(2 * (1 << 30), available // 10)
    return max(0, available - reserve)


def admit_host_allocation(required_bytes: int, phase: str) -> None:
    """Fail *before* materializing a host allocation/copy, never retry unchanged."""
    required = int(required_bytes)
    available = host_headroom_bytes()
    if required > available:
        raise HostMemoryResourceError(phase, required, available)


@contextmanager
def reserve_or_raise(phase: str):
    """Turn a failed lazy ``np.empty`` reservation into the typed resource error."""
    try:
        yield
    except MemoryError as error:
        raise HostMemoryResourceError(phase, 0, host_headroom_bytes()) from error


def admit_batch_download(out_bufs, slots, nseg: int, total: int) -> None:
    """Admit a per-electron batch's compact/download copy before any transfer.

    The compacted selection is another live copy beyond the destination arrays,
    so grow the destination first, then charge the copy itself.
    """
    if out_bufs and hasattr(out_bufs[0], "reserve"):
        out_bufs[0].reserve(nseg + total)
    admit_host_allocation(
        sum(total * a.dtype.itemsize * int(np.prod(a.shape[2:])) for a in slots if a is not None),
        "transport batch compact/download",
    )


class _GrowingColumn:
    """Minimal ndarray-like destination used only by the per-electron drivers."""

    def __init__(self, owner: GrowingHostRows, name: str):
        self.owner = owner
        self.name = name

    @property
    def shape(self):
        return self.owner.arrays[self.name].shape

    @property
    def dtype(self):
        return self.owner.arrays[self.name].dtype

    def reserve(self, rows: int) -> None:
        self.owner.reserve(rows)

    def __getitem__(self, index):
        return self.owner.arrays[self.name][index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            if index.stop is None or index.stop < 0:
                raise IndexError("growing output requires a nonnegative bounded slice")
            required = index.stop
        else:
            positions = np.asarray(index)
            required = int(positions.max()) + 1 if positions.size else 0
        self.reserve(required)
        self.owner.arrays[self.name][index] = value


class GrowingHostRows:
    """Keep all enabled columns at equal capacity; old+new copies are admitted.

    The transport drivers always write completed contiguous batches, or
    continuation scatter indices after the final count is known. Resizing
    happens only at these host-side boundaries, never inside compiled kernels.
    """

    def __init__(self, specs: dict[str, tuple[tuple[int, ...], Any]]):
        self.specs = specs
        self.capacity = 0
        self.arrays = {
            name: np.empty((0, *tail), dtype=dtype) for name, (tail, dtype) in specs.items()
        }

    def column(self, name: str) -> _GrowingColumn:
        return _GrowingColumn(self, name)

    def reserve(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        capacity = max(int(rows), self.capacity + max(1, self.capacity // 2))
        new_bytes = sum(
            capacity * int(np.prod(tail)) * np.dtype(dtype).itemsize
            for tail, dtype in self.specs.values()
        )
        # All old arrays and staged rounds are *still live* while new buffers
        # are allocated and copied. Charge their entire replacement payload.
        admit_host_allocation(new_bytes, "retained transport output grow/assembly")
        arrays = {}
        try:
            for name, (tail, dtype) in self.specs.items():
                new = np.empty((capacity, *tail), dtype=dtype)
                if self.capacity:
                    new[: self.capacity] = self.arrays[name]
                arrays[name] = new
        except MemoryError as error:
            # Do not replay the same impossible allocation. Any staged arrays
            # are dropped as the local mapping goes out of scope.
            raise HostMemoryResourceError(
                "retained transport output allocation", new_bytes, host_headroom_bytes()
            ) from error
        self.arrays = arrays
        self.capacity = capacity
