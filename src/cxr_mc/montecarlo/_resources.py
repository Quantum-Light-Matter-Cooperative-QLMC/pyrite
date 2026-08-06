"""Execution resource policies for host and accelerator admission."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._backend import ArrayBackend, BackendResourceError

GIB = 1 << 30
_VALID_POLICIES = ("auto", "conservative", "balanced", "throughput")


@dataclass(frozen=True)
class ResourcePolicy:
    """Resolved device/host limits; distinct from simulation sweep profiles."""

    requested: str
    name: str
    device_budget_bytes: int | None
    device_reserve_bytes: int | None
    host_fraction: float
    host_reserve_bytes: int
    oom_retries: int
    release_every: int


def resolve_resource_policy(backend: ArrayBackend, requested: str | None = None) -> ResourcePolicy:
    """Resolve named policy using device size when ``requested='auto'``."""

    requested = (requested or os.environ.get("CXR_MC_RESOURCE_POLICY", "auto")).strip().lower()
    if requested not in _VALID_POLICIES:
        raise BackendResourceError(
            f"CXR_MC_RESOURCE_POLICY must be one of {', '.join(_VALID_POLICIES)}; got {requested!r}"
        )
    total = backend.device.total_memory_bytes
    name = requested
    if name == "auto":
        name = "conservative" if total is not None and total < 8 * GIB else "balanced"
    if name == "conservative":
        # release_every=4: freeing the CuPy pool every case cost ~4% of GPU-phase
        # wall on a 3060 Ti (8 GiB -> conservative), 20.96 s -> 20.16 s at n=90
        # with FREE_EVERY=8. The per-case free is no longer load-bearing -- the
        # pool cap (device_budget_bytes) plus the OOM chunk-halving retry catch
        # growth -- so keep a cadence, just not every case. Half of balanced's 8.
        fraction, reserve, host_fraction, host_reserve, retries, release = (
            0.50,
            2 * GIB,
            0.60,
            4 * GIB,
            3,
            4,
        )
    elif name == "balanced":
        fraction, reserve, host_fraction, host_reserve, retries, release = (
            0.70,
            1 * GIB,
            0.75,
            2 * GIB,
            3,
            8,
        )
    else:
        fraction, reserve, host_fraction, host_reserve, retries, release = (
            0.85,
            0,
            0.85,
            1 * GIB,
            3,
            16,
        )
    if total is None:
        budget = device_reserve = None
    else:
        budget = max(0, min(int(total * fraction), total - reserve))
        device_reserve = total - budget
    return ResourcePolicy(
        requested=requested,
        name=name,
        device_budget_bytes=budget,
        device_reserve_bytes=device_reserve,
        host_fraction=host_fraction,
        host_reserve_bytes=host_reserve,
        oom_retries=retries,
        release_every=release,
    )


def admitted_chunk(
    *,
    requested_chunk: int,
    bins: int,
    itemsize: int,
    budget_bytes: int | None,
    intermediates: int = 3,
    minimum_chunk: int = 1_000,
) -> int:
    """Cap chunk before allocation; fail when even minimum cannot fit."""

    if budget_bytes is None or bins <= 0:
        return requested_chunk
    bytes_per_segment = intermediates * itemsize * bins
    cap = budget_bytes // bytes_per_segment
    if cap < minimum_chunk:
        raise BackendResourceError(
            f"device budget {budget_bytes / GIB:.2f} GiB cannot admit minimum "
            f"chunk {minimum_chunk} for {bins} bins"
        )
    return min(requested_chunk, int(cap))
