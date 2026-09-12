"""Host-memory admission and worker-pool sizing."""

import warnings

from ..._backend import BackendResourceError
from .chunking import _RESOURCE_POLICY


def _available_mem_mb():
    """Return available system memory in MB"""
    return _RESOURCE_POLICY.available_mem_mb()


def _mem_worker_cap(per_worker_mb=None):
    """Max workers host RAM allows at ``per_worker_mb`` each.

    ``min(MemAvailable, 0.9 * MemTotal) // per_worker_mb``. Binds BOTH worker
    pools (full-case CPU pool and GPU-pipeline transport pool) so neither can
    oversubscribe host RAM and re-create the 2026-07-18 qlmc OOM, where the
    kernel killed one worker and ``BrokenProcessPool`` lost the whole run. The
    two pools pass different budgets: ``_WORKER_MEM_MB`` for full-case workers
    (transport + spectrum state), ``_RESOURCE_POLICY.pipeline_worker_mem_mb`` for the much
    smaller transport-only pipeline workers. Default is ``_WORKER_MEM_MB``."""
    return min(_available_mem_mb(), int(_RESOURCE_POLICY.total_mem_mb * 0.9)) // (
        per_worker_mb or _RESOURCE_POLICY.worker_mem_mb
    )


def _admit_cpu_fallback():
    """Require one policy-budgeted host worker before accelerator fallback."""

    budget = min(
        _available_mem_mb(),
        max(
            0,
            int(_RESOURCE_POLICY.total_mem_mb * _RESOURCE_POLICY.host_fraction)
            - _RESOURCE_POLICY.host_reserve_bytes // 1_000_000,
        ),
    )
    if budget < _RESOURCE_POLICY.worker_mem_mb:
        raise BackendResourceError(
            f"{_RESOURCE_POLICY.name} policy cannot admit CPU fallback: "
            f"{budget} MiB budgeted, {_RESOURCE_POLICY.worker_mem_mb} MiB required"
        )


def _pipeline_slot_cap():
    """Concurrent host-RAM residencies the GPU pipeline may hold.

    A "slot" is one segment payload: either a transport worker building one, or
    an in-flight case whose payload the driver is already holding. Both are
    charged ``_RESOURCE_POLICY.pipeline_worker_mem_mb``, because that budget IS the payload --
    the measured 552-1033 MB child RSS is dominated by the segments the worker
    just built, and the driver holds a copy of exactly those from the moment the
    future completes until the case's spectrum phase runs."""
    return _mem_worker_cap(_RESOURCE_POLICY.pipeline_worker_mem_mb)


def _gpu_pipeline_workers(max_workers, n):
    """Size the GPU-pipeline transport pool (transport-only workers feeding the
    serial GPU). Auto = ~half the usable cores (transport is the tail); an
    explicit request is honored. BOTH are then clamped by host RAM: the pool
    spawns full worker processes just like the CPU pool, so without a cap
    ``ncpu // 2`` workers OOM'd a worker at pool startup and the first
    ``submit`` raised ``BrokenProcessPool`` (the CPU pool got this cap in the
    2026-07-18 fix; this path had been missing it). The budget is the
    transport-only one, NOT the full-case ``_WORKER_MEM_MB``: these workers
    never hold spectrum state, and charging them the full-case footprint capped
    the pool at 2 on a 23.4 GB / 24-core box.

    The cap covers the ``nw + _PIPELINE_PREFETCH_AHEAD`` in-flight payloads too,
    not just the ``nw`` workers -- ``2 * nw + _PIPELINE_PREFETCH_AHEAD`` slots in
    total. Budgeting workers alone is what let the 2026-08-08 `promising`/mose2
    pipeline arm run 16 workers with 18 cases in flight on a 45 GB box: 34 slots
    at 1536 MiB is 52 GB, and measured peak tree RSS was 50.3 GB with 12.9 GB of
    swap. Clamping an EXPLICIT request warns rather than doing it silently.
    Returns the worker count; the caller drops to serial below 2."""
    slots = _pipeline_slot_cap()
    cap = max(0, (slots - _RESOURCE_POLICY.pipeline_prefetch_ahead) // 2)
    if max_workers is None:
        ncpu = _RESOURCE_POLICY.n_cpus or 8
        nw = max(2, min(n, ncpu // 2))
    else:
        nw = min(max_workers, n)
        if cap < nw:
            warnings.warn(
                f"requested {max_workers} GPU-pipeline transport workers, "
                f"host RAM admits {cap} once their in-flight segment payloads "
                f"are charged too ({slots} slots at "
                f"{_RESOURCE_POLICY.pipeline_worker_mem_mb} "
                "MiB each); raise PYRITE_MC_PIPELINE_WORKER_MEM_MB only if the "
                "measured per-worker RSS is smaller than that budget",
                RuntimeWarning,
                stacklevel=2,
            )
    return min(nw, cap)


def _gpu_pipeline_prefetch(nw, n):
    """How many cases the GPU pipeline may hold in flight at once.

    ``nw`` keeps every worker fed and ``_PIPELINE_PREFETCH_AHEAD`` covers the
    handoff. ``_gpu_pipeline_workers`` already sized ``nw`` so this depth fits
    the host budget, but the budget is re-read here because free memory moves
    between the two calls -- and an unbudgeted depth is half of the 2026-08-08
    swap incident."""
    return max(
        1,
        min(
            nw + _RESOURCE_POLICY.pipeline_prefetch_ahead,
            max(1, _pipeline_slot_cap() - nw),
            n,
        ),
    )


def _cpu_pool_workers(max_workers, n):
    """Size the full-case CPU pool, capped so it cannot oversubscribe host RAM.

    Each full-case worker holds transport AND spectrum state (unlike the GPU
    pipeline's transport-only workers): ~5.5 GB anon-rss measured per worker at
    200 keV on qlmc, where the uncapped ncpu*3//4 = 24-worker pool OOM'd the
    45 GiB box (2026-07-18: kernel killed one worker, BrokenProcessPool lost
    the whole run after swap-thrash had already crawled it).

    max_workers: the caller's request -- None means size automatically from
        core count (ncpu*3//4), an integer requests that many (0 is handled by
        the caller, never seen here).
    n: number of cases (never spawn more workers than cases).

    The memory cap binds BOTH branches: even an explicit request is clamped to
    min(MemAvailable, 0.85*MemTotal) // _WORKER_MEM_MB, so a pinned count can't
    re-create the OOM. To deliberately run tighter than the measured budget,
    raise PYRITE_MC_WORKER_MEM_MB -- that's the knob for "my workload is smaller
    than the default assumes", not a bigger --max-workers.

    Returns the worker count to use, >= 1.
    """
    if _RESOURCE_POLICY.n_cpus is not None:
        _max_allowed_workers = _RESOURCE_POLICY.n_cpus * 3 // 4
    else:
        _max_allowed_workers = 6

    worker_cap = _mem_worker_cap()
    if max_workers is None:
        max_workers = _max_allowed_workers
    return max(1, min(max_workers, worker_cap, n))


def _case_progress_label(cases):
    """Name a tqdm case bar by its material when the batch is homogeneous."""
    materials = {case.get("crystal") for case in cases if case.get("crystal")}
    return f"{next(iter(materials))} cases" if len(materials) == 1 else "mixed cases"
