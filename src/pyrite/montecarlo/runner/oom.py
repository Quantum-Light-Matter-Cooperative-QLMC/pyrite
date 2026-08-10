"""Accelerator OOM tagging, pool release cadence, and pool limits."""

from .._backend import _GPU, BACKEND
from . import (
    _FREE_EVERY,
    _FREE_WATERMARK_MB,
    _GPU_POOL_FRAC,
    _GPU_POOL_SHARE,
)

_cases_since_free = 0
_pool_peak_bytes = 0
_pool_limit_set = False


class _SpectrumPhaseOOM(Exception):
    """Tag a catchable OOM with its owning spectrum phase."""

    def __init__(self, phase, error):
        super().__init__(str(error))
        self.phase = phase
        self.error = error


def _should_free(cases_since, every, reserved_bytes, watermark_mb):
    """A2 free-cadence predicate (pure, no CuPy so it unit-tests without a GPU).

    Free the pool when ``every`` cases have elapsed since the last free, OR when
    the reserved pool has crossed the watermark (``watermark_mb == 0`` -> the
    watermark is off). The default (every=1, watermark off) fires every case."""
    over_watermark = watermark_mb > 0 and reserved_bytes > watermark_mb * (1 << 20)
    return cases_since >= every or over_watermark


def _maybe_free_pool():
    """Return this case's GPU scratch to the device on the A2 cadence.

    ``free_all_blocks()`` releases the CuPy pool's free blocks back to the card so
    a long sweep can't let the reserved pool grow/fragment until it fills VRAM --
    but it forces a device sync + full realloc, so A2 stretches how often it runs
    (see :func:`_should_free`). The trigger reads ``total_bytes()`` (reserved), NOT
    ``used_bytes()``: by the time control reaches this inter-case point the case's
    CuPy temporaries are already dereferenced, so ``used_bytes()`` is ~0 and would
    never trip -- ``total_bytes()`` is the footprint that actually grows. Default
    (1 / off) reproduces the original per-case free exactly. Driver-process only,
    so the module counter needs no lock."""
    global _cases_since_free, _pool_peak_bytes
    stats = BACKEND.allocator_stats()
    reserved = int((stats["reserved_mib"] or 0) * (1 << 20))
    if reserved > _pool_peak_bytes:
        _pool_peak_bytes = reserved
    _cases_since_free += 1
    if _should_free(_cases_since_free, _FREE_EVERY, reserved, _FREE_WATERMARK_MB):
        BACKEND.release_memory()
        _cases_since_free = 0


def _ensure_pool_limit():
    """Set the CuPy pool fraction cap once per driver process.

    No-op on a CPU box or when cap is disabled (`_GPU_POOL_FRAC <= 0`).
    Idempotent: safe to call on every case; the module flag means the actual
    `set_limit` runs once. Makes an over-budget alloc raise a catchable
    OutOfMemoryError before the driver's own hard-OOM. The cap is divided by
    `_GPU_POOL_SHARE` so co-tenant scan processes on one GPU sum to _GPU_POOL_FRAC
    rather than oversubscribing it."""
    global _pool_limit_set
    if _pool_limit_set or not _GPU or _GPU_POOL_FRAC <= 0:
        return
    BACKEND.set_memory_limit(_GPU_POOL_FRAC / _GPU_POOL_SHARE)
    _pool_limit_set = True
