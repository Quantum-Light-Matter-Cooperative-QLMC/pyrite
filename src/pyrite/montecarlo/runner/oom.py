"""Accelerator OOM tagging, pool release cadence, and pool limits."""

from ..._backend import BACKEND
from .chunking import _RESOURCE_POLICY


class _SpectrumPhaseOOM(Exception):
    """Tag a catchable OOM with its owning spectrum phase."""

    def __init__(self, phase, error):
        super().__init__(str(error))
        self.phase = phase
        self.error = error


def _drop_traceback_locals(exc):
    """Release frame locals pinned by ``exc``'s traceback chain, keeping the text.

    A terminal OOM propagates with every callee frame (device arrays, suspended
    generators) alive through ``__traceback__``, so the pool cannot be returned
    until the exception dies. Clear finished frames and close pyrite generators
    they hold; traceback objects stay, so file/line/function still format.
    Still-executing frames (the raiser itself) raise RuntimeError and are skipped."""
    import inspect

    seen = set()
    stack = [exc]
    while stack:
        err = stack.pop()
        if err is None or id(err) in seen:
            continue
        seen.add(id(err))
        stack.extend((err.__cause__, err.__context__))
        tb = err.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            for value in list(frame.f_locals.values()):
                if inspect.isgenerator(value) and str(
                    value.gi_frame.f_globals.get("__name__", "") if value.gi_frame else ""
                ).startswith("pyrite"):
                    try:
                        value.close()
                    except (RuntimeError, ValueError):
                        pass
            try:
                frame.clear()
            except RuntimeError:
                pass
            tb = tb.tb_next


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
    stats = BACKEND.allocator_stats()
    reserved = int((stats["reserved_mib"] or 0) * (1 << 20))
    if reserved > _RESOURCE_POLICY.pool_peak_bytes:
        _RESOURCE_POLICY.pool_peak_bytes = reserved
    _RESOURCE_POLICY.cases_since_free += 1
    if _should_free(
        _RESOURCE_POLICY.cases_since_free,
        _RESOURCE_POLICY.free_every,
        reserved,
        _RESOURCE_POLICY.free_watermark_mb,
    ):
        BACKEND.release_memory()
        _RESOURCE_POLICY.cases_since_free = 0


def _ensure_pool_limit():
    """Set the CuPy pool fraction cap once per driver process.

    No-op on a CPU box or when cap is disabled (`_RESOURCE_POLICY.gpu_pool_fraction <= 0`).
    Idempotent: safe to call on every case; the module flag means the actual
    `set_limit` runs once. Makes an over-budget alloc raise a catchable
    OutOfMemoryError before the driver's own hard-OOM. The cap is divided by
    `_RESOURCE_POLICY.gpu_pool_share` so co-tenant scan processes on one GPU sum to _RESOURCE_POLICY.gpu_pool_fraction
    rather than oversubscribing it."""
    if (
        _RESOURCE_POLICY.pool_limit_set
        or not _RESOURCE_POLICY.gpu
        or _RESOURCE_POLICY.gpu_pool_fraction <= 0
    ):
        return
    BACKEND.set_memory_limit(_RESOURCE_POLICY.gpu_pool_fraction / _RESOURCE_POLICY.gpu_pool_share)
    _RESOURCE_POLICY.pool_limit_set = True
