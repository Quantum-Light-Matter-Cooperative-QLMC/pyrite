"""Accelerator OOM tagging, pool release cadence, and pool limits."""

from ..._backend import BACKEND
from .chunking import _RESOURCE_POLICY

#: Set to ``1`` to keep terminal-OOM traceback frames (and the device memory
#: their locals pin) for post-mortem debugging. Default off.
KEEP_OOM_FRAMES_ENV = "PYRITE_KEEP_OOM_FRAMES"


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
    Still-executing frames (the raiser itself) raise RuntimeError and are skipped.

    ``PYRITE_KEEP_OOM_FRAMES=1`` keeps every frame intact for post-mortem
    debugging (``pdb.pm()``), at the cost of the device memory they pin."""
    import inspect

    from ..._util.env import env_value

    if env_value(KEEP_OOM_FRAMES_ENV) == "1":
        return
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


#: Device bytes the transport ceiling leaves free beyond memory already used
#: outside the pool (CUDA context, other processes), for allocations made
#: outside the pool during transport (#1315; cf. ``stream_sizing.POOL_HEADROOM_BYTES``).
TRANSPORT_POOL_HEADROOM_BYTES = 1024**3 // 2


def _pool_cap_owned():
    """Whether the pool cap in force is the one this module last set.

    A scoped cap (``MemoryBlockSizer.pool_limit`` for streamed passes) replaces
    it for its duration; the phase caps below then leave the pool alone.
    """
    owned = _RESOURCE_POLICY.pool_limit_bytes
    return owned is None or BACKEND.memory_limit_bytes() == owned


def _ensure_pool_limit():
    """Cap the CuPy pool at the resource-policy budget for the spectrum phase.

    No-op on a CPU box or when cap is disabled (`_RESOURCE_POLICY.gpu_pool_fraction <= 0`).
    Idempotent: safe to call on every case. Makes an over-budget alloc raise a
    catchable OutOfMemoryError before the driver's own hard-OOM. The cap is
    divided by `_RESOURCE_POLICY.gpu_pool_share` so co-tenant scan processes on
    one GPU sum to _RESOURCE_POLICY.gpu_pool_fraction rather than
    oversubscribing it. Lowering the cap after the transport ceiling keeps the
    cached blocks; the pool returns them when an allocation crosses the cap."""
    if not _RESOURCE_POLICY.gpu or _RESOURCE_POLICY.gpu_pool_fraction <= 0:
        return
    if _RESOURCE_POLICY.pool_limit_phase == "budget" or not _pool_cap_owned():
        return
    BACKEND.set_memory_limit(_RESOURCE_POLICY.gpu_pool_fraction / _RESOURCE_POLICY.gpu_pool_share)
    _RESOURCE_POLICY.pool_limit_bytes = BACKEND.memory_limit_bytes()
    _RESOURCE_POLICY.pool_limit_phase = "budget"


def _transport_pool_ceiling():
    """Pool cap for CUDA transport: the card less memory outside the pool (#1315).

    The policy budget is sized for the spectrum phase. A resident adaptive join
    holds the kept blocks and their concatenation, about twice the rows:
    hopg 1000 keV/20M needs 11.3 GiB there, above the RTX 5080's 11.1 GiB
    balanced budget, and fell back to host rows (+6 s transport, line collection
    1.1 s -> 4 s). The ceiling still bounds the pool's cached reservation below
    the card (#1303). Never below the budget; divided by ``gpu_pool_share``.
    An explicit ``PYRITE_MC_GPU_POOL_FRAC`` or resource policy keeps the budget.
    """
    share = _RESOURCE_POLICY.gpu_pool_share
    total = BACKEND.device.total_memory_bytes
    budget = int(_RESOURCE_POLICY.gpu_pool_fraction * total / share) if total else 0
    outside = BACKEND.device_bytes_outside_pool()
    if not _RESOURCE_POLICY.transport_ceiling or not total or outside is None:
        return budget
    return max(budget, (total - outside - TRANSPORT_POOL_HEADROOM_BYTES) // share)


def _ensure_transport_pool_limit():
    """Cap the CuPy pool at :func:`_transport_pool_ceiling` before CUDA transport.

    Without a cap, CuPy keeps every freed block cached: streamed adaptive
    transport reserved 3.5x its use and oversubscribed the card (#1303). Same
    no-op and scoped-cap rules as :func:`_ensure_pool_limit`."""
    if not _RESOURCE_POLICY.gpu or _RESOURCE_POLICY.gpu_pool_fraction <= 0:
        return
    if _RESOURCE_POLICY.pool_limit_phase == "transport" or not _pool_cap_owned():
        return
    BACKEND.set_memory_limit_bytes(_transport_pool_ceiling())
    _RESOURCE_POLICY.pool_limit_bytes = BACKEND.memory_limit_bytes()
    _RESOURCE_POLICY.pool_limit_phase = "transport"
