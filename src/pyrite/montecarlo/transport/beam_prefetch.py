"""Hide the next block's host beam-entry sample under device work (#1454).

Block transport samples every block's entrance positions on the host
(:func:`.beam_entry.initial_beam_positions`) while the device waits. Those
draws are counter-addressed by global electron ID and every later operation
is row-wise, so a block ``[a, b)`` equals rows ``a..b`` of any larger draw and
of any row-wise concatenation of draws: the sampler's documented block
contract, pinned by the block-transport tests. Computing rows early on a
worker thread therefore cannot change a single bit.

:class:`BeamEntryPrefetch` owns one such worker for a block run. Each served
CUDA block schedules ``[stop, stop + GROWTH * Ne)`` with the same arguments
(adaptive chunks grow toward geometric decision points, and the next size is
only known after the current block). A request starting exactly at the
prefetched start, with an identical seed and arguments (identical objects,
equal scalars or equal frozen dataclasses), takes the rows it needs and
samples any remainder serially. A retry of the block just served (the
step-budget retry) samples serially and keeps the prefetch. Anything else samples serially as before: a miss costs time, never bits.

Concurrency (#1423): one worker, one pending future, caller ContextVars copied
into it, and only pure NumPy / ``nogil`` Numba work runs there. Served arrays
are fresh copies, so downstream in-place edits cannot leak between blocks.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import fields, is_dataclass

import numpy as np

_ACTIVE: ContextVar["BeamEntryPrefetch | None"] = ContextVar(
    "pyrite_beam_entry_prefetch", default=None
)
_SCALARS = (bool, int, float, str, type(None))
#: Prefetch span relative to the block just served.
GROWTH = 2


def sample_beam_positions(sample, p, transport_core, *, replaced=False, **kwargs):
    """``sample(p.seed, p.Ne, **kwargs)``, served by the active prefetcher if any.

    ``replaced`` marks a pass whose entrance comes from elsewhere (a secondary
    launch or a GDF beam): it bypasses the prefetcher so it neither consumes
    nor displaces the primary block's pending sample.
    """
    prefetch = _ACTIVE.get()
    if prefetch is None or replaced:
        return sample(p.seed, p.Ne, **kwargs)
    return prefetch.take(sample, p, transport_core, kwargs)


def _same(a, b) -> bool:
    """Identical objects, equal scalars, or frozen dataclasses with ``_same`` fields."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if type(a) in _SCALARS:
        return a == b
    if is_dataclass(a) and type(a).__dataclass_params__.frozen:
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    return False


def _prefetchable(p, transport_core, kwargs) -> bool:
    """CUDA blocks with a real counter-addressed sample, in the sampler's branch order."""
    if transport_core != "cuda" or getattr(p, "gdf_source", None) is not None:
        return False
    if (
        kwargs.get("transverse_distribution") is not None
        or kwargs.get("beam_fwhm_mm")
        or kwargs.get("beam_fwhm_y_mm")
    ):
        return True
    return kwargs.get("groove") is not None and bool(kwargs.get("counter_groove_phase"))


def _rows(result, a, b):
    """Rows ``[a, b)`` of a ``(pos, slopes, record)`` sample, as fresh arrays."""
    pos, slopes, record = result
    part = None if slopes is None else tuple(s[a:b].copy() for s in slopes)
    return pos[a:b].copy(), part, record


def _join(head, tail):
    """Row-wise concatenation of two consecutive samples (``record`` is row-free)."""
    slopes = None
    if head[1] is not None:
        slopes = tuple(np.concatenate(pair) for pair in zip(head[1], tail[1], strict=True))
    return np.concatenate((head[0], tail[0])), slopes, head[2]


class BeamEntryPrefetch:
    """Look-ahead host worker for block beam-entry samples of electrons ``< limit``."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.hits = 0  # served entirely from the prefetch
        self.partial = 0  # prefetch prefix plus a serial remainder
        self.misses = 0  # sampled serially although prefetching applied
        self._pool: ThreadPoolExecutor | None = None
        self._pending = None  # (sample, seed, kwargs without start, lo, hi, future)

    def __enter__(self):
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="pyrite-beam-entry")
        return self

    def __exit__(self, *exc):
        self._drop_pending()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        return False

    def run(self, simulate, *args, **kwargs):
        """Call ``simulate`` with this prefetcher serving its beam-entry samples."""
        token = _ACTIVE.set(self)
        try:
            return simulate(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    def take(self, sample, p, transport_core, kwargs):
        """Return the sample for ``p``'s block, then schedule the following rows."""
        if not _prefetchable(p, transport_core, kwargs):
            return sample(p.seed, p.Ne, **kwargs)
        seed, Ne = p.seed, int(p.Ne)
        start = int(kwargs.get("start", 0))
        result = self._serve(sample, seed, Ne, start, kwargs)
        if result is None:
            self.misses += 1
            result = sample(seed, Ne, **kwargs)
        pool = self._pool
        if pool is not None and self._pending is None:
            self._schedule(pool, sample, seed, Ne, kwargs)
        return result

    def _serve(self, sample, seed, Ne, start, kwargs):
        """The prefetched rows for ``[start, start + Ne)``, completed serially; or None."""
        pending = self._pending
        if pending is None or not self._matches(pending, sample, seed, kwargs):
            self._drop_pending()
            return None
        lo, hi, future = pending[3:]
        if start + Ne == lo:
            return None  # a retry of the block just served: keep the prefetch
        if start != lo:
            self._drop_pending()  # a replay from 0 or a skipped range
            return None
        self._pending = None
        got = future.result()
        if Ne <= hi - lo:
            self.hits += 1
            return _rows(got, 0, Ne)
        self.partial += 1
        tail = sample(seed, Ne - (hi - lo), **dict(kwargs, start=hi))
        return _join(got, tail)

    @staticmethod
    def _matches(pending, sample, seed, kwargs) -> bool:
        p_sample, p_seed, p_kwargs = pending[:3]
        rest = {k: v for k, v in kwargs.items() if k != "start"}
        return (
            p_sample is sample
            and _same(p_seed, seed)
            and p_kwargs.keys() == rest.keys()
            and all(_same(p_kwargs[k], rest[k]) for k in rest)
        )

    def _schedule(self, pool, sample, seed, Ne, kwargs):
        lo = int(kwargs.get("start", 0)) + Ne
        n = min(GROWTH * max(Ne, 1), self.limit - lo)
        if n <= 0:
            return
        context = contextvars.copy_context()
        future = pool.submit(context.run, sample, seed, n, **dict(kwargs, start=lo))
        rest = {k: v for k, v in kwargs.items() if k != "start"}
        self._pending = (sample, seed, rest, lo, lo + n, future)

    def _drop_pending(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            pending[-1].cancel()
