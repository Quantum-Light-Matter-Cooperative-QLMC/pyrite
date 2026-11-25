"""Hide the next block's host beam-entry sample under device work (#1454).

Block transport samples every block's entrance positions on the host
(:func:`.beam_entry.initial_beam_positions`) while the device waits. Those
draws are counter-addressed by global electron ID, so block ``[a, b)`` is a
pure function of its arguments: computing it early on a worker thread cannot
change a single bit. :class:`BeamEntryPrefetch` owns one such worker for a
block run. Each served CUDA block schedules ``[stop, stop + Ne)`` with the
same arguments; the next request takes the result only when its seed, range
and every argument match exactly (identical objects, or equal scalars), and
otherwise samples serially as before. A miss costs time, never correctness.

Concurrency (#1423): one worker, one pending future, caller ContextVars copied
into it, and only pure NumPy / ``nogil`` Numba work runs there. Served arrays
are never reused, so downstream in-place edits cannot leak between blocks.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

_ACTIVE: ContextVar["BeamEntryPrefetch | None"] = ContextVar(
    "pyrite_beam_entry_prefetch", default=None
)
_SCALARS = (bool, int, float, str, type(None))


def sample_beam_positions(sample, p, transport_core, **kwargs):
    """``sample(p.seed, p.Ne, **kwargs)``, served by the active prefetcher if any."""
    prefetch = _ACTIVE.get()
    if prefetch is None:
        return sample(p.seed, p.Ne, **kwargs)
    return prefetch.take(sample, p, transport_core, kwargs)


def _same(a, b) -> bool:
    return a is b or (type(a) in _SCALARS and type(a) is type(b) and a == b)


def _prefetchable(p, transport_core, kwargs) -> bool:
    """CUDA blocks with a real counter-addressed sample, not overwritten later."""
    if transport_core != "cuda" or getattr(p, "gdf_source", None) is not None:
        return False
    if kwargs.get("groove") is not None:
        return bool(kwargs.get("counter_groove_phase"))
    return bool(
        kwargs.get("transverse_distribution") is not None
        or kwargs.get("beam_fwhm_mm")
        or kwargs.get("beam_fwhm_y_mm")
    )


class BeamEntryPrefetch:
    """One-ahead host worker for block beam-entry samples of electrons ``< limit``."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.hits = 0
        self.misses = 0
        self._pool: ThreadPoolExecutor | None = None
        self._pending = None  # (sample, seed, Ne, kwargs, future)

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
        """Return the sample for ``p``'s block, then schedule the following block."""
        seed, Ne = p.seed, int(p.Ne)
        pending, self._pending = self._pending, None
        result = None
        if pending is not None:
            if self._matches(pending, sample, seed, Ne, kwargs):
                result = pending[-1].result()
            else:
                pending[-1].cancel()
        if result is None:
            if pending is not None or _prefetchable(p, transport_core, kwargs):
                self.misses += 1
            result = sample(seed, Ne, **kwargs)
        else:
            self.hits += 1
        pool = self._pool
        if pool is not None and _prefetchable(p, transport_core, kwargs):
            self._schedule(pool, sample, seed, Ne, kwargs)
        return result

    @staticmethod
    def _matches(pending, sample, seed, Ne, kwargs) -> bool:
        p_sample, p_seed, p_Ne, p_kwargs, _ = pending
        return (
            p_sample is sample
            and _same(p_seed, seed)
            and p_Ne == Ne
            and p_kwargs.keys() == kwargs.keys()
            and all(_same(p_kwargs[k], kwargs[k]) for k in kwargs)
        )

    def _schedule(self, pool, sample, seed, Ne, kwargs):
        start = int(kwargs.get("start", 0)) + Ne
        n = min(Ne, self.limit - start)
        if n <= 0:
            return
        nxt = dict(kwargs, start=start)
        context = contextvars.copy_context()
        future = pool.submit(context.run, sample, seed, n, **nxt)
        self._pending = (sample, seed, n, nxt, future)

    def _drop_pending(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            pending[-1].cancel()
