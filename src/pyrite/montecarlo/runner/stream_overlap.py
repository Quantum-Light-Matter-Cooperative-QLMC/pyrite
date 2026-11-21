"""Overlap live block transport with the previous block's scoring (#862).

Used by :func:`.block_transport._stream_electron_blocks` on CUDA when
:func:`overlap_requested` admits the run (``PYRITE_STREAM_OVERLAP=1``/``0``
forces it on/off). The main thread transports
block ``k + 1`` while one worker thread scores block ``k`` on its own
non-blocking device stream, so at most one transported block waits beside the
one being transported (bounded staging). Blocks are scored one at a time, in
order, by the same consumer, so the running totals see exactly the serial
sequence of additions: outputs equal serial streaming bit for bit.

The scoring stream waits on an event recorded after each block's transport and
is synchronized before the worker drops the block, so pool memory freed back
to the transport stream is never reused while scoring still reads it. Any
device OOM drains the pipeline and returns the uncommitted spans to the serial
loop, whose per-block recovery then applies unchanged.
"""

import contextvars
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..._backend import BACKEND

#: Smallest fixed block that overlaps by default (#862). Every measured
#: brem-only arm at >= 2**18 electrons gained 7-30 % on 16 and 96 GiB cards;
#: 2**16-2**17 gained <= 4 % on 16 GiB and 4096-8192 lost 2-10 %.
AUTO_MIN_BLOCK = 1 << 18


def overlap_requested(sizer, should_stop, n_electrons=0, block_electrons=0) -> bool:
    """Whether to overlap transport with scoring (#862).

    Only CUDA runs with a fixed block size and no early stop can overlap.
    ``PYRITE_STREAM_OVERLAP=1``/``0`` forces it on/off; otherwise it engages
    with at least two full blocks of :data:`AUTO_MIN_BLOCK` electrons. Overlap
    is bitwise equal to serial streaming, so the choice changes wall time and
    peak memory (about one extra block) only.
    """
    forced = os.environ.get("PYRITE_STREAM_OVERLAP")
    if forced == "0" or getattr(BACKEND, "name", None) != "cuda":
        return False
    if sizer is not None or should_stop is not None:
        return False
    if forced == "1":
        return True
    block = int(block_electrons)
    return block >= AUTO_MIN_BLOCK and int(n_electrons) >= 2 * block


def _score(consume, block, ready, stream) -> None:
    from ..spectrum.cxr._coverage import collect_line_data_coverage

    with stream:
        stream.wait_event(ready)
        with collect_line_data_coverage():
            consume(block)
        stream.synchronize()


def stream_overlapped(
    spans: Iterable[tuple[int, int]],
    simulate_block: Callable[[int, int], dict[str, Any]],
    consume: Callable[[dict[str, Any]], None],
    on_block,
    globalize: Callable[[dict[str, Any], int], dict[str, Any]],
    counts: dict[str, int],
) -> list[tuple[int, int]]:
    """Run ``spans`` overlapped, adding committed block counts to ``counts``.

    Returns the spans left for the serial loop: empty on success, or every
    span from the first one whose transport or scoring hit a device OOM.
    """
    from .. import runner
    from .oom import _drop_traceback_locals

    cp = BACKEND.xp
    spans = list(spans)
    pending = None  # (future, span index, block counts)
    failed_at = None

    def collect(entry) -> bool:
        future, index, block_counts = entry
        try:
            future.result()
        except Exception as error:
            if not runner._is_gpu_oom(error):
                raise
            _drop_traceback_locals(error)
            return False
        for key in counts:
            counts[key] += block_counts[key]
        return True

    with ThreadPoolExecutor(1, thread_name_prefix="pyrite-stream-score") as pool:
        stream = cp.cuda.Stream(non_blocking=True)
        try:
            for index, (start, stop) in enumerate(spans):
                try:
                    block = simulate_block(start, stop)
                except Exception as error:
                    if not runner._is_gpu_oom(error):
                        raise
                    _drop_traceback_locals(error)
                    failed_at = index
                    break
                if on_block is not None:
                    on_block(start, stop, block)
                block = globalize(block, start)
                block_counts = {key: int(block.get(key, 0)) for key in counts}
                ready = cp.cuda.Event(block=False, disable_timing=True)
                ready.record()
                if pending is not None:
                    previous, pending = pending, None
                    if not collect(previous):
                        failed_at = previous[1]
                        break
                    BACKEND.release_memory()
                context = contextvars.copy_context()
                future = pool.submit(context.run, _score, consume, block, ready, stream)
                pending = (future, index, block_counts)
                del block
            if pending is not None:
                previous, pending = pending, None
                if not collect(previous):
                    # Scored before the failed transport, so it fails first.
                    failed_at = previous[1]
        finally:
            if pending is not None:  # an error is propagating: wait, commit nothing
                pending[0].exception()
            BACKEND.release_memory()
    return [] if failed_at is None else spans[failed_at:]
