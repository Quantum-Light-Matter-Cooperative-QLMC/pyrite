"""#825 pageable D2H rate against device-pool occupancy (spill hypothesis).

Times ``cupy.ndarray.get()`` of 64 x 120 MB device arrays into fresh host
memory with 0, 7 and 13 GB held (or cached-free) in the default pool, plus a
pinned-destination control. Prints one JSON line per arm.
"""

import json
import mmap
import time

import cupy as cp
import numpy as np

CHUNK = 120 * 2**20
REPS = 64


def _mapped(n):
    buf = mmap.mmap(-1, n)
    if hasattr(mmap, "MADV_HUGEPAGE"):
        buf.madvise(mmap.MADV_HUGEPAGE)
    return np.frombuffer(buf, np.float64)


def rate(pinned=False, mapped=False):
    src = cp.arange(CHUNK // 8, dtype=np.float64)
    cp.cuda.Device().synchronize()
    out = None
    if pinned:
        mem = cp.cuda.alloc_pinned_memory(CHUNK)
        out = np.frombuffer(mem, np.float64, CHUNK // 8)
    t0 = time.perf_counter()
    keep = []
    for _ in range(REPS):
        if mapped:
            keep.append(src.get(out=_mapped(CHUNK)))
        else:
            keep.append(src.get(out=out) if pinned else src.get())
    wall = time.perf_counter() - t0
    del keep
    return REPS * CHUNK / wall / 1e9


pool = cp.get_default_memory_pool()
rate()  # warm-up
for held_gb in (0, 7, 13):
    for state in ("held", "cached"):
        blocks = [cp.empty(2**30, dtype=np.uint8) for _ in range(held_gb)]
        if state == "cached":
            del blocks
        free, total = cp.cuda.runtime.memGetInfo()
        line = {
            "held_gb": held_gb,
            "state": state,
            "driver_free_gb": free / 1e9,
            "pageable_GBps": rate(),
            "pinned_GBps": rate(pinned=True),
        }
        print(json.dumps(line), flush=True)
        blocks = None
        pool.free_all_blocks()
