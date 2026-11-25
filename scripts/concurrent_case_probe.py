"""Thread-scaling probe for concurrent cases in one interpreter (#1423).

Runs 8 independent jobs serially and on 1/2/4 threads, reports best-of-N wall
time, and checks every threaded result digest equals the serial one.

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYRITE_MC_BACKEND=cpu uv run python scripts/concurrent_case_probe.py \
        transport|case [Ne]

Jobs are the maintained #611 HOPG slab case (``checks/streamed_scoring_memory.py``:
ELSEPA elastic, BremsLib continuum with ``bremslib-soft-hard`` coupling, fixed
400-node line grid, per-electron core), one seed per job. ``transport`` times
``runner._transport_case``; ``case`` times the full ``run_case``. Both need the
fetched tables (``pyrite tables verify``). ``PROBE_WORKERS`` (default
``1,2,4``) and ``PROBE_REPEATS`` (default 3) narrow the sweep; ``PROBE_KIND=procs``
times a warmed ``spawn`` process pool instead of threads, including result
pickling. Each row also reports peak aggregate PSS of the process tree. Measurement
only: it changes no executor or default.
"""

import hashlib
import os
import platform
import runpy
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numba
import numpy as np
import psutil

from pyrite.montecarlo import runner

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "checks/streamed_scoring_memory.py"))["_case"]
ENERGY_KEV = float(os.environ.get("PROBE_ENERGY_KEV", "30"))


def case(seed):
    return BUILD("hopg", NE, ENERGY_KEV, 1.0e4, seed)


def digest(obj):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(repr(x).encode())

    walk(obj)
    return h.hexdigest()[:16]


MODE = sys.argv[1] if len(sys.argv) > 1 else "transport"
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 16
REPEATS = int(os.environ.get("PROBE_REPEATS", "3"))
JOBS = 8


def job(seed):
    if MODE == "transport":
        return runner._transport_case(case(seed), transport_core="per-electron")
    return runner.run_case(case(seed), transport_core="per-electron")


class PssSampler:
    """Peak summed PSS (MiB) of this process and its children while running."""

    def __init__(self, interval=0.05):
        self.peak, self._stop, self._interval = 0.0, threading.Event(), interval
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _total(self):
        me = psutil.Process()
        total = 0
        for proc in [me, *me.children(recursive=True)]:
            try:
                total += proc.memory_full_info().pss
            except (psutil.Error, AttributeError):
                pass
        return total / 2**20

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._total())
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._total())


def timed(pool, workers):
    t = time.perf_counter()
    if workers == 1:
        out = [job(s) for s in range(JOBS)]
    else:
        out = list(pool.map(job, range(JOBS)))
    return time.perf_counter() - t, out


def main():
    kind = os.environ.get("PROBE_KIND", "threads")
    print(
        f"python {platform.python_version()} gil={sys._is_gil_enabled()} numba {numba.__version__} "
        f"numba_threads={numba.get_num_threads()} mode={MODE} kind={kind} Ne={NE} jobs={JOBS} "
        f"pyrite={Path(runner.__file__).parents[2]}"
    )
    job(0)  # warm JIT / caches outside timing
    _, ref = timed(None, 1)
    ref_d = [digest(r) for r in ref]
    for workers in tuple(int(w) for w in os.environ.get("PROBE_WORKERS", "1,2,4").split(",")):
        if workers == 1:
            pool = None
        elif kind == "procs":
            pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
            list(pool.map(job, range(2 * workers)))  # import and JIT-cache load per worker
        else:
            pool = ThreadPoolExecutor(workers)
        times, same, peak = [], True, 0.0
        for _ in range(REPEATS):
            with PssSampler() as sampler:
                dt, out = timed(pool, workers)
            times.append(dt)
            peak = max(peak, sampler.peak)
            same &= [digest(r) for r in out] == ref_d
        if pool is not None:
            pool.shutdown()
        print(
            f"workers={workers} best={min(times):.2f}s samples={[round(x, 2) for x in times]} "
            f"peak_pss_MiB={peak:.0f} identical_to_serial={same}"
        )


if __name__ == "__main__":
    main()
