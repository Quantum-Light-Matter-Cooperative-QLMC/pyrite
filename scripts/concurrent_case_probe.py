"""Thread-scaling probe for concurrent cases in one interpreter (#1423).

Runs 8 independent jobs serially and on 1/2/4 threads, reports best-of-N wall
time, and checks every threaded result digest equals the serial one.

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYRITE_MC_BACKEND=cpu pixi run python scripts/concurrent_case_probe.py \
        transport|case [Ne]

Jobs are the maintained #611 HOPG slab case (``checks/streamed_scoring_memory.py``:
ELSEPA elastic, BremsLib continuum with ``bremslib-soft-hard`` coupling, fixed
400-node line grid, per-electron core), one seed per job. ``transport`` times
``runner._transport_case``; ``case`` times the full ``run_case``. Both need the
fetched tables (``pyrite tables verify``). ``PROBE_WORKERS`` (default
``1,2,4``) and ``PROBE_REPEATS`` (default 3) narrow the sweep; ``PROBE_KIND=procs``
times a warmed ``spawn`` process pool instead of threads, including result
pickling. Each row also reports peak aggregate PSS and USS of the process tree,
and parallel arms submit seeds in shuffled order, then compare digests by seed.
``PROBE_STOP=1`` instead runs a stop/drain trial per worker count: job 2 raises,
the coordinator sees the exception from ``as_completed``, cancels pending jobs
and drains started ones, and the row reports time to drain, jobs started,
completed and cancelled, and whether callbacks ran on the coordinator thread.
Measurement only: it changes no executor or default.
"""

import hashlib
import os
import platform
import random
import runpy
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    if seed < 0:
        raise RuntimeError("injected failure")
    if MODE == "transport":
        return runner._transport_case(case(seed), transport_core="per-electron")
    return runner.run_case(case(seed), transport_core="per-electron")


class PssSampler:
    """Peak summed PSS and USS (MiB) of this process and its children while running."""

    def __init__(self, interval=0.05):
        self.peak, self.peak_uss = 0.0, 0.0
        self._stop, self._interval = threading.Event(), interval
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _total(self):
        me = psutil.Process()
        pss = uss = 0
        for proc in [me, *me.children(recursive=True)]:
            try:
                info = proc.memory_full_info()
                pss, uss = pss + info.pss, uss + info.uss
            except (psutil.Error, AttributeError):
                pass
        return pss / 2**20, uss / 2**20

    def _sample(self):
        pss, uss = self._total()
        self.peak, self.peak_uss = max(self.peak, pss), max(self.peak_uss, uss)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def timed(pool, workers, rng):
    seeds = list(range(JOBS))
    t = time.perf_counter()
    if workers == 1:
        out = [job(s) for s in seeds]
    else:
        rng.shuffle(seeds)  # schedule independence: results are compared by seed
        by_seed = dict(zip(seeds, pool.map(job, seeds), strict=True))
        out = [by_seed[s] for s in range(JOBS)]
    return time.perf_counter() - t, out


def stop_trial(pool):
    """Fail job 2, cancel pending jobs, drain started ones; report the outcome."""
    seeds = [0, 1, -1, *range(3, JOBS)]
    coordinator, on_coordinator = threading.get_ident(), True
    cancelled = 0
    t = time.perf_counter()
    futures = [pool.submit(job, s) for s in seeds]
    error = None
    done_before = 0
    for fut in as_completed(futures):
        on_coordinator &= threading.get_ident() == coordinator  # callback site
        if fut.exception() is not None:
            error = fut.exception()
            cancelled = sum(f.cancel() for f in futures)
            break
        done_before += 1
    t_stop = time.perf_counter()
    pool.shutdown(wait=True, cancel_futures=True)
    drained = time.perf_counter()
    return {
        "error": repr(error),
        "to_error_s": round(t_stop - t, 2),
        "drain_s": round(drained - t_stop, 2),
        "started": len(seeds) - cancelled,
        "completed_before_error": done_before,
        "completed_ok": sum(1 for f in futures if not f.cancelled() and f.exception() is None),
        "cancelled": cancelled,
        "callbacks_on_coordinator": on_coordinator,
    }


def main():
    kind = os.environ.get("PROBE_KIND", "threads")
    print(
        f"python {platform.python_version()} gil={sys._is_gil_enabled()} numba {numba.__version__} "
        f"numba_threads={numba.get_num_threads()} mode={MODE} kind={kind} Ne={NE} jobs={JOBS} "
        f"pyrite={Path(runner.__file__).parents[2]}"
    )
    job(0)  # warm JIT / caches outside timing
    rng = random.Random(int(os.environ.get("PROBE_ORDER_SEED", "0")))
    stop = os.environ.get("PROBE_STOP") == "1"
    if not stop:
        _, ref = timed(None, 1, rng)
        ref_d = [digest(r) for r in ref]
    for workers in tuple(int(w) for w in os.environ.get("PROBE_WORKERS", "1,2,4").split(",")):
        if workers == 1 and not stop:
            pool = None
        elif kind == "procs":
            pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
            list(pool.map(job, range(2 * workers)))  # import and JIT-cache load per worker
        else:
            pool = ThreadPoolExecutor(workers)
        if stop:
            print(f"workers={workers} stop_trial={stop_trial(pool)}")
            continue
        times, same, peak, peak_uss = [], True, 0.0, 0.0
        for _ in range(REPEATS):
            with PssSampler() as sampler:
                dt, out = timed(pool, workers, rng)
            times.append(dt)
            peak, peak_uss = max(peak, sampler.peak), max(peak_uss, sampler.peak_uss)
            same &= [digest(r) for r in out] == ref_d
        if pool is not None:
            pool.shutdown()
        print(
            f"workers={workers} best={min(times):.2f}s samples={[round(x, 2) for x in times]} "
            f"peak_pss_MiB={peak:.0f} peak_uss_MiB={peak_uss:.0f} identical_to_serial={same}"
        )


if __name__ == "__main__":
    main()
