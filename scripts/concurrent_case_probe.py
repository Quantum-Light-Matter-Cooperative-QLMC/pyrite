"""Thread-scaling probe for concurrent cases in one interpreter (#1423).

Runs 8 independent jobs serially and on 1/2/4 threads, reports best-of-N wall
time, and checks every threaded result digest equals the serial one.

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYRITE_MC_BACKEND=cpu pixi run python scripts/concurrent_case_probe.py \
        transport|case [Ne]

Jobs are, by default (``PROBE_FAMILY=hopg``), the maintained #611 HOPG slab case (``checks/streamed_scoring_memory.py``:
ELSEPA elastic, BremsLib continuum with ``bremslib-soft-hard`` coupling, fixed
400-node line grid, per-electron core), one seed per job. ``transport`` times
``runner._transport_case``; ``case`` times the full ``run_case``. Both need the
fetched tables (``pyrite tables verify``). ``PROBE_FAMILY=layered`` uses the
#1432 MoS2-on-Si stack (100 keV, same physics), ``PROBE_FAMILY=adaptive`` the
HOPG slab with adaptive CXR and brem stopping (``PROBE_TARGET_RSE``, default
0.02, variance-of-variance guard ``PROBE_MAX_VOV``, default 0.05, between
``Ne/8`` and ``Ne`` histories; digests include the realized counts) and ``PROBE_FAMILY=channeling``
the admitted planar Si demo (17 MeV, 2 um, plane (1-10)) through the public
``pyrite.simulate`` with ``Numerics(seed=...)``; both run in ``case`` mode only. ``PROBE_WORKERS`` (default
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

import dataclasses
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
from pyrite.montecarlo.runner.adaptive import without_wall_clock

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "checks/streamed_scoring_memory.py"))["_case"]
ENERGY_KEV = float(os.environ.get("PROBE_ENERGY_KEV", "30"))
FAMILY = os.environ.get("PROBE_FAMILY", "hopg")


def case(seed):
    if FAMILY == "layered":
        return dataclasses.replace(_layered_case(), seed=seed)
    if FAMILY == "adaptive":
        return dataclasses.replace(_adaptive_case(), seed=seed)
    return BUILD("hopg", NE, ENERGY_KEV, 1.0e4, seed)


def _layered_case():
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    beam = pr.Beam(energy_keV=100, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch())
    stack = pr.geometry.Stack.on_substrate(
        "mos2", 1e4, "silicon", 1e5, tilt_deg=30, footprint=pr.geometry.Footprint(5, 5)
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000,
    )
    numerics = pr.Numerics(
        sampling=pr.config.FixedSampling(NE, NE),
        energy_model="midpoint",
        energy_bins=EnergyBins(cxr=np.linspace(1000, 20000, 400), brem=np.arange(1000, 1e5, 1000)),
    )
    return api.build_case(pr.Scene(beam, stack, pr.Detector(), physics=physics), numerics)


def _adaptive_case():
    import pyrite as pr
    from pyrite import api
    from pyrite._sampling import adaptive_sampling
    from pyrite.detectors import EnergyBins

    beam = pr.Beam(
        energy_keV=ENERGY_KEV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    slab = pr.geometry.Slab(
        "hopg", thickness_ang=1e4, tilt_deg=30.0, footprint=pr.geometry.Footprint(5, 5)
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000,
    )
    target = float(os.environ.get("PROBE_TARGET_RSE", "0.02"))
    precision = pr.config.AdaptiveSampling(
        target,
        NE // 8,
        NE,
        observables=("cxr", "brem"),
        brem_target_rse=target,
        brem_min_histories=NE // 8,
        brem_max_histories=NE,
        max_variance_of_variance=float(os.environ.get("PROBE_MAX_VOV", "0.05")),
    )
    numerics = pr.Numerics(
        sampling=adaptive_sampling(precision),
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.linspace(1000, 8000, 400), brem=np.arange(1000, ENERGY_KEV * 1e3, 500)
        ),
    )
    return api.build_case(pr.Scene(beam, slab, pr.Detector(), physics=physics), numerics)


def _channeling(seed):
    import pyrite as pr

    beam = pr.Beam(
        energy_keV=17_000, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=20_000,
        tilt_deg=3,
        tilt_azim_deg=45,
        footprint=pr.geometry.Footprint(5, 5),
    )
    detector = pr.Detector(
        0, acceptance=pr.detectors.SensorAcceptance(distance_mm=1_000, width_mm=3, height_mm=3)
    )
    result = pr.simulate(
        beam,
        target,
        detector,
        physics=pr.Physics(radiative_model="uncoupled"),
        channeling=pr.quantum.Channeling("planar", plane=(1, -1, 0)),
        numerics=pr.Numerics(
            seed=seed,
            sampling=pr.config.FixedSampling(cxr_histories=NE, brem_histories=NE),
            energy_bins=pr.detectors.EnergyBins(
                cxr=np.arange(1_000, 24_000, 50), brem=np.arange(1_000, 17_000_000, 500)
            ),
        ),
    )
    parts = ("characteristic", "brem", "channeling")
    out = {p: getattr(result, p) for p in parts}
    out["cxr_incoherent"], out["cxr_coherent"] = result.cxr.incoherent, result.cxr.coherent
    return {k: np.asarray(v.density) for k, v in out.items() if v is not None}


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
    if FAMILY == "channeling":
        return _channeling(seed)
    if MODE == "transport":
        return runner._transport_case(case(seed), transport_core="per-electron")
    out = runner.run_case(case(seed), transport_core="per-electron")
    if "adaptive_sampling" in out:  # wall-clock fields differ between replays
        out = {**out, "adaptive_sampling": without_wall_clock(out["adaptive_sampling"])}
    return out


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
        f"numba_threads={numba.get_num_threads()} mode={MODE} family={FAMILY} kind={kind} Ne={NE} jobs={JOBS} "
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
