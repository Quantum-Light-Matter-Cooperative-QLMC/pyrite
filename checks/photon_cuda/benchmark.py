"""Fixed-physics CUDA scaling, profiling and Optuna chunk experiments (#341).

Remote GPU allocation only. Run as a module so the original maintained
workload is reused: ``uv run --extra nvidia --extra optimization python -m
checks.photon_cuda.benchmark scaling|trace|tune``. Timings include host rates
and both transfer directions. No physical parameters or numerical defaults
are optimized. GPU-event elapsed time includes idle gaps between dispatches;
use the Nsight trace to distinguish active GPU kernels from host dispatch.
"""

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import resource
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from checks.photon_first_flight_cuda import SEED, _workload
from pyrite._backend import select_backend
from pyrite.materials.photon_cross_sections import EPDL_TABLE_SHA256
from pyrite.montecarlo.transport import photon_navigation as nav
from pyrite.montecarlo.transport.photon_cuda import scene_first_interactions_cuda


def workload(n, regions):
    args, kwargs = _workload(n)
    if regions != 3:
        step = 3e7 / (regions - 1)
        layers = [
            (i * step, (i + 0.8) * step, [("C", 0.08), ("Pb", 0.02)]) for i in range(regions - 1)
        ]
        args = (*args[:4], layers, args[5])
        nav.validate_photon_regions(layers, args[5], **kwargs)
    return args, kwargs


def cuda_host(cp, args, kwargs, chunk, implementation="fused", threads=128):
    n = len(args[0])
    out = {"distance_ang": np.empty(n)}
    out.update({key: np.empty(n, dtype=np.int64) for key in ("layer", "channel", "element")})
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        hit = scene_first_interactions_cuda(
            *(a[sl] for a in args[:4]),
            *args[4:],
            implementation=implementation,
            threads=threads,
            **kwargs,
        )
        for key, values in hit.items():
            out[key][sl] = cp.asnumpy(values)
    return out


def parity(expected, actual):
    for key in expected:
        if key == "distance_ang":
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(actual[key], expected[key])


def measure(run, repeats=5):
    run()
    run()
    values = []
    for _ in range(repeats):
        start = perf_counter()
        run()
        values.append(perf_counter() - start)
    return values


def scaling(cp):
    rows = []
    for regions in (3, 17):
        for n in (1024, 16384, 262144):
            args, kwargs = workload(n, regions)
            expected = nav.scene_first_interactions(*args, **kwargs)
            gpu = partial(cuda_host, cp, args, kwargs, n)
            parity(expected, gpu())
            cpu_times = measure(partial(nav.scene_first_interactions, *args, **kwargs))
            gpu_times = measure(gpu)
            array = partial(cuda_host, cp, args, kwargs, n, "array")
            parity(expected, array())
            array_times = measure(array)
            rows.append(
                {
                    "rays": n,
                    "regions": regions,
                    "cpu_seconds": cpu_times,
                    "cuda_seconds": gpu_times,
                    "array_seconds": array_times,
                    "parity": "pass",
                }
            )
    profile = cProfile.Profile()
    args, kwargs = workload(16384, 3)
    profile.runcall(cuda_host, cp, args, kwargs, 16384)
    stats = pstats.Stats(profile)
    top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:25]
    return {
        "scaling": rows,
        "cprofile_cumulative": [
            {
                "function": f"{Path(key[0]).name}:{key[1]}:{key[2]}",
                "calls": value[1],
                "exclusive_seconds": value[2],
                "cumulative_seconds": value[3],
            }
            for key, value in top
        ],
    }


def tune(cp):
    from pyrite.optimization import Categorical, Objective, OptunaOptimizer, ParameterSpace

    args, kwargs = workload(262144, 3)
    expected = nav.scene_first_interactions(*args, **kwargs)
    optimizer = OptunaOptimizer(
        ParameterSpace(
            {
                "chunk": Categorical((4096, 16384, 65536, 131072, 262144)),
                "threads": Categorical((32, 64, 128, 256, 512)),
            }
        ),
        (Objective("seconds", "minimize", "s"),),
        seed=SEED,
    )
    trials = []
    for _ in range(24):
        proposal = optimizer.ask()
        chunk = proposal.parameters["chunk"]
        threads = proposal.parameters["threads"]
        run = partial(cuda_host, cp, args, kwargs, chunk, "fused", threads)
        try:
            parity(expected, run())
            times = measure(run, repeats=3)
        except Exception:
            optimizer.tell(proposal, None, failed=True)
            raise
        optimizer.tell(proposal, (float(np.median(times)),))
        trials.append({"chunk": chunk, "threads": threads, "seconds": times})
    best = int(optimizer.study.best_params["chunk"])
    best_threads = int(optimizer.study.best_params["threads"])
    # Fresh alternating observations; optimizer trials are selection data only.
    holdout = {"unchunked": [], "selected": []}
    for repeat in range(9):
        arms = (("unchunked", 262144, 128), ("selected", best, best_threads))
        for name, chunk, threads in arms if repeat % 2 == 0 else reversed(arms):
            start = perf_counter()
            actual = cuda_host(cp, args, kwargs, chunk, "fused", threads)
            holdout[name].append(perf_counter() - start)
            parity(expected, actual)
    return {
        "tuning": {
            "rays": 262144,
            "regions": 3,
            "seed": SEED,
            "trials": trials,
            "selected_chunk": best,
            "selected_threads": best_threads,
            "holdout_seconds": holdout,
        }
    }


def trace(cp):
    from cupy.cuda import profiler

    args, kwargs = workload(16384, 3)
    for _ in range(2):
        cuda_host(cp, args, kwargs, 16384)
    profiler.start()
    for _ in range(5):
        cuda_host(cp, args, kwargs, 16384)
    cp.cuda.get_current_stream().synchronize()
    profiler.stop()
    return {"trace_calls": 5, "rays_per_call": 16384, "regions": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scaling", "trace", "tune"))
    args = parser.parse_args()
    backend = select_backend("cuda")
    cp = backend.xp
    result = {"scaling": scaling, "trace": trace, "tune": tune}[args.mode](cp)
    root = Path(__file__).resolve().parents[2]
    sources = (
        "checks/photon_cuda/benchmark.py",
        "checks/photon_first_flight_cuda.py",
        "src/pyrite/montecarlo/transport/photon_cuda.py",
        "src/pyrite/montecarlo/transport/_photon_cuda_kernel.py",
        "src/pyrite/montecarlo/transport/photon_navigation.py",
    )
    result.update(
        {
            "device": backend.device.name,
            "cupy": cp.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "seed": SEED,
            "epdl_table_sha256": EPDL_TABLE_SHA256,
            "warmups": 2,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "device_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
            "memory_scope": "combined process peak and retained device pool, not per-arm peaks",
            "source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sources
            },
        }
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
