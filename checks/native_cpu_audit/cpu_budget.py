"""CPU thread/process budget probes on four OS-affined CPUs (#1432); diagnostic only.

``topology``: identical four-case rounds under 1x1, 4x1, 1x4 and 2x2
(processes x Numba threads); separate spawned processes, simultaneous PSS/USS
sampling, exact result-hash agreement, no executor changes.

``pools``: whether ``threadpool_limits(limits=1)`` (the BLAS/OpenMP context
the Dask seams use) also caps the selected Numba layer on an existing CXR
reduction kernel; before/inside/after rounds with exact output hashes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import runpy
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def memory(pids):
    """Sum simultaneously observed process PSS/USS; do not sum lifetime RSS peaks."""
    total = dict(pss_bytes=0, uss_bytes=0, rss_bytes=0, os_threads=0)
    for pid in pids:
        fields = {}
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            parts = line.split()
            if parts[0].endswith(":") and len(parts) > 2 and parts[2] == "kB":
                fields[parts[0][:-1]] = int(parts[1]) * 1024
        total["pss_bytes"] += fields.get("Pss", 0)
        total["rss_bytes"] += fields.get("Rss", 0)
        total["uss_bytes"] += fields.get("Private_Clean", 0) + fields.get("Private_Dirty", 0)
        total["os_threads"] += len(list(Path(f"/proc/{pid}/task").iterdir()))
    return total


def worker(connection, case):
    try:
        lib = runpy.run_path(str(ROOT / "checks/native_cpu_audit.py"))
        from pyrite._jit_cache import NumbaCompileRecorder

        start = time.perf_counter()
        run, parameters = lib["case_workload"](case)
        with NumbaCompileRecorder() as recorder:
            reference = run()["arrays"]
        connection.send(
            {
                "ready": True,
                "pid": os.getpid(),
                "environment": lib["environment"](),
                "parameters": parameters,
                "prime_s": time.perf_counter() - start,
                "prime_compile_s": recorder.total_s,
                "arrays": reference,
            }
        )
        while True:
            count = connection.recv()
            if count == 0:
                return
            start, cpu = time.perf_counter(), time.process_time()
            for _ in range(count):
                if run()["arrays"] != reference:
                    raise RuntimeError("case output changed across batch/repeat")
            connection.send(
                {
                    "wall_s": time.perf_counter() - start,
                    "process_cpu_s": time.process_time() - cpu,
                    "cases": count,
                    "arrays": reference,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                }
            )
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def receive(connection):
    value = connection.recv()
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def arm(case, processes, threads):
    """Warm each worker, then time identical four-case rounds with sampled memory."""
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    context = mp.get_context("spawn")
    workers, pipes = [], []
    try:
        for _ in range(processes):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, case))
            process.start()
            child.close()
            workers.append(process)
            pipes.append(parent)
        ready = [receive(pipe) for pipe in pipes]
        expected = int(os.environ["AUDIT1432_CPUS"])
        for info in ready:
            if len(info["environment"]["affinity"]) != expected:
                raise RuntimeError("worker escaped CPU affinity")
            if info["environment"]["numba_threads"] != threads:
                raise RuntimeError("worker Numba mask differs from frozen topology")
        if any(info["arrays"] != ready[0]["arrays"] for info in ready):
            raise RuntimeError("workers produced different results")
        rounds = []
        pids = [os.getpid(), *[p.pid for p in workers]]
        for _ in range(3):
            start = time.perf_counter()
            for pipe in pipes:
                pipe.send(4 // processes)
            waiting, results = set(range(processes)), []
            samples = []
            while waiting:
                samples.append(memory(pids))
                for index in list(waiting):
                    if pipes[index].poll():
                        results.append(receive(pipes[index]))
                        waiting.remove(index)
                if waiting:
                    time.sleep(0.02)
            wall = time.perf_counter() - start
            rounds.append(
                {
                    "cases": 4,
                    "wall_s": wall,
                    "cases_per_s": 4 / wall,
                    "process_cpu_s": sum(r["process_cpu_s"] for r in results),
                    "sampled_peak": {k: max(s[k] for s in samples) for k in samples[0]},
                    "workers": results,
                }
            )
        return {
            "processes": processes,
            "numba_threads_per_process": threads,
            "ready": ready,
            "rounds": rounds,
            "memory_sampling_interval_s": 0.02,
            "timing_scope": "four warm complete run_case calls incl. identical result hashing and parent dispatch/20ms polling; excludes worker setup/prime",
        }
    finally:
        for pipe, process in zip(pipes, workers, strict=True):
            if process.is_alive():
                try:
                    pipe.send(0)
                except (BrokenPipeError, OSError):
                    pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


def topology(parser, args):
    if int(os.environ.get("AUDIT1432_CPUS", "0")) != 4 or len(os.sched_getaffinity(0)) != 4:
        parser.error(
            "this frozen protocol requires exactly four assigned CPUs and matching OS affinity"
        )
    report = {"schema": "pyrite.cpu-topology-audit.v1", "case": args.case, "arms": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for processes, threads in ((1, 1), (4, 1), (1, 4), (2, 2)):
        result = arm(args.case, processes, threads)
        if (
            report["arms"]
            and result["ready"][0]["arrays"] != report["arms"][0]["ready"][0]["arrays"]
        ):
            raise RuntimeError("output changed across topologies")
        report["arms"].append(result)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {
                    "processes": processes,
                    "threads": threads,
                    "round_wall_s": [r["wall_s"] for r in result["rounds"]],
                }
            ),
            flush=True,
        )


def pools(parser, args):
    import hashlib

    import numba
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    from pyrite.montecarlo.spectrum.cxr._cpu_reduce import _sincsq_reduce

    if len(os.sched_getaffinity(0)) != 4 or numba.get_num_threads() != 4:
        parser.error("requires four OS-affined CPUs and four Numba threads")
    nodes = 2048
    energies = np.linspace(1000.0, 20000.0, nodes)
    widths = np.full(nodes, 0.005)
    weights = np.ones((3, nodes))
    grid = energies.copy()
    out = np.zeros((3, nodes))
    _sincsq_reduce(energies, widths, weights, grid, out)
    lib = runpy.run_path(str(ROOT / "checks/native_cpu_audit.py"))
    report = {
        "schema": "pyrite.numba-pool-budget.v1",
        "environment": lib["environment"](),
        "input": {
            "nodes": nodes,
            "lines": nodes,
            "energy_eV": [1000.0, 20000.0],
            "width_per_eV": 0.005,
            "three_weight_rows": "all ones",
        },
        "samples": [],
    }

    def measure(mode):
        out.fill(0)
        start, cpu = time.perf_counter(), time.process_time()
        _sincsq_reduce(energies, widths, weights, grid, out)
        return {
            "mode": mode,
            "wall_s": time.perf_counter() - start,
            "cpu_s": time.process_time() - cpu,
            "numba_mask": numba.get_num_threads(),
            "native_pools": threadpool_info(),
            "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        }

    # Interleave before/inside/after the context; no numerical candidate change.
    for _ in range(3):
        report["samples"].append(measure("base-before"))
        with threadpool_limits(limits=1):
            report["samples"].append(measure("limited"))
        report["samples"].append(measure("base-after"))
    if len({s["output_sha256"] for s in report["samples"]}) != 1:
        raise RuntimeError("pool control changed numerical output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "layer": numba.threading_layer(),
                "masks": [s["numba_mask"] for s in report["samples"]],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="probe", required=True)
    top = sub.add_parser("topology")
    top.add_argument("--case", choices=("thin30", "crystal300"), default="crystal300")
    top.add_argument("--out", type=Path, required=True)
    top.set_defaults(run=topology)
    pool = sub.add_parser("pools")
    pool.add_argument("--out", type=Path, required=True)
    pool.set_defaults(run=pools)
    args = parser.parse_args()
    args.run(parser, args)


if __name__ == "__main__":
    main()
