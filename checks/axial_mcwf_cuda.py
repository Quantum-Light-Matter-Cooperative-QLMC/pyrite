"""Real-hardware axial MCWF benchmark; run only in bounded remote GPU allocation.

Stage with ``pyrite remote sync``; use ``uv run --extra nvidia python
checks/axial_mcwf_cuda.py``. Complex128 CPU/CUDA, same crystal/grid/seed/depth,
CPU and CUDA independent Philox algorithms. Target declared before tuning:
>= 2x warm end-to-end speedup on the 48²/128-electron/100-step workload.
Print cold/warm timing, setup, memory, device and transfer evidence as JSON.
Validation: cr-axial-realspace-propagation (measurement)
Validation: cr-axial-realspace-dissipator (measurement)
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.states import gaussian


def measure(shape, electrons, depth, cp, *, cells=1, energy=1e6, cpu_electrons=None, repeats=3):
    """Time one workload; CPU may run fewer electrons (throughput compared per electron-step)."""
    setup = perf_counter()
    profile = axial_profile("silicon", (0, 0, 1))
    d = axial_dissipator(profile, energy, (cells, cells), (shape, shape))
    potential = profile.on_grid((cells, cells), d.grid.shape)
    setup = perf_counter() - setup
    entry = gaussian(waist_ang=0.6)
    counts = {"cpu": min(electrons, cpu_electrons or electrons), "cuda": electrons}
    timings = {}
    results = {}
    for backend in ("cpu", "cuda"):
        solver = AxialMCWF(d, energy, potential, backend=backend)
        args = dict(
            depths_ang=[0, depth / 2, depth], n_electrons=counts[backend], seed=17, step_ang=10
        )
        times = []
        for _repeat in range(repeats + 1):
            cp.cuda.get_current_stream().synchronize()
            start = perf_counter()
            result = solver.run(entry, **args)
            cp.cuda.get_current_stream().synchronize()
            times.append(perf_counter() - start)
        warm = float(np.median(times[1:]))
        timings[backend] = {
            "electrons": counts[backend],
            "cold_s": times[0],
            "warm_s": times[1:],
            "warm_median_s": warm,
            "electron_steps_per_s": counts[backend] * round(depth / 10) / warm,
        }
        results[backend] = {"surviving": result.surviving.tolist(), "jumps": result.jumps}
    start = perf_counter()
    device = cp.asarray(result.observables)
    cp.cuda.get_current_stream().synchronize()
    upload = perf_counter() - start
    start = perf_counter()
    cp.asnumpy(device)
    transfer = perf_counter() - start
    return {
        "shape": [shape, shape],
        "cells": [cells, cells],
        "energy_eV": energy,
        "electrons": electrons,
        "depth_ang": depth,
        "step_ang": 10,
        "seed": 17,
        "precision": "complex128",
        "setup_s": setup,
        "timings": timings,
        "results": results,
        "warm_speedup": timings["cpu"]["warm_median_s"] / timings["cuda"]["warm_median_s"]
        if counts["cpu"] == counts["cuda"]
        else None,
        "throughput_speedup": timings["cuda"]["electron_steps_per_s"]
        / timings["cpu"]["electron_steps_per_s"],
        "result_upload_s": upload,
        "result_download_s": transfer,
        "pool_used_bytes": cp.get_default_memory_pool().used_bytes(),
        "pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workload",
        action="append",
        help="CELLS,POINTS,ELECTRONS,DEPTH_ANG (repeatable; default: the two hardware-target rows)",
    )
    parser.add_argument("--energy-eV", type=float, default=1e6)
    parser.add_argument("--cpu-max-electrons", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    ns = parser.parse_args(argv)
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if ns.workload is None:
        rows = [measure(12, 8, 100, cp), measure(48, 128, 1000, cp)]
    else:
        rows = []
        for spec in ns.workload:
            cells, points, electrons, depth = spec.split(",")
            rows.append(
                measure(
                    int(cells) * int(points),
                    int(electrons),
                    float(depth),
                    cp,
                    cells=int(cells),
                    energy=ns.energy_eV,
                    cpu_electrons=ns.cpu_max_electrons,
                    repeats=ns.repeats,
                )
            )
    print(
        json.dumps(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "device": name.decode() if isinstance(name, bytes) else name,
                "cpu": platform.processor(),
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "target_warm_speedup": 2,
                "repeats": ns.repeats,
                "warmup": 1,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "workloads": rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
