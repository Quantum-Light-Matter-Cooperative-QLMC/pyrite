"""Remote-GPU batching benchmark of energy-resolved axial MCWF (#627).

Run only in a bounded remote GPU allocation from a synced checkout:
``uv run --extra nvidia python checks/axial_energy_mcwf_cuda.py``.
Same crystal, grid, depths, seed and incident-energy quadrature on both
backends (complex128). The CPU reference runs one electron per solver call
and may be capped by ``--cpu-max-electrons``; throughput is compared per
electron-step. CUDA rows scan the admission ``chunk_size`` (largest device
batch per energy group). Energy and entrance draws are shared; trajectory
streams differ, so result summaries are statistical, not paired.
Prints one JSON document. Performance evidence only; no physics verdict.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from pyrite.montecarlo.channeling import (
    AxialEnergyMCWFOptions,
    IncidentEnergyEnsemble,
    axial_energy_mcwf,
)
from pyrite.montecarlo.channeling.axial_propagation import AxialPropagationConfig


def _summary(result):
    return {
        "bin_counts": result.bin_counts.tolist(),
        "surviving_final": float(result.surviving[-1]),
        "surviving_by_bin_final": result.surviving_by_bin[:, -1].tolist(),
        "jumps": {k: int(v) for k, v in sorted(result.jumps.items())},
    }


def _timed(cp, beam, config, options, repeats):
    times, result = [], None
    for _ in range(repeats + 1):
        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        result = axial_energy_mcwf(beam, config, options=options)
        cp.cuda.get_current_stream().synchronize()
        times.append(perf_counter() - start)
    return times, result


def measure(cp, points, electrons, nodes, depth, chunks, *, energy, cpu_electrons, repeats):
    """One workload: CPU reference and a CUDA admission-chunk scan."""
    spread = 0.0 if nodes == 1 else 0.01
    beam = (
        IncidentEnergyEnsemble(energy)
        if nodes == 1
        else IncidentEnergyEnsemble(energy, energy_spread_frac=spread, order=nodes)
    )
    config = AxialPropagationConfig(
        mode="mcwf",
        kinetic_energy_eV=energy,
        shape=(points, points),
        depths_ang=(0.0, depth / 2, depth),
        step_ang=10.0,
    )
    steps = round(depth / 10.0)
    cached = min(max(nodes, 1), 8)
    rows = []
    cpu_n = min(electrons, cpu_electrons)
    times, result = _timed(
        cp,
        beam,
        config,
        AxialEnergyMCWFOptions(
            n_electrons=cpu_n,
            seed=17,
            chunk_size=cpu_n,
            max_cached_solvers=cached,
            work_limit=10**12,
        ),
        repeats,
    )
    warm = float(np.median(times[1:]))
    rows.append(
        {
            "backend": "cpu",
            "electrons": cpu_n,
            "chunk_size": cpu_n,
            "cold_s": times[0],
            "warm_s": times[1:],
            "warm_median_s": warm,
            "electron_steps_per_s": cpu_n * steps / warm,
            "results": _summary(result),
        }
    )
    pool = cp.get_default_memory_pool()
    for chunk in chunks:
        pool.free_all_blocks()
        options = AxialEnergyMCWFOptions(
            n_electrons=electrons,
            seed=17,
            chunk_size=min(chunk, electrons),
            max_cached_solvers=cached,
            backend="cuda",
            work_limit=10**12,
        )
        times, result = _timed(cp, beam, config, options, repeats)
        warm = float(np.median(times[1:]))
        rows.append(
            {
                "backend": "cuda",
                "electrons": electrons,
                "chunk_size": options.chunk_size,
                "cold_s": times[0],
                "warm_s": times[1:],
                "warm_median_s": warm,
                "electron_steps_per_s": electrons * steps / warm,
                "pool_reserved_bytes": pool.total_bytes(),
                "results": _summary(result),
            }
        )
    cpu_rate = rows[0]["electron_steps_per_s"]
    for row in rows[1:]:
        row["throughput_speedup_vs_cpu"] = row["electron_steps_per_s"] / cpu_rate
    return {
        "shape": [points, points],
        "energy_eV": energy,
        "energy_nodes": nodes,
        "energy_spread_frac": spread,
        "max_cached_solvers": cached,
        "depth_ang": depth,
        "step_ang": 10.0,
        "seed": 17,
        "precision": "complex128",
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workload",
        action="append",
        help="POINTS,ELECTRONS,ENERGY_NODES,DEPTH_ANG (repeatable)",
    )
    parser.add_argument("--chunks", default="16,64,256,1024")
    parser.add_argument("--energy-eV", type=float, default=1e6)
    parser.add_argument("--cpu-max-electrons", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    ns = parser.parse_args(argv)
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    chunks = [int(c) for c in ns.chunks.split(",")]
    specs = ns.workload or ["32,256,1,1000", "32,256,3,1000", "32,1024,7,1000"]
    rows = []
    for spec in specs:
        points, electrons, nodes, depth = spec.split(",")
        rows.append(
            measure(
                cp,
                int(points),
                int(electrons),
                int(nodes),
                float(depth),
                chunks,
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
