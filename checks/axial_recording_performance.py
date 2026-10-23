"""Axial app workload timing/parity artifact; run in a bounded remote allocation.

Stage with ``pyrite remote sync``, then run with the project's NVIDIA extra:
``uv run python checks/axial_recording_performance.py --recording all --output run``.
Use identical arguments in baseline/candidate checkouts. Timing excludes imports
and one cold run; JSON records each warm repeat and nvidia-smi utilization.
The NPZ holds observables for direct seeded before/after comparisons. Wrap a
separate invocation in ``nsys profile --trace=cuda`` for API/kernel summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--electrons", type=int, default=16)
    parser.add_argument("--step", type=float, default=5)
    parser.add_argument("--depth", type=float, default=10000)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--recording", choices=("off", "spectral", "all"), default="off")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import cupy as cp

    from pyrite.apps.axial_channeling import _level_observable
    from pyrite.montecarlo.channeling import axial_state_diagnostics as spectral
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticConfig,
        axial_diagnostic,
    )
    from pyrite.montecarlo.channeling.states import gaussian

    # Measurement wrappers expose eigensolver time and fallback frequency while
    # leaving the production signatures and numerical results unchanged.
    spectral_stats = dict(host_s=0.0, host_solves=0, device_s=0.0, accepted=0, fallback=0)
    host_solve = spectral.transverse_spectrum

    def timed_host(*pos, **kw):
        start = perf_counter()
        solved = host_solve(*pos, **kw)
        spectral_stats["host_s"] += perf_counter() - start
        spectral_stats["host_solves"] += 1
        return solved

    spectral.transverse_spectrum = timed_host
    try:
        from pyrite.montecarlo.channeling import _axial_spectral_batch as batch
    except ImportError:
        batch = None
    if batch is not None:
        device_solve = batch.transverse_spectra_batch

        def timed_device(*pos, **kw):
            start = perf_counter()
            solved = device_solve(*pos, **kw)
            spectral_stats["device_s"] += perf_counter() - start
            spectral_stats["accepted"] += sum(s is not None for s in solved)
            spectral_stats["fallback"] += sum(s is None for s in solved)
            return solved

        batch.transverse_spectra_batch = timed_device

    depths = tuple(
        np.unique(np.rint(np.linspace(0, args.depth / args.step, args.samples))) * args.step
    )
    config = AxialDiagnosticConfig(
        mode="mcwf",
        backend="cuda",
        shape=(args.points,) * 2,
        cells=(args.cells,) * 2,
        n_electrons=args.electrons,
        step_ang=args.step,
        depths_ang=depths,
        seed=args.seed,
        spectral_states=0 if args.recording == "off" else 16,
        record_coherence=args.recording == "all",
    )
    observer = _level_observable(config) if args.recording == "all" else None
    entry = gaussian(waist_ang=0.5)

    def run():
        return axial_diagnostic(
            config, entry=entry, observable=observer, record_metrics=args.recording == "all"
        )

    start = perf_counter()
    run()
    cp.cuda.get_current_stream().synchronize()
    cold = perf_counter() - start
    spectral_stats.update({key: 0 for key in spectral_stats})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gpu_log = args.output.with_suffix(".gpu.csv")
    with gpu_log.open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=log,
        )
        times = []
        try:
            for _ in range(args.repeats):
                start = perf_counter()
                result = run()
                cp.cuda.get_current_stream().synchronize()
                times.append(perf_counter() - start)
        finally:
            monitor.terminate()
            monitor.wait()
    fields = (
        "density_per_ang2",
        "surviving",
        "numerical_sink",
        "observables",
        "spectral_populations",
        "spectral_represented",
        "coherence_numerator_per_ang2",
        "coherence_second_moment",
    )
    arrays = {name: getattr(result, name) for name in fields if getattr(result, name) is not None}
    arrays.update({"jumps_" + k: v for k, v in result.jump_history.items()})
    if result.metrics is not None:
        arrays.update(
            {
                "metrics_" + k: v
                for k, v in vars(result.metrics).items()
                if isinstance(v, np.ndarray)
            }
        )
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    rows = [line.split(", ") for line in gpu_log.read_text().splitlines() if line.count(",") == 2]
    report = {
        "argv": sys.argv,
        "config": asdict(config),
        "recording": args.recording,
        "precision": "complex128",
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cold_s": cold,
        "warm_s": times,
        "warm_median_s": float(np.median(times)),
        "gpu_util_mean": float(np.mean([float(row[1]) for row in rows])) if rows else None,
        "gpu_memory_max_MiB": max(float(row[2]) for row in rows) if rows else None,
        "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
        "jumps": result.jumps,
        "spectral_solver_totals": spectral_stats,
        "source_sha256": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in sys.modules.copy().items()
            if name.startswith("pyrite.montecarlo.channeling.")
            and getattr(module, "__file__", None)
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
