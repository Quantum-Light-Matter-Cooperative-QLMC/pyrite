"""Profile sweep retention with real campaign cases and synthetic spectra.

Run: uv run python checks/checkpoint_sweep_memory.py
Each arm runs in a fresh CPU process and empty checkpoint directory. This
isolates orchestration/storage from Monte Carlo; it is not a transport benchmark.
Linux only: RSS and /proc I/O counters, with an 8 GiB virtual-memory cap.
"""

import argparse
import contextlib
import io
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def io_counts():
    return {
        key: int(value)
        for key, value in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def measure(configs, bins, retain):
    import numpy as np

    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
    from pyrite.detectors import EnergyBins
    from pyrite.runs import run

    grid = np.linspace(100.0, 10000.0, bins)
    cases = build_cases(
        Sweep(
            material="hopg",
            thickness_ang=np.linspace(1000.0, 10000.0, configs),
            beam=BeamSpec(energy_keV=[20.0, 30.0]),
            energy_bins=EnergyBins(cxr=grid, brem=grid),
        )
    )
    # Warm the small per-case derived-scalar path equally in each arm.
    from pyrite.results import store_result

    def payload(index):
        return {
            "E_grid": grid,
            "spec": grid * (index + 1) * 1e-6,
            "brem": grid * (index + 1) * 1e-7,
            "spec_characteristic": grid * (index + 1) * 1e-8,
            "E_grid_brem": grid,
            "brem_wide": grid * (index + 1) * 1e-7,
            "eta": 0.05,
        }

    store_result({}, cases[0], payload(0))
    progress_rss = []

    def synthetic_runner(cases, callback, **kwargs):
        for index, case in enumerate(cases):
            callback(index, case, payload(index))
            progress_rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    run.run_cases = synthetic_runner
    timings = []
    before_io = io_counts()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyrite-sweep-memory-") as directory:
        results = {}
        with contextlib.redirect_stdout(io.StringIO()):
            assert run.run_sweep(
                cases,
                results,
                checkpoint_dir=directory,
                retain_results=retain,
                progress=False,
                on_timing=timings.append,
            )
        elapsed = time.perf_counter() - started
        artifact_bytes = sum(
            path.stat().st_size for path in Path(directory).rglob("*") if path.is_file()
        )
    after_io = io_counts()
    return {
        "configs": configs,
        "records": len(cases),
        "bins": bins,
        "retain_results": retain,
        "seconds": elapsed,
        "peak_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "frontier_peak_MiB": progress_rss[-1],
        "shard_seconds": sum(item.get("shard_write_seconds", 0) for item in timings),
        "consolidation_seconds": sum(item.get("consolidation_seconds", 0) for item in timings),
        "read_MiB": (after_io["rchar"] - before_io["rchar"]) / 1024**2,
        "write_MiB": (after_io["wchar"] - before_io["wchar"]) / 1024**2,
        "artifact_MiB": artifact_bytes / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--bins", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arm", choices=("retained", "disk"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(*args.configs, args.bins, args.repeats) < 1:
        parser.error("counts must be positive")
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    if args.arm:
        print(json.dumps(measure(args.configs[0], args.bins, args.arm == "retained")), flush=True)
    else:
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "backend": "CPU synthetic spectral payloads",
                    "seed": "deterministic",
                    "cache": "fresh directories/readers, common scalar warmup",
                    "repeats": args.repeats,
                    "configs": args.configs,
                    "bins": args.bins,
                }
            ),
            flush=True,
        )
        for configs in args.configs:
            for _ in range(args.repeats):
                for arm in ("retained", "disk"):
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--configs",
                            str(configs),
                            "--bins",
                            str(args.bins),
                            "--arm",
                            arm,
                        ],
                        check=True,
                    )


if __name__ == "__main__":
    main()
