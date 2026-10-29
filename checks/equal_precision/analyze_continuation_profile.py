"""Summarize a ``continuation-profile.sbatch`` evidence folder (#789).

Usage: ``python analyze_continuation_profile.py FOLDER --output summary.json``.

For each nsys stem it reports, for the captured measured request only: the
NVTX range totals (request, transport core and its launch/stage/compact/
reassembly children), device-to-host and host-to-device bytes, counts and copy
time, memsets, and the counts and host time of synchronizing and copying CUDA
API calls. For each ncu stem it reports per-launch duration, grid size, SM
active-cycle fraction over elapsed time, achieved active warps and SIMT thread
efficiency, plus the duration-weighted means. Track-length quantiles, survivor
fractions and occupancy-API theoretical occupancy come from the benchmark
records written alongside. Profiled timings carry profiler overhead and are
not benchmark results.
"""

import argparse
import csv
import io
import json
from pathlib import Path

NVTX = [
    "bench.measured",
    "pyrite.transport.core",
    "pyrite.transport.launch",
    "pyrite.transport.compact",
    "pyrite.transport.continuation.launch",
    "pyrite.transport.continuation.stage",
    "pyrite.transport.continuation.assemble",
    "pyrite.transport.output",
]
API = ["cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaMemcpyAsync", "cuLaunchKernel"]
NCU = {
    "gpu__time_duration.sum": "duration",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__occupancy_limit_registers": "occupancy_limit_registers_blocks",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": "sm_active_cycles_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "active_threads_per_warp_inst",
}


def rows(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def nsys_summary(folder, stem):
    out = {"nvtx_s": {}, "memory": {}, "api": {}}
    for row in rows(folder / f"{stem}_nvtx_sum.csv"):
        name = row["Range"].lstrip(":")
        if name in NVTX:
            out["nvtx_s"][name] = {
                "total": int(row["Total Time (ns)"]) * 1e-9,
                "instances": int(row["Instances"]),
            }
    sizes = {r["Operation"]: r for r in rows(folder / f"{stem}_cuda_gpu_mem_size_sum.csv")}
    times = {r["Operation"]: r for r in rows(folder / f"{stem}_cuda_gpu_mem_time_sum.csv")}
    for operation, size in sizes.items():
        key = operation.strip("[]").replace("CUDA ", "")
        out["memory"][key] = {
            "MB": float(size["Total (MB)"]),
            "count": int(size["Count"]),
            "device_time_s": int(times[operation]["Total Time (ns)"]) * 1e-9,
        }
    for row in rows(folder / f"{stem}_cuda_api_sum.csv"):
        if row["Name"] in API:
            out["api"][row["Name"]] = {
                "calls": int(row["Num Calls"]),
                "host_time_s": int(row["Total Time (ns)"]) * 1e-9,
            }
    for row in rows(folder / f"{stem}_cuda_gpu_kern_sum.csv"):
        if row["Name"] == "_transport_kernel":
            out["transport_kernel"] = {
                "instances": int(row["Instances"]),
                "device_time_s": int(row["Total Time (ns)"]) * 1e-9,
            }
    nvtx = out["nvtx_s"]
    if "bench.measured" in nvtx and "pyrite.transport.core" in nvtx:
        out["outside_transport_core_s"] = (
            nvtx["bench.measured"]["total"] - nvtx["pyrite.transport.core"]["total"]
        )
    return out


def ncu_summary(path):
    text = path.read_text()
    start = text.find('"ID"')
    if start < 0:
        return {"status": "no metrics", "log_tail": text[-400:]}
    table = list(csv.reader(io.StringIO(text[start:])))
    header, body = table[0], [r for r in table[2:] if len(r) == len(table[0])]
    launches = []
    for row in body:
        record = dict(zip(header, row, strict=True))
        launch = {}
        for metric, name in NCU.items():
            if metric in record:
                try:
                    launch[name] = float(record[metric].replace(",", ""))
                except ValueError:
                    launch[name] = record[metric]
        launches.append(launch)
    total = sum(launch.get("duration", 0.0) for launch in launches)
    weighted = {}
    for name in ("sm_active_cycles_pct", "achieved_occupancy_pct", "active_threads_per_warp_inst"):
        if total and all(isinstance(launch.get(name), float) for launch in launches):
            weighted[name] = sum(launch["duration"] * launch[name] for launch in launches) / total
    return {"launches": launches, "duration_weighted": weighted, "n_launches": len(launches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder
    summary = {"schema": "pyrite.transport-continuation.profile.v1", "nsys": {}, "ncu": {}}
    for report in sorted(folder.glob("*_nvtx_sum.csv")):
        stem = report.name.removesuffix("_nvtx_sum.csv")
        summary["nsys"][stem] = nsys_summary(folder, stem)
    for path in sorted(folder.glob("ncu-*.csv")):
        summary["ncu"][path.stem] = ncu_summary(path)
    for path in sorted(folder.glob("ncu-*.failed.json")):
        log = path.with_name(path.name.replace(".failed.json", ".log"))
        tail = log.read_text()[-600:] if log.exists() else ""
        summary["ncu"][path.name.removesuffix(".failed.json")] = {
            "status": "failed",
            "log_tail": tail,
        }
    records = {}
    for path in sorted(folder.glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("schema") != "pyrite.transport-continuation.benchmark.v1":
            continue
        sample = record["samples"][0]
        records[path.stem] = {
            "policy": sample["policy"]["name"],
            "track_rows": sample["measured"].get("track_rows"),
            "round_populations": [r["launched"] for r in sample["measured"]["rounds"]],
            "launch_populations": [x["electrons"] for x in sample["measured"]["launches"]],
            "compiled_kernel_resources": record["compiled_kernel_resources"],
        }
    summary["benchmark_records"] = records
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for stem, entry in summary["nsys"].items():
        mem = entry["memory"].get("memcpy Device-to-Host", {})
        sync = entry["api"].get("cudaDeviceSynchronize", {})
        print(
            stem,
            f"outside_core={entry.get('outside_transport_core_s', float('nan')):.3f}s",
            f"DtoH={mem.get('MB', 0):.0f}MB/{mem.get('count', 0)}x/{mem.get('device_time_s', 0):.3f}s",
            f"sync={sync.get('calls', 0)}x",
        )
    for stem, entry in summary["ncu"].items():
        print(stem, entry.get("status", ""), entry.get("duration_weighted", ""))


if __name__ == "__main__":
    main()
