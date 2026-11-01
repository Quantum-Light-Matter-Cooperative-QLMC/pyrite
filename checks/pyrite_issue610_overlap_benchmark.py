#!/usr/bin/env python3
"""Benchmark PyRITE issue #610 capture policies on an SSH CUDA host.

Local (streams this script over SSH, no scp needed):
    uv run python checks/pyrite_issue610_overlap_benchmark.py --host qlmc --ne 8000 --repeats 3

Already on a GPU host in a PyRITE checkout:
    PYRITE_MC_BACKEND=cuda uv run python checks/pyrite_issue610_overlap_benchmark.py --local

Benchmark source: identical, seeded CUDA-transport result for every write.
Run GPU work in a scheduler allocation after the remote-gpu-jobs pre-flight.
The throughput policy requires the issue #610 patch on the remote checkout.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="qlmc", help="SSH host alias (default: qlmc)")
    p.add_argument(
        "--remote-dir", default="~/pyrite", help="Remote PyRITE checkout (default: ~/pyrite)"
    )
    p.add_argument("--uv", default="~/.local/bin/uv", help="Remote uv executable")
    p.add_argument("--local", action="store_true", help="Run in this Python process (do not SSH)")
    p.add_argument(
        "--ne", type=int, default=8000, help="Electron count for one CUDA run (default: 8000)"
    )
    p.add_argument(
        "--thickness-ang",
        type=float,
        default=200_000.0,
        help="Slab thickness in Å (default: 200000 = 20 µm)",
    )
    p.add_argument(
        "--energy-kev", type=float, default=30.0, help="Incident energy in keV (default: 30)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3, help="Paired writes per policy")
    p.add_argument(
        "--residency",
        choices=("device", "host", "both"),
        default="device",
        help="Source segments (default: device)",
    )
    p.add_argument(
        "--out-dir",
        default="pyrite-output/issue610-bench",
        help="Where to save remote CSV, JSON and temporary artifacts",
    )
    p.add_argument("--chunk-rows", type=int, default=4096, help="HDF5 rows per chunk")
    p.add_argument(
        "--device-block-mib", type=int, default=64, help="Writer transfer block limit in MiB"
    )
    p.add_argument("--read-block", type=int, default=65536, help="Max streamed rows per block")
    p.add_argument(
        "--keep", action="store_true", help="Keep large HDF5 benchmark files (off by default)"
    )
    p.add_argument(
        "--overlap-test",
        action="store_true",
        help="Also compare serialized write + next CUDA transport against threaded overlap (exploratory)",
    )
    p.add_argument(
        "--overlap-only",
        action="store_true",
        help="Skip the earlier compact/throughput microbenchmark; run only the overlap experiment",
    )
    return p.parse_args()


def shell_path(p):
    if p == "~":
        return '"$HOME"'
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


def ssh_launch(args):
    # Pipe this same file over SSH so the remote checkout need not contain it.
    pass_args = [
        "--local",
        "--ne",
        str(args.ne),
        "--thickness-ang",
        str(args.thickness_ang),
        "--energy-kev",
        str(args.energy_kev),
        "--seed",
        str(args.seed),
        "--repeats",
        str(args.repeats),
        "--residency",
        args.residency,
        "--out-dir",
        args.out_dir,
        "--read-block",
        str(args.read_block),
        "--chunk-rows",
        str(args.chunk_rows),
        "--device-block-mib",
        str(args.device_block_mib),
    ]
    if args.keep:
        pass_args += ["--keep"]
    if args.overlap_test:
        pass_args += ["--overlap-test"]
    if args.overlap_only:
        pass_args += ["--overlap-only"]
    checkout = shell_path(args.remote_dir)
    uv = shell_path(args.uv)
    command = (
        f"cd {checkout} && "
        f'PYRITE_MC_BACKEND=cuda PYRITE_HOME="$PWD" '
        f"{uv} run python - {shlex.join(pass_args)}"
    )
    print(f"Connecting to {args.host}; remote checkout: {args.remote_dir}", flush=True)
    print(
        "NOTE: Run `uv run pyrite remote sync` locally first if the #610 patch is not on qlmc.",
        flush=True,
    )
    data = Path(__file__).read_bytes()
    try:
        result = subprocess.run(["ssh", "-T", args.host, command], input=data, check=False)
    except FileNotFoundError as exc:
        raise SystemExit("ssh not found on PATH") from exc
    raise SystemExit(result.returncode)


def linux_rss_bytes():
    # No psutil dependency; works on qlmc Linux.
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class PeakSampler:
    """Sampled process RSS and CuPy pool usage; sampling may miss narrow spikes."""

    def __init__(self, cp, period=0.05):
        self.cp = cp
        self.period = period
        self.stop = threading.Event()
        self.thread = None
        self.before_rss = linux_rss_bytes()
        self.peak_rss = self.before_rss or 0
        self.peak_pool_used = 0
        self.peak_pool_reserved = 0
        free, total = cp.cuda.runtime.memGetInfo()
        self.before_vram = total - free
        self.peak_vram = self.before_vram

    def poll(self):
        pool = self.cp.get_default_memory_pool()
        while not self.stop.is_set():
            rss = linux_rss_bytes()
            if rss is not None:
                self.peak_rss = max(self.peak_rss, rss)
            self.peak_pool_used = max(self.peak_pool_used, pool.used_bytes())
            self.peak_pool_reserved = max(self.peak_pool_reserved, pool.total_bytes())
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.peak_vram = max(self.peak_vram, total - free)
            self.stop.wait(self.period)

    def __enter__(self):
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, linux_rss_bytes() or 0)


def host_copy_tree(obj, cp, memo=None):
    """Copy CUDA arrays once, retaining aliases so HDF5 doesn't duplicate them."""
    if memo is None:
        memo = {}
    key = id(obj)
    if key in memo:
        return memo[key]
    if isinstance(obj, cp.ndarray):
        result = cp.asnumpy(obj)
    elif isinstance(obj, dict):
        result = {k: host_copy_tree(v, cp, memo) for k, v in obj.items()}
    elif isinstance(obj, list):
        result = [host_copy_tree(v, cp, memo) for v in obj]
    elif isinstance(obj, tuple):
        result = tuple(host_copy_tree(v, cp, memo) for v in obj)
    else:
        return obj
    memo[key] = result
    return result


def check_sample(path, original, cp, traj):
    """Verify exact bytes/dtypes/ordering of representative row columns."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as h:
        assert h.attrs["complete"] and h.attrs["schema_version"] == 3
        view = traj.transport_view(h)
        n = int(original["electron_id"].shape[0])
        order = h["transport_order"] if "transport_order" in h else None
        for lo, hi in ((0, min(12, n)), (max(0, n - 12), n)):
            original_rows = order[lo:hi] if order is not None else np.arange(lo, hi)
            for key in ("electron_id", "r_mid", "L_ang", "E_start_keV"):
                if key not in original:
                    continue
                sample = original[key]
                if isinstance(sample, cp.ndarray):
                    expected = cp.asnumpy(sample[cp.asarray(original_rows)])
                else:
                    expected = np.asarray(sample)[original_rows]
                actual = view[key][lo:hi]
                np.testing.assert_array_equal(actual, expected, err_msg=key)
                assert actual.dtype == expected.dtype, key


def run_overlap_benchmark(
    args, cp, traj, simulate_trajectories, source, transport_kwargs, case, out_dir
):
    """Compare two-case throughput with one bounded writer thread.

    This is NOT a production async implementation or a spectrum overlap
    measurement: it overlaps capture of case A with CUDA transport of B.
    The shared A payload is read-only and remains live until future.result().
    The worker count is one, no capture queue is allowed to grow, and writer
    exceptions propagate before an artifact is counted as complete.
    """
    runs = []
    print(
        "\nOVERLAP EXPERIMENT: capture A + transport B, sequential versus concurrent.", flush=True
    )
    print(
        "Only one writer can be in flight; source A stays resident. This does NOT benchmark spectrum scoring.",
        flush=True,
    )
    for trial in range(args.repeats):
        next_kwargs = dict(transport_kwargs, seed=args.seed + 1000 + trial)
        for policy in ("throughput", "compact"):
            modes = ("sequential", "overlap") if trial % 2 == 0 else ("overlap", "sequential")
            for mode in modes:
                path = out_dir / f"overlap-{policy}-{mode}-{trial:02d}.h5"
                if path.exists():
                    raise SystemExit(f"Refusing to overwrite {path}")

                def writer(path=path, policy=policy):
                    a = time.perf_counter()
                    traj.write_trajectory_artifact(path, source, case=case, storage_policy=policy)
                    return time.perf_counter() - a

                with PeakSampler(cp) as sample:
                    start = time.perf_counter()
                    if mode == "sequential":
                        write_s = writer()
                        transport_start = time.perf_counter()
                        next_result = simulate_trajectories(**next_kwargs)
                        cp.cuda.Stream.null.synchronize()
                        next_s = time.perf_counter() - transport_start
                    else:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            # One outstanding write, never an unbounded queue.
                            future = executor.submit(writer)
                            transport_start = time.perf_counter()
                            try:
                                next_result = simulate_trajectories(**next_kwargs)
                                cp.cuda.Stream.null.synchronize()
                                next_s = time.perf_counter() - transport_start
                            finally:
                                # A writer exception cannot be lost. If transport
                                # also fails, joining still closes any .partial.
                                try:
                                    write_s = future.result()
                                except BaseException:
                                    raise
                    wall_s = time.perf_counter() - start
                assert path.is_file() and path.stat().st_size > 0
                header = traj.read_trajectory_header(path)
                assert header["complete"]
                assert header["segment_count"] == len(source["L_ang"])
                del next_result
                gc.collect()
                if not args.keep:
                    path.unlink()
                record = dict(
                    trial=trial + 1,
                    policy=policy,
                    mode=mode,
                    wall_s=round(wall_s, 6),
                    write_s=round(write_s, 6),
                    next_transport_s=round(next_s, 6),
                    peak_rss_mib=round(sample.peak_rss / 2**20, 1),
                    peak_vram_mib=round(sample.peak_vram / 2**20, 1),
                    peak_pool_reserved_mib=round(sample.peak_pool_reserved / 2**20, 1),
                )
                runs.append(record)
                print(
                    f"  {policy:10s} {mode:10s} #{trial + 1}: "
                    f"pair {wall_s:.3f}s, write {write_s:.3f}s, "
                    f"next transport {next_s:.3f}s, "
                    f"peak RSS {record['peak_rss_mib']:.0f} MiB",
                    flush=True,
                )
    for policy in ("throughput", "compact"):
        seq = statistics.median(
            r["wall_s"] for r in runs if r["policy"] == policy and r["mode"] == "sequential"
        )
        ovl = statistics.median(
            r["wall_s"] for r in runs if r["policy"] == policy and r["mode"] == "overlap"
        )
        print(
            f"  {policy}: sequential {seq:.3f}s, overlap {ovl:.3f}s, "
            f"wall gain {seq - ovl:+.3f}s ({seq / ovl:.2f}x)",
            flush=True,
        )
    csv_path = out_dir / "overlap_issue610.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    (out_dir / "overlap_issue610.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(f"Overlap results saved to {csv_path}", flush=True)
    print(
        "This isolates write/transport concurrency; it does not establish an "
        "end-to-end speedup for real spectrum-scoring runs.",
        flush=True,
    )


def run_benchmark(args):
    if args.ne <= 0 or args.repeats <= 0 or args.read_block <= 0 or args.thickness_ang <= 0:
        raise SystemExit("--ne, --repeats, --read-block and --thickness-ang must be positive")
    if args.chunk_rows <= 0 or args.device_block_mib <= 0:
        raise SystemExit("--chunk-rows and --device-block-mib must be positive")
    if args.overlap_only:
        args.overlap_test = True

    # Override host-specific catalog settings for this synthetic transport test.
    # PyRITE's ordinary remote launcher explicitly selects the synchronized
    # catalog and user layer; a direct `uv run python` doesn't. Older user-layer
    # crystals may still carry removed E_grid keys. This benchmark needs neither
    # user profiles nor crystal catalog entries, so use a private empty layer.
    # Set all variables BEFORE importing PyRITE (which may load the catalog).
    checkout = Path.cwd()
    bundled_catalog = checkout / "src/pyrite/data/catalog"
    if not bundled_catalog.is_dir():
        raise SystemExit(
            f"Expected PyRITE checkout in {checkout}; missing {bundled_catalog}. "
            "Run from the checkout, or launch with --host."
        )
    empty_catalog = tempfile.TemporaryDirectory(prefix="pyrite-issue610-user-catalog-")
    atexit.register(empty_catalog.cleanup)
    os.environ["PYRITE_CATALOG"] = str(bundled_catalog)
    os.environ["PYRITE_USER_CATALOG"] = empty_catalog.name
    os.environ["PYRITE_MC_BACKEND"] = "cuda"
    import cupy as cp

    from pyrite import _backend
    from pyrite.montecarlo import trajectories as traj
    from pyrite.montecarlo.transport import simulate_trajectories

    if _backend.BACKEND.name != "cuda":
        raise SystemExit(f"Requested CUDA backend, got {_backend.BACKEND.name!r}")
    signature = inspect.signature(traj.write_trajectory_artifact)
    if "storage_policy" not in signature.parameters:
        raise SystemExit(
            "The remote PyRITE checkout does not have the #610 throughput patch.\n"
            "Apply the patch locally and sync it with `uv run pyrite remote sync`,\n"
            "or update the remote checkout before trying again."
        )

    traj._ROW_CHUNK = args.chunk_rows
    traj._DEVICE_BLOCK_BYTES = args.device_block_mib * 2**20
    import h5py
    import numpy as np

    stamp = checkout / ".pyrite-sync"
    metadata = dict(
        arguments=vars(args),
        python=sys.version,
        platform=platform.platform(),
        cupy=cp.__version__,
        numpy=np.__version__,
        h5py=h5py.__version__,
        hdf5=h5py.version.hdf5_version,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        trajectory_source_sha256=hashlib.sha256(Path(traj.__file__).read_bytes()).hexdigest(),
        sync_stamp=stamp.read_text() if stamp.is_file() else None,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        memory_sampling_period_s=0.05,
        vram_metric="device total minus free; includes context/allocator and other processes",
        cache_state="warm CUDA; filesystem page cache not dropped; no fsync",
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = cp.cuda.runtime.getDeviceProperties(0).get("name", "CUDA GPU")
    if isinstance(gpu, bytes):
        gpu = gpu.decode(errors="replace")
    metadata["gpu"] = str(gpu)
    (out_dir / "benchmark_environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"GPU: {gpu} | backend: {_backend.BACKEND.name} | CuPy: {cp.__version__}", flush=True)
    print(f"Python: {sys.version.split()[0]} | code: {Path(traj.__file__).resolve()}", flush=True)
    print("Catalog: synchronized bundled catalog with isolated empty user layer", flush=True)
    print(f"Output: {out_dir} | N_e={args.ne:,} | repetitions={args.repeats}", flush=True)

    theta, azimuth = math.radians(45), math.radians(180)
    # Exactly the sample-frame incident direction from tilted_geometry for a
    # normal tilted by (theta, azimuth), without importing crystal geometry.
    # The previous geometry import unnecessarily initialized the full material
    # catalog, causing a schema error on hosts with older user catalog files.
    beam_dir = (
        -math.sin(theta) * math.cos(azimuth),
        -math.sin(theta) * math.sin(azimuth),
        math.cos(theta),
    )
    transport_kwargs = dict(
        E0_keV=args.energy_kev,
        Ne=args.ne,
        thickness_ang=args.thickness_ang,
        composition=[("C", 0.11)],
        elastic_model="sr",
        inelastic_model="continuous",
        transport_core="cuda",
        keep_segments_on_device=True,
        beam_dir=beam_dir,
        tilt_polar_rad=theta,
        tilt_azim_rad=azimuth,
        beam_fwhm_mm=0.001,
        seed=args.seed,
    )
    case = {
        "name": "issue610-gpu-benchmark",
        "E0_keV": args.energy_kev,
        "Ne": args.ne,
        "thickness_ang": args.thickness_ang,
        "seed": args.seed,
        "elastic_model": "sr",
    }

    print("Warming CUDA transport (32 electrons)...", flush=True)
    warm = dict(transport_kwargs)
    warm["Ne"] = 32
    _ = simulate_trajectories(**warm)
    cp.cuda.Stream.null.synchronize()
    del _, warm
    gc.collect()

    print("Generating ONE GPU-resident trajectory set for all policy comparisons...", flush=True)
    with PeakSampler(cp) as sample:
        start = time.perf_counter()
        device_segments = simulate_trajectories(**transport_kwargs)
        cp.cuda.Stream.null.synchronize()
        transport_time = time.perf_counter() - start
    num_rows = int(device_segments["L_ang"].shape[0])
    if not isinstance(device_segments["r_mid"], cp.ndarray):
        raise RuntimeError("Transport did not retain r_mid on CUDA device")
    print(
        f"CUDA transport: {transport_time:.3f}s; rows={num_rows:,}; "
        f"sampled peak RSS={sample.peak_rss / 2**20:.0f} MiB; "
        f"sampled peak CuPy pool={sample.peak_pool_reserved / 2**20:.0f} MiB",
        flush=True,
    )
    if num_rows == 0:
        raise SystemExit("No segments generated; increase slab thickness / adjust inputs")

    staging_s = 0.0
    if args.residency in ("host", "both"):
        print("Making a host-resident copy once (outside timed captures)...", flush=True)
        start = time.perf_counter()
        host_segments = host_copy_tree(device_segments, cp)
        cp.cuda.Stream.null.synchronize()
        staging_s = time.perf_counter() - start
        print(
            f"Host staging: {staging_s:.3f}s (not included in writer timings)",
            flush=True,
        )
    else:
        host_segments = None

    # Host-only runs model a pipeline that has already released its GPU payload.
    # Use separate device/host invocations for comparable isolated memory peaks.
    if args.residency == "host" and not args.overlap_test:
        del device_segments
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    if not args.overlap_only:
        entries = []
        for residency in ("device", "host") if args.residency == "both" else (args.residency,):
            source = device_segments if residency == "device" else host_segments
            assert source is not None
            for trial in range(args.repeats):
                # Alternate policy order to mitigate caching/thermal/order effects.
                policies = (
                    ("compact", "throughput") if trial % 2 == 0 else ("throughput", "compact")
                )
                for policy in policies:
                    name = f"{residency}-{policy}-{trial:02d}.h5"
                    path = out_dir / name
                    if path.exists():
                        raise SystemExit(
                            f"Refusing existing benchmark file: {path} (remove it first)"
                        )
                    traffic = {"bytes": 0, "calls": 0, "max_bytes": 0}
                    old_to_cpu = traj._to_cpu

                    def tracked_to_cpu(array, traffic=traffic, old_to_cpu=old_to_cpu):
                        if traj.is_device_array(array):
                            count = int(array.nbytes)
                            traffic["bytes"] += count
                            traffic["calls"] += 1
                            traffic["max_bytes"] = max(traffic["max_bytes"], count)
                        return old_to_cpu(array)

                    try:
                        traj._to_cpu = tracked_to_cpu
                        with PeakSampler(cp) as sample:
                            start = time.perf_counter()
                            cpu_start = time.process_time()
                            traj.write_trajectory_artifact(
                                path, source, case=case, storage_policy=policy
                            )
                            cp.cuda.Stream.null.synchronize()
                            elapsed = time.perf_counter() - start
                            cpu_elapsed = time.process_time() - cpu_start
                    finally:
                        traj._to_cpu = old_to_cpu
                    size = path.stat().st_size
                    check_sample(path, source, cp, traj)
                    with PeakSampler(cp):
                        start = time.perf_counter()
                        read_count = 0
                        # Bounded streaming of *all* row fields; excludes shared values.
                        for block in traj.iter_electron_row_blocks(
                            path, args.read_block, include_shared=False
                        ):
                            read_count += len(block["electron_id"])
                        read_elapsed = time.perf_counter() - start
                    assert read_count == num_rows, (read_count, num_rows)
                    row = dict(
                        residency=residency,
                        policy=policy,
                        trial=trial,
                        ne=args.ne,
                        rows=num_rows,
                        transport_s=round(transport_time, 6),
                        staging_s=round(staging_s, 6),
                        chunk_rows=args.chunk_rows,
                        device_block_mib=args.device_block_mib,
                        write_s=round(elapsed, 6),
                        cpu_write_s=round(cpu_elapsed, 6),
                        read_stream_s=round(read_elapsed, 6),
                        size_mib=round(size / 2**20, 3),
                        bytes_per_row=round(size / num_rows, 3),
                        transferred_mib=round(traffic["bytes"] / 2**20, 3),
                        download_calls=traffic["calls"],
                        max_download_mib=round(traffic["max_bytes"] / 2**20, 3),
                        before_rss_mib=round(sample.before_rss / 2**20, 1),
                        peak_rss_mib=round(sample.peak_rss / 2**20, 1),
                        before_vram_mib=round(sample.before_vram / 2**20, 1),
                        peak_vram_mib=round(sample.peak_vram / 2**20, 1),
                        peak_pool_reserved_mib=round(sample.peak_pool_reserved / 2**20, 1),
                    )
                    entries.append(row)
                    print(
                        f"{residency:6s} {policy:10s} #{trial + 1} | "
                        f"write {elapsed:7.3f}s | stream {read_elapsed:7.3f}s | "
                        f"{size / 2**20:8.1f} MiB | D→H {traffic['bytes'] / 2**20:8.1f} MiB",
                        flush=True,
                    )
                    if not args.keep:
                        path.unlink()
                    gc.collect()
        # Summary: paired, per residency; throughput/compact write ratio and size ratio.
        print("\nMEDIANS (lower write / read is better):", flush=True)
        for residency in ("device", "host") if args.residency == "both" else (args.residency,):
            by_policy = {
                p: [r for r in entries if r["residency"] == residency and r["policy"] == p]
                for p in ("compact", "throughput")
            }
            t0 = statistics.median(r["write_s"] for r in by_policy["compact"])
            t1 = statistics.median(r["write_s"] for r in by_policy["throughput"])
            print(
                f"  {residency:6s}: compact {t0:.3f}s, throughput {t1:.3f}s | "
                f"speedup {t0 / t1:.2f}x, "
                f"size ratio {statistics.median(r['size_mib'] for r in by_policy['throughput']) / statistics.median(r['size_mib'] for r in by_policy['compact']):.2f}x",
                flush=True,
            )
        csv_path = out_dir / "capture_issue610.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
        json_path = out_dir / "capture_issue610.json"
        json_path.write_text(
            json.dumps(
                dict(gpu=str(gpu), metadata=metadata, transport_s=transport_time, records=entries),
                indent=2,
            )
            + "\n"
        )
        print(f"\nSaved remote results: {csv_path} and {json_path}", flush=True)
        print(
            "Memory peaks are sampled and include the preexisting transport buffers; "
            "D→H counts only calls made by the HDF5 writer, not transport itself.",
            flush=True,
        )
    if args.overlap_test:
        run_overlap_benchmark(
            args, cp, traj, simulate_trajectories, device_segments, transport_kwargs, case, out_dir
        )


if __name__ == "__main__":
    parsed = cli()
    if not parsed.local:
        ssh_launch(parsed)
    run_benchmark(parsed)
