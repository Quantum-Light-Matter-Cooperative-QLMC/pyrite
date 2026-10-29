"""Benchmark bounded CUDA transport continuation against full-history launches (#789).

One process runs every requested policy for one scene and seed, interleaved in
the given order. Each policy performs a complete same-workload warmup at
``seed + 1000000`` and then the measured request; both are device-synchronized.
Per request it records synchronized launch timers and populations, executed vs
retained rows, continuation rounds, scratch bytes passed to each launch, peak
device-pool bytes above the pre-request level (a CuPy memory hook), retained
output bytes and a SHA-256 over every returned array. Compiled kernel
resources are recorded once. Diagnostic timings only: launch timers include
dispatch and synchronization, and hashing/scoring sit outside the request timer.

Policies (``--policies``, comma-separated):

- ``reference``: current default driver, 128 threads, initial capacity 512.
- ``learned32``: #786's request-local capacity hint (1.25 x largest observed
  track, carried from warmup) with 32-thread blocks.
- ``reference-B<n>`` / ``learned32-B<n>``: the same with ``max_batch=n``, a
  full-history arm at the concurrency of a continuation population.
- ``cont-Q<rows>-A<active|all>-T<threads>[-D]``: continuation rounds; ``-D``
  stages retained rows on the device.

Run on the lab box from a synced checkout after ``prepare_stopping.py``; the
continuation policies need a branch that provides
``pyrite.montecarlo.transport.continuation``.
"""

import argparse
import cProfile
import hashlib
import json
import math
import pstats
import resource
import runpy
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np

from pyrite.montecarlo.transport import TransportLUTConfig, _jit_launch, simulate_trajectories
from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
from pyrite.xsgen.sbethe.catalog import resolve_composition_table

HERE = Path(__file__).resolve().parent
ADAPTER = runpy.run_path(str(HERE / "pyrite_adapter.py"))


class PeakPoolHook(cp.cuda.memory_hook.MemoryHook):
    """Live and peak bytes handed out by the CuPy device pool."""

    name = "PeakPoolHook"

    def __init__(self):
        self.live = 0
        self.peak = 0
        self.sizes = {}

    def reset(self):
        self.peak = self.live

    def malloc_postprocess(self, **kwargs):
        self.sizes[kwargs["mem_ptr"]] = kwargs["mem_size"]
        self.live += kwargs["mem_size"]
        self.peak = max(self.peak, self.live)

    def free_postprocess(self, **kwargs):
        self.live -= self.sizes.pop(kwargs["mem_ptr"], kwargs["mem_size"])


def parse_policy(name):
    kind, _, limit = name.partition("-B")
    if kind in {"reference", "learned32"} and (not limit or limit.isdigit()):
        threads = 128 if kind == "reference" else 32
        return {"name": name, "kind": kind, "threads": threads, "max_batch": int(limit or 0)}
    fields = {part[:1]: part[1:] for part in name.split("-")[1:]}
    if not name.startswith("cont-") or set(fields) - {"D"} != {"Q", "A", "T"}:
        raise SystemExit(f"unknown policy {name!r}")
    return {
        "name": name,
        "kind": "continuation",
        "rows_per_round": int(fields["Q"]),
        "active": None if fields["A"] == "all" else int(fields["A"]),
        "threads": int(fields["T"]),
        "device_staging": "D" in fields,
    }


def array_digest(result):
    digest = hashlib.sha256()
    for key, value in sorted(result.items()):
        if isinstance(value, cp.ndarray):
            value = cp.asnumpy(value)
        if isinstance(value, np.ndarray):
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def transport_kwargs(spec, histories, scratch_mib):
    comp = spec["composition"]
    elements = [element for element, _ in comp]
    entries = resolve_layer_tables(comp)
    stopping = resolve_composition_table(spec["catalog_key"], comp)
    brem = load_bremsstrahlung_tables(elements)
    angle = math.radians(spec["angle_deg"])
    return dict(
        E0_keV=spec["energy_keV"],
        Ne=histories,
        composition=comp,
        thickness_ang=spec["thickness_um"] * 1e4,
        crystal_width_mm=spec["width_mm"],
        crystal_height_mm=spec["width_mm"],
        beam_dir=np.array([math.sin(angle), 0, math.cos(angle)]),
        E_cut_keV=spec["electron_cutoff_keV"],
        energy_model="midpoint",
        max_dE_frac=0.05,
        max_steps=20000,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in entries]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=spec["straggling"],
        **ADAPTER["radiative_arguments"](
            "source", "bremslib-soft-hard", spec["hard_photon_cutoff_eV"], brem
        ),
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=False),
        per_electron_config=PerElectronTransportConfig(
            scratch_budget_bytes=scratch_mib * 1024**2, probe_electrons=0
        ),
    )


def run_policy(policy, base, seed, hook, original_factory, cprofile=False):
    launches, rounds = [], []
    state = {"hint": 512}

    def factory(*args, **kwargs):
        kwargs["config"] = replace(
            _jit_launch.DEFAULT_TRANSPORT_KERNEL_CONFIG, nthreads=policy["threads"]
        )
        core, xp = original_factory(*args, **kwargs)

        def timed(*core_args, **core_kwargs):
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            core(*core_args, **core_kwargs)
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start
            counts = cp.asnumpy(core_args[7][0])
            run = core_args[0]
            launches.append(
                {
                    "electrons": int(run[1]),
                    "capacity": int(run[2]),
                    "rows": int(counts.sum()),
                    "max_rows": int(counts.max()) if counts.size else 0,
                    "overflowed": bool(counts.size and counts.max() > run[2]),
                    "scratch_bytes": int(sum(a.nbytes for a in core_args[6])),
                    "pool_live_bytes": hook.live,
                    "synchronized_wall_s": elapsed,
                }
            )
            if policy["kind"] == "learned32" and counts.size:
                state["hint"] = max(state["hint"], int(np.ceil(1.25 * counts.max())))

        return timed, xp

    def request(run_seed, label):
        config = base["per_electron_config"]
        if policy.get("max_batch"):
            config = replace(config, max_batch=policy["max_batch"])
        if policy["kind"] == "learned32":
            config = replace(config, seg_capacity=state["hint"])
        elif policy["kind"] == "continuation":
            from pyrite.montecarlo.transport.continuation import TransportContinuationConfig

            config = replace(
                config,
                continuation=TransportContinuationConfig(
                    rows_per_round=policy["rows_per_round"],
                    active_histories=policy["active"],
                    device_staging=policy["device_staging"],
                    on_round=rounds.append,
                ),
            )
        launches.clear()
        rounds.clear()
        cp.cuda.Device().synchronize()
        before = hook.live
        hook.reset()
        # NVTX names the request so nsys/ncu can capture the measured one only.
        cp.cuda.nvtx.RangePush(f"bench.{label}")
        profiler = cProfile.Profile() if cprofile and label == "measured" else None
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        result = simulate_trajectories(**dict(base, seed=run_seed, per_electron_config=config))
        cp.cuda.Device().synchronize()
        if profiler is not None:
            profiler.disable()
        wall = time.perf_counter() - start
        cp.cuda.nvtx.RangePop()
        retained = sum(
            int(np.asarray(v).nbytes) for v in result.values() if isinstance(v, np.ndarray)
        )
        record = {
            "seed": run_seed,
            "initial_capacity": int(config.seg_capacity),
            "request_s": wall,
            "launch_s": sum(launch["synchronized_wall_s"] for launch in launches),
            "launches": list(launches),
            "rounds": [r for r in rounds if r.get("phase", "round") == "round"],
            "assembly_s": sum(r["wall_s"] for r in rounds if r.get("phase") == "assembly"),
            "executed_rows": sum(launch["rows"] for launch in launches),
            "retained_rows": int(np.asarray(result["electron_id"]).size),
            "retained_host_bytes": retained,
            "peak_pool_above_request_bytes": hook.peak - before,
            "max_launch_scratch_bytes": max(launch["scratch_bytes"] for launch in launches),
            "track_rows": track_distribution(result["electron_id"], base["Ne"]),
            "exits": {
                key: int(result[key])
                for key in ["n_backscattered", "n_transmitted", "n_side_exited", "n_cutoff_stopped"]
            },
        }
        hashed = time.perf_counter()
        record["arrays_sha256"] = array_digest(result)
        record["hash_s"] = time.perf_counter() - hashed
        if profiler is not None:
            stats = pstats.Stats(profiler)
            record["cprofile_cumulative_s"] = {
                f"{path.rsplit('/', 1)[-1]}:{name}": entry[3]
                for (path, _line, name), entry in stats.stats.items()
                if entry[3] > 0.02 * wall
            }
        del result
        return record

    _jit_launch.make_cuda_transport_core = factory
    try:
        warmup = request(seed + 1_000_000, "warmup")
        measured = request(seed, "measured")
    finally:
        _jit_launch.make_cuda_transport_core = original_factory
    return {"policy": policy, "warmup": warmup, "measured": measured}


def track_distribution(electron_id, n_histories):
    """Per-history row-count quantiles and the surviving-history curve.

    A history with ``n`` rows is still running at its ``n``-th step, so
    ``surviving[f]`` is the fraction of histories longer than ``f`` times the
    longest: the lanes a full-history launch keeps busy at that point.
    """
    rows = np.bincount(np.asarray(electron_id), minlength=n_histories)
    longest = int(rows.max())
    return {
        "mean": float(rows.mean()),
        "quantiles": {str(q): float(np.quantile(rows, q)) for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)},
        "zero_row_histories": int(np.count_nonzero(rows == 0)),
        "surviving": {
            f"{f:.1f}": float(np.mean(rows > f * longest)) for f in np.arange(0.1, 1.0, 0.1)
        },
    }


def kernel_resources():
    """Compiled attributes, plus theoretical occupancy from the occupancy API."""
    out = []
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    for kernel, _ in _jit_launch._transport_kernel._cache.values():
        entry = {
            name: cp.cuda.driver.funcGetAttribute(code, kernel.ptr)
            for name, code in [
                ("registers_per_thread", 4),
                ("local_bytes_per_thread", 3),
                ("max_threads_per_block", 0),
            ]
        }
        occupancy = {}
        for block in (32, 64, 128, 256):
            try:
                blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    kernel.ptr, block, 0
                )
            except Exception as error:  # older CuPy: record why, keep the run
                occupancy[str(block)] = repr(error)
                continue
            occupancy[str(block)] = {
                "blocks_per_sm": blocks,
                "theoretical": blocks * block / properties["maxThreadsPerMultiProcessor"],
            }
        entry["theoretical_occupancy_by_block"] = occupancy
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--scratch-mib", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="cProfile the measured request (adds overhead; diagnostics only)",
    )
    args = parser.parse_args()
    _raw, spec, _path = ADAPTER["load_scene"](args.scene)
    base = transport_kwargs(spec, args.histories, args.scratch_mib)
    hook = PeakPoolHook()
    original_factory = _jit_launch.make_cuda_transport_core
    samples = []
    with hook:
        for name in args.policies.split(","):
            policy = parse_policy(name)
            samples.append(
                run_policy(policy, base, args.seed, hook, original_factory, args.cprofile)
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            partial = {"status": "partial", "samples": samples}
            args.output.write_text(json.dumps(partial, indent=2) + "\n")
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    sync = Path.cwd() / ".pyrite-sync"
    if not revision and sync.exists():  # a synced lab checkout is not a git work tree
        fields = dict(line.split(": ", 1) for line in sync.read_text().splitlines() if ": " in line)
        revision = fields.get("code_revision", "")
    record = {
        "schema": "pyrite.transport-continuation.benchmark.v1",
        "status": "ok",
        "scene": args.scene,
        "histories": args.histories,
        "seed": args.seed,
        "scratch_mib": args.scratch_mib,
        "probe_electrons": 0,
        "git_revision": revision or None,
        "device": {
            "name": properties["name"].decode(),
            "multiProcessorCount": properties["multiProcessorCount"],
            "maxThreadsPerMultiProcessor": properties["maxThreadsPerMultiProcessor"],
        },
        "compiled_kernel_resources": kernel_resources(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "samples": samples,
        "interpretation": (
            "Diagnostic. Warmup and measurement share one policy; launch timers include "
            "dispatch and synchronization. Peak pool bytes are allocations above the "
            "pre-request pool level, not driver-reported VRAM."
        ),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for sample in samples:
        m = sample["measured"]
        print(
            sample["policy"]["name"],
            f"request={m['request_s']:.3f}s launch={m['launch_s']:.3f}s",
            f"launches={len(m['launches'])} rows={m['executed_rows']}/{m['retained_rows']}",
            f"peak_pool={m['peak_pool_above_request_bytes'] / 2**20:.1f}MiB",
            m["arrays_sha256"][:12],
        )


if __name__ == "__main__":
    main()
