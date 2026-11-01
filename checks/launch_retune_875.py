"""Launch-shape and compile-option retune of the exact CUDA kernel, both precisions (#875).

Arms are named ``<precision>-t<threads>[-<build>]``, e.g. ``fp64-t128``,
``mixed-t32``, ``mixed-t32-ro+r168``. ``<build>`` is a
:mod:`transport_kernel_builds` build name (``ro``, ``r168``, ``lb128x2``, ...;
absent: the cupyx.jit kernel). Threads per block go through
``PerElectronTransportConfig.kernel_threads``; precision through
``transport_precision``.

``timing``
    For every (scene, schedule, seed) all arms run back to back in rotating
    order after an untimed warmup of each. Per run: wall time of
    ``simulate_trajectories`` (warm transport) and the summed CUDA-event time
    of the transport-kernel launches, plus a sha256 over every returned array.
    Every arm must match the same-precision ``t128`` jit arm bitwise (launch
    shape and read-only loads must not change bits; a cap that does is
    reported).

``attributes``
    Registers, local (stack) bytes and theoretical occupancy (blocks per SM at
    32/64/128 threads) of each arm's compiled kernel.

Needs a CUDA device and the catalog; run through ``pyrite remote``. No
validation claim.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import transport_kernel_builds as _builds  # noqa: E402
from transport_kernel_builds import (  # noqa: E402
    _function_attributes,
    _functions,
    _scene_kwargs,
    parse_build,
)
from transport_kernel_variants import _digest, _max_threads_per_sm, _sync  # noqa: E402

# #821 anchors (scenes.json) with histories sized for a few seconds per warm run.
_builds.SCENES.update(
    {
        "si-1MeV-1mm": ("Si", "silicon", 0.049942993373787194, 1000.0, 1000.0, 4_000),
        "w-1MeV-100um": ("W", "W", 0.06322199557658834, 1000.0, 100.0, 4_000),
    }
)
SCENES = ("si-300keV-100um", "w-300keV-10um", "si-1MeV-1mm", "w-1MeV-100um")


def parse_arm(name):
    """``(precision, threads, build_name)`` of an arm name."""
    parts = name.split("-", 2)
    precision, threads = parts[0], parts[1]
    if precision not in ("fp64", "mixed") or not threads.startswith("t"):
        raise ValueError(f"arm {name!r}: expected <fp64|mixed>-t<threads>[-<build>]")
    return precision, int(threads[1:]), parts[2] if len(parts) > 2 else "jit"


class Timed:
    """Kernel proxy recording a CUDA-event pair around every launch."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def __call__(self, grid, block, args, shared_mem=0):
        import cupy

        start, stop = cupy.cuda.Event(), cupy.cuda.Event()
        start.record()
        self.inner(grid, block, args, shared_mem)
        stop.record()
        self.sink.append((start, stop))

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Arms:
    def __init__(self, names):
        from pyrite.montecarlo.transport import _jit_launch
        from pyrite.montecarlo.transport._jit_rawmodule import RawModuleKernel

        self.launch, self.names = _jit_launch, names
        self.original = _jit_launch.transport_kernel_variant
        self.spec = {n: parse_arm(n) for n in names}
        self.builds = {n: parse_build(self.spec[n][2]) for n in names}
        self.kernels = {n: {} for n in names}
        self.events = []
        self._wrap = RawModuleKernel

    def select(self, name):
        build, kernels = self.builds[name], self.kernels[name]

        def variant(switches, precision="fp64"):
            key = (switches, precision)
            if key not in kernels:
                jit_kernel = self.original(switches, precision)
                kernels[key] = jit_kernel if build is None else self._wrap(jit_kernel, build)
            return Timed(kernels[key], self.events)

        self.launch._SPECIALIZE_TRANSPORT_KERNEL = True
        self.launch.transport_kernel_variant = variant

    def kernel_ms(self):
        total = sum(float(start.elapsed_time(stop)) for start, stop in self.events)
        launches = len(self.events)
        self.events.clear()
        return total, launches


def _spread(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def timing(args):
    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.montecarlo.transport.batching import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG

    names = args.arms
    arms = Arms(names)
    rows = []
    for scene in args.scenes:
        kwargs = _scene_kwargs(scene)
        histories = args.histories or _builds.SCENES[scene][-1]
        for schedule in args.schedules:
            base = replace(DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG, schedule=schedule)

            def run(name, seed, n, kwargs=kwargs, base=base):
                precision, threads, _ = arms.spec[name]
                config = replace(base, kernel_threads=threads)
                arms.select(name)
                arms.events.clear()
                _sync()
                start = time.perf_counter()
                result = simulate_trajectories(
                    **kwargs,
                    Ne=n,
                    seed=seed,
                    per_electron_config=config,
                    transport_precision=precision,
                )
                _sync()
                wall = time.perf_counter() - start
                return wall, arms.kernel_ms(), result

            for name in names:  # warm: transpile, compile, tables
                run(name, args.seeds[0] + 1_000_000, min(histories, 2000))
            for name in names:  # full size: pool growth outside the timings
                run(name, args.seeds[0] + 1_000_000, histories)
            for index, seed in enumerate(args.seeds):
                shift = index % len(names)
                order = names[shift:] + names[:shift]
                record = dict(scene=scene, schedule=schedule, seed=seed, histories=histories)
                for name in order:
                    wall, (kernel_ms, launches), result = run(name, seed, histories)
                    record[f"{name}_s"] = wall
                    record[f"{name}_kernel_ms"] = kernel_ms
                    record[f"{name}_launches"] = launches
                    record[f"{name}_digest"] = _digest(result)
                    record["rows"] = int(np.asarray(result["electron_id"]).size)
                    del result
                record["identical"] = {
                    n: record[f"{n}_digest"] == record[f"{arms.spec[n][0]}-t128_digest"]
                    for n in names
                    if f"{arms.spec[n][0]}-t128" in names
                }
                rows.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    summary = {}
    for scene in args.scenes:
        for schedule in args.schedules:
            group = [r for r in rows if r["scene"] == scene and r["schedule"] == schedule]
            entry = {}
            for n in names:
                base = f"{arms.spec[n][0]}-t128"
                item = dict(
                    seconds=_spread([r[f"{n}_s"] for r in group]),
                    kernel_ms=_spread([r[f"{n}_kernel_ms"] for r in group]),
                    all_identical=all(r["identical"].get(n, True) for r in group),
                )
                if base in names:
                    item["wall_ratio_vs_t128"] = _spread(
                        [r[f"{base}_s"] / r[f"{n}_s"] for r in group]
                    )
                    item["kernel_ratio_vs_t128"] = _spread(
                        [r[f"{base}_kernel_ms"] / r[f"{n}_kernel_ms"] for r in group]
                    )
                entry[n] = item
            summary[f"{scene}/{schedule}"] = entry
    return dict(mode="timing", arms=names, rows=rows, summary=summary)


def attributes(args):
    from pyrite.montecarlo.transport import simulate_trajectories

    kwargs = _scene_kwargs(args.scenes[0])
    arms = Arms(args.arms)
    records = {}
    for name in args.arms:
        precision, threads, _ = arms.spec[name]
        arms.select(name)
        simulate_trajectories(**kwargs, Ne=500, seed=1, transport_precision=precision)
        _sync()
        found = []
        for kernel in arms.kernels[name].values():
            for function, _source in _functions(kernel):
                info = _function_attributes(function)
                from cupy_backends.cuda.api import driver

                info["occupancy"] = {
                    t: driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, t, 0)
                    * t
                    / _max_threads_per_sm()
                    for t in (32, 64, 128)
                }
                info["blocks_per_sm"] = {
                    t: driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, t, 0)
                    for t in (32, 64, 128)
                }
                found.append(info)
        records[name] = found
        print(json.dumps({name: found}), file=sys.stderr, flush=True)
    return dict(mode="attributes", arms=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["timing", "attributes"])
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--histories", type=int, default=0, help="0: the scene's")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    parser.add_argument("--schedules", nargs="+", default=["full-history", "continuation"])
    parser.add_argument("--output")
    args = parser.parse_args()
    import cupy

    record = timing(args) if args.mode == "timing" else attributes(args)
    record["device"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    record["cupy"] = cupy.__version__
    text = json.dumps(record, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
