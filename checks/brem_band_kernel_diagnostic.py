"""Inspect precision specializations and time staged #938 kernels separately.

Run inside a CUDA allocation. Output, CUDA sources and compiler cache must be
gitignored. Replay uses identical captured production bands; raw timings reuse
one staged panel per node count, excluding uploads and host preparation. These
panel timings diagnose kernels, not end-to-end production speedup.
"""

import argparse
import json
import time
from pathlib import Path

import brem_primary_band_benchmark as benchmark
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    benchmark.ignored(args.output / "diagnostic.json")
    args.output.mkdir(parents=True, exist_ok=True)

    import cupy as cp

    from pyrite._backend import _GPU, REAL
    from pyrite.montecarlo.spectrum import brem_band_jit as kernels

    assert _GPU and REAL == np.float32
    original_factory = kernels._band_kernel
    original_group = kernels._group_rows
    pending, panels, compiled = [], {}, {}

    class TimedKernel:
        def __init__(self, kernel, kind, dtype):
            self.kernel, self.kind, self.dtype = kernel, kind, dtype

        def __call__(self, grid, block, arguments, **kwargs):
            label = f"{self.kind}-{np.dtype(self.dtype).name}"
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            cp.cuda.nvtx.RangePush(label)
            start.record()
            self.kernel(grid, block, arguments, **kwargs)
            stop.record()
            cp.cuda.nvtx.RangePop()
            pending.append((label, start, stop))
            if self.kind == "quadrature":
                key = (np.dtype(self.dtype).name, int(arguments[-2]))
            else:
                key = (np.dtype(self.dtype).name, "group")
            panels.setdefault(key, (self.kernel, grid, block, arguments, kwargs))
            compiled[label] = self.kernel

    def factory(dtype):
        return TimedKernel(original_factory(dtype), "quadrature", dtype)

    def group(grid, block, arguments, **kwargs):
        return TimedKernel(original_group, "primary", arguments[0].dtype)(
            grid, block, arguments, **kwargs
        )

    kernels._band_kernel = factory
    kernels._group_rows = group
    replay_start = time.perf_counter()
    try:
        benchmark.replay(
            argparse.Namespace(
                capture=args.capture,
                output=args.output / "replay",
                repeats=args.repeats,
                profile=False,
                nodes=[],
            )
        )
    finally:
        kernels._band_kernel = original_factory
        kernels._group_rows = original_group
    cp.cuda.get_current_stream().synchronize()
    natural = [
        dict(kind=kind, device_ms=cp.cuda.get_elapsed_time(start, stop))
        for kind, start, stop in pending
    ]
    # Persist every instantiated signature. Distinct row-buffer dtypes alone
    # are insufficient: inspect accumulator declarations and reduction bodies.
    sources = []
    for label, kernel in compiled.items():
        for index, (signature, source) in enumerate(kernel.cached_codes.items()):
            target = args.output / f"{label}-{index}.cu"
            target.write_text(source)
            sources.append(dict(file=target.name, signature=str(signature)))

    raw = []
    for key, (kernel, grid, block, arguments, kwargs) in panels.items():
        for _ in range(3):
            kernel(grid, block, arguments, **kwargs)
        cp.cuda.get_current_stream().synchronize()
        samples = []
        for _ in range(args.repeats):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            for _ in range(10):
                kernel(grid, block, arguments, **kwargs)
            stop.record()
            stop.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start, stop) / 10)
        raw.append(dict(dtype=key[0], nodes=key[1], grid=grid, block=block, device_ms=samples))
    report = dict(
        source_sha=benchmark.source_pin(),
        capture=str(args.capture),
        gpu=str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        cupy=cp.__version__,
        replay_wall_s=time.perf_counter() - replay_start,
        natural_launches=natural,
        raw_panels=raw,
        sources=sources,
        limitations="natural launch events include launch starvation; raw panels exclude staging; "
        "primary atomics reuse one panel and retain contention; not an end-to-end speedup",
    )
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
