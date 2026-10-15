"""Isolated grid-search/replay diagnostics, excluded from benchmark summaries."""

import hashlib
import json
import runpy
import sys
import time
from dataclasses import replace
from pathlib import Path

import cupy as xp
import numpy as np

from pyrite.montecarlo import transport
from pyrite.montecarlo._cupy_jit import jit
from pyrite.montecarlo.transport import _jit_launch, _jit_radiative_device, batching
from pyrite.montecarlo.transport._jit_device import F64_INF, F64_ONE, I32_ONE, I32_TWO, I32_ZERO


@jit.rawkernel(device=True)
def indexed_next_node(x, nominal, nom_base, n_red, top_a, top_b, cut_reduced):
    lo = I32_ZERO
    hi = n_red - I32_ONE
    while lo < hi:
        mid = (lo + hi) // I32_TWO
        if nominal[nom_base + mid] <= x:
            lo = mid + I32_ONE
        else:
            hi = mid
    best = F64_INF
    if lo < n_red - I32_ONE:
        best = nominal[nom_base + lo]
    if top_a > x and top_a < best:
        best = top_a
    if top_b > x and top_b < best:
        best = top_b
    if cut_reduced > x and cut_reduced < best:
        best = cut_reduced
    if F64_ONE > x and F64_ONE < best:
        best = F64_ONE
    return best


@jit.rawkernel(device=True)
def indexed_row_chi(reduced, nominal, nom_base, n_red, top_r, chi, chi_row):
    if reduced >= top_r:
        return chi[chi_row + n_red - I32_ONE]
    lo = I32_ZERO
    hi = n_red - I32_ONE
    while hi - lo > I32_ONE:
        mid = (lo + hi) // I32_TWO
        if reduced >= nominal[nom_base + mid]:
            lo = mid
        else:
            hi = mid
    lower = lo
    x0 = nominal[nom_base + lower]
    x1 = nominal[nom_base + lower + I32_ONE]
    if lower == n_red - I32_TWO:
        x1 = top_r
    weight = (reduced - x0) / (x1 - x0)
    return chi[chi_row + lower] + weight * (chi[chi_row + lower + I32_ONE] - chi[chi_row + lower])


def main():
    variant = sys.argv.pop(1)
    audit_path = Path(sys.argv.pop(1))
    if variant not in {
        "baseline",
        "nodes",
        "both",
        "capacity",
        "both-capacity",
        "threads32",
        "both-capacity32",
        "radiation-off",
    }:
        raise ValueError(variant)
    if variant in {"nodes", "both", "both-capacity", "both-capacity32"}:
        _jit_radiative_device._rad_next_node = indexed_next_node
    if variant in {"both", "both-capacity", "both-capacity32"}:
        _jit_radiative_device._rad_row_chi = indexed_row_chi
    if variant in {"capacity", "both-capacity", "both-capacity32"}:
        original_config = batching.PerElectronTransportConfig

        def configured(*args, **kwargs):
            return replace(original_config(*args, **kwargs), seg_capacity=12500)

        batching.PerElectronTransportConfig = configured
    launches = []
    if variant == "radiation-off":
        original_simulate = transport.simulate_trajectories

        def without_radiation(**kwargs):
            kwargs.update(
                radiative_model="uncoupled", radiative_cutoff_eV=None, bremslib_tables=None
            )
            return original_simulate(**kwargs)

        transport.simulate_trajectories = without_radiation
    original_factory = _jit_launch.make_cuda_transport_core

    def factory(*args, **kwargs):
        if variant in {"threads32", "both-capacity32"}:
            kwargs["config"] = replace(_jit_launch.DEFAULT_TRANSPORT_KERNEL_CONFIG, nthreads=32)
        original, array_module = original_factory(*args, **kwargs)

        def core(*core_args):
            xp.cuda.Device().synchronize()
            start = time.perf_counter()
            original(*core_args)
            xp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start
            counts = xp.asnumpy(core_args[7][0])
            cap = int(core_args[0][2])
            launches.append(
                {
                    "electrons": int(core_args[0][1]),
                    "capacity": cap,
                    "segments": int(counts.sum()),
                    "max_segments": int(counts.max()),
                    "replayed": bool(counts.max() > cap),
                    "synchronized_wall_s": elapsed,
                    "scratch_bytes": sum(a.nbytes for a in core_args[6]),
                }
            )

        return core, array_module

    _jit_launch.make_cuda_transport_core = factory
    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    original_score = namespace["score_segments"]
    blocks = []

    def score(result, count):
        digest = hashlib.sha256()
        for key, value in sorted(result.items()):
            if isinstance(value, np.ndarray):
                digest.update(key.encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        ids = np.asarray(result["electron_id"])
        blocks.append(
            {
                "primaries": count,
                "segments": int(ids.size),
                "arrays_sha256": digest.hexdigest(),
                "launch_end": len(launches),
            }
        )
        if variant == "radiation-off":
            result["hard_radiative_k_eV"] = np.zeros(ids.size)
        return original_score(result, count)

    namespace["main"].__globals__["score_segments"] = score
    namespace["main"]()
    sample_path = Path(sys.argv[sys.argv.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    sample.update(schema="pyrite.equal-precision.diagnostic-sample.v1", diagnostic_variant=variant)
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")
    properties = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
    device = {
        key: properties[key]
        for key in [
            "multiProcessorCount",
            "warpSize",
            "maxThreadsPerMultiProcessor",
            "major",
            "minor",
        ]
    }
    device["name"] = properties["name"].decode()
    device["single_to_double_precision_perf_ratio"] = xp.cuda.runtime.deviceGetAttribute(
        87, xp.cuda.Device().id
    )
    resources = []
    for kernel, _ in _jit_launch._transport_kernel._cache.values():
        resources.append(
            {
                name: xp.cuda.driver.funcGetAttribute(code, kernel.ptr)
                for name, code in [
                    ("registers_per_thread", 4),
                    ("local_bytes_per_thread", 3),
                    ("max_threads_per_block", 0),
                ]
            }
        )
    for code in _jit_launch._transport_kernel.cached_codes.values():
        audit_path.with_suffix(".cu").write_text(code)
    audit_path.write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.kernel-diagnostic.v1",
                "variant": variant,
                "command": sys.argv,
                "launches": launches,
                "blocks": blocks,
                "device": device,
                "compiled_kernel_resources": resources,
                "threads_per_block": 32 if variant in {"threads32", "both-capacity32"} else 128,
                "interpretation": "Diagnostic only. First request is warmup; hashes/reductions are outside transport timers. Initial capacity 12500 is not a safe bound. Launch timers include dispatch and synchronization; first includes JIT.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
