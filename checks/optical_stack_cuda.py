"""Real CUDA parity and throughput of the O(N) many-layer stack solve (#962).

Run on a clean committed tree with a CUDA device:
``uv run --extra nvidia pyrite-dev validation-records -- checks/optical_stack_cuda.py``.
Missing CUDA fails; there is no skip or CPU fallback. Prints JSON evidence.

Parity compares the complex top-wave amplitude and interface response of the
NumPy recursion with the fused kernels, and the recursion with the dense solve
where that fits. Throughput is the best of three warm runs at each layer count;
set ``OPENBLAS_NUM_THREADS`` so the dense CPU column is not oversubscribed.
Validation: optical-stack-fast-solver (measurement only).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime

import cupy
import numpy as np

from pyrite.montecarlo.spectrum import transition_radiation as tr

PAIRS = 256
PERIOD = [(2.1 + 0.0j, 800.0), (6.0 + 0.01j, 500.0)]
TOLERANCE = 1e-10


def _inputs(pairs=PAIRS):
    energy = np.linspace(1.0, 3.0, pairs)
    theta = np.linspace(0.0, 1.3, pairs)
    n = np.stack([np.sin(theta), np.zeros(pairs), np.cos(theta)], axis=-1)
    return energy, n, np.broadcast_to([0.05, 0.0, 0.7], n.shape)


def _device_name():
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"]
    return subprocess.check_output(query, text=True).strip()


def _peak_difference(device, host):
    return float(np.max(np.abs(cupy.asnumpy(device) - host)) / np.max(np.abs(host)))


def _parity(periods):
    energy, n, v = _inputs()
    layers = PERIOD * periods
    _, host = tr.planar_transition_radiation(energy, n, v, layers, return_amplitude=True)
    _, device = tr.planar_transition_radiation(energy, n, v, layers, xp=cupy, return_amplitude=True)
    host_stack = tr.interface_response(energy, n, layers, 1.0)
    device_stack = tr.interface_response(energy, n, layers, 1.0, xp=cupy)
    return {
        "layers": 2 * periods,
        "amplitude": _peak_difference(device, host),
        "response": _peak_difference(device_stack.response, host_stack.response),
    }


def _warm(run, repeats=3):
    run()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def _throughput(periods, *, dense):
    energy, n, v = _inputs()
    layers = PERIOD * periods

    def run(xp):
        result = tr.planar_transition_radiation(energy, n, v, layers, xp=xp)
        float(result.sum())

    row = {"layers": 2 * periods}
    saved = tr.FAST_STACK_MIN_LAYERS
    try:
        tr.FAST_STACK_MIN_LAYERS = 1
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.cuda.runtime.deviceSynchronize()
        row["fast_gpu_s"] = _warm(lambda: run(cupy))
        row["peak_pool_bytes"] = int(cupy.get_default_memory_pool().total_bytes())
        if periods <= 500:
            row["fast_cpu_s"] = _warm(lambda: run(np), 1)
        if dense:
            tr.FAST_STACK_MIN_LAYERS = 10**9
            row["dense_cpu_s"] = _warm(lambda: run(np), 1)
            row["dense_gpu_s"] = _warm(lambda: run(cupy), 1)
    finally:
        tr.FAST_STACK_MIN_LAYERS = saved
    return row


def main() -> int:
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("no CUDA device")
    parity = [_parity(periods) for periods in (10, 150, 1000)]
    timing = [
        _throughput(periods, dense=periods <= 25) for periods in (5, 25, 50, 100, 500, 2500, 10000)
    ]
    print(
        json.dumps(
            {
                "validation_ids": ["optical-stack-fast-solver"],
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "device": _device_name(),
                "cupy": cupy.__version__,
                "pairs": PAIRS,
                "tolerance": TOLERANCE,
                "parity_peak_normalized": parity,
                "timing": timing,
            },
            indent=1,
        )
    )
    worst = max(max(row["amplitude"], row["response"]) for row in parity)
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
