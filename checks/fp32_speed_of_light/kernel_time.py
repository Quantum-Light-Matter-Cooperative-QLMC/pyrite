"""Kernel-only time of the production exact transport kernel on a #821 scene (#1013).

Wraps ``_jit_launch.transport_kernel_variant`` so every launch of the kernel
that ``pyrite_bench.py`` runs is bracketed by CUDA events, then runs the
harness unchanged (``--kernel specialized``). The kernel and its arguments are
the production ones for ``--transport-precision``; only the timing is added.
Writes the kernel seconds and retained rows of the measured (warm) request:

    uv run python checks/fp32_speed_of_light/kernel_time.py --stats out.stats.json \\
        si-3MeV-1mm --histories 7000 --seed 1 --rows-per-round 1024 --device-staging \\
        --transport-precision mixed --no-audit --output out.json

Unlike ``precision_variants.py`` (#831, which rewrites the generated source),
this measures the shipped kernels, so the cost of a kernel change is compared
tree against tree at the same arguments.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Timed:
    """A launched kernel whose calls record CUDA events under the current seed."""

    def __init__(self, kernel, events, state):
        self.kernel, self.events, self.state = kernel, events, state

    def __call__(self, grid, block, args, *rest, **kw):
        import cupy

        start, stop = cupy.cuda.Event(), cupy.cuda.Event()
        start.record()
        self.kernel(grid, block, args, *rest, **kw)
        stop.record()
        self.events.setdefault(self.state["seed"], []).append((start, stop))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    args, bench_args = parser.parse_known_args()

    import cupy

    import pyrite.montecarlo.transport as transport
    from pyrite.montecarlo.transport import _jit_launch

    events, state, wrapped = {}, {"seed": None}, {}
    original = _jit_launch.transport_kernel_variant

    def timed_variant(switches, precision="fp64"):
        key = (switches, precision)
        if key not in wrapped:
            wrapped[key] = _Timed(original(switches, precision), events, state)
        return wrapped[key]

    _jit_launch.transport_kernel_variant = timed_variant
    simulate = transport._simulate_trajectories

    def tagged(*a, **kw):
        state["seed"] = int(kw["seed"])
        return simulate(*a, **kw)

    transport._simulate_trajectories = tagged
    sys.argv = [str(HERE.parent / "geant4_performance" / "pyrite_bench.py"), *bench_args]
    sys.argv += ["--kernel", "specialized"]
    runpy.run_path(sys.argv[0], run_name="__main__")
    cupy.cuda.Device().synchronize()
    per_seed = {
        str(seed): {
            "launches": len(ev),
            "kernel_s": sum(cupy.cuda.get_elapsed_time(s, e) for s, e in ev) / 1e3,
        }
        for seed, ev in events.items()
    }
    out = {"variants": [f"{s}:{p}" for s, p in wrapped], "per_seed": per_seed}
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
