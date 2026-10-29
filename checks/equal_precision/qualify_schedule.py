"""Qualify the CUDA transport schedule on production-default runner cases (#799).

One process measures one (material, histories, seed) case under the schedule
named by ``PYRITE_MC_TRANSPORT_SCHEDULE``, which the runner reads at import
like any production run. Cases are built by the campaign defaults: ELSEPA
elastic, ``auto`` inelastic (shell soft/hard on covered materials, continuous
stopping otherwise), coupled BremsLib radiation, midpoint transport. A full
same-workload warmup at ``seed + 1000000`` precedes the measured request.

Per request it records end-to-end ``run_case`` wall time, the runner's
transport interval, peak CuPy pool bytes above the pre-request level, process
peak RSS, the schedule the transport resolved (from the INFO log) and a
SHA-256 per returned array. Diagnostic timings: one process per sample, no
profiler attached.
"""

import argparse
import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import cupy as cp
import numpy as np

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner

SCENES = {  # material, thickness [angstrom], energy [keV]
    "silicon": ("silicon", 1.0e6, 300.0),
    "tungsten": ("tungsten", 1.0e5, 300.0),
    "pbwo4": ("pbwo4", 1.0e5, 300.0),
}


class PeakPoolHook(cp.cuda.memory_hook.MemoryHook):
    """Live and peak bytes handed out by the CuPy device pool."""

    name = "PeakPoolHook"

    def __init__(self):
        self.live = 0
        self.peak = 0
        self.sizes = {}

    def malloc_postprocess(self, **kwargs):
        self.sizes[kwargs["mem_ptr"]] = kwargs["mem_size"]
        self.live += kwargs["mem_size"]
        self.peak = max(self.peak, self.live)

    def free_postprocess(self, **kwargs):
        self.live -= self.sizes.pop(kwargs["mem_ptr"], kwargs["mem_size"])


class ScheduleLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def build_case(scene, histories, seed):
    material, thickness_ang, energy_keV = SCENES[scene]
    sweep = Sweep(
        material=material,
        thickness_ang=thickness_ang,
        beam=BeamSpec(energy_keV=energy_keV),
        n_electrons=histories,
        n_electrons_brem=histories,
    )
    case = dict(build_cases(sweep)[0])
    case["seed"] = seed
    return case


def digest(out):
    hashes = {}
    for key in sorted(out):
        value = out[key]
        if key.startswith("_t_") or not isinstance(value, np.ndarray) or value.dtype == object:
            continue
        hashes[key] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return hashes


def request(case, hook, log):
    log.messages.clear()
    cp.cuda.Device().synchronize()
    before = hook.live
    hook.peak = hook.live
    start = time.perf_counter()
    out = runner.run_case(case, record_timing=True, transport_core="cuda")
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - start
    return {
        "wall_s": wall,
        "transport_s": out.get("_t_transport"),
        "peak_pool_bytes": hook.peak - before,
        "schedule_log": sorted(set(m for m in log.messages if "transport schedule" in m)),
        "digest": digest(out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=sorted(SCENES))
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log = ScheduleLog()
    logger = logging.getLogger("pyrite.montecarlo.transport.continuation")
    logger.addHandler(log)
    logger.setLevel(logging.INFO)
    hook = PeakPoolHook()
    with hook:
        warmup = request(build_case(args.scene, args.histories, args.seed + 1_000_000), hook, log)
        measured = request(build_case(args.scene, args.histories, args.seed), hook, log)
    record = {
        "scene": args.scene,
        "histories": args.histories,
        "seed": args.seed,
        "schedule": os.environ.get("PYRITE_MC_TRANSPORT_SCHEDULE", ""),
        "warmup_wall_s": warmup["wall_s"],
        "measured": measured,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
