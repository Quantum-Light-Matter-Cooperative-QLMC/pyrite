"""#907 probe: streamed-block transport population vs kernel time and memory.

Usage: python stream907.py MODE N BLOCK SCRATCH_MIB THREADS BATCH OUT_PREFIX
  MODE  joined | streamed | transport   (BLOCK ignored for joined)
        transport: electron-block transport only, each block discarded
        unscored (probe for populations whose scoring would not fit)
  BATCH electron_batch cap (concurrent continuation histories); 0 = default

Scene (#907 benchmark): W 1 MeV, 10 um slab tilted 45 deg, point beam, scalar
detector at 90 deg, 750-1000 eV CXR window (251 bins), n_families=1, seed 7,
brem electrons N//16, CUDA core pinned. One arm per process: ru_maxrss is
this arm's host peak. Transport kernel GPU time is bracketed per continuation
launch with CUDA events (no host sync added inside the loop). VRAM: CuPy pool
reserved/used peak and driver-level used peak (total - free), sampled 5 ms.
Prints one JSON line; writes OUT_PREFIX.json and OUT_PREFIX.npz (spectra).

Sweep driver (one subprocess per arm, so each arm's ru_maxrss is its own):
  python stream907.py sweep rtx6000 OUTDIR   built-in RTX PRO 6000 arm list
  python stream907.py sweep ARMFILE OUTDIR   lines "MODE N BLOCK SCRATCH THREADS BATCH [K=V...]"
Env PROBE_ROWS sets continuation_rows (with schedule="auto").
"""

import json
import os
import resource
import sys
import threading
import time

os.environ.setdefault("PYRITE_MC_BACKEND", "cuda")

import cupy
import numpy as np


def scene_for(electrons, block, scratch, threads, batch):
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, PointSpot
    from pyrite.config import (
        CxrReflections,
        ExecutionConfig,
        FixedSampling,
        Numerics,
        TransportKernelTuning,
    )
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.units import ureg

    rows = int(os.environ.get("PROBE_ROWS", "0")) or None
    tuning = TransportKernelTuning(
        kernel_threads=threads,
        scratch_mib=scratch,
        electron_batch=batch or None,
        schedule=None if rows is None else "auto",
        continuation_rows=rows,
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_electrons=electrons, brem_electrons=max(1_024, electrons // 16)),
        seed=7,
        energy_bins=EnergyBins().with_cxr_energies(np.linspace(750, 1_000, 251) * ureg.eV),
        cxr_reflections=CxrReflections(n_families=1),
    ).with_kernel_tuning(
        ExecutionConfig(transport_core="cuda", transport=tuning, stream_block_electrons=block)
    )
    return Scene(
        Beam(kinetic_energy=1 * ureg.MeV, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab(
            "tungsten",
            thickness=10 * ureg.um,
            tilt=45 * ureg.degree,
            footprint=Footprint(width=5 * ureg.mm, height=5 * ureg.mm),
        ),
        Detector(observation_angle=90 * ureg.degree),
        numerics=numerics,
    )


class Launches:
    """CUDA-event brackets around every continuation core launch."""

    def __init__(self):
        self.events = []  # (population, start, stop)
        self.drives = []  # per continuation pass: n_active

    def install(self):
        from pyrite.montecarlo.transport import batching, continuation

        real = batching._drive_continuation_rounds
        real_pop = continuation.active_population
        probe = self

        def active_population(*args, **kwargs):
            out = real_pop(*args, **kwargs)
            probe.drives.append(int(out[0]))
            return out

        continuation.active_population = active_population

        def drive(core, *args, **kwargs):
            def timed(launch, *a, **k):
                start, stop = cupy.cuda.Event(), cupy.cuda.Event()
                start.record()
                out = core(launch, *a, **k)
                stop.record()
                probe.events.append((int(launch[1]), start, stop))
                return out

            return real(timed, *args, **kwargs)

        batching._drive_continuation_rounds = drive

    def summary(self):
        cupy.cuda.Device().synchronize()
        pops = np.array([p for p, _, _ in self.events], dtype=np.int64)
        ms = np.array([cupy.cuda.get_elapsed_time(a, b) for _, a, b in self.events])
        return {
            "launches": len(self.events),
            "kernel_s": float(ms.sum() / 1e3),
            "pop_max": int(pops.max(initial=0)),
            "pop_time_weighted": float((pops * ms).sum() / max(ms.sum(), 1e-30)),
            "n_active_per_pass": sorted(set(self.drives)),
            "passes": len(self.drives),
            "launch_pop_ms": [[int(p), round(float(t), 3)] for p, t in zip(pops, ms, strict=True)],
        }


class Memory:
    def __init__(self):
        self.pool = cupy.get_default_memory_pool()
        free, self.total = cupy.cuda.runtime.memGetInfo()
        self.dev_base = self.dev_peak = self.total - free
        self.res_peak = self.used_peak = 0
        self.phase = "transport"
        self.by_phase = {}
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.wait(0.005):
            self._sample()

    def _sample(self):
        free, _ = cupy.cuda.runtime.memGetInfo()
        self.dev_peak = max(self.dev_peak, self.total - free)
        self.res_peak = max(self.res_peak, self.pool.total_bytes())
        self.used_peak = max(self.used_peak, self.pool.used_bytes())
        prev = self.by_phase.get(self.phase, 0)
        self.by_phase[self.phase] = max(prev, self.total - free - self.dev_base)

    def __enter__(self):
        self.res_peak, self.used_peak = 0, 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def transport_only(scene, block):
    """Block transport with every block released unscored; counts only."""
    from pyrite import api
    from pyrite._backend import BACKEND
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.transport_options import transport_tuning

    case = api.build_case(scene, scene.numerics)
    seen = {"rows": 0, "max": 0}

    def consume(segs):
        rows = int(segs["L_ang"].shape[0])
        seen["rows"] += rows
        seen["max"] = max(seen["max"], rows)
        BACKEND.release_memory()

    with transport_tuning(scene.numerics, streaming=True):
        tp = runner._transport_case(
            case, transport_core="cuda", block_electrons=block, block_consumer=consume
        )
    counts = tp["segs"]
    eta = int(counts["n_backscattered"]) / int(counts["Ne"])
    return {"n_segments": seen["rows"], "eta": eta, "max_rows": seen["max"]}


def main():
    mode, n, block, scratch, threads, batch, prefix = sys.argv[1:8]
    n, block, scratch, threads, batch = map(int, (n, block, scratch, threads, batch))
    import pyrite
    from pyrite import api
    from pyrite.montecarlo.runner import streamed_scoring

    pyrite.fetch_tables()
    props = cupy.cuda.runtime.getDeviceProperties(0)
    sms = int(props["multiProcessorCount"])
    # Warm: compile the transport/scoring kernels at this launch shape.
    pyrite.simulate(scene_for(4096, 4096, scratch, threads, batch))
    cupy.get_default_memory_pool().free_all_blocks()

    captured = {}
    real_result = api._scalar_result

    def scalar_result(request, case, xsgen_tables, output, *a, **k):
        captured["out"] = output
        return real_result(request, case, xsgen_tables, output, *a, **k)

    api._scalar_result = scalar_result
    score = {"s": 0.0, "blocks": 0, "max_rows": 0}
    real_add = streamed_scoring.BlockSpectrumScorer.add

    def add(self, segs):
        rows = int(segs["L_ang"].shape[0])
        cupy.cuda.Device().synchronize()
        memory[0]._sample()
        memory[0].phase = "score"
        t = time.perf_counter()
        real_add(self, segs)
        cupy.cuda.Device().synchronize()
        memory[0]._sample()
        memory[0].phase = "transport"
        score["s"] += time.perf_counter() - t
        score["blocks"] += 1
        score["max_rows"] = max(score["max_rows"], rows)

    memory = [None]
    streamed_scoring.BlockSpectrumScorer.add = add
    launches = Launches()
    launches.install()
    scene = scene_for(n, None if mode == "joined" else block, scratch, threads, batch)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    with Memory() as mem:
        memory[0] = mem
        if mode == "transport":
            captured["out"] = transport_only(scene, block)
        else:
            pyrite.simulate(scene)
        cupy.cuda.Device().synchronize()
    wall = time.perf_counter() - t0
    out = captured["out"]
    k = launches.summary()
    if mode == "transport":
        score["max_rows"] = out.pop("max_rows")
    rows = int(out["n_segments"])
    rec = {
        "mode": mode,
        "N": n,
        "block": block if mode == "streamed" else None,
        "scratch_mib": scratch,
        "kernel_threads": threads,
        "electron_batch": batch or None,
        "continuation_rows": int(os.environ.get("PROBE_ROWS", "0")) or 256,
        "device": props["name"].decode(),
        "sms": sms,
        "fp64": os.environ.get("PYRITE_FP64") == "1",
        "wall_s": wall,
        "score_s": score["s"],
        "score_blocks": score["blocks"],
        "max_block_rows": score["max_rows"],
        **k,
        "kernel_ns_per_row": k["kernel_s"] / max(rows, 1) * 1e9,
        "lanes_per_sm_max": k["pop_max"] / sms,
        "n_segments": rows,
        "eta": float(out["eta"]),
        "host_peak_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "host_before_GiB": rss0 / 2**20,
        "vram_driver_peak_GiB": (mem.dev_peak - mem.dev_base) / 2**30,
        "vram_pool_reserved_peak_GiB": mem.res_peak / 2**30,
        "vram_pool_used_peak_GiB": mem.used_peak / 2**30,
        "vram_driver_peak_by_phase_GiB": {k: v / 2**30 for k, v in mem.by_phase.items()},
        "sha": os.environ.get("PROBE_SHA", "?"),
    }
    if mode != "transport":
        spectra = ("spec", "spec_characteristic", "brem_wide")
        np.savez(prefix + ".npz", **{key: np.asarray(out[key]) for key in spectra})
    with open(prefix + ".json", "w") as handle:
        json.dump(rec, handle)
    print("RESULT " + json.dumps(rec), flush=True)


#: RTX PRO 6000 (96 GB, 188 SMs) arms: populations past the 5080's 16k-block
#: ceiling, at the #907 fixed total of 262,144 electrons, plus 5080-matched
#: reference arms (block 16,384) and an fp32 joined reference at 32,768.
RTX6000_ARMS = """
streamed 262144 16384 2048 32 0
streamed 262144 32768 4096 32 0
streamed 262144 65536 8192 32 0
streamed 262144 131072 16384 32 131072
streamed 262144 65536 8192 64 0
streamed 262144 65536 8192 32 0 PROBE_ROWS=1024
streamed 262144 65536 8192 32 0
joined 32768 0 4096 32 0
streamed 32768 8192 4096 32 0
"""


def sweep(arms, outdir):
    import shlex
    import subprocess

    text = RTX6000_ARMS if arms == "rtx6000" else open(arms).read()
    os.makedirs(outdir, exist_ok=True)
    lines = [line.split() for line in text.splitlines() if line.strip() and line[0] != "#"]
    for i, (mode, n, block, scratch, threads, batch, *envs) in enumerate(lines, 1):
        tag = f"{i}-{mode}-{n}-{block}-{scratch}-{threads}-{batch}"
        env = {**os.environ, **dict(e.split("=", 1) for e in envs)}
        cmd = [sys.executable, "-u", os.path.abspath(__file__), mode, n, block, scratch, threads]
        cmd += [batch, os.path.join(outdir, tag)]
        print("ARM", tag, " ".join(envs), flush=True)
        with open(os.path.join(outdir, tag + ".log"), "w") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        tail = open(os.path.join(outdir, tag + ".log")).read().splitlines()[-3:]
        print(f"rc={rc}", *(t for t in tail if t.startswith(("RESULT", "cupy", "Error"))), flush=True)
        print(shlex.join(cmd), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep(sys.argv[2], sys.argv[3])
    else:
        main()
