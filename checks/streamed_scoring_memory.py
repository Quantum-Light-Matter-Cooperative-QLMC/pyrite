"""Peak memory and wall time of joined vs live streamed scoring (#611).

Usage:
  streamed_scoring_memory.py run MODE NE BLOCK OUT_JSON [MATERIAL [E0_KEV [CORE [THICKNESS_ANG]]]]
  streamed_scoring_memory.py compare REF_JSON OTHER_JSON...

MODE: ``joined`` (``run_case``: full trajectory, then spectra; BLOCK ignored)
or ``streamed`` (``run_case_streamed`` with BLOCK electrons per block).
CORE (default ``per-electron``; ``cuda`` on the GPU box) is pinned for both
arms, because ``auto`` can resolve to the block-incapable lockstep core.
One arm per process, so ``ru_maxrss`` is that arm's host peak. Device peak is
the CuPy pool's reserved and used bytes, sampled every 5 ms above the
pre-run level (reserved bytes are never returned, so they include
fragmentation; used bytes are live arrays).

The case is a production-style coupled BremsLib midpoint run on a slab
(1 um unless THICKNESS_ANG is given) with explicit line and continuum bins (no case-local grid resolution).
"""

import json
import os
import resource
import sys
import threading
import time
from dataclasses import replace

import numpy as np

SPECTRA = ("spec", "spec_characteristic", "brem_wide")


def _case(material, ne, e0_kev, thickness_ang=1.0e4):
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    numerics = pr.Numerics(
        cxr_trials=ne,
        brem_trials=ne,
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.linspace(1000.0, 20000.0, 400), brem=np.arange(1000.0, e0_kev * 1e3, 1000.0)
        ),
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
    )
    target = pr.geometry.Slab(
        material,
        thickness_ang=thickness_ang,
        tilt_deg=30.0,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=e0_kev, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        target,
        pr.Detector(),
        physics=physics,
    )
    case = api.build_case(scene, numerics)
    assert case.get("line_grid_policy") is None, "benchmark needs a final line grid"
    return replace(case, E_cut_lines_keV=5.0, E_cut_brem_keV=5.0)


class _PoolPeak:
    """Sample CuPy pool reserved bytes on a thread; no-op without CuPy."""

    def __init__(self):
        try:
            import cupy

            self.pool = cupy.get_default_memory_pool()
        except Exception:
            self.pool = None
        self.base = self.peak = 0 if self.pool is None else self.pool.total_bytes()
        self.used_base = self.used_peak = 0 if self.pool is None else self.pool.used_bytes()
        self._stop = threading.Event()

    def __enter__(self):
        if self.pool is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.005):
            self.peak = max(self.peak, self.pool.total_bytes())
            self.used_peak = max(self.used_peak, self.pool.used_bytes())

    def __exit__(self, *exc):
        self._stop.set()
        if self.pool is not None:
            self._thread.join()
            self.peak = max(self.peak, self.pool.total_bytes())
            self.used_peak = max(self.used_peak, self.pool.used_bytes())

    @property
    def above_base(self):
        return int(self.peak - self.base)


def run(
    mode,
    ne,
    block,
    out_path,
    material="silicon",
    e0_kev=300.0,
    core="per-electron",
    thickness_ang=1.0e4,
):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed

    case = _case(material, int(ne), float(e0_kev), float(thickness_ang))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    with _PoolPeak() as pool:
        if mode == "joined":
            out = runner.run_case(case, transport_core=core)
        elif mode == "streamed":
            out = run_case_streamed(case, int(block), transport_core=core)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    wall = time.perf_counter() - t0
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "mode": mode,
        "Ne": int(ne),
        "block": int(block) if mode == "streamed" else None,
        "material": material,
        "E0_keV": float(e0_kev),
        "core": core,
        "thickness_ang": float(thickness_ang),
        "wall_s": wall,
        "host_peak_MiB": rss_peak / 1024,
        "host_peak_before_run_MiB": rss_before / 1024,
        "device_pool_peak_above_base_MiB": pool.above_base / 2**20,
        "device_used_peak_above_base_MiB": (pool.used_peak - pool.used_base) / 2**20,
        "fp64": os.environ.get("PYRITE_FP64") == "1",
        "n_segments": int(out["n_segments"]),
        "eta": float(out["eta"]),
        **{key: np.asarray(out[key]).tolist() for key in SPECTRA},
    }
    with open(out_path, "w") as handle:
        json.dump(record, handle)
    summary = {k: v for k, v in record.items() if k not in SPECTRA}
    print(json.dumps(summary))


def compare(ref_path, *others):
    ref = json.load(open(ref_path))
    worst = 0.0
    for path in others:
        other = json.load(open(path))
        assert other["n_segments"] == ref["n_segments"] and other["eta"] == ref["eta"], path
        for key in SPECTRA:
            a, b = np.asarray(other[key]), np.asarray(ref[key])
            scale = np.maximum(np.abs(b), np.finfo(float).tiny)
            rel = float(np.max(np.abs(a - b) / scale, initial=0.0))
            worst = max(worst, rel)
            print(f"{path} {key}: max rel diff {rel:.3e}")
    print(f"worst max rel diff {worst:.3e}")
    return 0 if worst <= 1e-12 else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif sys.argv[1] == "compare":
        sys.exit(compare(*sys.argv[2:]))
    else:
        raise SystemExit(__doc__)
